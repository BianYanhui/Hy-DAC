"""
KV-Cache 复用模块 (KV-Cache Reuse Module)

该模块实现了KV-Cache的复用机制。
在设备离线后进行任务重分配时，能够复用已有的KV-Cache，
只对新分配的任务（没有缓存的Heads）进行重计算。
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import threading
import json
from pathlib import Path
from safetensors import safe_open


@dataclass
class KVCacheEntry:
    """单个Head的KV-Cache条目"""
    head_id: int
    k_cache: Optional[torch.Tensor] = None  # [seq_len, head_dim]
    v_cache: Optional[torch.Tensor] = None  # [seq_len, head_dim]
    seq_length: int = 0
    is_valid: bool = False
    last_update_time: float = 0.0
    
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """更新KV-Cache"""
        self.k_cache = k
        self.v_cache = v
        self.seq_length = k.shape[0] if k is not None else 0
        self.is_valid = True
        self.last_update_time = time.time()
    
    def clear(self):
        """清空缓存"""
        self.k_cache = None
        self.v_cache = None
        self.seq_length = 0
        self.is_valid = False


class SimpleLlamaAttention(nn.Module):
    """
    简化的Llama Attention模块，用于演示KV-Cache计算
    """
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # GQA的重复因子
        
        # 简化的投影层
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def compute_kv_for_heads(self, x: torch.Tensor, 
                              head_ids: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算指定Heads的KV值
        
        Args:
            x: 输入张量 [batch, seq_len, dim]
            head_ids: 要计算的KV Head ID列表
            
        Returns:
            {head_id: (k, v)}
        """
        batch, seq_len, _ = x.shape
        
        # 计算所有的K和V
        k = self.wk(x)  # [batch, seq_len, n_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, n_kv_heads * head_dim]
        
        # 重塑为 [batch, seq_len, n_kv_heads, head_dim]
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # 提取指定的heads
        result = {}
        for head_id in head_ids:
            if 0 <= head_id < self.n_kv_heads:
                result[head_id] = (
                    k[:, :, head_id, :].clone(),  # [batch, seq_len, head_dim]
                    v[:, :, head_id, :].clone()
                )
        
        return result


class KVCacheManager:
    """
    KV-Cache管理器
    
    管理单个设备上的所有KV-Cache，支持：
    - 缓存存储和检索
    - 选择性重计算（只计算缺失的Heads）
    - 缓存迁移（用于任务重分配）
    """
    
    def __init__(self, device_id: str, 
                 n_layers: int = 16, 
                 n_kv_heads: int = 8,
                 head_dim: int = 64):
        """
        初始化KV-Cache管理器
        
        Args:
            device_id: 设备ID
            n_layers: Transformer层数
            n_kv_heads: KV Head数量
            head_dim: 每个Head的维度
        """
        self.device_id = device_id
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        # 分配给该设备的Heads
        self.assigned_heads: List[int] = []
        
        # KV-Cache存储: {layer_id: {head_id: KVCacheEntry}}
        self.cache: Dict[int, Dict[int, KVCacheEntry]] = {
            layer: {} for layer in range(n_layers)
        }
        
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_computations = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def set_assigned_heads(self, head_ids: List[int]):
        """设置分配给该设备的Heads"""
        with self.lock:
            self.assigned_heads = sorted(head_ids)
    
    def add_heads(self, new_head_ids: List[int]):
        """添加新的Heads（用于任务重分配后）"""
        with self.lock:
            for head_id in new_head_ids:
                if head_id not in self.assigned_heads:
                    self.assigned_heads.append(head_id)
            self.assigned_heads.sort()
    
    def get_assigned_heads(self) -> List[int]:
        """获取分配的Heads"""
        return self.assigned_heads.copy()
    
    def _has_cache_unlocked(self, layer_id: int, head_id: int) -> bool:
        """检查是否有指定Head的缓存（内部使用，不加锁）"""
        if layer_id in self.cache and head_id in self.cache[layer_id]:
            return self.cache[layer_id][head_id].is_valid
        return False
    
    def has_cache(self, layer_id: int, head_id: int) -> bool:
        """检查是否有指定Head的缓存"""
        with self.lock:
            return self._has_cache_unlocked(layer_id, head_id)
    
    def get_cache(self, layer_id: int, head_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取指定Head的KV-Cache"""
        with self.lock:
            if self._has_cache_unlocked(layer_id, head_id):
                entry = self.cache[layer_id][head_id]
                self.cache_hits += 1
                return (entry.k_cache, entry.v_cache)
            self.cache_misses += 1
            return None
    
    def set_cache(self, layer_id: int, head_id: int, 
                  k: torch.Tensor, v: torch.Tensor):
        """设置指定Head的KV-Cache"""
        with self.lock:
            if layer_id not in self.cache:
                self.cache[layer_id] = {}
            
            if head_id not in self.cache[layer_id]:
                self.cache[layer_id][head_id] = KVCacheEntry(head_id=head_id)
            
            self.cache[layer_id][head_id].update(k, v)
    
    def get_heads_needing_computation(self) -> List[int]:
        """
        获取需要计算的Heads（没有缓存的Heads）
        
        Returns:
            需要重新计算KV-Cache的Head ID列表
        """
        heads_needing_computation = []
        
        with self.lock:
            for head_id in self.assigned_heads:
                # 检查是否所有层都有缓存
                has_all_layers = all(
                    self._has_cache_unlocked(layer_id, head_id)
                    for layer_id in range(self.n_layers)
                )
                
                if not has_all_layers:
                    heads_needing_computation.append(head_id)
        
        return heads_needing_computation
    
    def get_heads_with_cache(self) -> List[int]:
        """获取有缓存的Heads"""
        heads_with_cache = []
        
        with self.lock:
            for head_id in self.assigned_heads:
                # 检查是否所有层都有缓存
                has_all_layers = all(
                    self._has_cache_unlocked(layer_id, head_id)
                    for layer_id in range(self.n_layers)
                )
                
                if has_all_layers:
                    heads_with_cache.append(head_id)
        
        return heads_with_cache
    
    def clear_cache_for_heads(self, head_ids: List[int]):
        """清除指定Heads的缓存"""
        with self.lock:
            for layer_id in self.cache:
                for head_id in head_ids:
                    if head_id in self.cache[layer_id]:
                        self.cache[layer_id][head_id].clear()
    
    def clear_all_cache(self):
        """清除所有缓存"""
        with self.lock:
            for layer_id in self.cache:
                for entry in self.cache[layer_id].values():
                    entry.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_cache_statistics(self) -> Dict:
        """获取缓存统计信息"""
        with self.lock:
            total_entries = 0
            valid_entries = 0
            total_memory = 0
            
            for layer_id in self.cache:
                for entry in self.cache[layer_id].values():
                    total_entries += 1
                    if entry.is_valid:
                        valid_entries += 1
                        if entry.k_cache is not None:
                            total_memory += entry.k_cache.numel() * entry.k_cache.element_size()
                        if entry.v_cache is not None:
                            total_memory += entry.v_cache.numel() * entry.v_cache.element_size()
            
            hit_rate = (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0
            )
            
            return {
                "device_id": self.device_id,
                "assigned_heads": self.assigned_heads,
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate_percent": hit_rate,
                "memory_bytes": total_memory,
                "memory_mb": total_memory / (1024 * 1024)
            }


class KVCacheReuseEngine:
    """
    KV-Cache复用引擎
    
    协调多个设备的KV-Cache计算和复用，支持：
    - 初始KV-Cache计算
    - 设备离线后的选择性重计算
    - 与全量重计算的性能对比
    """
    
    def __init__(self, 
                 model_params_path: str,
                 device: str = "cpu"):
        """
        初始化KV-Cache复用引擎
        
        Args:
            model_params_path: 模型参数文件路径 (params.json)
            device: 计算设备
        """
        # 加载模型参数
        with open(model_params_path, 'r') as f:
            params = json.load(f)
        
        self.dim = params['dim']
        self.n_layers = params['n_layers']
        self.n_heads = params['n_heads']
        self.n_kv_heads = params['n_kv_heads']
        self.head_dim = self.dim // self.n_heads
        self.device = device
        
        # 创建简化的attention模块用于KV计算
        self.attention = SimpleLlamaAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads
        )
        
        # 设备的KV-Cache管理器
        self.cache_managers: Dict[str, KVCacheManager] = {}
        
        # 计算统计
        self.computation_times: List[float] = []
        
        print(f"[KV-Cache Engine] Initialized with {self.n_kv_heads} KV heads, "
              f"{self.n_layers} layers, head_dim={self.head_dim}")
    
    def register_device(self, device_id: str, assigned_heads: List[int]):
        """注册设备并分配Heads"""
        manager = KVCacheManager(
            device_id=device_id,
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim
        )
        manager.set_assigned_heads(assigned_heads)
        self.cache_managers[device_id] = manager
        print(f"[KV-Cache Engine] Registered {device_id} with heads {assigned_heads}")
    
    def compute_kv_cache(self, 
                         input_ids: torch.Tensor,
                         device_id: str,
                         head_ids: Optional[List[int]] = None,
                         force_recompute: bool = False) -> Tuple[float, int]:
        """
        计算KV-Cache
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            device_id: 设备ID
            head_ids: 要计算的Head IDs，如果不指定则计算所有分配的heads
            force_recompute: 是否强制重计算（忽略缓存）
            
        Returns:
            (computation_time, num_heads_computed): 计算时间和计算的Head数量
        """
        if device_id not in self.cache_managers:
            raise ValueError(f"Unknown device: {device_id}")
        
        manager = self.cache_managers[device_id]
        
        # 确定需要计算的heads
        if head_ids is None:
            if force_recompute:
                heads_to_compute = manager.get_assigned_heads()
            else:
                heads_to_compute = manager.get_heads_needing_computation()
        else:
            heads_to_compute = head_ids
        
        if not heads_to_compute:
            return (0.0, 0)
        
        # 创建随机嵌入（在实际应用中应该用真实的embedding）
        batch_size, seq_len = input_ids.shape
        x = torch.randn(batch_size, seq_len, self.dim)
        
        start_time = time.time()
        
        # 模拟逐层计算KV-Cache
        for layer_id in range(self.n_layers):
            # 计算指定heads的KV
            kv_results = self.attention.compute_kv_for_heads(x, heads_to_compute)
            
            # 存储到缓存
            for head_id, (k, v) in kv_results.items():
                manager.set_cache(layer_id, head_id, k, v)
            
            # 模拟层间的计算延迟
            time.sleep(0.001)  # 1ms per layer
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return (computation_time, len(heads_to_compute))
    
    def compute_all_devices(self, 
                            input_ids: torch.Tensor,
                            force_recompute: bool = False) -> Dict[str, Tuple[float, int]]:
        """
        计算所有设备的KV-Cache
        
        Args:
            input_ids: 输入token IDs
            force_recompute: 是否强制重计算
            
        Returns:
            {device_id: (time, num_heads)}
        """
        results = {}
        
        for device_id in self.cache_managers:
            time_taken, num_heads = self.compute_kv_cache(
                input_ids, device_id, force_recompute=force_recompute
            )
            results[device_id] = (time_taken, num_heads)
        
        return results
    
    def handle_device_offline(self, 
                              offline_device_id: str,
                              target_device_id: str,
                              reassigned_heads: List[int],
                              input_ids: torch.Tensor) -> Dict:
        """
        处理设备离线 - 使用KV-Cache复用策略
        
        Args:
            offline_device_id: 离线设备ID
            target_device_id: 接收任务的设备ID
            reassigned_heads: 被重分配的Head IDs
            input_ids: 输入token IDs
            
        Returns:
            包含计算统计的字典
        """
        if target_device_id not in self.cache_managers:
            raise ValueError(f"Unknown target device: {target_device_id}")
        
        target_manager = self.cache_managers[target_device_id]
        
        # 记录重分配前的状态
        original_heads = target_manager.get_assigned_heads()
        heads_with_cache = target_manager.get_heads_with_cache()
        
        # 添加新的heads到目标设备
        target_manager.add_heads(reassigned_heads)
        
        # 计算需要重新计算的heads（新分配的，没有缓存）
        heads_needing_computation = reassigned_heads  # 新分配的heads肯定没有缓存
        heads_reused = heads_with_cache  # 原有的heads可以复用
        
        print(f"\n[KV-Cache Reuse] Device {offline_device_id} went offline")
        print(f"  - Reassigning heads {reassigned_heads} to {target_device_id}")
        print(f"  - {target_device_id} can reuse cache for heads: {heads_reused}")
        print(f"  - Need to recompute heads: {heads_needing_computation}")
        
        # 只计算新分配的heads
        start_time = time.time()
        recompute_time, num_recomputed = self.compute_kv_cache(
            input_ids, target_device_id, head_ids=heads_needing_computation
        )
        total_time = time.time() - start_time
        
        return {
            "strategy": "kv_cache_reuse",
            "offline_device": offline_device_id,
            "target_device": target_device_id,
            "original_heads": original_heads,
            "reassigned_heads": reassigned_heads,
            "heads_reused": heads_reused,
            "heads_recomputed": heads_needing_computation,
            "num_heads_reused": len(heads_reused),
            "num_heads_recomputed": num_recomputed,
            "recompute_time": recompute_time,
            "total_time": total_time
        }
    
    def handle_device_offline_full_recompute(self,
                                              offline_device_id: str,
                                              input_ids: torch.Tensor) -> Dict:
        """
        处理设备离线 - 传统的全量重计算策略
        
        所有设备清除缓存，完全重新计算
        
        Args:
            offline_device_id: 离线设备ID
            input_ids: 输入token IDs
            
        Returns:
            包含计算统计的字典
        """
        print(f"\n[Full Recompute] Device {offline_device_id} went offline")
        print("  - Clearing all caches and recomputing everything...")
        
        # 清除所有设备的缓存
        total_heads_to_compute = 0
        for device_id, manager in self.cache_managers.items():
            if device_id != offline_device_id:
                manager.clear_all_cache()
                total_heads_to_compute += len(manager.get_assigned_heads())
        
        # 重新计算所有设备的KV-Cache
        start_time = time.time()
        results = self.compute_all_devices(input_ids, force_recompute=True)
        total_time = time.time() - start_time
        
        total_compute_time = sum(t for t, _ in results.values())
        total_heads_computed = sum(n for _, n in results.values())
        
        return {
            "strategy": "full_recompute",
            "offline_device": offline_device_id,
            "devices_recomputed": list(results.keys()),
            "total_heads_computed": total_heads_computed,
            "per_device_times": {k: v[0] for k, v in results.items()},
            "total_compute_time": total_compute_time,
            "total_time": total_time
        }
    
    def get_all_cache_statistics(self) -> Dict[str, Dict]:
        """获取所有设备的缓存统计"""
        return {
            device_id: manager.get_cache_statistics()
            for device_id, manager in self.cache_managers.items()
        }
    
    def print_cache_status(self):
        """打印缓存状态"""
        print("\n" + "="*70)
        print("KV-Cache Status")
        print("="*70)
        
        for device_id, manager in self.cache_managers.items():
            stats = manager.get_cache_statistics()
            print(f"\n  {device_id}:")
            print(f"    Assigned Heads: {stats['assigned_heads']}")
            print(f"    Cached Heads: {manager.get_heads_with_cache()}")
            print(f"    Needs Computation: {manager.get_heads_needing_computation()}")
            print(f"    Cache Hit Rate: {stats['hit_rate_percent']:.1f}%")
            print(f"    Memory Usage: {stats['memory_mb']:.2f} MB")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # 测试代码
    print("Testing KV-Cache Reuse Module")
    
    # 创建测试参数文件路径
    import tempfile
    import os
    
    # 创建临时参数文件
    test_params = {
        "dim": 2048,
        "n_layers": 4,  # 使用较少的层进行测试
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim_multiplier": 1.5,
        "multiple_of": 256,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "use_scaled_rope": True
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_params, f)
        temp_params_path = f.name
    
    try:
        # 创建KV-Cache复用引擎
        engine = KVCacheReuseEngine(temp_params_path)
        
        # 注册设备
        engine.register_device("Device_0", [0, 1])
        engine.register_device("Device_1", [2, 3])
        engine.register_device("Device_2", [4, 5])
        engine.register_device("Device_3", [6, 7])
        
        # 初始计算
        print("\nInitial KV-Cache computation...")
        input_ids = torch.randint(0, 1000, (1, 32))  # batch=1, seq=32
        results = engine.compute_all_devices(input_ids)
        
        for device_id, (time_taken, num_heads) in results.items():
            print(f"  {device_id}: computed {num_heads} heads in {time_taken*1000:.2f}ms")
        
        engine.print_cache_status()
        
        # 模拟设备离线 - KV-Cache复用策略
        print("\n" + "="*70)
        print("Simulating Device_1 going offline (KV-Cache Reuse Strategy)")
        print("="*70)
        
        reuse_result = engine.handle_device_offline(
            offline_device_id="Device_1",
            target_device_id="Device_3",
            reassigned_heads=[2, 3],
            input_ids=input_ids
        )
        
        print(f"\nKV-Cache Reuse Result:")
        print(f"  Heads reused: {reuse_result['num_heads_reused']}")
        print(f"  Heads recomputed: {reuse_result['num_heads_recomputed']}")
        print(f"  Recompute time: {reuse_result['recompute_time']*1000:.2f}ms")
        
        engine.print_cache_status()
        
    finally:
        os.unlink(temp_params_path)
    
    print("\nTest completed!")
