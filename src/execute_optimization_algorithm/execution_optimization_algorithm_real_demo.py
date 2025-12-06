"""
真实模型分布式推理优化Demo (Real Model Distributed Inference Optimization Demo)

该Demo使用真实的 Llama-3.2-1B 模型进行 KV-Cache 计算和性能对比。
验证 KV-Cache 复用策略在设备离线场景下的优化效果。

使用方法：
    python execution_optimization_algorithm_real_demo.py
"""

import os
import sys
import time
import json
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# 导入本地模块
from llama_model import LlamaModelLoader, LlamaModel, LlamaConfig


@dataclass
class DeviceKVCache:
    """设备的KV-Cache存储"""
    device_id: str
    assigned_heads: List[int] = field(default_factory=list)
    # {layer_id: {head_id: (k, v)}}
    cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    is_online: bool = True
    
    def has_cache_for_head(self, head_id: int, n_layers: int) -> bool:
        """检查是否有指定head的完整缓存"""
        for layer_id in range(n_layers):
            if layer_id not in self.cache:
                return False
            if head_id not in self.cache[layer_id]:
                return False
        return True
    
    def get_cached_heads(self, n_layers: int) -> List[int]:
        """获取有完整缓存的heads"""
        cached = []
        for head_id in self.assigned_heads:
            if self.has_cache_for_head(head_id, n_layers):
                cached.append(head_id)
        return cached
    
    def get_heads_needing_computation(self, n_layers: int) -> List[int]:
        """获取需要计算的heads"""
        return [h for h in self.assigned_heads 
                if not self.has_cache_for_head(h, n_layers)]
    
    def set_cache(self, layer_id: int, head_id: int, k: torch.Tensor, v: torch.Tensor):
        """设置缓存"""
        if layer_id not in self.cache:
            self.cache[layer_id] = {}
        self.cache[layer_id][head_id] = (k.clone(), v.clone())
    
    def clear_cache(self):
        """清除所有缓存"""
        self.cache = {}
    
    def add_heads(self, new_heads: List[int]):
        """添加新的heads"""
        for h in new_heads:
            if h not in self.assigned_heads:
                self.assigned_heads.append(h)
        self.assigned_heads.sort()


class RealModelKVCacheEngine:
    """
    真实模型 KV-Cache 引擎
    
    使用真实的 Llama 模型计算 KV-Cache
    """
    
    def __init__(self, 
                 model_path: str,
                 params_path: str,
                 device: str = "cpu"):
        """
        初始化引擎
        
        Args:
            model_path: model.safetensors 路径
            params_path: params.json 路径
            device: 计算设备
        """
        self.device = device
        
        # 加载模型
        print("\n" + "="*60)
        print("Loading Real Llama-3.2-1B Model...")
        print("="*60)
        
        self.loader = LlamaModelLoader(model_path, params_path, device)
        self.model = self.loader.load_model()
        self.config = self.loader.config
        
        # 设备管理
        self.devices: Dict[str, DeviceKVCache] = {}
        
        print(f"\n[Engine] Model ready with {self.config.n_kv_heads} KV heads, "
              f"{self.config.n_layers} layers")
    
    def register_device(self, device_id: str, assigned_heads: List[int]):
        """注册设备"""
        self.devices[device_id] = DeviceKVCache(
            device_id=device_id,
            assigned_heads=sorted(assigned_heads),
            cache={},
            is_online=True
        )
        print(f"[Engine] Registered {device_id} with heads {assigned_heads}")
    
    def compute_kv_cache_for_device(self, 
                                     tokens: torch.Tensor,
                                     device_id: str,
                                     head_ids: Optional[List[int]] = None,
                                     force_recompute: bool = False) -> Tuple[float, int]:
        """
        使用真实模型为设备计算 KV-Cache
        
        Args:
            tokens: 输入 token IDs [batch, seq_len]
            device_id: 设备ID
            head_ids: 要计算的 head IDs，None 表示计算需要计算的
            force_recompute: 是否强制重计算
            
        Returns:
            (computation_time, num_heads_computed)
        """
        if device_id not in self.devices:
            raise ValueError(f"Unknown device: {device_id}")
        
        device_cache = self.devices[device_id]
        
        # 确定需要计算的 heads
        if head_ids is None:
            if force_recompute:
                heads_to_compute = device_cache.assigned_heads
            else:
                heads_to_compute = device_cache.get_heads_needing_computation(self.config.n_layers)
        else:
            heads_to_compute = head_ids
        
        if not heads_to_compute:
            return (0.0, 0)
        
        # 使用真实模型计算 KV-Cache
        tokens = tokens.to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # 使用模型计算所有层的 KV
            all_kv = self.model.compute_all_kv(tokens, heads_to_compute)
            
            # 存储到设备缓存
            for layer_id, layer_kv in all_kv.items():
                for head_id, (k, v) in layer_kv.items():
                    device_cache.set_cache(layer_id, head_id, k, v)
        
        computation_time = time.time() - start_time
        
        return (computation_time, len(heads_to_compute))
    
    def compute_all_devices(self, 
                            tokens: torch.Tensor,
                            force_recompute: bool = False) -> Dict[str, Tuple[float, int]]:
        """计算所有设备的 KV-Cache"""
        results = {}
        
        for device_id in self.devices:
            if self.devices[device_id].is_online:
                time_taken, num_heads = self.compute_kv_cache_for_device(
                    tokens, device_id, force_recompute=force_recompute
                )
                results[device_id] = (time_taken, num_heads)
        
        return results
    
    def simulate_device_offline(self, device_id: str):
        """模拟设备离线"""
        if device_id in self.devices:
            self.devices[device_id].is_online = False
            print(f"[Engine] {device_id} is now OFFLINE")
    
    def simulate_device_online(self, device_id: str):
        """恢复设备在线"""
        if device_id in self.devices:
            self.devices[device_id].is_online = True
            print(f"[Engine] {device_id} is now ONLINE")
    
    def reassign_heads(self, from_device: str, to_device: str) -> List[int]:
        """重新分配 heads"""
        if from_device not in self.devices or to_device not in self.devices:
            return []
        
        reassigned = self.devices[from_device].assigned_heads.copy()
        self.devices[to_device].add_heads(reassigned)
        self.devices[from_device].assigned_heads = []
        
        return reassigned
    
    def print_status(self):
        """打印状态"""
        print("\n" + "-"*60)
        print("System Status")
        print("-"*60)
        
        for device_id, device in self.devices.items():
            status = "ONLINE" if device.is_online else "OFFLINE"
            cached = device.get_cached_heads(self.config.n_layers)
            needs = device.get_heads_needing_computation(self.config.n_layers)
            
            print(f"  {device_id}: [{status}]")
            print(f"    Assigned heads: {device.assigned_heads}")
            print(f"    Cached heads: {cached}")
            print(f"    Needs computation: {needs}")
        
        print("-"*60)


def run_real_model_comparison():
    """运行真实模型的性能对比"""
    
    # 模型路径
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors"
    params_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json"
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "REAL MODEL DEMO" + " "*20 + "#")
    print("#" + " "*10 + "KV-Cache Reuse Optimization" + " "*10 + "#")
    print("#"*70)
    
    # 创建引擎
    engine = RealModelKVCacheEngine(model_path, params_path, device="cpu")
    
    # 设置设备（4个设备，8个KV heads）
    num_devices = 4
    n_kv_heads = engine.config.n_kv_heads  # 8
    heads_per_device = n_kv_heads // num_devices  # 2
    
    print(f"\n[Setup] {num_devices} devices, {n_kv_heads} KV heads")
    print(f"[Setup] Each device handles {heads_per_device} heads")
    
    for i in range(num_devices):
        device_id = f"Device_{i}"
        start_head = i * heads_per_device
        assigned_heads = list(range(start_head, start_head + heads_per_device))
        engine.register_device(device_id, assigned_heads)
    
    # 创建测试输入
    seq_length = 64
    tokens = torch.randint(0, 1000, (1, seq_length))
    
    print(f"\n[Input] Sequence length: {seq_length}")
    
    # ============================================================
    # 阶段1: 初始 KV-Cache 计算
    # ============================================================
    print("\n" + "="*60)
    print("Phase 1: Initial KV-Cache Computation (All Devices)")
    print("="*60)
    
    initial_results = {}
    total_initial_time = 0.0
    
    for device_id in engine.devices:
        time_taken, num_heads = engine.compute_kv_cache_for_device(tokens, device_id)
        initial_results[device_id] = (time_taken, num_heads)
        total_initial_time += time_taken
        print(f"  {device_id}: computed {num_heads} heads in {time_taken*1000:.2f}ms")
    
    print(f"\nTotal initial computation time: {total_initial_time*1000:.2f}ms")
    
    engine.print_status()
    
    # 保存初始状态用于对比测试
    initial_device_states = {
        device_id: {
            "assigned_heads": device.assigned_heads.copy(),
            "cached_heads": device.get_cached_heads(engine.config.n_layers).copy()
        }
        for device_id, device in engine.devices.items()
    }
    
    # ============================================================
    # 阶段2: 设备离线 - KV-Cache 复用策略
    # ============================================================
    print("\n" + "="*60)
    print("Phase 2: Device Offline - KV-Cache REUSE Strategy")
    print("="*60)
    
    offline_device = "Device_1"
    target_device = "Device_3"
    
    print(f"\n[Scenario] {offline_device} goes offline")
    print(f"[Action] Reassign its heads to {target_device}")
    
    # 记录 target 设备重分配前的状态
    target_cached_before = engine.devices[target_device].get_cached_heads(engine.config.n_layers)
    
    # 模拟离线并重分配
    engine.simulate_device_offline(offline_device)
    reassigned_heads = engine.reassign_heads(offline_device, target_device)
    
    print(f"\n[Reassignment]")
    print(f"  Reassigned heads: {reassigned_heads}")
    print(f"  {target_device} original cached: {target_cached_before}")
    print(f"  {target_device} now assigned: {engine.devices[target_device].assigned_heads}")
    
    # 使用复用策略 - 只计算新分配的 heads
    reuse_start = time.time()
    
    heads_needing_compute = engine.devices[target_device].get_heads_needing_computation(engine.config.n_layers)
    print(f"\n[KV-Cache Reuse] Only computing new heads: {heads_needing_compute}")
    
    if heads_needing_compute:
        reuse_time, reuse_heads = engine.compute_kv_cache_for_device(
            tokens, target_device, head_ids=heads_needing_compute
        )
    else:
        reuse_time, reuse_heads = 0.0, 0
    
    reuse_total_time = time.time() - reuse_start
    
    print(f"\n[KV-Cache Reuse Result]")
    print(f"  Heads reused (from cache): {len(target_cached_before)}")
    print(f"  Heads recomputed: {reuse_heads}")
    print(f"  Computation time: {reuse_time*1000:.2f}ms")
    print(f"  Total time: {reuse_total_time*1000:.2f}ms")
    
    engine.print_status()
    
    # ============================================================
    # 阶段3: 重置并测试全量重计算策略
    # ============================================================
    print("\n" + "="*60)
    print("Phase 3: Device Offline - FULL RECOMPUTE Strategy (Traditional)")
    print("="*60)
    
    # 重置设备状态
    print("\n[Reset] Restoring initial state for fair comparison...")
    
    for device_id, device in engine.devices.items():
        device.assigned_heads = initial_device_states[device_id]["assigned_heads"].copy()
        device.is_online = True
        device.cache = {}  # 清除缓存
    
    # 重新计算初始缓存
    for device_id in engine.devices:
        engine.compute_kv_cache_for_device(tokens, device_id)
    
    # 现在模拟相同的离线场景，但使用全量重计算
    engine.simulate_device_offline(offline_device)
    reassigned_heads = engine.reassign_heads(offline_device, target_device)
    
    print(f"\n[Scenario] {offline_device} goes offline (same as before)")
    print(f"[Strategy] Full recompute - clear all caches and recompute everything")
    
    # 全量重计算 - 清除所有在线设备的缓存
    full_start = time.time()
    
    for device_id, device in engine.devices.items():
        if device.is_online:
            device.clear_cache()
    
    print("\n[Full Recompute] Clearing all caches...")
    print("[Full Recompute] Recomputing all heads for all online devices...")
    
    full_results = {}
    full_compute_time = 0.0
    total_heads_computed = 0
    
    for device_id, device in engine.devices.items():
        if device.is_online:
            time_taken, num_heads = engine.compute_kv_cache_for_device(
                tokens, device_id, force_recompute=True
            )
            full_results[device_id] = (time_taken, num_heads)
            full_compute_time += time_taken
            total_heads_computed += num_heads
            print(f"  {device_id}: recomputed {num_heads} heads in {time_taken*1000:.2f}ms")
    
    full_total_time = time.time() - full_start
    
    print(f"\n[Full Recompute Result]")
    print(f"  Total heads recomputed: {total_heads_computed}")
    print(f"  Total computation time: {full_compute_time*1000:.2f}ms")
    print(f"  Total time: {full_total_time*1000:.2f}ms")
    
    # ============================================================
    # 阶段4: 性能对比
    # ============================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON RESULTS (Real Model)")
    print("="*70)
    
    speedup = full_total_time / reuse_total_time if reuse_total_time > 0 else float('inf')
    time_saved = full_total_time - reuse_total_time
    time_saved_percent = (time_saved / full_total_time * 100) if full_total_time > 0 else 0
    
    heads_saved = total_heads_computed - reuse_heads
    heads_saved_percent = (heads_saved / total_heads_computed * 100) if total_heads_computed > 0 else 0
    
    print(f"\n{'Strategy':<35} {'Time (ms)':<15} {'Heads Computed':<20}")
    print("-"*70)
    print(f"{'KV-Cache Reuse (Optimized)':<35} {reuse_total_time*1000:<15.2f} {reuse_heads:<20}")
    print(f"{'Full Recompute (Traditional)':<35} {full_total_time*1000:<15.2f} {total_heads_computed:<20}")
    print("-"*70)
    
    print(f"\n{'Metric':<40} {'Value':<25}")
    print("-"*65)
    print(f"{'Speedup':<40} {speedup:.2f}x")
    print(f"{'Time Saved':<40} {time_saved*1000:.2f}ms ({time_saved_percent:.1f}%)")
    print(f"{'Heads Reused (Cache Hit)':<40} {len(target_cached_before)}")
    print(f"{'Computation Saved':<40} {heads_saved} heads ({heads_saved_percent:.1f}%)")
    print("-"*65)
    
    print(f"\n{'='*70}")
    print(f"✅ KV-Cache Reuse strategy is {speedup:.2f}x FASTER than Full Recompute!")
    print(f"✅ Saved {heads_saved_percent:.1f}% of computation by reusing cached KV values.")
    print(f"✅ This is REAL MODEL computation, not simulation!")
    print(f"{'='*70}")
    
    # 返回结果
    return {
        "reuse_time_ms": reuse_total_time * 1000,
        "full_time_ms": full_total_time * 1000,
        "speedup": speedup,
        "time_saved_percent": time_saved_percent,
        "heads_saved_percent": heads_saved_percent
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Real Model Distributed Inference Optimization Demo")
    print("Using Llama-3.2-1B for actual KV-Cache computation")
    print("="*70)
    
    try:
        results = run_real_model_comparison()
        
        print("\n" + "="*70)
        print("Demo Completed Successfully!")
        print("="*70)
        print("\nKey Findings (Real Model):")
        print(f"  - Speedup: {results['speedup']:.2f}x")
        print(f"  - Time saved: {results['time_saved_percent']:.1f}%")
        print(f"  - Computation saved: {results['heads_saved_percent']:.1f}%")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
