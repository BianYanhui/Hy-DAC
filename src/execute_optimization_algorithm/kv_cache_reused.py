"""
KV-Cache复用模块
负责管理和复用KV-Cache，减少重计算开销
"""
import torch
from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from llama_model_loader import LlamaModel


class KVCacheManager:
    """KV-Cache管理器（支持真实模型）"""
    
    def __init__(self, llama_model: Optional['LlamaModel'] = None, num_layers: int = 16, hidden_size: int = 64):
        """
        初始化KV-Cache管理器
        
        Args:
            llama_model: Llama模型实例（用于真实推理）
            num_layers: 模型层数
            hidden_size: 隐藏层大小
        """
        self.llama_model = llama_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # worker_id -> {head_id -> List[Tuple(k_cache, v_cache)] for each layer}
        self.kv_caches: Dict[str, Dict[int, List[Tuple]]] = {}
        
    def initialize_worker_cache(self, worker_id: str, heads: List[int], 
                               input_text: str = "Hello, this is a test.", seq_length: int = 32):
        """
        为Worker初始化KV-Cache（真实计算）
        
        Args:
            worker_id: Worker ID
            heads: 该Worker负责的头部列表
            input_text: 输入文本（用于真实推理）
            seq_length: 序列长度（缩短以加速）
        """
        if worker_id not in self.kv_caches:
            self.kv_caches[worker_id] = {}
        
        print(f"[KVCacheManager] Worker {worker_id} 开始初始化 KV-Cache for Heads {heads}")
        
        if self.llama_model is not None:
            # 真实推理：批量计算所有heads的KV-Cache（加速）
            start_time = time.time()
            
            # 创建简单的输入
            input_ids = torch.randint(0, 1000, (1, seq_length), device=self.llama_model.device)
            
            # 批量计算所有heads（而不是一个一个计算）
            with torch.no_grad():
                _, kv_cache_list = self.llama_model.compute_with_heads(
                    input_ids, heads, kv_cache_list=None
                )
            
            # 为所有heads保存相同的KV cache（简化）
            for head in heads:
                if head not in self.kv_caches[worker_id]:
                    self.kv_caches[worker_id][head] = kv_cache_list
            
            elapsed = time.time() - start_time
            print(f"[KVCacheManager] Worker {worker_id} 完成 KV-Cache 初始化，耗时: {elapsed:.3f}秒")
        else:
            # 模拟模式
            for head in heads:
                if head not in self.kv_caches[worker_id]:
                    # 为每一层创建KV-Cache
                    kv_cache_list = []
                    for _ in range(self.num_layers):
                        k_cache = torch.randn(1, seq_length, 8, self.hidden_size)
                        v_cache = torch.randn(1, seq_length, 8, self.hidden_size)
                        kv_cache_list.append((k_cache, v_cache))
                    self.kv_caches[worker_id][head] = kv_cache_list
            
            print(f"[KVCacheManager] Worker {worker_id} 初始化 KV-Cache for Heads {heads} (模拟模式)")
    
    def compute_kv_cache_for_heads(self, worker_id: str, heads: List[int], 
                                   seq_length: int = 32, input_text: str = "Test input") -> float:
        """
        为指定的头部计算KV-Cache（真实计算）
        
        Args:
            worker_id: Worker ID
            heads: 需要计算的头部列表
            seq_length: 序列长度
            input_text: 输入文本
            
        Returns:
            计算耗时（秒）
        """
        start_time = time.time()
        
        if worker_id not in self.kv_caches:
            self.kv_caches[worker_id] = {}
        
        print(f"[KVCacheManager] Worker {worker_id} 开始计算 Heads {heads} 的 KV-Cache...")
        
        if self.llama_model is not None:
            # 真实推理：批量计算
            input_ids = torch.randint(0, 1000, (1, seq_length), device=self.llama_model.device)
            
            # 批量计算所有heads
            with torch.no_grad():
                _, kv_cache_list = self.llama_model.compute_with_heads(
                    input_ids, heads, kv_cache_list=None
                )
            
            # 保存所有heads的KV cache
            for head in heads:
                self.kv_caches[worker_id][head] = kv_cache_list
        else:
            # 模拟模式
            for head in heads:
                kv_cache_list = []
                for _ in range(self.num_layers):
                    k_cache = torch.randn(1, seq_length, 8, self.hidden_size)
                    v_cache = torch.randn(1, seq_length, 8, self.hidden_size)
                    kv_cache_list.append((k_cache, v_cache))
                self.kv_caches[worker_id][head] = kv_cache_list
                
                # 模拟计算时间
                time.sleep(0.05)
        
        elapsed = time.time() - start_time
        print(f"[KVCacheManager] Worker {worker_id} 完成 Heads {heads} 的 KV-Cache 计算，耗时: {elapsed:.3f}秒")
        
        return elapsed
    
    def reuse_cache_and_compute_new(self, worker_id: str, old_heads: List[int], 
                                    new_heads: List[int], seq_length: int = 128) -> Tuple[float, int, int]:
        """
        复用已有的KV-Cache并计算新的KV-Cache
        
        Args:
            worker_id: Worker ID
            old_heads: 已有的头部列表（可以复用KV-Cache）
            new_heads: 新分配的头部列表（需要重新计算KV-Cache）
            seq_length: 序列长度
            
        Returns:
            (计算耗时, 复用的头部数量, 重新计算的头部数量)
        """
        # 确定可以复用的头部和需要重新计算的头部
        reusable_heads = []
        heads_to_compute = []
        
        if worker_id in self.kv_caches:
            for head in old_heads:
                if head in self.kv_caches[worker_id]:
                    reusable_heads.append(head)
        
        for head in new_heads:
            if worker_id not in self.kv_caches or head not in self.kv_caches[worker_id]:
                heads_to_compute.append(head)
        
        print(f"[KVCacheManager] Worker {worker_id} 复用:")
        print(f"  ✓ 可复用的 Heads: {reusable_heads} ({len(reusable_heads)} 个)")
        print(f"  ✗ 需要重新计算的 Heads: {heads_to_compute} ({len(heads_to_compute)} 个)")
        
        # 只需要计算新的头部
        compute_time = 0.0
        if heads_to_compute:
            compute_time = self.compute_kv_cache_for_heads(
                worker_id, heads_to_compute, seq_length
            )
        else:
            print(f"[KVCacheManager] Worker {worker_id} 无需重新计算，完全复用!")
        
        return compute_time, len(reusable_heads), len(heads_to_compute)
    
    def remove_worker_cache(self, worker_id: str):
        """移除Worker的所有KV-Cache"""
        if worker_id in self.kv_caches:
            removed_heads = list(self.kv_caches[worker_id].keys())
            del self.kv_caches[worker_id]
            print(f"[KVCacheManager] 移除 Worker {worker_id} 的 KV-Cache (Heads: {removed_heads})")
    
    def get_worker_cached_heads(self, worker_id: str) -> List[int]:
        """获取Worker已缓存的头部列表"""
        if worker_id in self.kv_caches:
            return list(self.kv_caches[worker_id].keys())
        return []
    
    def calculate_reuse_ratio(self, worker_id: str, required_heads: List[int]) -> float:
        """
        计算KV-Cache复用率
        
        Args:
            worker_id: Worker ID
            required_heads: 需要的头部列表
            
        Returns:
            复用率（0.0-1.0）
        """
        if not required_heads:
            return 0.0
        
        cached_heads = self.get_worker_cached_heads(worker_id)
        reusable_count = len([h for h in required_heads if h in cached_heads])
        
        return reusable_count / len(required_heads)
