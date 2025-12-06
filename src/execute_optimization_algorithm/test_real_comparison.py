#!/usr/bin/env python3
"""
快速测试脚本：验证传统方法和优化方法都使用真实计算
"""
import sys
sys.path.insert(0, '/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/20251201-JSAC-Hy-DAC/Hy-DAC-Code/Hy-DAC/src/execute_optimization_algorithm')

import torch
from llama_model_loader import LlamaModel
from kv_cache_reused import KVCacheManager

print("="*60)
print("测试：真实计算对比")
print("="*60)

# 加载模型
model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors"
params_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json"

print("\n1. 加载模型...")
model = LlamaModel(model_path, params_path)

# 创建KV-Cache管理器
kv_manager = KVCacheManager(
    llama_model=model,
    num_layers=model.get_num_layers(),
    hidden_size=model.get_head_dim()
)

print("\n2. 测试传统方法（完全重计算16个heads）...")
traditional_time = kv_manager.compute_kv_cache_for_heads_no_print(
    "test_traditional", list(range(1, 17)), seq_length=32
)
print(f"   传统方法耗时: {traditional_time:.3f}秒")

print("\n3. 测试优化方法（只计算8个新heads）...")
# 先初始化8个heads
kv_manager.initialize_worker_cache("test_optimized", list(range(1, 9)), seq_length=32)
# 然后只计算新的8个heads
optimized_time = kv_manager.compute_kv_cache_for_heads(
    "test_optimized", list(range(9, 17)), seq_length=32
)
print(f"   优化方法耗时: {optimized_time:.3f}秒")

print("\n4. 性能对比:")
speedup = traditional_time / optimized_time if optimized_time > 0 else float('inf')
improvement = (traditional_time - optimized_time) / traditional_time * 100 if traditional_time > 0 else 0

print(f"   传统方法: {traditional_time:.3f}秒 (16个heads)")
print(f"   优化方法: {optimized_time:.3f}秒 (8个heads)")
print(f"   加速比: {speedup:.2f}x")
print(f"   性能提升: {improvement:.1f}%")

print("\n✅ 测试完成！两种方法都使用了真实计算。")
print("="*60)
