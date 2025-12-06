#!/usr/bin/env python3
"""
测试脚本：验证真实模型加载和推理
"""
import sys
sys.path.insert(0, '/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/20251201-JSAC-Hy-DAC/Hy-DAC-Code/Hy-DAC/src/execute_optimization_algorithm')

from llama_model_loader import LlamaModel
import torch

print("="*60)
print("测试 Llama-3.2-1B 模型加载")
print("="*60)

# 模型路径
model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors"
params_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json"

# 加载模型
model = LlamaModel(model_path, params_path)

print("\n测试推理...")
input_ids = torch.randint(0, 1000, (1, 32))

# 测试计算head 1, 2, 3
print("计算 Heads [1, 2, 3]...")
with torch.no_grad():
    output, kv_cache = model.compute_with_heads(input_ids, [1, 2, 3], kv_cache_list=None)

print(f"输出shape: {output.shape}")
print(f"KV cache层数: {len(kv_cache)}")

print("\n✅ 测试成功！模型可以正常加载和推理")
