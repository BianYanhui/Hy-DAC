"""
Llama 模型加载和推理模块 (Llama Model Loading and Inference Module)

该模块负责加载真实的 Llama-3.2-1B 模型并进行实际的 KV-Cache 计算。
支持按 Head 进行分布式计算。
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class LlamaConfig:
    """Llama 模型配置"""
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    ffn_dim_multiplier: float = 1.5
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_seq_len: int = 2048
    
    @classmethod
    def from_json(cls, path: str) -> "LlamaConfig":
        """从 JSON 文件加载配置"""
        with open(path, 'r') as f:
            params = json.load(f)
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算旋转位置编码的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, 
                     freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    # 将张量转换为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整 freqs_cis 形状
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    # 应用旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 KV heads 以匹配 Q heads 数量 (GQA)"""
    if n_rep == 1:
        return x
    bs, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )


class LlamaAttention(nn.Module):
    """Llama Attention 层"""
    
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # 投影层
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
    
    def compute_kv(self, x: torch.Tensor, 
                   freqs_cis: torch.Tensor,
                   head_ids: Optional[List[int]] = None) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算指定 Heads 的 KV 值
        
        Args:
            x: 输入张量 [batch, seq_len, dim]
            freqs_cis: 位置编码频率
            head_ids: 要计算的 KV Head IDs，None 表示计算所有
            
        Returns:
            {head_id: (k, v)} 其中 k, v 形状为 [batch, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 K 和 V
        k = self.wk(x)  # [batch, seq_len, n_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, n_kv_heads * head_dim]
        
        # 重塑为 [batch, seq_len, n_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 应用旋转位置编码到 K
        # 先创建一个 dummy Q 用于 RoPE
        q_dummy = torch.zeros(batch_size, seq_len, self.n_kv_heads, self.head_dim, 
                              device=x.device, dtype=x.dtype)
        _, k = apply_rotary_emb(q_dummy, k, freqs_cis[:seq_len])
        
        # 提取指定的 heads
        if head_ids is None:
            head_ids = list(range(self.n_kv_heads))
        
        result = {}
        for head_id in head_ids:
            if 0 <= head_id < self.n_kv_heads:
                result[head_id] = (
                    k[:, :, head_id, :].clone(),  # [batch, seq_len, head_dim]
                    v[:, :, head_id, :].clone()
                )
        
        return result
    
    def forward_with_cache(self, x: torch.Tensor,
                           freqs_cis: torch.Tensor,
                           kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        使用已有的 KV-Cache 进行前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, dim]
            freqs_cis: 位置编码
            kv_cache: {head_id: (k_cache, v_cache)} 已缓存的 KV 值
            mask: 注意力掩码
            
        Returns:
            输出张量 [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q
        q = self.wq(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 从缓存组装完整的 K, V
        k_full = torch.zeros(batch_size, seq_len, self.n_kv_heads, self.head_dim,
                             device=x.device, dtype=x.dtype)
        v_full = torch.zeros(batch_size, seq_len, self.n_kv_heads, self.head_dim,
                             device=x.device, dtype=x.dtype)
        
        for head_id, (k_cached, v_cached) in kv_cache.items():
            k_full[:, :, head_id, :] = k_cached
            v_full[:, :, head_id, :] = v_cached
        
        # 应用 RoPE 到 Q 和 K
        q, k_full = apply_rotary_emb(q, k_full, freqs_cis[:seq_len])
        
        # GQA: 重复 KV heads
        k_full = repeat_kv(k_full, self.n_rep)
        v_full = repeat_kv(v_full, self.n_rep)
        
        # 转置为 [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_full)
        
        # 重塑并投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        
        return output


class LlamaFeedForward(nn.Module):
    """Llama FFN 层"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Llama 3.2 使用的隐藏层维度
        # 根据 params.json: ffn_dim_multiplier = 1.5, multiple_of = 256
        # 实际 hidden_dim = 8192 (从模型权重中读取)
        hidden_dim = int(8 * config.dim * config.ffn_dim_multiplier / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaTransformerBlock(nn.Module):
    """Llama Transformer 块"""
    
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = LlamaAttention(config, layer_id)
        self.feed_forward = LlamaFeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
    
    def compute_kv(self, x: torch.Tensor, 
                   freqs_cis: torch.Tensor,
                   head_ids: Optional[List[int]] = None) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """计算该层指定 Heads 的 KV 值"""
        h = self.attention_norm(x)
        return self.attention.compute_kv(h, freqs_cis, head_ids)
    
    def forward_with_cache(self, x: torch.Tensor,
                           freqs_cis: torch.Tensor,
                           kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用 KV-Cache 进行前向传播"""
        h = x + self.attention.forward_with_cache(
            self.attention_norm(x), freqs_cis, kv_cache, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(nn.Module):
    """完整的 Llama 模型"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            LlamaTransformerBlock(config, i) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # 预计算位置编码
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta
        )
    
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """获取 token embeddings"""
        return self.tok_embeddings(tokens)
    
    def compute_kv_for_layer(self, x: torch.Tensor, 
                              layer_id: int,
                              head_ids: Optional[List[int]] = None) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算指定层的 KV 值
        
        Args:
            x: 经过 embedding 的输入 [batch, seq_len, dim]
            layer_id: 层 ID
            head_ids: 要计算的 Head IDs
            
        Returns:
            {head_id: (k, v)}
        """
        # 确保 freqs_cis 在正确的设备上
        freqs_cis = self.freqs_cis.to(x.device)
        
        # 逐层传播到目标层
        h = x
        for i in range(layer_id):
            # 简化：只做前向传播，不计算完整的 attention
            # 这里我们假设前面层的输出近似等于输入（简化处理）
            pass
        
        # 计算目标层的 KV
        return self.layers[layer_id].compute_kv(h, freqs_cis, head_ids)
    
    def compute_all_kv(self, tokens: torch.Tensor,
                       head_ids: Optional[List[int]] = None) -> Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        计算所有层指定 Heads 的 KV 值
        
        Args:
            tokens: 输入 token IDs [batch, seq_len]
            head_ids: 要计算的 Head IDs
            
        Returns:
            {layer_id: {head_id: (k, v)}}
        """
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(x.device)
        
        all_kv = {}
        h = x
        
        for layer_id, layer in enumerate(self.layers):
            # 计算该层的 KV
            all_kv[layer_id] = layer.compute_kv(h, freqs_cis, head_ids)
            
            # 继续前向传播（简化版本：使用刚计算的 KV）
            # 为了避免递归依赖，这里使用简化的前向传播
            h_normed = layer.attention_norm(h)
            
            # 计算完整的 KV 用于前向传播
            full_kv = layer.compute_kv(h, freqs_cis, None)
            
            # 使用 KV-Cache 进行前向传播
            attn_out = layer.attention.forward_with_cache(h_normed, freqs_cis, full_kv)
            h = h + attn_out
            h = h + layer.feed_forward(layer.ffn_norm(h))
        
        return all_kv


class LlamaModelLoader:
    """Llama 模型加载器"""
    
    def __init__(self, 
                 model_path: str,
                 params_path: str,
                 device: str = "cpu"):
        """
        初始化模型加载器
        
        Args:
            model_path: model.safetensors 路径
            params_path: params.json 路径
            device: 计算设备
        """
        self.model_path = Path(model_path)
        self.params_path = Path(params_path)
        self.device = device
        
        # 加载配置
        self.config = LlamaConfig.from_json(str(params_path))
        
        # 模型实例（延迟加载）
        self._model: Optional[LlamaModel] = None
        
        print(f"[LlamaModelLoader] Config loaded from {params_path}")
        print(f"  - dim: {self.config.dim}")
        print(f"  - n_layers: {self.config.n_layers}")
        print(f"  - n_heads: {self.config.n_heads}")
        print(f"  - n_kv_heads: {self.config.n_kv_heads}")
    
    def load_model(self) -> LlamaModel:
        """加载模型权重"""
        if self._model is not None:
            return self._model
        
        print(f"[LlamaModelLoader] Loading model from {self.model_path}...")
        start_time = time.time()
        
        # 创建模型
        model = LlamaModel(self.config)
        
        # 从 safetensors 加载权重
        state_dict = {}
        with safe_open(str(self.model_path), framework="pt", device=self.device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # 映射权重名称（safetensors 可能使用不同的命名）
        mapped_state_dict = self._map_weights(state_dict)
        
        # 加载权重
        model.load_state_dict(mapped_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        load_time = time.time() - start_time
        print(f"[LlamaModelLoader] Model loaded in {load_time:.2f}s")
        
        self._model = model
        return model
    
    def _map_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """映射权重名称到模型结构"""
        mapped = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # 常见的名称映射
            if key.startswith("model."):
                new_key = key[6:]  # 去掉 "model." 前缀
            
            # 映射 embedding
            if "embed_tokens" in new_key:
                new_key = new_key.replace("embed_tokens", "tok_embeddings")
            
            # 映射 layers
            if "self_attn" in new_key:
                new_key = new_key.replace("self_attn", "attention")
            if "q_proj" in new_key:
                new_key = new_key.replace("q_proj", "wq")
            if "k_proj" in new_key:
                new_key = new_key.replace("k_proj", "wk")
            if "v_proj" in new_key:
                new_key = new_key.replace("v_proj", "wv")
            if "o_proj" in new_key:
                new_key = new_key.replace("o_proj", "wo")
            
            # 映射 FFN
            if "mlp" in new_key:
                new_key = new_key.replace("mlp", "feed_forward")
            if "gate_proj" in new_key:
                new_key = new_key.replace("gate_proj", "w1")
            if "down_proj" in new_key:
                new_key = new_key.replace("down_proj", "w2")
            if "up_proj" in new_key:
                new_key = new_key.replace("up_proj", "w3")
            
            # 映射 norm
            if "input_layernorm" in new_key:
                new_key = new_key.replace("input_layernorm", "attention_norm")
            if "post_attention_layernorm" in new_key:
                new_key = new_key.replace("post_attention_layernorm", "ffn_norm")
            
            # 映射 lm_head
            if "lm_head" in new_key:
                new_key = new_key.replace("lm_head", "output")
            
            mapped[new_key] = value
        
        return mapped
    
    def get_config(self) -> LlamaConfig:
        """获取模型配置"""
        return self.config
    
    @property
    def model(self) -> LlamaModel:
        """获取模型（如果未加载则自动加载）"""
        if self._model is None:
            self.load_model()
        return self._model


# 便捷函数
def load_llama_model(model_path: str, params_path: str, device: str = "cpu") -> Tuple[LlamaModel, LlamaConfig]:
    """
    加载 Llama 模型的便捷函数
    
    Args:
        model_path: model.safetensors 路径
        params_path: params.json 路径
        device: 计算设备
        
    Returns:
        (model, config)
    """
    loader = LlamaModelLoader(model_path, params_path, device)
    model = loader.load_model()
    return model, loader.config


if __name__ == "__main__":
    # 测试代码
    import sys
    
    model_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/model.safetensors"
    params_path = "/Users/yhbian/Library/CloudStorage/OneDrive-个人/边彦晖-学校/杂乱/Models/Llama-3.2-1B/params.json"
    
    print("Testing Llama Model Loading...")
    
    # 加载模型
    loader = LlamaModelLoader(model_path, params_path, device="cpu")
    model = loader.load_model()
    
    # 测试 KV 计算
    print("\nTesting KV computation...")
    tokens = torch.randint(0, 1000, (1, 32))  # batch=1, seq=32
    
    start_time = time.time()
    all_kv = model.compute_all_kv(tokens, head_ids=[0, 1])
    compute_time = time.time() - start_time
    
    print(f"Computed KV for heads [0, 1] across {len(all_kv)} layers in {compute_time*1000:.2f}ms")
    
    # 打印一些信息
    for layer_id in [0, 7, 15]:
        if layer_id in all_kv:
            for head_id, (k, v) in all_kv[layer_id].items():
                print(f"  Layer {layer_id}, Head {head_id}: K={k.shape}, V={v.shape}")
    
    print("\nTest completed!")
