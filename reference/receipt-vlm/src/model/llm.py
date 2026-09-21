"""
自制小型 decoder-only LLM（对齐 MiniMind 实现，保证训练数值稳定）

与早期手写版的区别（也是早期全量预训练发散的根因修复）：
- 注意力改用 `F.scaled_dot_product_attention`（SDPA）：融合、数值稳定、causal 内部处理，
  不再显式构造带 `float('-inf')` 的大分数矩阵，避免 bf16 下 softmax 溢出/NaN。
- 归一化用 RMSNorm（fp32 计算），FFN 用 SwiGLU——与 MiniMind 一致。
- 支持 GQA（num_kv_heads，默认等于 num_heads 即 MHA）。

对外接口保持不变（供 src/model/vlm.py 组装 VLM、src/pretrain_lm.py 预训练）：
  MiniLLM(config, gradient_checkpointing=False)
  .forward(input_ids, attention_mask=None) / .forward_with_embeddings(embeddings, attention_mask=None)
  .embed_tokens / .layers / .ln_f / .lm_head / .num_parameters / .trainable_parameters
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LLMConfig:
    """LLM 配置（对齐 MiniMind）"""
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: Optional[int] = None      # GQA；None → 等于 num_heads（MHA）
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    dropout: float = 0.0                    # 预训练默认 0（与 MiniMind 一致，更稳）
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6


# ---------------------------------------------------------------------------- #
# 基础组件
# ---------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """RMSNorm，fp32 计算以保证稳定。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(dim: int, max_pos: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)            # [max_pos, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)     # [max_pos, dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, L, D]; cos,sin: [L, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, L, D] -> [B, n_kv*n_rep, L, D]（GQA）。"""
    if n_rep == 1:
        return x
    b, n_kv, l, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, l, d).reshape(b, n_kv * n_rep, l, d)


class Attention(nn.Module):
    """SDPA 注意力 + RoPE，支持 GQA。"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.n_heads = config.num_heads
        self.n_kv = config.num_kv_heads or config.num_heads
        assert config.hidden_size % self.n_heads == 0
        assert self.n_heads % self.n_kv == 0
        self.head_dim = config.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # attn_mask is None -> 纯 causal（更快、最稳）；否则用显式 bool 掩码（含 padding）
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
        )
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU FFN。"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm + SDPA-Attention + SwiGLU。"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attention(self.attention_norm(x), cos, sin, attn_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------- #
# 主模型
# ---------------------------------------------------------------------------- #
class MiniLLM(nn.Module):
    """自制小型 decoder-only LLM（MiniMind 风格）。"""

    def __init__(self, config: LLMConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_dim = config.hidden_size // config.num_heads
        cos, sin = precompute_rope(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.post_init()

    def post_init(self):
        self.apply(self._init_weights)
        self._scale_residual_init()
        # 权重绑定
        self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_init(self):
        """
        残差分支的输出投影按 1/sqrt(2·num_layers) 缩放（GPT-2 / LLaMA 的做法）。

        不做这一步时，每层往残差流里加的量方差相同，残差流的范数随深度累积：
        实测 16 层放大 83 倍、20 层放大 117 倍，层数越深越糟。表现出来就是
        预训练早期数值不稳。缩放后末层 RMS 与输入同量级。
        """
        std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.feed_forward.down_proj.weight, mean=0.0, std=std)

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def _build_attn_mask(self, attention_mask, L, device):
        """attention_mask [B,L]（1=valid）→ [B,1,L,L] bool（True=可注意）；含 causal。
        若 attention_mask 为 None，返回 None（走 SDPA 的 is_causal 路径）。"""
        if attention_mask is None:
            return None
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        pad = attention_mask.bool()[:, None, None, :]          # [B,1,1,L] 针对 key
        return causal[None, None, :, :] & pad                  # [B,1,L,L]

    def _run(self, h, attention_mask):
        B, L, _ = h.shape
        cos = self.rope_cos[:L].to(h.dtype)
        sin = self.rope_sin[:L].to(h.dtype)
        attn_mask = self._build_attn_mask(attention_mask, L, h.device)
        h = self.drop(h)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, cos, sin, attn_mask, use_reentrant=False)
            else:
                h = layer(h, cos, sin, attn_mask)
        h = self.ln_f(h)
        return self.lm_head(h)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """input_ids [B,L] → logits [B,L,V]。"""
        return self._run(self.embed_tokens(input_ids), attention_mask)

    def forward_with_embeddings(self, embeddings: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """直接吃 pre-computed embeddings（VLM 注入视觉特征用）。"""
        return self._run(embeddings, attention_mask)

    def freeze_layers(self, unfrozen_indices: Optional[List[int]] = None):
        for p in self.parameters():
            p.requires_grad = False
        if unfrozen_indices is None:
            return
        if 0 in unfrozen_indices:
            for p in self.embed_tokens.parameters():
                p.requires_grad = True
        if -1 in unfrozen_indices:
            for p in self.layers[-1].parameters():
                p.requires_grad = True
            for p in self.ln_f.parameters():
                p.requires_grad = True
            # 注意：lm_head.weight 与 embed_tokens.weight 是同一个 Parameter（权重绑定），
            # 解冻 lm_head 必然连带解冻 embedding。只有 0 也在解冻列表里时才允许，
            # 否则「冻住 embedding」只是错觉。
            if 0 in unfrozen_indices:
                for p in self.lm_head.parameters():
                    p.requires_grad = True

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing MiniLLM (MiniMind-aligned)...")
    cfg = LLMConfig(vocab_size=12000, hidden_size=1024, num_layers=16, num_heads=16, intermediate_size=2752)
    m = MiniLLM(cfg)
    print(f"Total parameters: {m.num_parameters:,}")

    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    # 纯 causal
    logits = m(ids)
    assert logits.shape == (2, 64, cfg.vocab_size)
    # 带 padding mask
    mask = torch.ones(2, 64, dtype=torch.long)
    mask[0, 50:] = 0
    logits2 = m(ids, mask)
    loss = F.cross_entropy(logits2[:, :-1].float().reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss).item()
    print(f"forward+backward OK, loss={loss.item():.4f}")
    print("MiniLLM test passed!")
