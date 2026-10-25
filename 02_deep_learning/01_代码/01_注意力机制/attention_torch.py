"""
attention_torch.py —— 注意力 / 多头注意力的 PyTorch 实现（nn.Module，可训练）

与 attention_numpy.py 的区别：
  · 这里带 autograd，能反传、能训练；
  · 参数排布按 PyTorch 习惯（把 W_q/W_k/W_v 合成一个 in_proj，方便一次矩阵乘），
    但提供 load_from_numpy() / to_numpy_params() 与 NumPy 版逐位对拍。

形状约定与 NumPy 版完全一致：输入 (B, T, d)，输出 (B, T, d)，权重 (B, h, T, S)。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "TransformerBlock",
    "causal_mask_bool",
]


def causal_mask_bool(T: int, S: int | None = None, device=None) -> torch.Tensor:
    """(T, S) 布尔掩码，True=允许注意（下三角）。与 numpy 版 causal_mask 等价。"""
    S = T if S is None else S
    return torch.ones(T, S, dtype=torch.bool, device=device).tril()


class ScaledDotProductAttention(nn.Module):
    """softmax(QKᵀ/√dk + mask) V

    注意 mask 的加法位置：**必须在 softmax 之前**加 -inf。
    若改成 softmax 之后再乘 0，被屏蔽位置的概率质量已经计入分母，
    剩下的有效位置权重之和不等于 1，输出会被系统性缩小。
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (B,h,T,S)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        return torch.matmul(w, v), w


class MultiHeadAttention(nn.Module):
    """多头注意力。参数量 4d²，与头数无关（d 固定时）。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        # 一次算出 [Q|K|V]，再切三块（PyTorch nn.MultiheadAttention 也是这么做的）
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B,T,d) -> (B,h,T,dk)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.dk).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B,h,T,dk) -> (B,T,d)"""
        B, h, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * dk)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = True,
    ):
        x_kv = x_q if x_kv is None else x_kv
        B, T, _ = x_q.shape
        S = x_kv.size(1)
        d = self.d_model

        if x_kv is x_q:  # 自注意力：一次投影出 Q|K|V
            qkv = self.in_proj(x_q)  # (B,T,3d)
            q = self._split(qkv[..., :d])
            k = self._split(qkv[..., d : 2 * d])
            v = self._split(qkv[..., 2 * d :])
        else:  # 交叉注意力：Q 与 KV 分别投影
            q = self._split(self.in_proj(x_q)[..., :d])
            kv = self.in_proj(x_kv)  # (B,S,3d)
            k = self._split(kv[..., d : 2 * d])
            v = self._split(kv[..., 2 * d :])

        if is_causal:
            m = causal_mask_bool(T, S, device=x_q.device)
            mask = m if mask is None else (mask & m)

        ctx, w = self.attn(q, k, v, mask)  # (B,h,T,dk), (B,h,T,S)
        out = self.out_proj(self._merge(ctx))  # (B,T,d)
        return (out, w) if need_weights else out

    # ---- 与 NumPy 版对拍 ----
    def load_from_numpy(self, m) -> "MultiHeadAttention":
        """把 attention_numpy.MultiHeadAttention 的四个 d×d 矩阵拷进来。"""
        with torch.no_grad():
            W = [
                torch.as_tensor(m.W_q, dtype=torch.float32),
                torch.as_tensor(m.W_k, dtype=torch.float32),
                torch.as_tensor(m.W_v, dtype=torch.float32),
            ]
            self.in_proj.weight.copy_(torch.cat(W, dim=1).t())  # nn.Linear 存的是 (out,in)
            self.in_proj.bias.zero_()
            self.out_proj.weight.copy_(torch.as_tensor(m.W_o, dtype=torch.float32).t())
            self.out_proj.bias.zero_()
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerBlock(nn.Module):
    """一个最小 Transformer 块：LN → MHA → 残差 → LN → FFN → 残差。

    放这个类是为了说明多头注意力的**位置**：它只是两个子层之一，
    另一半是前馈网络（逐 token 的 MLP），两者都靠残差与 LayerNorm 串起来。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        a, w = self.mha(self.ln1(x), is_causal=is_causal)  # Pre-LN 结构
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, w
