"""
attention_numpy.py —— 注意力 / 多头注意力 的纯 NumPy 实现（从零写，不调库）

设计原则
--------
1. 每个函数都标注张量形状，形状错了立刻报错，不靠猜。
2. 公式与实现一一对应，注释里给出对应书页码：
      d2l 第 10 章     p.399–430（《动手学深度学习》中文版）
      Bishop 12.1      p.374–384（Deep Learning: Foundations and Concepts, 2024）
      从零构建 LLM 中文版 p.64–97（第 3 章 编码注意力机制）
3. 只依赖 numpy。对拍用的 PyTorch 版在 attention_torch.py，两边权重可以互拷。

符号约定
--------
B  batch size
T  序列长度（query 侧）
S  序列长度（key/value 侧；自注意力时 S == T）
d  d_model，模型宽度
h  head 数
dk d_model // h，每个头的维度
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "causal_mask",
    "MultiHeadAttention",
    "sinusoidal_positional_encoding",
]


def softmax(x: np.ndarray, axis: int = -1, mask: np.ndarray | None = None) -> np.ndarray:
    """数值稳定的 softmax。

    mask: 与 x 同形（可广播），True 表示「允许注意」，False 表示「屏蔽」。
          实现方式是把被屏蔽位置先置为 -inf 再归一化 —— 这是与「先 softmax 再乘 mask」
          的关键区别：后者会让被屏蔽位置的权重在归一化分母里仍然占份额。
    """
    if mask is not None:
        x = np.where(mask, x, -np.inf)
        # 整行全屏蔽时 softmax 会得到 nan；置 0 处理，避免污染下游
        all_masked = ~np.any(mask, axis=axis, keepdims=True)
        x = np.where(all_masked, 0.0, x)

    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    out = e / np.where(s == 0, 1.0, s)
    return out


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """缩放点积注意力（d2l p.415 / 从零构建 LLM p.84）。

        Attention(Q, K, V) = softmax(Q Kᵀ / √dk) V

    为什么除以 √dk：q·k = Σ q_i k_i，若 q、k 各分量独立且方差为 1，则点积方差为 dk。
    不缩放时 dk 越大点积越极端，softmax 进入饱和区、梯度接近 0（Bishop 12.1.5）。

    参数
    ----
    Q : (..., T, dk)
    K : (..., S, dk)
    V : (..., S, dv)
    mask : (T, S) 或可广播到 (..., T, S)，True=允许注意

    返回
    ----
    out     : (..., T, dv)
    weights : (..., T, S)  每行和为 1（整行被屏蔽时为 0）
    """
    dk = Q.shape[-1]
    if K.shape[-1] != dk:
        raise ValueError(f"Q 与 K 的最后一维必须相同，得到 {Q.shape} 与 {K.shape}")

    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)  # (..., T, S)
    weights = softmax(scores, axis=-1, mask=mask)  # (..., T, S)
    out = weights @ V  # (..., T, dv)
    return out, weights


def causal_mask(T: int, S: int | None = None) -> np.ndarray:
    """因果掩码：位置 i 只能看到 j <= i（GPT 类解码器用，从零构建 LLM p.147）。"""
    S = T if S is None else S
    idx_q = np.arange(T)[:, None]
    idx_k = np.arange(S)[None, :]
    return idx_k <= idx_q  # (T, S)，下三角为 True


class MultiHeadAttention:
    """多头注意力（Bishop 12.1.6 p.382 / d2l 10.5 p.422）。

    与「把同一个注意力算 h 遍」的区别：每个头有**独立的投影矩阵**，
    因此能同时关注不同的表示子空间。因为 dk = d/h，总参数量仍约为 4d²，
    与单头基本相同 —— 多头的代价是并行度而不是参数量。

    参数量：W_q(d×d) + W_k(d×d) + W_v(d×d) + W_o(d×d) = 4d²
    """

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator | None = None):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        rng = rng or np.random.default_rng(0)
        # Xavier 风格缩放，避免初始 score 方差过大
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    # ---- 形状变换 ----
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d) -> (B, h, T, dk)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.dk).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, h, T, dk) -> (B, T, d)"""
        B, h, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, h * dk)

    # ---- 前向 ----
    def forward(
        self,
        x_q: np.ndarray,
        x_kv: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """x_q: (B, T, d)；x_kv: (B, S, d)，None 表示自注意力。

        返回 out (B, T, d) 与 weights (B, h, T, S)。
        """
        x_kv = x_q if x_kv is None else x_kv
        if x_q.ndim != 3 or x_kv.ndim != 3:
            raise ValueError("输入必须是 (B, T, d) 三维张量")

        Q = self._split_heads(x_q @ self.W_q)  # (B, h, T, dk)
        K = self._split_heads(x_kv @ self.W_k)  # (B, h, S, dk)
        V = self._split_heads(x_kv @ self.W_v)  # (B, h, S, dk)

        ctx, w = scaled_dot_product_attention(Q, K, V, mask)  # (B,h,T,dk), (B,h,T,S)
        out = self._merge_heads(ctx) @ self.W_o  # (B, T, d)
        return out, w

    __call__ = forward

    def num_parameters(self) -> int:
        return sum(p.size for p in (self.W_q, self.W_k, self.W_v, self.W_o))


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """正弦位置编码（d2l 10.6）。

    必要性：自注意力本身对顺序**不敏感** —— 打乱词序，输出只是跟着重排，
    语义不变。位置编码把「位置」注入 token 表示，才让模型知道先后。
    """
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]  # (max_len, 1)
    i = np.arange(0, d_model, 2)[None, :]  # (1, d/2)
    angle = pos / np.power(10000.0, i / d_model)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : d_model // 2])
    return pe  # (max_len, d_model)
