"""A4 交叉注意力的最小实现。

来源：Vaswani et al. *Attention Is All You Need*. NeurIPS 2017
（decoder 中的 encoder-decoder attention）。

用一路信息出 Query，另一路信息出 Key / Value，做「查询—检索—加权取回」。
本目录可整体拷到别的项目，不依赖本仓库其它模块。

    python cross_attention.py
    python -m unittest test_cross_attention -v
"""
import torch
import torch.nn as nn

__all__ = ["CrossAttention"]


class CrossAttention(nn.Module):
    """单头交叉注意力。

    Args:
        d_q: Query 一路的特征维度。
        d_kv: Key/Value 一路的特征维度（可与 d_q 不同）。
        d_model: 投影后的公共维度 m。
        dropout: 输出 dropout。

    形状：``q_in [B, Nq, d_q]``、``kv_in [B, Nk, d_kv]`` → 输出 ``[B, Nq, d_q]``。

    注意：输出单元数跟着 **Query** 走，与 Nk 无关。这是选型时的关键判据。

    ⚠️ 若 Nq = Nk = 1（两路都是单向量），相似度只有一个数，softmax 后权重恒为 1，
    注意力完全失效，退化成一次线性变换。要让它工作，某一路必须先拆成多个单元。
    """

    def __init__(self, d_q: int, d_kv: int, d_model: int = 32, dropout: float = 0.0):
        super().__init__()
        if min(d_q, d_kv, d_model) < 1:
            raise ValueError("各维度必须为正")
        self.d_q = d_q
        self.d_kv = d_kv
        self.d_model = d_model

        self.q = nn.Linear(d_q, d_model)
        self.k = nn.Linear(d_kv, d_model)
        self.v = nn.Linear(d_kv, d_model)
        self.o = nn.Linear(d_model, d_q)
        self.drop = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                return_weights: bool = False):
        if q_in.dim() != 3 or q_in.size(-1) != self.d_q:
            raise ValueError(f"q_in 期望 [B, Nq, {self.d_q}]，收到 {tuple(q_in.shape)}")
        if kv_in.dim() != 3 or kv_in.size(-1) != self.d_kv:
            raise ValueError(f"kv_in 期望 [B, Nk, {self.d_kv}]，收到 {tuple(kv_in.shape)}")
        if q_in.size(0) != kv_in.size(0):
            raise ValueError("两路 batch size 不一致")

        q = self.q(q_in)                                  # [B, Nq, m]
        k = self.k(kv_in)                                 # [B, Nk, m]
        v = self.v(kv_in)                                 # [B, Nk, m]
        score = q @ k.transpose(-1, -2) * self.scale      # [B, Nq, Nk]
        w = torch.softmax(score, dim=-1)                  # 每行和为 1
        out = self.o(self.drop(w @ v))                    # [B, Nq, d_q]
        return (out, w) if return_weights else out


if __name__ == "__main__":
    torch.manual_seed(0)
    ca = CrossAttention(d_q=8, d_kv=5, d_model=16)

    q, kv = torch.randn(2, 4, 8), torch.randn(2, 6, 5)
    out, w = ca(q, kv, return_weights=True)
    print(f"正常      q{tuple(q.shape)} kv{tuple(kv.shape)} → {tuple(out.shape)}")
    print(f"          权重 {tuple(w.shape)}  "
          f"行和 {[round(float(v), 3) for v in w[0].sum(-1).detach()]}")

    q1, kv1 = torch.randn(2, 1, 8), torch.randn(2, 1, 5)
    _, w1 = ca(q1, kv1, return_weights=True)
    print(f"两路单 token  权重 {w1.flatten().tolist()}  ← 恒为 1，注意力失效")

    q2 = torch.randn(2, 1, 8)
    _, w2 = ca(q2, kv, return_weights=True)
    print(f"单查询多候选  权重 {[round(float(v), 3) for v in w2[0, 0].detach()]}  ← 仍有效")
