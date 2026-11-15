"""A5 条件调制（FiLM）的最小实现。

来源：Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.

用一路信息（cond）生成「缩放 γ + 平移 β」，去调整另一路信息（x）。
本目录可整体拷到别的项目，不依赖本仓库其它模块。

    python film.py
    python -m unittest test_film -v
"""
import torch
import torch.nn as nn

__all__ = ["FiLM"]


class FiLM(nn.Module):
    """用 cond 调制 x。

    Args:
        cond_dim: 条件向量维度 d，与 n_features 无关。
        n_features: 被调制特征的列数 C。

    形状：x ``[B, C]``、cond ``[B, d]`` → 输出 ``[B, C]``。

    恒等初始化：γ 的权重置零、偏置置 1，β 全部置零，
    于是初始 γ≡1、β≡0，输出与输入逐元素相等。
    这使「加 FiLM 前后」成为干净对照——差异只来自训练中学到的调制。
    """

    def __init__(self, cond_dim: int, n_features: int):
        super().__init__()
        if cond_dim < 1 or n_features < 1:
            raise ValueError(
                f"cond_dim 与 n_features 必须为正，收到 {cond_dim}, {n_features}")

        self.cond_dim = cond_dim
        self.n_features = n_features
        self.to_gamma = nn.Linear(cond_dim, n_features)
        self.to_beta = nn.Linear(cond_dim, n_features)
        self.identity_init()

    def identity_init(self) -> None:
        """把模块复位成恒等映射：γ≡1、β≡0。"""
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError(f"x 期望 [B, {self.n_features}]，收到 {tuple(x.shape)}")
        if cond.dim() != 2 or cond.size(1) != self.cond_dim:
            raise ValueError(f"cond 期望 [B, {self.cond_dim}]，收到 {tuple(cond.shape)}")
        if x.size(0) != cond.size(0):
            raise ValueError("x 与 cond 的 batch size 不一致")

        gamma = self.to_gamma(cond)      # [B, C]
        beta = self.to_beta(cond)        # [B, C]
        return gamma * x + beta


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    cond = torch.randn(4, 5)
    film = FiLM(cond_dim=5, n_features=16)
    out = film(x, cond)

    print(f"x {tuple(x.shape)} + cond {tuple(cond.shape)} → {tuple(out.shape)}")
    print(f"初始是否恒等: {torch.allclose(out, x)}")
    print(f"参数量 {sum(p.numel() for p in film.parameters())}  "
          f"= 2 × (5×16 + 16)")

    film.to_gamma.bias.data.fill_(0.0)   # 破坏 γ≡1
    print(f"把 γ 偏置改成 0 后是否恒等: {torch.allclose(film(x, cond), x)}"
          f"  ← 恒等是靠初始化造出来的，不是结构自带")
