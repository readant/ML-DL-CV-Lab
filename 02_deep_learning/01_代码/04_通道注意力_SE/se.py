"""A1 通道注意力（Squeeze-and-Excitation）的最小实现。

来源：Hu, Shen, Sun. *Squeeze-and-Excitation Networks*. CVPR 2018.

本目录可整体拷到别的项目直接用，不依赖本仓库任何其它模块。

    python se.py                    # 自检，打印形状与参数量
    python -m unittest test_se -v   # 单元测试
"""
import torch
import torch.nn as nn

__all__ = ["SE1D"]


class SE1D(nn.Module):
    """一维特征上的通道注意力。

    Args:
        n_features: 特征列数 C。
        reduction: 压缩比 r，瓶颈宽度为 ``max(1, C // r)``。常用 8 或 16；
            ``r=1`` 时不压缩（即 sSE 变体）。

    形状：输入 ``[B, C]`` → 输出 ``([B, C], 门控 [B, C])``。

    注意：在纯向量输入上不存在「Squeeze」这一步（每列本来就是一个数），
    所以这里没有 ``AdaptiveAvgPool``——照抄图像版代码会多此一举。
    """

    def __init__(self, n_features: int, reduction: int = 8):
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features 必须为正，收到 {n_features}")
        if reduction < 1:
            raise ValueError(f"reduction 必须为正，收到 {reduction}")

        hidden = max(1, n_features // reduction)
        self.n_features = n_features
        self.reduction = reduction
        self.gate = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError(
                f"期望 [B, {self.n_features}]，收到 {tuple(x.shape)}")
        w = self.gate(x)          # [B, C]，值域 (0, 1)
        return x * w, w


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    se = SE1D(16, reduction=4)
    out, w = se(x)

    print(f"输入     {tuple(x.shape)}")
    print(f"输出     {tuple(out.shape)}")
    print(f"门控范围 {w.min().item():.3f} ~ {w.max().item():.3f}")
    print(f"参数量   {sum(p.numel() for p in se.parameters())}  (reduction=4)")
    print(f"参数量   {sum(p.numel() for p in SE1D(16, reduction=1).parameters())}  (reduction=1)")
    print(f"初始是否恒等: {torch.allclose(out, x, atol=1e-3)}  "
          f"← False 说明 sigmoid(0)=0.5 把输入整体压了一半")
