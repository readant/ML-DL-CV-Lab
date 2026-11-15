"""分组注意力（Group Attention）—— 与具体项目无关的算法实现。

定位
----
本文件只包含算法本体。它不知道「电池」「SOH」「PINN」是什么，
也不知道组数一定是 4、某组一定是 17 维。这些都属于**调用方的配置**。

这样划分的理由：算法实现要能被别的任务复用、被独立检验；
一旦把某个项目的维度常量写进来，它就从算法退化成项目附件了。

机制
----
输入是若干组特征张量（各组列数可不同）：

    1. 每组经独立线性投影 → 一个 d_model 维 token
    2. 全部 token 做多头自注意力 → 组间交换信息
    3. 每个 token 生成标量门控 gᵢ（组内中心化，均值和恒为 1）
    4. 用 gᵢ 缩放第 i 组特征

与「逐维通道注意力」（如 SE-Net）的差别：逐维门控只能回答
「第 k 个特征多重要」；分组门控回答「这一类量整体该被放大还是压低」，
当输入天然带有语义分组时，后者的结论可解释。

三条硬约束（每条都有对应测试）
------------------------------
1. 恒等初始化：``gate_head`` 零初始化 → ``tanh(0)=0`` → ``gᵢ ≡ 1``。
   训练起点对输入是严格恒等变换，故「加与不加」的差别全部来自学到的偏离量，
   使「该模块是否有效」成为可证伪的问题。
2. 预算守恒：门控经组内中心化，``Σ(gᵢ − 1) ≡ 0``。
   只允许相对分配，不允许整体缩放——否则模块会退化成偷偷改学习率。
3. 幅度受限：``gate ∈ (1−2γ, 1+2γ)``，且要求 ``γ < 0.5``。
   早期版本取 γ=1，权重饱和时 ``tanh`` 返回精确 −1，gate 归零会把某一组
   整组抹掉且不产生任何报错。门控只能是微调，不能是开关。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["GroupAttention", "GroupGatedNet"]


class GroupAttention(nn.Module):
    """按预设分组对特征做组级注意力门控。

    Args:
        group_dims: 各组的特征列数，如 ``(17, 1, 16, 1)``。只声明结构，
            不携带任何领域含义。
        d_model: 每组投影到的 token 维度，须能被 ``nhead`` 整除。
        nhead: 多头注意力的头数。
        gate_scale: 门控幅度系数 γ，须在 (0, 0.5)。
        dropout: 注意力内部 dropout。

    形状：
        groups: list of [B, d_i]，共 len(group_dims) 个
        return: (list of [B, d_i], gate [B, n_group])
    """

    def __init__(self, group_dims: Sequence[int], d_model: int = 16,
                 nhead: int = 2, gate_scale: float = 0.25, dropout: float = 0.0):
        super().__init__()
        group_dims = tuple(int(d) for d in group_dims)
        if len(group_dims) < 2:
            raise ValueError(f"至少需要 2 组，收到 {len(group_dims)} 组")
        if any(d <= 0 for d in group_dims):
            raise ValueError(f"各组维度必须为正，收到 {group_dims}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
        if not 0 < gate_scale < 0.5:
            raise ValueError(
                f"gate_scale 须在 (0, 0.5)，收到 {gate_scale}。"
                f"取 0.5 会让 gate 在饱和时触及 0，把整组特征清零")

        self.group_dims = group_dims
        self.d_model = d_model
        self.gate_scale = gate_scale

        # 各组维度不同，无法共享投影权重
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in group_dims])
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          dropout=dropout, batch_first=True)
        self.gate_head = nn.Linear(d_model, 1)
        self._init_identity()

    def _init_identity(self) -> None:
        """使初始状态严格等价于恒等变换。"""
        for layer in self.proj:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.in_proj_bias, 0)
        nn.init.xavier_normal_(self.attn.out_proj.weight)
        nn.init.constant_(self.attn.out_proj.bias, 0)
        # 恒等性的唯一来源；改动此行会破坏整个设计前提
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def _make_gate(self, attn_out: torch.Tensor) -> torch.Tensor:
        """由注意力输出生成门控，并强制组内中心化。"""
        raw = torch.tanh(self.gate_head(attn_out)).squeeze(-1)   # [B, G] ∈ (−1, 1)
        centered = raw - raw.mean(dim=1, keepdim=True)           # 组内均值为 0
        return 1.0 + self.gate_scale * centered                  # 均值为 1

    def forward(self, groups: Sequence[torch.Tensor]):
        groups = list(groups)
        if len(groups) != len(self.group_dims):
            raise ValueError(
                f"期望 {len(self.group_dims)} 组，收到 {len(groups)} 组")
        for i, (g, d) in enumerate(zip(groups, self.group_dims)):
            if g.dim() != 2 or g.size(-1) != d:
                raise ValueError(
                    f"第 {i} 组期望 [B, {d}]，收到 {tuple(g.shape)}")
        batch = groups[0].size(0)
        if any(g.size(0) != batch for g in groups):
            raise ValueError("各组 batch size 不一致")

        toks = torch.stack([p(g) for p, g in zip(self.proj, groups)], dim=1)  # [B,G,d]
        attn_out, attn_w = self.attn(toks, toks, toks,
                                     need_weights=True, average_attn_weights=True)
        gate = self._make_gate(attn_out)                                     # [B,G]

        # gate[:, i:i+1] 为 [B,1]，与第 i 组 [B, d_i] 按列广播相乘。
        # 不要写成 split 后 unsqueeze(1)：那会得到 [B,1,1]，广播成 [B,1,d_i]。
        gated = [g * gate[:, i:i + 1] for i, g in enumerate(groups)]

        self.last_gate = gate
        self.last_attn_w = attn_w
        return gated, gate


class GroupGatedNet(nn.Module):
    """把 :class:`GroupAttention` 插到任意网络之前。

    用于「网络已经在跑、只想在输入侧加一个分组门控」的场景。
    ``net`` 的输入维度必须等于 ``sum(group_dims)``。

    前向：``z → 按 group_dims 切分 → 门控 → 拼回 → net``

    对外接口与 ``net`` 一致，可直接原位替换。恒等初始化保证替换后
    首步输出与原 ``net`` 相同（eval 模式下逐元素相等）。
    """

    def __init__(self, net: nn.Module, group_dims: Sequence[int], **ga_kwargs):
        super().__init__()
        group_dims = tuple(int(d) for d in group_dims)
        self.group_dims = group_dims
        self.gate_mod = GroupAttention(group_dims, **ga_kwargs)
        self.net = net
        self.last_gate = None

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        """只做门控，返回同形状张量。便于单独检查中间结果。"""
        if z.dim() != 2 or z.size(-1) != sum(self.group_dims):
            raise ValueError(
                f"期望 [B, {sum(self.group_dims)}]，收到 {tuple(z.shape)}")
        groups = torch.split(z, self.group_dims, dim=-1)
        gated, gate = self.gate_mod(list(groups))
        self.last_gate = gate
        return torch.cat(gated, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.gate(z))

    @torch.no_grad()
    def gate_summary(self, z: torch.Tensor) -> torch.Tensor:
        """返回 [B, G] 门控系数，供解释用（不参与梯度）。"""
        self.gate(z)
        return self.last_gate
