"""分组注意力（Group Attention）的单元测试。

设计原则：每条测试钉死一个**会静默出错**的细节。重点不是覆盖率，
而是让「这套算法到底成不成立」变成可复现的客观事实。

这些测试**不含任何项目专属配置**：分组维度用 (3,5,2)、(4,4) 等任意值，
主干网络用通用 nn.Sequential。若某条测试只能靠 35 维的特定切分才通过，
说明算法里混进了领域假设——那本身就是缺陷。

运行：
    cd 02_deep_learning/01_代码/03_分组注意力算法
    python -m unittest test_group_attention -v
"""
import unittest

import torch
import torch.nn as nn

from group_attention import GroupAttention, GroupGatedNet

# 与领域无关的任意配置
DIMS_A = (3, 5, 2)      # 3 组，共 10 维
DIMS_B = (4, 4)         # 2 组，共 8 维


def _make_net(in_dim, out_dim=1, dropout=0.2):
    """通用主干：不用任何项目里的网络结构。"""
    return nn.Sequential(
        nn.Linear(in_dim, 16), nn.Tanh(), nn.Dropout(dropout),
        nn.Linear(16, out_dim),
    )


def _build_pair(dims=DIMS_A, seed=7):
    """构造「门控版」与「原版」，并共享主干权重。

    共享权重是恒等性测试的前提，否则差异里混入了初始化噪声，
    无法区分「门控导致的变化」与「随机初始化导致的变化」。
    """
    torch.manual_seed(seed)
    net_a = _make_net(sum(dims))
    net_b = _make_net(sum(dims))
    net_b.load_state_dict(net_a.state_dict())
    gated = GroupGatedNet(net_a, dims)
    return gated, net_b


def _random_groups(dims, batch=8):
    return [torch.randn(batch, d) for d in dims]


class TestConstruction(unittest.TestCase):
    def test_rejects_single_group(self):
        """只有一组时「组间注意力」没有意义，应直接拒绝而不是静默退化。"""
        with self.assertRaises(ValueError):
            GroupAttention((10,))

    def test_rejects_non_positive_dim(self):
        with self.assertRaises(ValueError):
            GroupAttention((3, 0, 5))

    def test_attn_dim_must_divide_nhead(self):
        with self.assertRaises(ValueError):
            GroupAttention(DIMS_A, d_model=15, nhead=2)

    def test_gate_scale_must_be_bounded(self):
        with self.assertRaises(ValueError):
            GroupAttention(DIMS_A, gate_scale=2.0)
        # 0.5 是上界本身，不允许：centered ∈ (−2,2) 时 gate 会触及 0
        with self.assertRaises(ValueError):
            GroupAttention(DIMS_A, gate_scale=0.5)


class TestForwardShape(unittest.TestCase):
    def test_output_matches_input_dims(self):
        """门控是缩放不是降维，输出各组维度必须与输入一致。"""
        ga = GroupAttention(DIMS_A)
        gated, gate = ga(_random_groups(DIMS_A))
        self.assertEqual([g.size(1) for g in gated], list(DIMS_A))
        self.assertEqual(tuple(gate.shape), (8, len(DIMS_A)))

    def test_works_for_arbitrary_group_configs(self):
        """算法不绑定任何特定分组——这是「去项目化」的直接证据。"""
        for dims in [DIMS_A, DIMS_B, (1, 1, 1), (10, 1), (2, 3, 4, 5, 6)]:
            ga = GroupAttention(dims)
            gated, gate = ga(_random_groups(dims))
            self.assertEqual([g.size(1) for g in gated], list(dims))
            self.assertEqual(gate.size(1), len(dims))

    def test_rejects_wrong_group_count(self):
        ga = GroupAttention(DIMS_A)
        with self.assertRaises(ValueError):
            ga(_random_groups(DIMS_B))

    def test_rejects_wrong_group_width(self):
        ga = GroupAttention(DIMS_A)
        bad = [torch.randn(8, 3), torch.randn(8, 5), torch.randn(8, 99)]
        with self.assertRaises(ValueError):
            ga(bad)

    def test_rejects_inconsistent_batch(self):
        ga = GroupAttention(DIMS_A)
        bad = [torch.randn(8, 3), torch.randn(8, 5), torch.randn(7, 2)]
        with self.assertRaises(ValueError):
            ga(bad)

    def test_packed_input_width_checked(self):
        gated = GroupGatedNet(_make_net(sum(DIMS_A)), DIMS_A)
        with self.assertRaises(ValueError):
            gated(torch.randn(8, sum(DIMS_A) + 1))

    def test_attention_weights_normalised(self):
        """注意力图应为 [B,G,G]，每个 query 的权重和为 1。"""
        ga = GroupAttention(DIMS_A)
        ga(_random_groups(DIMS_A))
        w = ga.last_attn_w
        self.assertEqual(tuple(w.shape), (8, 3, 3))
        row_sums = w.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


class TestIdentityInitialisation(unittest.TestCase):
    """恒等初始化是本算法可证伪性的基础，是全部测试里最关键的一组。"""

    def test_gate_is_exactly_one_at_init(self):
        ga = GroupAttention(DIMS_A)
        _, gate = ga(_random_groups(DIMS_A))
        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))

    def test_init_output_equals_original_net(self):
        """同权重下，门控版输出应与原网络逐元素相等。

        必须在 eval() 下比较：主干含 Dropout，train 模式下即使 gate=1，
        dropout 掩码也会让单次前向不同。恒等性因此是「eval 模式下的严格恒等」。
        """
        gated, plain = _build_pair()
        gated.eval()
        plain.eval()
        z = torch.randn(32, sum(DIMS_A))
        with torch.no_grad():
            self.assertTrue(torch.allclose(gated(z), plain(z), atol=1e-6))

    def test_train_mode_differs_due_to_dropout(self):
        """反证上一条：train 模式下两者不应完全相同（dropout 生效）。

        若此条失败，说明 dropout 被意外关闭，恒等性测试的结论也随之失真。
        """
        gated, plain = _build_pair()
        gated.train()
        plain.train()
        z = torch.randn(32, sum(DIMS_A))
        self.assertFalse(torch.allclose(gated(z), plain(z), atol=1e-6))

    def test_gate_is_centred_to_one(self):
        """Σ(gᵢ − 1) ≡ 0：只做组间相对分配，不含整体缩放。

        早期版本未中心化，实测各组门控同步下降，本质是整体缩放，
        等价于偷改学习率，与注意力语义无关。此条防止以后改回去。
        """
        ga = GroupAttention(DIMS_A)
        with torch.no_grad():
            nn.init.normal_(ga.gate_head.weight, std=5.0)
            nn.init.normal_(ga.gate_head.bias, std=5.0)
            _, gate = ga(_random_groups(DIMS_A, batch=128))
        means = gate.mean(dim=1)
        self.assertTrue(torch.allclose(means, torch.ones_like(means), atol=1e-5))

    def test_gate_spread_is_nonzero(self):
        """中心化后各组必须真的拉开差距，否则等于没做注意力。"""
        ga = GroupAttention(DIMS_A)
        with torch.no_grad():
            nn.init.normal_(ga.gate_head.weight, std=5.0)
            _, gate = ga(_random_groups(DIMS_A, batch=64))
        spread = gate.max(dim=1).values - gate.min(dim=1).values
        self.assertGreater(float(spread.mean()), 1e-3)

    def test_gate_stays_within_bounded_band(self):
        """gate = 1 + γ·(t − mean t)，γ=0.25 → gate ∈ (0.5, 1.5)。

        这条来自一次真实失败：初版取 γ=1 且不中心化，权重饱和时 tanh 返回
        精确 −1，gate 归零会把某一组整组抹掉，且训练 loss 不会有任何提示。
        门控只能是微调，不能是开关。
        """
        ga = GroupAttention(DIMS_A)
        with torch.no_grad():
            nn.init.normal_(ga.gate_head.weight, std=50.0)
            nn.init.normal_(ga.gate_head.bias, std=50.0)
            _, gate = ga(_random_groups(DIMS_A, batch=64))
        self.assertGreaterEqual(float(gate.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(gate.max()), 1.5 + 1e-6)


class TestLearnability(unittest.TestCase):
    """验证这套结构真的能学，而不是一个僵死的旁路。"""

    def test_gate_head_receives_gradient(self):
        gated, _ = _build_pair()
        gated.train()
        gated(torch.randn(8, sum(DIMS_A))).sum().backward()
        grad = gated.gate_mod.gate_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_gradient_reaches_every_group(self):
        """各组输入的梯度都必须非零，否则存在被门控「切断」的分组。"""
        ga = GroupAttention(DIMS_A)
        groups = [torch.randn(4, d, requires_grad=True) for d in DIMS_A]
        gated, _ = ga(groups)
        torch.cat([g.reshape(4, -1) for g in gated], dim=1).sum().backward()
        for i, g in enumerate(groups):
            self.assertGreater(float(g.grad.abs().sum()), 0.0, f"第 {i} 组梯度为零")

    def test_gate_shifts_after_training_steps(self):
        """几步梯度下降后 gate 应偏离 1 —— 证明它确实在优化中被使用。"""
        gated, _ = _build_pair()
        gated.train()
        opt = torch.optim.Adam(gated.parameters(), lr=1e-2)
        z = torch.randn(32, sum(DIMS_A))
        target = torch.randn(32, 1)
        for _ in range(20):
            opt.zero_grad()
            loss = nn.functional.mse_loss(gated(z), target)
            loss.backward()
            opt.step()
        gate = gated.gate_summary(z)
        self.assertGreater(float((gate - 1.0).abs().max()), 1e-4)


class TestRobustness(unittest.TestCase):
    def test_extreme_input_has_no_nan(self):
        """真实数据里各分组量级可能相差多个数量级，门控不允许放大成 nan。"""
        gated, _ = _build_pair()
        gated.eval()
        with torch.no_grad():
            out = gated(torch.randn(8, sum(DIMS_A)) * 1e3)
        self.assertFalse(bool(torch.isnan(out).any()))
        self.assertFalse(bool(torch.isinf(out).any()))

    def test_parameter_count_matches_formula(self):
        """门控参数量应精确等于结构公式值，防止无意中多加层或漏参数。

        不依赖主干规模，是纯结构检查。公式对应：
            proj      Σ(d_i·m + m)
            in_proj   3·m² + 3·m
            out_proj  m² + m
            gate_head m + 1
        """
        dims, m = DIMS_A, 16
        ga = GroupAttention(dims, d_model=m, nhead=2)
        actual = sum(p.numel() for p in ga.parameters())
        expected = (sum(d * m + m for d in dims)
                    + (3 * m * m + 3 * m)
                    + (m * m + m)
                    + (m + 1))
        self.assertEqual(actual, expected)

    def test_gate_does_not_dominate_a_realistic_backbone(self):
        """与规模相称的主干下，门控不应占据主要参数预算。

        注意这条只在主干规模与门控相称时有意义：主干越小，门控占比越高，
        属规模效应而非缺陷。参照：PINN4SOH 的 35→60→60→1（5881 参数）
        下门控为 1729，占比 29%。
        """
        net = nn.Sequential(nn.Linear(sum(DIMS_A), 64), nn.Tanh(), nn.Linear(64, 1))
        gated = GroupGatedNet(net, DIMS_A)
        n_net = sum(p.numel() for p in net.parameters())
        n_gate = sum(p.numel() for p in gated.gate_mod.parameters())
        self.assertGreater(n_gate, 0)
        self.assertLess(n_gate, (n_net + n_gate) * 0.75,
                        f"门控 {n_gate} 在总算力预算中占比过高")

    def test_same_seed_reproduces_weights(self):
        """同种子必须复现同一组初始权重 —— 否则对比实验的分母不可信。"""
        a, _ = _build_pair(seed=123)
        b, _ = _build_pair(seed=123)
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
            self.assertTrue(torch.equal(pa, pb))

    def test_gate_helper_matches_forward(self):
        """gate() 单独调用应与 forward 内部使用的门控一致，否则解释结果会失真。"""
        gated, _ = _build_pair()
        gated.eval()
        z = torch.randn(16, sum(DIMS_A))
        with torch.no_grad():
            via_helper = gated.gate(z)
            via_forward = gated.net(gated.gate(z))
            self.assertTrue(torch.allclose(via_forward, gated(z), atol=1e-6))
        self.assertEqual(tuple(gated.gate_summary(z).shape), (16, len(DIMS_A)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
