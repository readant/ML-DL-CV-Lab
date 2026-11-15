"""A1 通道注意力（SE）的单元测试。

运行：python -m unittest test_se -v
"""
import unittest

import torch

from se import SE1D


class TestSE1D(unittest.TestCase):
    def test_shape_preserved(self):
        """门控是缩放不是降维，输入输出同形状。"""
        se = SE1D(16, reduction=4)
        out, w = se(torch.randn(4, 16))
        self.assertEqual(tuple(out.shape), (4, 16))
        self.assertEqual(tuple(w.shape), (4, 16))

    def test_gate_in_open_unit_interval(self):
        """sigmoid 输出严格落在 (0, 1)，不会出现 0 或 1。"""
        se = SE1D(16, reduction=4)
        _, w = se(torch.randn(64, 16))
        self.assertGreater(float(w.min()), 0.0)
        self.assertLess(float(w.max()), 1.0)

    def test_output_is_input_times_gate(self):
        """输出的定义就是输入逐列乘门控，这条钉死语义。"""
        se = SE1D(16, reduction=4)
        x = torch.randn(4, 16)
        out, w = se(x)
        self.assertTrue(torch.allclose(out, x * w))

    def test_param_count_follows_formula(self):
        """参数量 = C·h + h + h·C + C，其中 h = max(1, C//r)。

        不依赖具体数值，防止以后改动结构导致参数悄悄变化。
        """
        for c, r in [(16, 4), (16, 1), (32, 8), (7, 8), (5, 2)]:
            h = max(1, c // r)
            expected = c * h + h + h * c + c
            actual = sum(p.numel() for p in SE1D(c, r).parameters())
            self.assertEqual(actual, expected, f"C={c}, r={r}")

    def test_not_identity_at_init(self):
        """反证：SE 初始**不**恒等。

        sigmoid(0) = 0.5，所以刚初始化时输出约为输入的一半。
        这意味着「加 SE 前后」不是干净对照——效果变差时无法区分
        是「SE 没用」还是「初始缩放破坏了训练」。
        """
        se = SE1D(16, reduction=4)
        x = torch.randn(4, 16)
        out, _ = se(x)
        self.assertFalse(torch.allclose(out, x, atol=1e-3))

    def test_reduction_one_gives_most_params(self):
        """r=1 即 sSE 变体（不压缩），参数量最大——对应文档里的动手练习。"""
        wide = sum(p.numel() for p in SE1D(16, reduction=1).parameters())
        narrow = sum(p.numel() for p in SE1D(16, reduction=4).parameters())
        self.assertGreater(wide, narrow)

    def test_rejects_wrong_input_width(self):
        se = SE1D(16, reduction=4)
        with self.assertRaises(ValueError):
            se(torch.randn(4, 15))

    def test_rejects_invalid_reduction(self):
        with self.assertRaises(ValueError):
            SE1D(16, reduction=0)

    def test_rejects_invalid_features(self):
        with self.assertRaises(ValueError):
            SE1D(0)

    def test_deterministic_under_seed(self):
        torch.manual_seed(1)
        a = SE1D(16, reduction=4)
        torch.manual_seed(1)
        b = SE1D(16, reduction=4)
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
            self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main(verbosity=2)
