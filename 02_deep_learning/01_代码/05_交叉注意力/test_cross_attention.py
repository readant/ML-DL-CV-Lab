"""A4 交叉注意力的单元测试。

运行：python -m unittest test_cross_attention -v
"""
import unittest

import torch

from cross_attention import CrossAttention


class TestCrossAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ca = CrossAttention(d_q=8, d_kv=5, d_model=16)

    def test_output_shape_follows_query(self):
        """输出单元数跟着 Query 走，与 Nk 无关 —— 这是选型的关键判据。"""
        out = self.ca(torch.randn(2, 4, 8), torch.randn(2, 6, 5))
        self.assertEqual(tuple(out.shape), (2, 4, 8))

    def test_output_shape_unaffected_by_nk(self):
        for nk in (2, 6, 17):
            out = self.ca(torch.randn(2, 4, 8), torch.randn(2, nk, 5))
            self.assertEqual(tuple(out.shape), (2, 4, 8))

    def test_weights_shape_and_row_normalised(self):
        _, w = self.ca(torch.randn(2, 4, 8), torch.randn(2, 6, 5), return_weights=True)
        self.assertEqual(tuple(w.shape), (2, 4, 6))
        row_sums = w.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_degenerates_when_both_sides_single_token(self):
        """两路都只有 1 个单元时，权重恒为 1，注意力完全失效。

        这是最常见的误用：把两个向量送进来，以为在做融合，其实只做了一次线性变换。
        """
        _, w = self.ca(torch.randn(2, 1, 8), torch.randn(2, 1, 5), return_weights=True)
        self.assertTrue(torch.allclose(w, torch.ones_like(w), atol=1e-6))

    def test_still_works_with_single_query_many_candidates(self):
        """只要有一路是多单元，注意力依然有效。

        与上一条对比：真正失效的是「Nq=1 且 Nk=1」，不是「Nq=1」。
        """
        _, w = self.ca(torch.randn(2, 1, 8), torch.randn(2, 6, 5), return_weights=True)
        self.assertFalse(torch.allclose(w, torch.ones_like(w), atol=1e-3))
        self.assertAlmostEqual(float(w[0, 0].sum()), 1.0, places=5)

    def test_param_count_follows_formula(self):
        """参数量 = 2(d_q·m + m) + 2(d_kv·m + m) ... 具体为 q/k/v/o 四项之和。"""
        d_q, d_kv, m = 8, 5, 16
        expected = ((d_q * m + m) + (d_kv * m + m) + (d_kv * m + m) + (m * d_q + d_q))
        actual = sum(p.numel() for p in self.ca.parameters())
        self.assertEqual(actual, expected)

    def test_no_nan_on_large_input(self):
        out = self.ca(torch.randn(2, 4, 8) * 1e3, torch.randn(2, 6, 5) * 1e3)
        self.assertFalse(bool(torch.isnan(out).any()))
        self.assertFalse(bool(torch.isinf(out).any()))

    def test_rejects_2d_query(self):
        """单向量输入必须先 unsqueeze 成 [B, 1, d]，否则会被拦下。"""
        with self.assertRaises(ValueError):
            self.ca(torch.randn(2, 8), torch.randn(2, 6, 5))

    def test_rejects_wrong_feature_width(self):
        with self.assertRaises(ValueError):
            self.ca(torch.randn(2, 4, 9), torch.randn(2, 6, 5))

    def test_rejects_batch_mismatch(self):
        with self.assertRaises(ValueError):
            self.ca(torch.randn(2, 4, 8), torch.randn(3, 6, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
