"""A5 条件调制（FiLM）的单元测试。

运行：python -m unittest test_film -v
"""
import unittest

import torch
import torch.nn as nn

from film import FiLM


class TestFiLM(unittest.TestCase):
    def test_shape_preserved(self):
        """调制是逐列变换，形状不变。"""
        film = FiLM(cond_dim=5, n_features=16)
        out = film(torch.randn(4, 16), torch.randn(4, 5))
        self.assertEqual(tuple(out.shape), (4, 16))

    def test_identity_at_init(self):
        """初始 γ≡1、β≡0，输出与输入逐元素相等 —— 干净的对照起点。"""
        film = FiLM(cond_dim=5, n_features=16)
        x = torch.randn(4, 16)
        self.assertTrue(torch.allclose(film(x, torch.randn(4, 5)), x))

    def test_identity_comes_from_init_not_structure(self):
        """反证：把 γ 偏置改成 0，恒等立刻消失，输出变成全 0。

        这条说明恒等是「初始化造出来的」，不是结构自带的性质。
        改动初始化就会失去它——所以别随手替换 init 写法。
        """
        film = FiLM(cond_dim=5, n_features=16)
        film.to_gamma.bias.data.fill_(0.0)
        x = torch.randn(4, 16)
        out = film(x, torch.randn(4, 5))
        self.assertFalse(torch.allclose(out, x, atol=1e-3))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_identity_init_can_restore(self):
        """identity_init() 必须能把被改坏的模块复位。"""
        film = FiLM(cond_dim=5, n_features=16)
        with torch.no_grad():
            nn.init.normal_(film.to_gamma.weight, std=3.0)
            nn.init.normal_(film.to_beta.bias, std=3.0)
        film.identity_init()
        x = torch.randn(4, 16)
        self.assertTrue(torch.allclose(film(x, torch.randn(4, 5)), x))

    def test_cond_dim_independent_of_features(self):
        """条件维度与特征维度无关，这是 FiLM 相对 SE 的关键差异。"""
        for d, c in [(1, 64), (5, 16), (32, 3)]:
            film = FiLM(cond_dim=d, n_features=c)
            out = film(torch.randn(2, c), torch.randn(2, d))
            self.assertEqual(tuple(out.shape), (2, c))

    def test_param_count_follows_formula(self):
        """参数量 = 2 × (d·C + C)。"""
        for d, c in [(5, 16), (1, 8), (32, 4)]:
            expected = 2 * (d * c + c)
            actual = sum(p.numel() for p in FiLM(d, c).parameters())
            self.assertEqual(actual, expected, f"d={d}, C={c}")

    def test_gamma_and_beta_actually_used(self):
        """γ 与 β 都必须参与计算，否则等于白加参数。"""
        film = FiLM(cond_dim=5, n_features=16)
        x = torch.randn(4, 16)
        cond = torch.randn(4, 5)
        out = film(x, cond)
        expected = film.to_gamma(cond) * x + film.to_beta(cond)
        self.assertTrue(torch.allclose(out, expected))

    def test_rejects_wrong_x_width(self):
        film = FiLM(cond_dim=5, n_features=16)
        with self.assertRaises(ValueError):
            film(torch.randn(4, 15), torch.randn(4, 5))

    def test_rejects_wrong_cond_width(self):
        film = FiLM(cond_dim=5, n_features=16)
        with self.assertRaises(ValueError):
            film(torch.randn(4, 16), torch.randn(4, 4))

    def test_rejects_batch_mismatch(self):
        film = FiLM(cond_dim=5, n_features=16)
        with self.assertRaises(ValueError):
            film(torch.randn(4, 16), torch.randn(3, 5))

    def test_deterministic_under_seed(self):
        torch.manual_seed(2)
        a = FiLM(5, 16)
        torch.manual_seed(2)
        b = FiLM(5, 16)
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
            self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main(verbosity=2)
