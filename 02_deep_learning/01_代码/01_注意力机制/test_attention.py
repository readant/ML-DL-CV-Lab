"""
test_attention.py —— 用可执行的断言验证「注意力为什么这样设计」

测试不是走过场。下面每条都对应一个容易搞错或容易被论文含糊带过的结论：

  [1] 权重每行和为 1                     —— 归一化的定义
  [2] 因果掩码后，未来位置的权重必须恰为 0  —— mask 必须加在 softmax 之前
  [3] 「先 softmax 再乘 mask」会让剩余权重和 < 1 —— 用数字证明 [2] 的必要性
  [4] 参数量 = 4d²，与头数无关             —— 多头不是 h 倍参数
  [5] h=1 时多头退化为单头（逐位相等）      —— 多头是单头的推广而非替代
  [6] 无位置编码时自注意力对置换等变        —— 这就是必须加位置编码的原因
  [7] 加上正弦位置编码后置换等变性被破坏     —— 位置编码确实注入了顺序信息
  [8] NumPy 版与 PyTorch 版逐位一致        —— 两份实现互相校验
  [9] PyTorch 版与官方 F.scaled_dot_product_attention 一致 —— 与工业实现对齐
  [10] 梯度可回传且有限                    —— 能训练

运行：python -m unittest test_attention -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# 让测试在任意工作目录下都能导入同目录模块（仓库根目录、模块目录均可）
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import attention_numpy as an  # noqa: E402

try:
    import attention_torch as at  # noqa: E402
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False


def _rng(seed=42):
    return np.random.default_rng(seed)


class TestNumPyAttention(unittest.TestCase):
    def setUp(self):
        self.rng = _rng()
        self.B, self.T, self.d, self.h = 2, 5, 16, 4
        self.x = self.rng.normal(size=(self.B, self.T, self.d))
        self.m = an.MultiHeadAttention(self.d, self.h, self.rng)

    # [1]
    def test_rows_sum_to_one(self):
        _, w = self.m(self.x)
        self.assertEqual(w.shape, (self.B, self.h, self.T, self.T))
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    # [2]
    def test_causal_mask_zeroes_future(self):
        _, w = self.m(self.x, mask=an.causal_mask(self.T))
        fut = np.triu_indices(self.T, k=1)
        np.testing.assert_array_equal(w[..., fut[0], fut[1]], 0.0)
        # 且剩余部分依然归一化
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    # [3]
    def test_softmax_then_mask_breaks_normalization(self):
        """先 softmax 再乘 mask 会让有效位置的权重之和 < 1，输出被系统性缩小。"""
        Q = self.rng.normal(size=(1, 1, 1, self.d))
        K = self.rng.normal(size=(1, 1, self.T, self.d))
        V = self.rng.normal(size=(1, 1, self.T, self.d))
        m3 = an.causal_mask(1, self.T)  # 第 0 行只看得到第 0 个 key

        # 正确做法：mask 进 softmax
        _, w_ok = an.scaled_dot_product_attention(Q, K, V, mask=m3)
        self.assertAlmostEqual(float(w_ok.sum()), 1.0, places=12)

        # 错误做法：softmax 之后乘 0/1
        scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(self.d)
        w_bad = an.softmax(scores) * m3
        self.assertLess(float(w_bad.sum()), 1.0 - 1e-6)
        # 差值就是被屏蔽位置原本占的概率质量
        self.assertAlmostEqual(float(1.0 - w_bad.sum()), float(an.softmax(scores)[..., ~m3].sum()), places=12)

    # [4]
    def test_parameter_count_is_4d2(self):
        self.assertEqual(self.m.num_parameters(), 4 * self.d * self.d)
        m1 = an.MultiHeadAttention(self.d, 1, self.rng)
        self.assertEqual(m1.num_parameters(), self.m.num_parameters())

    # [5]
    def test_single_head_equals_manual_attention(self):
        d = 8
        m1 = an.MultiHeadAttention(d, 1, _rng(7))
        x = _rng(11).normal(size=(3, 4, d))

        Q, K, V = x @ m1.W_q, x @ m1.W_k, x @ m1.W_v
        scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
        manual = (an.softmax(scores) @ V) @ m1.W_o

        out, w = m1(x)
        np.testing.assert_allclose(out, manual, atol=1e-12)
        self.assertEqual(w.shape, (3, 1, 4, 4))

    # [6]
    def test_permutation_equivariance_without_position(self):
        """打乱输入顺序，输出只是跟着重排 —— 自注意力本身看不见顺序。"""
        perm = self.rng.permutation(self.T)
        out_a, _ = self.m(self.x)
        out_b, _ = self.m(self.x[:, perm, :])
        np.testing.assert_allclose(out_b, out_a[:, perm, :], atol=1e-12)

    # [7]
    def test_positional_encoding_breaks_equivariance(self):
        """位置编码绑定在「位置槽位」上：只打乱词、位置不动，输出必须改变。

        注意别写成「词和位置编码一起置换」—— 那仍然是等变的，测不出东西。
        """
        pe = an.sinusoidal_positional_encoding(self.T, self.d)[None, :, :]
        perm = self.rng.permutation(self.T)

        out_a, _ = self.m(self.x + pe)  # 原顺序
        out_b, _ = self.m(self.x[:, perm, :] + pe)  # 词打乱，位置编码仍在原槽位
        self.assertFalse(
            np.allclose(out_b, out_a[:, perm, :], atol=1e-6), "词序被打乱但输出只是重排 —— 说明位置编码没起作用"
        )

        # 反过来：词与位置编码同步置换，则仍应等变（位置标签跟着词走）
        out_c, _ = self.m(self.x[:, perm, :] + pe[:, perm, :])
        np.testing.assert_allclose(out_c, out_a[:, perm, :], atol=1e-12)

    def test_softmax_numerically_stable(self):
        big = np.array([[1e4, 0.0, -1e4]])
        w = an.softmax(big)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)


@unittest.skipUnless(HAS_TORCH, "未安装 torch，跳过 PyTorch 相关断言")
class TestTorchAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.d, self.h = 2, 5, 16, 4
        self.x = torch.randn(self.B, self.T, self.d)

    # [8]
    def test_numpy_and_torch_match(self):
        rng = _rng(3)
        mn = an.MultiHeadAttention(self.d, self.h, rng)
        xn = rng.normal(size=(self.B, self.T, self.d))

        mt = at.MultiHeadAttention(self.d, self.h).load_from_numpy(mn)
        out_np, w_np = mn(xn)
        out_t, w_t = mt(torch.as_tensor(xn, dtype=torch.float32))
        # 精度差异来自 float64 vs float32
        np.testing.assert_allclose(out_t.detach().numpy(), out_np, atol=1e-5)
        np.testing.assert_allclose(w_t.detach().numpy(), w_np, atol=1e-6)

    # [9]
    def test_matches_torch_official_sdpa(self):
        """与 torch.nn.functional.scaled_dot_product_attention 对齐（B,H,T,dk 布局）。"""
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 4, 5, 8)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn = at.ScaledDotProductAttention()
        out, _ = attn(q, k, v, mask=at.causal_mask_bool(5))
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)

    # [10]
    def test_gradients_flow(self):
        mha = at.MultiHeadAttention(self.d, self.h)
        out, w = mha(self.x, is_causal=True)
        loss = (out**2).mean()
        loss.backward()
        grads = [p.grad for p in mha.parameters() if p.grad is not None]
        self.assertTrue(grads, "没有任何梯度")
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())

    def test_causal_mask_blocks_gradient_to_future(self):
        """因果掩码不仅置零权重，还要切断对未来位置的梯度。"""
        mha = at.MultiHeadAttention(8, 2)
        x = torch.randn(1, 4, 8, requires_grad=True)
        out, w = mha(x, is_causal=True)
        out[0, 0].sum().backward()  # 只对第 0 个输出位置求导
        g = x.grad[0]  # (T, d)
        self.assertTrue(
            torch.allclose(g[1:], torch.zeros_like(g[1:]), atol=1e-7), "第 0 个位置不该影响后面的 token，但梯度泄漏了"
        )

    def test_transformer_block_shape(self):
        blk = at.TransformerBlock(self.d, self.h, d_ff=32)
        y, w = blk(self.x)
        self.assertEqual(tuple(y.shape), (self.B, self.T, self.d))
        self.assertEqual(tuple(w.shape), (self.B, self.h, self.T, self.T))

    def test_cross_attention_shapes(self):
        mha = at.MultiHeadAttention(self.d, self.h)
        kv = torch.randn(self.B, 7, self.d)
        out, w = mha(self.x, x_kv=kv)
        self.assertEqual(tuple(out.shape), (self.B, self.T, self.d))
        self.assertEqual(tuple(w.shape), (self.B, self.h, self.T, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
