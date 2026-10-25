"""
demo_attention.py —— 把注意力权重打出来看（纯 NumPy，不需要训练）

演示 4 件事，全部可复现、不依赖任何预训练权重：
  ① 形状与归一化：权重矩阵长什么样，每行和为 1
  ② 因果掩码：只能看左边，未来位置权重恰为 0
  ③ **W_q 与 W_k 必须分开**：绑定时每个词最强关注自己（对角线主导），分开才有机动性
  ④ 多头：不同头的关注点不同（随机初始化，差异本身没有语义含义）

⚠ 声明：词向量与投影矩阵都是随机生成的，本文件演示的是「机制」而不是「效果」。
   网上流传的「it 指向 animal」来自训练好的模型，不能用手写权重复现，
   所以这里不做那种演示，改为展示可验证的结构性质。

运行：python demo_attention.py（在本目录或仓库根目录都可以）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让脚本在任意工作目录下都能找到同目录的 attention_numpy
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import attention_numpy as an  # noqa: E402

TOKENS = ["the", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
T = len(TOKENS)
SHADES = " .:-=+*#%@"  # 10 级灰度


def toy_embeddings(d: int = 16, seed: int = 7) -> np.ndarray:
    """随机词向量（固定种子，可复现）。"""
    return np.random.default_rng(seed).normal(0, 1, size=(1, T, d))


def bar(v: float, width: int = 24) -> str:
    n = int(round(v * width))
    return "█" * max(n, 0) + "·" * max(width - n, 0)


def show_row(w: np.ndarray, q: int, head: int = 0, topk: int = 4) -> None:
    row = w[0, head, q]
    print(f"  查询词 «{TOKENS[q]}»（head {head}）")
    for j in np.argsort(-row)[:topk]:
        print(f"    → {TOKENS[j]:<10} {row[j]:.3f}  {bar(float(row[j]))}")
    print(f"    权重和 {row.sum():.3f}\n")


def show_matrix(w: np.ndarray, head: int = 0) -> None:
    """每行归一到 [0,1] 再映射到 10 级灰度，避免整片空白。"""
    raw = w[0, head]
    M = raw / (raw.max(axis=1, keepdims=True) + 1e-12)  # 仅用于上色
    print("         " + "".join(f"{t[:2]:>3}" for t in TOKENS) + "       行内最强者（真实权重）")
    for i in range(T):
        cells = "".join(f"  {SHADES[min(9, int(v * 9.99))]:>2}" for v in M[i])
        am = int(np.argmax(raw[i]))
        print(f"  {TOKENS[i]:<8}{cells}   {TOKENS[am]}({raw[i, am]:.3f})")
    print()


def main() -> None:
    d = 16
    x = toy_embeddings(d)

    # ---------- ① 形状与归一化 ----------
    print("=" * 80)
    print(f"① 形状与归一化：输入 {x.shape} → 权重 (B,h,T,T)")
    print("=" * 80)
    m = an.MultiHeadAttention(d, num_heads=4, rng=np.random.default_rng(0))
    out, w = m(x)
    print(f"  输出 {out.shape}   权重 {w.shape}")
    print(f"  每行权重和：min={w.sum(-1).min():.6f}  max={w.sum(-1).max():.6f}（都应为 1）")
    print(f"  参数量 {m.num_parameters()} = 4×d² = {4 * d * d}\n")
    show_matrix(w, head=0)

    # ---------- ② 因果掩码 ----------
    print("=" * 80)
    print("② 因果掩码：位置 i 只能看到 j ≤ i")
    print("=" * 80)
    _, wc = m(x, mask=an.causal_mask(T))
    fut = np.triu_indices(T, k=1)
    print(f"  未来位置（上三角）权重之和 = {wc[..., fut[0], fut[1]].sum():.1e}  ← 必须为 0")
    print(f"  每行权重和仍为 1：{np.allclose(wc.sum(-1), 1.0)}")
    show_row(wc, q=TOKENS.index("it"))
    show_row(wc, q=0)  # 第一个词只能看自己 → 权重必然是 1.000
    show_matrix(wc, head=0)

    # ---------- ③ W_q = W_k 的退化 ----------
    print("=" * 80)
    print("③ 为什么 W_q 和 W_k 必须分开？")
    print("=" * 80)
    xn = x[0]  # (T,d)
    W = np.random.default_rng(1).normal(0, 1 / np.sqrt(d), (d, d))

    # 绑定：Q = K = xW。分数矩阵对称，对角线 x_iᵀWWᵀx_i 通常最大
    z = xn @ W
    w_tied = an.softmax(z @ z.T / np.sqrt(d))
    print(
        f"  绑定 Q=K ：对角线（自己→自己）平均权重 {np.mean(np.diag(w_tied)):.3f}；"
        f"{np.mean(np.argmax(w_tied, axis=1) == np.arange(T)):.0%} 的词最强关注自己"
    )

    # 分开：Q、K 用两个独立矩阵
    Wq = np.random.default_rng(2).normal(0, 1 / np.sqrt(d), (d, d))
    Wk = np.random.default_rng(3).normal(0, 1 / np.sqrt(d), (d, d))
    w_sep = an.softmax((xn @ Wq) @ (xn @ Wk).T / np.sqrt(d))
    print(
        f"  分离 Q≠K：对角线平均权重 {np.mean(np.diag(w_sep)):.3f}；"
        f"{np.mean(np.argmax(w_sep, axis=1) == np.arange(T)):.0%} 的词最强关注自己"
    )
    print()
    print("  绑定时分数矩阵对称、每个词与自己最相似，注意力退化成「照镜子」；")
    print("  分开投影后 Q 空间与 K 空间不同，词才能去关注别的词 ——")
    print("  这就是每个头都要有独立 W_q、W_k 的原因（Bishop 12.1.6 p.382）。\n")

    # ---------- ④ 多头 ----------
    print("=" * 80)
    print("④ 多头：四个头看的东西不一样（随机初始化，差异无语义含义）")
    print("=" * 80)
    q = TOKENS.index("it")
    for h in range(4):
        row = w[0, h, q]
        top = TOKENS[int(np.argmax(row))]
        ent = float(-np.sum(row * np.log(row + 1e-12)))
        print(f"  head {h}: «it» 最关注 «{top}»（{row.max():.3f}）　分布熵 {ent:.2f} （上限 ln{T} = {np.log(T):.2f}）")
    print()
    print("  熵越接近上限越平均。随机初始化的头差异是噪声；")
    print("  训练后才会分化成「有的抓句法、有的抓共指」—— 见 Voita et al. 2019 的头消融实验。")


if __name__ == "__main__":
    main()
