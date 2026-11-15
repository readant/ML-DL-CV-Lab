# 06 条件调制（FiLM）· 实现

技术点 **A5** 的可运行实现。技术点本身的讲解（机制、形状变化、何时用别用、变体对比）在笔记里：

[`../../00_笔记/01_注意力机制/技术点A5-条件调制_FiLM.md`](../../00_笔记/01_注意力机制/技术点A5-条件调制_FiLM.md)

## 接口

```python
from film import FiLM

film = FiLM(cond_dim=5, n_features=16)
out = film(x, cond)     # x: [B,16] + cond: [B,5] → [B,16]

film.identity_init()    # 复位为恒等映射（γ≡1, β≡0）
```

## 跑

```bash
python film.py                    # 自检：含恒等性与破坏恒等的演示
python -m unittest test_film -v   # 11 条测试
```

## 依赖

仅 `torch`。不 import 本仓库任何其它模块，整个目录可单独拷走。

## 已知特性

- **初始恒等**：γ 权重置零、偏置置 1，β 全零 → 输出与输入逐元素相等。
  这使「加 FiLM 前后」成为干净对照。
- 恒等**只靠初始化**，不是结构自带的性质。替换 init 写法就会失去它（测试里有反证）。
- 条件维度 `cond_dim` 与被调制维度 `n_features` 完全无关。
