# 04 通道注意力（SE）· 实现

技术点 **A1** 的可运行实现。技术点本身的讲解（机制、形状变化、何时用别用、变体对比）在笔记里：

[`../../00_笔记/01_注意力机制/技术点A1-通道注意力_SE.md`](../../00_笔记/01_注意力机制/技术点A1-通道注意力_SE.md)

## 接口

```python
from se import SE1D

se = SE1D(n_features=16, reduction=4)
out, w = se(x)        # x: [B, 16] → out: [B, 16]（已缩放）, w: [B, 16]（门控）
```

## 跑

```bash
python se.py                    # 自检：打印形状、门控范围、参数量
python -m unittest test_se -v   # 10 条测试
```

## 依赖

仅 `torch`。不 import 本仓库任何其它模块，整个目录可单独拷走。

## 已知特性

- **初始不恒等**：`sigmoid(0)=0.5`，刚初始化时输出约为输入的一半。
  若需要干净对照，得手工把 `gate` 末层 bias 调大造恒等起点（笔记第 5 节有说明）。
- 纯向量输入下**没有 Squeeze 这一步**，故未使用 `AdaptiveAvgPool`。
