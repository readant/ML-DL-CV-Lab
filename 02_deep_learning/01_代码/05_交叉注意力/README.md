# 05 交叉注意力 · 实现

技术点 **A4** 的可运行实现。技术点本身的讲解（机制、形状变化、何时用别用、变体对比）在笔记里：

[`../../00_笔记/01_注意力机制/技术点A4-交叉注意力.md`](../../00_笔记/01_注意力机制/技术点A4-交叉注意力.md)

## 接口

```python
from cross_attention import CrossAttention

ca = CrossAttention(d_q=8, d_kv=5, d_model=16)
out = ca(q_in, kv_in)                              # [B,Nq,8] + [B,Nk,5] → [B,Nq,8]
out, w = ca(q_in, kv_in, return_weights=True)      # w: [B,Nq,Nk]，每行和为 1
```

输出单元数跟着 **Query** 走，与 `Nk` 无关。

## 跑

```bash
python cross_attention.py                    # 自检：含退化演示
python -m unittest test_cross_attention -v   # 10 条测试
```

## 依赖

仅 `torch`。不 import 本仓库任何其它模块，整个目录可单独拷走。

## 已知特性

- **两路都是单 token 时会退化**：权重恒为 1，注意力什么也没做，只剩一次线性变换。
  要让它工作，某一路必须先拆成多个单元（序列、分组、多位置均可）。
- 单头实现，便于看清数据流。要多头换成 `nn.MultiheadAttention` 即可，笔记第 5 节有说明。
