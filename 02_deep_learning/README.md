# 深度学习

本阶段为深度学习（规划中），以 PyTorch 为核心，构建在机器学习基础之上。

## 环境激活

```bash
conda activate main
```

## 目录结构

```plaintext
02_deep_learning/
├── README.md          # 本使用指引
├── 00_笔记/
│   └── 01_注意力机制/  # 注意力/多头注意力：PDF 页码索引、中文文献、摘要笔记（离线 HTML）
├── 01_代码/
│   └── 01_注意力机制/  # NumPy/PyTorch 实现 + 14 条断言 + 字符热力图演示
└── 02_项目/           # 项目实战
```

模型与数据统一放在仓库根级：模型 `../models/`、数据 `../data/`。

## 规划路线

1. PyTorch 张量与自动求导
2. 神经网络搭建与训练
3. CNN / RNN 等经典架构
4. 深度学习项目实战

## 已完成模块

### 注意力机制（Attention / Multi-Head Attention）

不依赖预训练权重与 GPU，几秒内可跑完，用 14 条断言把易错细节钉死。

```bash
conda activate main
cd 02_deep_learning/01_代码/01_注意力机制
python -m unittest test_attention -v   # Ran 14 tests ... OK
python demo_attention.py               # 打印注意力权重矩阵
```

| 位置 | 内容 |
| :--- | :--- |
| [`01_代码/01_注意力机制/`](01_代码/01_注意力机制/) | `attention_numpy.py`（从零实现）、`attention_torch.py`（可训练）、`test_attention.py`（14 条断言）、`demo_attention.py`、实现笔记 HTML |
| [`00_笔记/01_注意力机制/`](00_笔记/01_注意力机制/) | 技术点索引（PDF 页码）、中文文献清单（18 条题录）、文献笔记（7 篇摘要提炼） |

三个已验证的结论：`W_q` 与 `W_k` 必须分开投影（绑定后 91% 的词只关注自己，分开后降到 9%）；多头参数量恒为 4d²、与头数无关；mask 必须加在 softmax 之前。

## Notebook 中使用根级模型/数据

Notebook 的 CWD 取决于启动位置，`__file__` 不可用。在首个 cell 运行 bootstrap 定位仓库根目录：

```python
# notebook 标准 bootstrap cell
from pathlib import Path
import sys

def _find_root():
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "config.py").exists():
            return d

ROOT = _find_root()
if ROOT and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODELS_DIR, DATA_DIR
```

之后即可用 `MODELS_DIR` / `DATA_DIR` 访问根级模型与数据。