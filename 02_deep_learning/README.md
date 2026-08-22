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
├── 00_笔记/           # 学习笔记（md）
├── 01_代码/           # 代码/notebook
└── 02_项目/           # 项目实战
```

模型与数据统一放在仓库根级：模型 `../models/`、数据 `../data/`。

## 规划路线

1. PyTorch 张量与自动求导
2. 神经网络搭建与训练
3. CNN / RNN 等经典架构
4. 深度学习项目实战

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