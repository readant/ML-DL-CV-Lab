# 机器学习

本阶段覆盖机器学习全栈：数据科学基础、数学基础、核心算法与项目实战。

## 环境激活

```bash
conda activate main
```

## 目录结构

```plaintext
01_machine_learning/
├── README.md          # 本使用指引
├── 00_笔记/           # 学习笔记（md）
├── 01_代码/           # 代码/notebook
│   ├── 01_基础库/            # 基础库与数据分析（NumPy/Pandas/Matplotlib + 数据分析实战）
│   └── 02_算法/              # sklearn 算法
└── 02_项目/           # 项目实战
```

数据集统一放在仓库根级 `../data/`。

## 学习路线

1. 数据科学基础（NumPy、Pandas、Matplotlib/Seaborn）
2. 数学基础（线性代数、概率统计、梯度下降）
3. 核心算法（线性回归、逻辑回归、KNN、决策树、随机森林、SVM、K-Means）
4. 项目实战（完整流程：清洗→特征工程→训练→可视化）

## Notebook 中使用根级数据

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

from config import DATA_DIR
```

之后即可用 `DATA_DIR` 访问根级数据。