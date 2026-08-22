# MediaPipe 姿态估计

本阶段使用 MediaPipe 实现人体姿态、手部检测与手势识别、面部特征分析。

## 环境激活

```bash
conda activate main
```

## 目录结构

```plaintext
04_pose_estimation/
├── README.md          # 本使用指引
├── 00_笔记/           # 学习笔记（md）
├── 01_代码/           # 练习代码
│   ├── base_test.py   # 测试基类（复用检测/绘制逻辑）
│   ├── 身体/          # 身体姿态
│   ├── 面部/          # 面部特征
│   └── 手部/          # 手部姿态
└── 02_项目/           # 项目实战
```

模型与数据统一放在仓库根级：

- 模型：仓库根 `../models/`
- 数据：仓库根 `../data/`

## 模型下载

```bash
# 将模型文件放入仓库根 models/
cd <仓库根>
python 下载模型.py   # 若有则运行；否则手动放入 models/
```

## 代码结构说明

- `01_代码/手部/` — 手部检测、手势识别、双手协同、置信度估计
- `01_代码/面部/` — 面部检测、表情识别、面部特征分析
- `01_代码/身体/` — 全身检测、姿态检测

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

之后即可用 `MODELS_DIR / "模型文件名"` 访问根级模型，`DATA_DIR` 访问根级数据。

## 常见问题

- 模型文件放入仓库根 `models/`，不要放在本阶段内
- 模型路径必须是纯 ASCII，不能包含中文
- 若遇路径错误，检查模型文件是否在仓库根 `models/` 的正确位置