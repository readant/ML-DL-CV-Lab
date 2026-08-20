# 计算机视觉（YOLO）

本阶段以 YOLO 目标检测为核心，涵盖 OpenCV 视觉基础、YOLO 推理、摄像头实时检测、视频处理与自定义训练。

## 环境激活

```bash
conda activate main
```

## 目录结构

```plaintext
03_computer_vision/
├── README.md          # 本使用指引
├── 00_notes/          # 学习笔记（md）
├── 01_code/           # 练习代码
│   └── video_processing/    # 视频处理/实时检测
└── 02_projects/       # 项目实战
```

模型与数据统一放在仓库根级：

- 模型：仓库根 `../models/`
- 数据：仓库根 `../data/`

## 常用命令

### YOLO 推理

```python
from ultralytics import YOLO
model = YOLO('../models/yolov8n.pt')
results = model('图片路径')
```

### 摄像头实时检测

```python
model = YOLO('../models/yolov8n.pt')
model.predict(source=0, show=True)
```

### 训练自己的数据

```python
model = YOLO('../models/yolov8n.pt')
model.train(data='数据集.yaml', epochs=100)
```

## Notebook 中使用根级模型/数据

Notebook 的 CWD 取决于启动位置，`__file__` 不可用。在首个 cell 运行下面的 bootstrap 定位仓库根目录：

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

之后即可用 `MODELS_DIR / "yolov8n.pt"` 访问根级模型，`DATA_DIR` 访问根级数据。

## 常见问题

- 模型文件放入仓库根 `models/`，不要放在本阶段内
- 模型路径须为纯 ASCII，避免中文路径导致加载失败