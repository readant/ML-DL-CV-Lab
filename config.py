"""仓库统一的路径基建。

仅供 .py 脚本使用（依赖 __file__）。notebook 请用各阶段 README 中的
锚点查找 bootstrap cell 定位根目录后导入本模块。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ---- data/ 子路径（与仓库实际目录骨架保持一致） ----
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
DATASETS_DIR = DATA_DIR / "datasets"
MEDIAPIPE_DIR = DATA_DIR / "mediapipe"

# images/ 主题子目录
IMAGES_FACES_DIR = IMAGES_DIR / "faces"
IMAGES_SCENES_DIR = IMAGES_DIR / "scenes"
IMAGES_OBJECTS_DIR = IMAGES_DIR / "objects"
IMAGES_GESTURES_DIR = IMAGES_DIR / "gestures"
IMAGES_HAND_DIR = IMAGES_DIR / "hand"

# videos/ 子目录
VIDEOS_RAW_DIR = VIDEOS_DIR / "raw"
VIDEOS_PROCESSED_DIR = VIDEOS_DIR / "processed"
VIDEOS_SAMPLE_DIR = VIDEOS_DIR / "sample"

# datasets/ 子目录
DATASETS_POSE_DIR = DATASETS_DIR / "pose"
DATASETS_DETECTION_DIR = DATASETS_DIR / "detection"
DATASETS_CLASSIFICATION_DIR = DATASETS_DIR / "classification"

# mediapipe/ 子目录
MEDIAPIPE_IMAGES_DIR = MEDIAPIPE_DIR / "images"
MEDIAPIPE_VIDEOS_DIR = MEDIAPIPE_DIR / "videos"