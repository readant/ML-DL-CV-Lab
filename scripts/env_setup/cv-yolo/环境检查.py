#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ cv-yolo 环境检查脚本
专为 RTX 5060 + cv-yolo 环境设计
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import print_header, print_item

import warnings
warnings.filterwarnings('ignore')

print_header("cv-yolo 机器视觉环境检查")

print_header("1. Python 版本")
print_item("Python", True, sys.version.split()[0])

print_header("2. PyTorch & GPU 加速")
try:
    import torch
    print_item("PyTorch", True, torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print_item("CUDA 可用", cuda_ok)
    if cuda_ok:
        print_item("GPU 型号", True, torch.cuda.get_device_name(0))
except ImportError:
    print_item("PyTorch", False, "未安装")

print_header("3. 核心视觉库")
try:
    import cv2
    print_item("OpenCV", True, cv2.__version__)
except ImportError:
    print_item("OpenCV", False, "未安装")

try:
    import ultralytics
    print_item("YOLO (Ultralytics)", True, ultralytics.__version__)
except ImportError:
    print_item("YOLO (Ultralytics)", False, "未安装")

try:
    import numpy
    print_item("NumPy", True, numpy.__version__)
except ImportError:
    print_item("NumPy", False, "未安装")

try:
    import PIL
    print_item("Pillow", True, PIL.__version__)
except ImportError:
    print_item("Pillow", False, "未安装")

print_header("4. 开发环境")
try:
    import jupyterlab
    print_item("JupyterLab", True, "已安装")
except ImportError:
    print_item("JupyterLab", False, "未安装")

print_header("检查完成")
if 'torch' in dir() and torch.cuda.is_available():
    print("\n🎉 恭喜！cv-yolo 环境非常完美！")
    print("   GPU 已激活，可以满血跑 YOLO 了！")
else:
    print("\n⚠️  环境存在问题，请检查上面的 ❌ 项")
print("=" * 60 + "\n")
