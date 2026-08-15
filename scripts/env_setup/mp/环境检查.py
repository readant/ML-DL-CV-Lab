#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ mp 环境检查脚本
专为 MediaPipe 姿态估计环境设计
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import print_header, print_item

print_header("mp MediaPipe 环境检查")

print_header("1. Python 版本")
print_item("Python", True, sys.version.split()[0])

print_header("2. MediaPipe 核心库")
try:
    import mediapipe
    print_item("MediaPipe", True, mediapipe.__version__)
except ImportError:
    print_item("MediaPipe", False, "未安装")

print_header("3. 视觉依赖")
try:
    import cv2
    print_item("OpenCV", True, cv2.__version__)
except ImportError:
    print_item("OpenCV", False, "未安装")

try:
    import numpy
    print_item("NumPy", True, numpy.__version__)
except ImportError:
    print_item("NumPy", False, "未安装")

try:
    import matplotlib
    print_item("Matplotlib", True, matplotlib.__version__)
except ImportError:
    print_item("Matplotlib", False, "未安装")

print_header("4. 辅助工具")
try:
    import requests
    print_item("Requests", True, requests.__version__)
except ImportError:
    print_item("Requests", False, "未安装")

try:
    import tqdm
    print_item("tqdm", True, tqdm.__version__)
except ImportError:
    print_item("tqdm", False, "未安装")

print_header("检查完成")
print("\n🎉 mp 环境配置完成！")
print("=" * 60 + "\n")
