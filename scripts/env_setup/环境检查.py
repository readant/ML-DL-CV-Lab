#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ main 统一学习环境检查脚本
覆盖机器学习 / 深度学习 / 机器视觉 / MediaPipe 姿态估计全部学习内容
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import print_header, print_item

import warnings

warnings.filterwarnings("ignore")


def check_import(name, label=None):
    """尝试导入指定模块并报告版本。"""
    label = label or name
    try:
        module = __import__(name)
        print_item(label, True, getattr(module, "__version__", "已安装"))
    except ImportError:
        print_item(label, False, "未安装")


def main():
    print_header("main 统一学习环境检查")

    print_header("1. Python 版本")
    print_item("Python", True, sys.version.split()[0])

    print_header("2. 数据科学 & 机器学习")
    check_import("numpy")
    check_import("pandas")
    check_import("matplotlib")
    check_import("seaborn")
    check_import("scipy")
    check_import("sklearn", "scikit-learn")
    check_import("skimage", "scikit-image")
    check_import("PIL", "Pillow")
    check_import("openpyxl")
    check_import("tqdm")
    check_import("requests")
    check_import("yaml", "PyYAML")

    print_header("3. 深度学习 & GPU 加速")
    try:
        import torch

        print_item("PyTorch", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        print_item("CUDA 可用", cuda_ok)
        if cuda_ok:
            print_item("GPU 型号", True, torch.cuda.get_device_name(0))
    except ImportError:
        print_item("PyTorch", False, "未安装")

    print_header("4. 机器视觉 & YOLO")
    check_import("cv2", "OpenCV")
    check_import("ultralytics", "YOLO (Ultralytics)")

    print_header("5. MediaPipe 姿态估计")
    check_import("mediapipe")

    print_header("6. 开发环境")
    check_import("jupyterlab")

    print_header("检查完成")
    try:
        if torch.cuda.is_available():
            print("🎉 main 环境配置完成，GPU 已激活，可满血运行全部学习内容！")
        else:
            print("✅ main 环境配置完成（未检测到 GPU，将使用 CPU 运行）")
    except NameError:
        print("⚠️  环境存在问题，请检查上面的 ❌ 项")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()