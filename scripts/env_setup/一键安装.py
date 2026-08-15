#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 一键安装所有虚拟环境
运行前请确保已安装 Miniconda
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n📦 {description}")
    print("-" * 50)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e.stderr}")
        return False

def main():
    print("=" * 60)
    print("  🚀 ML-DL-CV-Lab 学习仓库 - 一键环境安装工具")
    print("=" * 60)
    print("\n本脚本将为您创建三个虚拟环境：")
    print("  1. ml-full - 机器学习全栈环境")
    print("  2. cv-yolo - 机器视觉YOLO环境")
    print("  3. mp - MediaPipe姿态估计环境")
    print("\n⚠️  注意：安装过程可能需要10-30分钟，请耐心等待")
    
    confirm = input("\n是否继续安装? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消安装")
        return
    
    # 创建 ml-full 环境
    print("\n" + "=" * 60)
    print("  安装环境1: ml-full")
    print("=" * 60)
    
    run_command("conda create -n ml-full python=3.11 -y", "创建 ml-full 环境")
    run_command("conda activate ml-full && conda install numpy pandas matplotlib scikit-learn scipy seaborn openpyxl tqdm pyyaml -y", "安装核心依赖")
    run_command("conda activate ml-full && pip install jupyterlab jupyterlab-language-pack-zh-CN", "安装开发工具")
    
    # 创建 cv-yolo 环境
    print("\n" + "=" * 60)
    print("  安装环境2: cv-yolo")
    print("=" * 60)
    
    run_command("conda create -n cv-yolo python=3.11 -y", "创建 cv-yolo 环境")
    run_command("conda activate cv-yolo && conda install numpy pillow scipy scikit-image matplotlib tqdm pyyaml -y", "安装基础依赖")
    run_command("conda activate cv-yolo && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128", "安装 PyTorch (CUDA128)")
    run_command("conda activate cv-yolo && pip install ultralytics opencv-python opencv-python-headless jupyterlab jupyterlab-language-pack-zh-CN moviepy", "安装视觉工具")
    
    # 创建 mp 环境
    print("\n" + "=" * 60)
    print("  安装环境3: mp")
    print("=" * 60)
    
    run_command("conda create -n mp python=3.11 -y", "创建 mp 环境")
    run_command("conda activate mp && conda install numpy matplotlib requests tqdm -y", "安装基础依赖")
    run_command("conda activate mp && pip install mediapipe opencv-python", "安装 MediaPipe")
    
    print("\n" + "=" * 60)
    print("  🎉 安装完成！")
    print("=" * 60)
    print("\n已成功创建以下虚拟环境：")
    print("  - ml-full (机器学习)")
    print("  - cv-yolo (机器视觉)")
    print("  - mp (MediaPipe)")
    print("\n📌 使用方法：")
    print("  conda activate ml-full    # 激活机器学习环境")
    print("  conda activate cv-yolo    # 激活机器视觉环境")
    print("  conda activate mp         # 激活MediaPipe环境")
    print("\n验证安装：")
    print("  python 脚本工具/env_setup/ml-full/环境检查.py")
    print("  python 脚本工具/env_setup/cv-yolo/环境检查.py")
    print("  python 脚本工具/env_setup/mp/环境检查.py")

if __name__ == "__main__":
    main()
