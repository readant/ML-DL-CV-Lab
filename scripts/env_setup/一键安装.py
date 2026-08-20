#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 一键安装统一学习环境 (main)
学习阶段采用统一的 Conda main 环境，覆盖全部学习内容：
机器学习 + 深度学习 + 机器视觉(YOLO) + MediaPipe 姿态估计。
环境定义见仓库根目录 environment.yml。
运行前请确保已安装 Miniconda。
"""

import os
import subprocess

ENV_NAME = "main"
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "environment.yml")


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
    print("  🚀 ML-DL-CV-Lab 统一环境安装工具")
    print("=" * 60)
    print(f"\n将基于 environment.yml 创建统一环境：{ENV_NAME}")
    print("  覆盖：机器学习 / 深度学习 / YOLO机器视觉 / MediaPipe姿态估计")
    print("\n⚠️  注意：安装过程可能需要10-30分钟，请耐心等待")

    confirm = input("\n是否继续安装? (y/n): ").strip().lower()
    if confirm != "y":
        print("取消安装")
        return

    if not os.path.exists(ENV_FILE):
        print(f"❌ 未找到 environment.yml：{ENV_FILE}")
        print("   请确认仓库结构完整后重试。")
        return

    print(f"\n使用环境定义文件：{ENV_FILE}")

    # 判断环境是否已存在
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    exists = f" {ENV_NAME} " in result.stdout or result.stdout.count(f"{ENV_NAME} ") > 0

    if exists:
        print(f"\n⚠️  检测到 {ENV_NAME} 环境已存在，将更新依赖...")
        ok = run_command(f"conda env update -f \"{ENV_FILE}\" --prune", "更新 main 环境")
    else:
        ok = run_command(f"conda env create -f \"{ENV_FILE}\"", "创建 main 环境")

    if not ok:
        print("\n❌ 环境配置失败，请检查错误信息。")
        return

    print("\n" + "=" * 60)
    print("  🎉 安装完成！")
    print("=" * 60)
    print("\n激活环境：")
    print("  conda activate main")
    print("\n验证安装：")
    print("  python scripts/env_setup/环境检查.py")
    print("\n切换学习内容时无需切换环境，main 环境已覆盖全部内容。")


if __name__ == "__main__":
    main()