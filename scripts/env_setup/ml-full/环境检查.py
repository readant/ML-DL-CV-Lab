#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
✅ ml-full 环境检查脚本
专为机器学习全栈环境设计
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import print_header, print_item

print_header("ml-full 机器学习环境检查")

print_header("1. Python 版本")
print_item("Python", True, sys.version.split()[0])

print_header("2. 核心数据科学库")
try:
    import numpy
    print_item("NumPy", True, numpy.__version__)
except ImportError:
    print_item("NumPy", False, "未安装")

try:
    import pandas
    print_item("Pandas", True, pandas.__version__)
except ImportError:
    print_item("Pandas", False, "未安装")

try:
    import matplotlib
    print_item("Matplotlib", True, matplotlib.__version__)
except ImportError:
    print_item("Matplotlib", False, "未安装")

try:
    import sklearn
    print_item("scikit-learn", True, sklearn.__version__)
except ImportError:
    print_item("scikit-learn", False, "未安装")

try:
    import scipy
    print_item("SciPy", True, scipy.__version__)
except ImportError:
    print_item("SciPy", False, "未安装")

try:
    import seaborn
    print_item("Seaborn", True, "已安装")
except ImportError:
    print_item("Seaborn", False, "未安装")

print_header("3. 开发环境")
try:
    import jupyterlab
    print_item("JupyterLab", True, "已安装")
except ImportError:
    print_item("JupyterLab", False, "未安装")

print_header("检查完成")
print("\n🎉 ml-full 环境配置完成！")
print("=" * 60 + "\n")
