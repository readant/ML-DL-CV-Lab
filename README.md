<div align="center">

# ML-DL-CV-Lab

**机器学习 · 深度学习 · 机器视觉 从入门到实战全记录**

从零基础到 YOLO 目标检测与 MediaPipe 姿态估计的完整开源学习路径，配套统一环境一键复现、结构化笔记与可运行实战代码。

<p>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://github.com/readant/Sum"><img alt="License" src="https://img.shields.io/badge/license-MIT-4DABF7?style=flat-square"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-RTX%2050%20Series-76B900?style=flat-square&logo=nvidia">
</p>

<p>
  <b>RTX 50 系显卡完美适配</b> ·
  <b>统一环境一键复现</b> ·
  <b>结构化学习笔记</b> ·
  <b>可复现实战代码</b>
</p>

</div>

---

## 目录

- [仓库亮点](#仓库亮点)
- [适合人群](#适合人群)
- [仓库结构](#仓库结构)
- [环境配置](#环境配置)
- [学习路线](#学习路线)
- [快速开始](#快速开始)
- [使用与提交规范](#使用与提交规范)
- [免责声明](#免责声明)
- [交流与反馈](#交流与反馈)

---

## 仓库亮点

1. **零基础友好**：全程以大一新生视角记录，无跳步、无黑箱。每一步学习都配有对应笔记与可运行代码。
2. **统一环境一键复现**：机器学习、深度学习和机器视觉全部依赖集成在 Conda `main` 单一环境，通过 `environment.yml` 一键创建，避免多环境切换干扰学习。
3. **新硬件适配**：针对 RTX 50 系显卡（RTX 5060 Laptop，sm_120 算力）提供踩坑后的 CUDA 兼容适配方案，解决新显卡常见的 CUDA 报错。
4. **结构化管理**：笔记、代码、数据集、模型严格分离，长期学习也不会出现文件混乱。
5. **全流程闭环**：从环境搭建 → 基础语法 → 核心算法 → 实战项目，覆盖大学 AI 课程、学科竞赛、毕业设计的核心需求。
6. **多技术栈覆盖**：涵盖机器学习、YOLO 目标检测、MediaPipe 姿态估计等主流 AI 视觉技术。

---

## 适合人群

- 零基础入门机器学习 / 机器视觉的在校大学生
- 物联网、计算机、自动化相关专业的 AI 学习新手
- 持有 RTX 50 系新显卡，受 CUDA / PyTorch 兼容问题困扰的同学
- 希望系统学习 YOLO 目标检测，完成课程设计 / 毕设 / 竞赛的同学
- 对人体姿态估计、手部 / 面部识别感兴趣的开发者

---

## 仓库结构

遵循「学习阶段隔离、笔记代码分离、数据模型统一管理」的原则：

```plaintext
ML-DL-CV-Lab/
├── config.py                       # 路径基建（统一定位根级 models/data）
├── environment.yml                 # main 统一环境定义
├── pyproject.toml                  # 项目工具链配置
├── test_model_paths.py            # 模型路径测试脚本
├── .gitignore                      # Git 忽略规则
│
├── 00_prerequisites/               # 前置基础：Python、数据科学、数学
│   ├── 00_python_basics/           # Python 基础（速查表/控制流/函数/OOP）
│   ├── 01_data_processing/         # NumPy & Pandas 数据处理
│   ├── 02_data_visualization/      # Matplotlib 可视化
│   └── 03_math_basics/             # 数学基础（线性代数、概率统计）
│
├── 01_machine_learning/            # 第一阶段：机器学习
│   ├── README.md                   # 阶段使用指引
│   ├── 01_code/                    # 代码/notebook
│   │   ├── 00_intro/               # 数据科学入门
│   │   ├── 01_basics_library/      # 基础库（NumPy/Pandas/Matplotlib）
│   │   └── 02_algorithms/          # sklearn 算法
│   └── 02_projects/                # 项目实战
│
├── 02_deep_learning/               # 第二阶段：深度学习（规划中）
├── 03_computer_vision/             # 第三阶段：计算机视觉（YOLO）
│   └── 01_code/video_processing/   # 视频处理/实时检测
│
├── 04_pose_estimation/             # 第四阶段：MediaPipe 姿态估计
│   ├── base_test.py                # 测试基类
│   ├── body/                       # 身体姿态
│   ├── face/                       # 面部特征/表情识别
│   └── hand/                       # 手部检测/手势识别
│
├── data/                           # 根级共享数据集（不入库）
├── models/                         # 根级共享预训练模型（不入库）
├── docs/                           # 文档
├── scripts/                        # 工具脚本
│   ├── env_setup/                  # 环境配置（环境检查/一键安装）
│   └── utils/                      # 通用工具（摄像头测试）
└── README.md                       # 本文档
```

### 路径规范

- 数据与模型**统一放在根级 `data/`、`models/`**，各阶段不单独存放。
- `.py` 脚本通过 `pathlib.Path(__file__)` 上溯定位根级目录，不受运行目录影响。
- Notebook 因 `__file__` 不可用，请在首个 cell 运行各阶段 README 中的 bootstrap cell 定位仓库根目录后导入 `config`。

---

## 环境配置

学习阶段采用统一的 Conda `main` 环境，避免多环境切换对学习思路的干扰。所有依赖（PyTorch、YOLO、MediaPipe 等）均安装在同一环境中，`environment.yml` 一键创建。若后续因项目部署或严重依赖冲突需拆分环境，再按需隔离。

### 前置依赖

- 安装 [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)（推荐 Python 3.11+ 版本）。
- Windows 用户需提前更新 NVIDIA 显卡驱动至最新版（RTX 50 系显卡需驱动版本 ≥ 572.97）。

### 一键创建统一环境（推荐）

```bash
cd ML-DL-CV-Lab
conda env create -f environment.yml
conda activate main
```

### 更新已有环境

```bash
conda env update -f environment.yml --prune
```

### 环境验证

```bash
conda activate main
python scripts/env_setup/环境检查.py
```

---

## 学习路线

遵循「先基础后实战、先机器学习后机器视觉再到姿态估计」的循序渐进路线，总周期约 **8 个月**。

### 第一阶段：机器学习全栈入门（8 周）

打牢数据科学与机器学习核心基础，为后续机器视觉学习做底层铺垫，对应 `01_machine_learning/`。

| 学习模块 | 周期 | 核心内容 | 验收标准 |
| :--- | :---: | :--- | :--- |
| Python 数据科学基础 | 2 周 | NumPy 数组运算、Pandas 数据清洗、Matplotlib/Seaborn 可视化 | 独立完成表格数据的读取、清洗、特征处理与可视化 |
| 机器学习数学基础 | 1 周 | 线性代数、概率统计、梯度下降与优化原理 | 能看懂算法数学公式，手动实现梯度下降 |
| 机器学习核心算法 | 4 周 | 线性回归、逻辑回归、KNN、决策树、随机森林、SVM、K-Means、模型评估 | 独立完成分类 / 回归 / 聚类任务，完成模型训练与调优 |
| 机器学习项目实战 | 1 周 | 完整项目全流程：需求分析 → 数据清洗 → 特征工程 → 模型训练 → 可视化输出 | 完成 2 个完整实战项目，输出可复用代码与可视化报告 |

### 第二阶段：机器视觉 & YOLO 实战（16 周）

掌握计算机视觉核心能力，精通 YOLO 目标检测全流程，发挥 RTX 5060 GPU 的硬件加速优势，对应 `03_computer_vision/`。

| 学习模块 | 周期 | 核心内容 | 验收标准 |
| :--- | :---: | :--- | :--- |
| OpenCV 视觉基础 | 3 周 | 图像读写与像素操作、图像增强、特征提取、摄像头/视频流处理 | 独立完成图像/视频的基础处理，熟练使用 OpenCV 核心 API |
| YOLOv8 目标检测入门 | 3 周 | YOLO 核心原理、模型加载与推理、参数调优、摄像头实时检测、视频批量处理 | 独立完成图片/视频/摄像头的目标检测，按需调优参数 |
| YOLOv8 进阶与自定义训练 | 4 周 | 自定义数据集、模型微调、GPU 训练优化、模型评估、实例分割/姿态估计/多目标跟踪 | 独立完成数据集制作 → 模型训练 → 实时检测全流程 |
| 视频处理与高级视觉应用 | 2 周 | 多路视频流、目标计数、越界检测、轨迹追踪、人流量统计 | 实现业务化视觉功能，不止于基础目标检测 |
| 视觉项目全栈实战 | 4 周 | 完整视觉项目全流程开发，对应课程设计/竞赛/毕设需求 | 完成 2 个完整落地项目，输出可运行程序与项目文档 |

### 第三阶段：MediaPipe 姿态估计（8 周）

掌握 MediaPipe 人体姿态估计、手部检测、面部识别等核心技术，对应 `04_pose_estimation/`。

| 学习模块 | 周期 | 核心内容 | 验收标准 |
| :--- | :---: | :--- | :--- |
| MediaPipe 基础 | 1 周 | 框架介绍、模型下载与配置、基础 API 使用 | 成功搭建环境，运行基础检测示例 |
| 手部姿态学习 | 2 周 | 手部关键点、手势识别、双手协同、置信度估计 | 实现实时手部检测与手势识别，准确率达 85% 以上 |
| 面部特征学习 | 2 周 | 面部关键点、表情识别、面部特征分析 | 实现实时面部检测与表情识别，支持多种表情分类 |
| 身体姿态学习 | 2 周 | 身体关键点、全身检测、姿态分析 | 实现实时全身姿态检测，绘制完整骨架 |
| MediaPipe 项目实战 | 1 周 | 结合 OpenCV 与 MediaPipe 开发综合应用 | 完成 1-2 个完整姿态估计应用项目 |

---

## 快速开始

```bash
# 1. 克隆仓库
git clone <your-repo-url> ML-DL-CV-Lab
cd ML-DL-CV-Lab

# 2. 创建统一环境
conda env create -f environment.yml

# 3. 激活环境
conda activate main

# 4. 验证环境
python scripts/env_setup/环境检查.py
```

环境就绪后，可从 `00_prerequisites/` 开始逐阶段学习。

---

## 使用与提交规范

### 文件使用规范

1. 学习笔记**仅存放于对应阶段的 `00_notes/` 文件夹**（前置基础阶段直接放在模块目录），按学习模块分子文件夹管理。
2. 数据集、模型文件**统一存放于根级 `data/` 或 `models/` 文件夹**，绝不与代码、笔记混放。
3. 纯代码练习、工具脚本存放于对应模块的代码文件夹，不与笔记混杂。
4. 整个学习阶段统一使用 `conda activate main` 激活环境，无需在多个环境间切换。

### Git 提交规范

提交信息严格遵循 `[类型]: 提交内容` 的格式，便于回溯学习历程：

| 类型 | 用途 |
| :--- | :--- |
| `feat` | 新增学习笔记 / 代码 / 项目 |
| `fix` | 修复代码 bug / 环境问题 |
| `docs` | 更新 README / 学习文档 |
| `refactor` | 重构代码 / 优化笔记结构 |
| `chore` | 配置更新 / 工具脚本优化 |

---

## 免责声明

1. 本仓库仅用于个人学习记录，所有代码与笔记均为学习过程产出，请勿用于商业用途。
2. 所有预训练模型均来自 Ultralytics 和 Google MediaPipe 官方开源项目，模型版权归原作者所有。
3. 本仓库提供的环境配置方案仅为个人踩坑经验，仅供参考；因硬件/软件版本差异导致的问题，需自行排查解决。

---

## 交流与反馈

如果你也是 AI 学习新手，或者遇到了 RTX 50 系显卡的环境适配问题，欢迎通过 [Issue](https://github.com/readant/Sum/issues) 交流学习，一起踩坑一起进步！

---

<div align="center">

**ML-DL-CV-Lab** · Keep Learning, Keep Coding 🚀

</div>