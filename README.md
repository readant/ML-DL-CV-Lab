# ML-DL-CV-Lab | 机器学习&深度学习&机器视觉 从入门到实战全记录
> 物联网工程专业从大一开始的AI学习全历程开源仓库 | 零基础入门 | 三环境隔离 | RTX 50系显卡完美适配 | 结构化学习笔记 | 可复现实战代码

本仓库全程记录从零基础入门，到机器学习基础、机器视觉YOLO实战、MediaPipe姿态估计、项目落地的完整学习路径。配套保姆级环境配置教程、结构化学习笔记、可直接运行的实战代码、完整项目源码，为零基础入门AI视觉/机器学习的同学提供可参考、可复现的学习范本。

---

## 📋 目录
- [🌟 仓库核心亮点](#-仓库核心亮点)
- [🎯 适合人群](#-适合人群)
- [📁 仓库文件夹结构](#-仓库文件夹结构)
- [🛠️ 环境配置说明](#️-环境配置说明)
- [📅 学习路线规划](#-学习路线规划)
- [📌 使用与提交规范](#-使用与提交规范)
- [📄 免责声明](#-免责声明)
- [🤝 交流与反馈](#-交流与反馈)

---

## 🌟 仓库核心亮点
1.  **新手友好**：全程以大一零基础视角记录，无跳步、无黑箱，每一步学习都有对应的笔记与可运行代码
2.  **三环境隔离**：机器学习、机器视觉、MediaPipe姿态估计三套独立虚拟环境，彻底解决Python包版本冲突问题
3.  **新硬件适配**：针对RTX 50系显卡（RTX 5060 Laptop sm_120算力）提供踩坑后的完美适配方案，解决新显卡CUDA兼容报错难题
4.  **结构化管理**：严格的文件管理逻辑，笔记、代码、数据集、模型完全分离，长期学习也不会出现文件混乱
5.  **全流程闭环**：从环境搭建→基础语法→核心算法→实战项目，完整覆盖大学AI相关课程、学科竞赛、毕业设计的核心需求
6.  **多技术栈覆盖**：涵盖机器学习、YOLO目标检测、MediaPipe姿态估计等主流AI视觉技术，满足不同场景需求

---

## 🎯 适合人群
- 零基础入门机器学习/机器视觉的在校大学生
- 物联网、计算机、自动化相关专业的AI学习新手
- 持有RTX 50系新显卡，被CUDA/PyTorch兼容问题困扰的同学
- 想要系统学习YOLO目标检测，完成课程设计/毕设/竞赛的同学
- 对人体姿态估计、手部/面部识别感兴趣的开发者

---

## 📁 仓库文件夹结构
严格遵循「学习阶段隔离、笔记代码分离、数据模型统一管理」的原则，结构如下：

```plaintext
ML-DL-CV-Lab/
├── 00_prerequisites/              # 前置基础：Python、数据科学、数学
│   ├── 00_python_basics/          # Python基础语法
│   ├── 01_data_processing/        # NumPy & Pandas 数据处理
│   ├── 02_data_visualization/     # Matplotlib & Seaborn 可视化
│   └── 03_math_basics/           # 数学基础（线性代数、概率统计）
│
├── 01_machine_learning/           # 第一阶段：机器学习
│   ├── 00_notes/                  # 学习笔记（按主题分类）
│   │   └── 01_intro/              # 机器学习入门
│   ├── 01_code/                   # 代码练习
│   │   ├── 01_numpy_practice/     # NumPy练习
│   │   ├── 02_sklearn/            # sklearn算法实现
│   │   └── 03_visualization/     # 可视化代码
│   ├── 02_projects/               # 项目实战
│   └── data/                      # 统一数据区
│
├── 02_deep_learning/              # 第二阶段：深度学习（规划中，暂无内容）
│   ├── 00_notes/
│   ├── 01_code/
│   ├── 02_projects/
│   └── data/
│
├── 03_computer_vision/            # 第三阶段：计算机视觉
│   ├── 00_notes/                  # 学习笔记
│   ├── 01_code/                   # 代码练习
│   │   ├── opencv/                # OpenCV代码
│   │   ├── yolo/                  # YOLO代码
│   │   └── video_processing/      # 视频处理代码
│   ├── 02_projects/               # 项目实战
│   └── models/                    # YOLO模型存放区
│
├── 04_pose_estimation/            # 第四阶段：MediaPipe姿态估计
│   ├── 00_notes/                  # 学习笔记
│   ├── 01_code/                   # 代码练习
│   │   ├── hand/                  # 手部姿态
│   │   ├── face/                  # 面部特征
│   │   └── body/                  # 身体姿态
│   ├── 02_projects/               # 项目实战
│   └── models/                    # MediaPipe模型（纯ASCII路径）
│
├── 05_projects/                   # 第五阶段：综合项目
│   ├── complete_projects/         # 完整项目
│   ├── competition/               # 竞赛项目
│   └── portfolio/                 # 作品集
│
├── docs/                          # 文档
│   └── MediaPipe错误与解决方案.md  # 错误记录
│
├── scripts/                       # 工具脚本
│   ├── env_setup/                 # 环境配置
│   │   ├── ml-full/               # ml-full环境检查
│   │   ├── cv-yolo/               # cv-yolo环境检查
│   │   ├── mp/                    # MediaPipe环境检查
│   │   └── 一键安装.py            # 一键安装所有环境
│   └── utils/                     # 通用工具
│       └── 摄像头测试.py          # 摄像头测试
│
├── test_model_paths.py            # 模型路径测试脚本
├── .gitignore                     # Git忽略规则
└── README.md                      # 本文档
```

---

## 🛠️ 环境配置说明
本仓库分为三套完全独立的虚拟环境，基于Miniconda搭建，避免版本冲突，以下为一键安装指令。

### 前置依赖
- 安装 [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)（推荐Python 3.11+版本）
- Windows用户需提前更新NVIDIA显卡驱动至最新版（RTX 50系显卡需驱动版本≥572.97）

---

#### 环境1：ml-full 机器学习专属环境
用于第一阶段机器学习全流程学习，适配所有经典机器学习算法、数据处理、可视化需求。
```bash
# 1. 创建并激活环境
conda create -n ml-full python=3.11 -y && conda activate ml-full

# 2. 使用conda安装核心科学计算库（稳定可靠，版本兼容性好）
conda install numpy pandas matplotlib scikit-learn scipy seaborn openpyxl tqdm pyyaml -y

# 3. 使用pip安装开发工具（获取最新功能，体验更好）
pip install jupyterlab jupyterlab-language-pack-zh-CN

# 或使用 requirements 文件一键安装
pip install -r requirements-ml-full.txt
```

---

#### 环境2：cv-yolo 机器视觉YOLO专属环境
**重点适配RTX 50系显卡（sm_120算力）**，解决新显卡CUDA兼容报错问题，实现GPU满血加速。
```bash
# 1. 创建并激活环境
conda create -n cv-yolo python=3.11 -y && conda activate cv-yolo

# 2. 使用conda安装核心科学计算库（稳定可靠，版本兼容性好）
conda install numpy pillow scipy scikit-image matplotlib tqdm pyyaml -y

# 3. 安装核心：RTX50系专属PyTorch GPU版（CUDA128 预览版，原生支持sm_120）
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. 使用pip安装视觉工具（获取最新功能，体验更好）
pip install ultralytics opencv-python opencv-python-headless jupyterlab jupyterlab-language-pack-zh-CN moviepy

# 或使用 requirements 文件一键安装（不含PyTorch，需单独安装）
pip install -r requirements-cv-yolo.txt
```

---

#### 环境3：mp MediaPipe姿态估计专属环境
用于第四阶段MediaPipe姿态估计学习，专注于人体姿态、手部、面部特征检测。
```bash
# 1. 创建并激活环境
conda create -n mp python=3.11 -y && conda activate mp

# 2. 使用conda安装核心科学计算库（稳定可靠，版本兼容性好）
conda install numpy matplotlib requests tqdm -y

# 3. 使用pip安装视觉工具（获取最新功能，体验更好）
pip install mediapipe opencv-python

# 或使用 requirements 文件一键安装
pip install -r requirements-mp.txt
```

---

#### 环境验证
安装完成后，运行对应环境的检查脚本，即可验证环境是否完美配置：
```bash
# 验证ml-full环境
conda activate ml-full
python scripts/env_setup/ml-full/环境检查.py

# 验证cv-yolo环境
conda activate cv-yolo
python scripts/env_setup/cv-yolo/环境检查.py

# 验证mp环境
conda activate mp
python scripts/env_setup/mp/环境检查.py
```

---

## 📅 学习路线规划
本仓库按照「先基础后实战、先机器学习后机器视觉再到姿态估计」的循序渐进路线规划，适配大学课余学习节奏，总周期约8个月。

### 第一阶段：机器学习全栈入门（8周）
打牢数据科学与机器学习核心基础，为后续机器视觉学习做底层铺垫，对应`01_machine_learning/`文件夹。

| 学习模块 | 学习周期 | 核心学习内容 | 验收标准 |
| :--- | :--- | :--- | :--- |
| Python数据科学基础 | 2周 | NumPy数组运算、Pandas数据清洗、Matplotlib/Seaborn数据可视化 | 独立完成表格数据的读取、清洗、特征处理与可视化 |
| 机器学习数学基础 | 1周 | 线性代数基础、概率统计核心、梯度下降与优化原理 | 能看懂算法数学公式，手动实现梯度下降 |
| 机器学习核心算法 | 4周 | 线性回归、逻辑回归、KNN、决策树、随机森林、SVM、K-Means聚类、模型评估 | 独立完成分类/回归/聚类任务，完成模型训练与调优 |
| 机器学习项目实战 | 1周 | 完整走完机器学习项目全流程：需求分析→数据清洗→特征工程→模型训练→可视化输出 | 完成2个完整实战项目，输出可复用代码与可视化报告 |

---

### 第二阶段：机器视觉&YOLO实战（16周）
掌握计算机视觉核心能力，精通YOLO目标检测全流程，发挥RTX 5060 GPU的硬件加速优势，对应`03_computer_vision/`文件夹。

| 学习模块 | 学习周期 | 核心学习内容 | 验收标准 |
| :--- | :--- | :--- | :--- |
| OpenCV视觉基础 | 3周 | 图像读写与像素操作、图像增强、特征提取、摄像头/视频流处理 | 独立完成图像/视频的基础处理，熟练使用OpenCV核心API |
| YOLOv8目标检测入门 | 3周 | YOLO核心原理、模型加载与推理、参数调优、摄像头实时检测、视频批量处理 | 独立完成图片/视频/摄像头的目标检测，按需调优参数 |
| YOLOv8进阶与自定义训练 | 4周 | 自定义数据集制作、模型微调训练、GPU训练优化、模型评估、实例分割/姿态估计/多目标跟踪 | 独立完成自定义数据集制作→模型训练→实时检测全流程 |
| 视频处理与高级视觉应用 | 2周 | 多路视频流处理、目标计数、越界检测、轨迹追踪、人流量统计 | 实现业务化视觉功能，不止于基础目标检测 |
| 视觉项目全栈实战 | 4周 | 完整视觉项目全流程开发，对应课程设计/竞赛/毕设需求 | 完成2个完整落地项目，输出可运行的完整程序与项目文档 |

---

### 第三阶段：MediaPipe姿态估计（8周）
掌握MediaPipe人体姿态估计、手部检测、面部识别等核心技术，对应`04_pose_estimation/`文件夹。

| 学习模块 | 学习周期 | 核心学习内容 | 验收标准 |
| :--- | :--- | :--- | :--- |
| MediaPipe基础 | 1周 | MediaPipe框架介绍、模型下载与配置、基础API使用 | 成功搭建MediaPipe环境，运行基础检测示例 |
| 手部姿态学习 | 2周 | 手部关键点检测、手势识别、双手协同、置信度估计 | 实现实时手部检测与手势识别，准确率达到85%以上 |
| 面部特征学习 | 2周 | 面部关键点检测、表情识别、面部特征分析 | 实现实时面部检测与表情识别，支持多种表情分类 |
| 身体姿态学习 | 2周 | 身体关键点检测、全身检测、姿态分析 | 实现实时全身姿态检测，绘制完整骨架 |
| MediaPipe项目实战 | 1周 | 结合OpenCV与MediaPipe，开发综合应用 | 完成1-2个完整的姿态估计应用项目 |

---

## 📌 使用与提交规范
### 文件使用规范
1.  学习笔记**仅存放于对应阶段的`00_notes/`文件夹**，按学习模块分子文件夹管理
2.  数据集、模型文件**仅存放于对应阶段的`data/`或`models/`文件夹**，绝不和代码、笔记混放
3.  纯代码练习、工具脚本存放于对应模块的代码文件夹，不与笔记混杂
4.  切换学习内容前，必须先激活对应的虚拟环境，避免版本冲突

### Git提交规范
提交信息严格遵循`[类型]: 提交内容`的格式，便于回溯学习历程：
- `feat: 新增XX学习笔记/代码/项目`
- `fix: 修复XX代码bug/环境问题`
- `docs: 更新README/学习文档`
- `refactor: 重构XX代码/优化笔记结构`
- `chore: 配置更新/工具脚本优化`

---

## 📄 免责声明
1.  本仓库仅用于个人学习记录，所有代码与笔记均为学习过程产出，请勿用于商业用途
2.  所有预训练模型均来自Ultralytics和Google MediaPipe官方开源项目，模型版权归原作者所有
3.  本仓库提供的环境配置方案仅为个人踩坑经验，仅供参考，因硬件/软件版本差异导致的问题，需自行排查解决

---

## 🤝 交流与反馈
如果你也是AI学习新手，或者遇到了RTX 50系显卡的环境适配问题，欢迎通过 [Issue](https://github.com/readant/Sum/issues) 交流学习，一起踩坑一起进步！

---
