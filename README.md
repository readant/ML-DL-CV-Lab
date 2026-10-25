<div align="center">

# ML-DL-CV-Lab

**机器学习 · 深度学习 · 机器视觉 科研工作台**

面向科研复现与文献管理的深度学习仓库：方法复现、文献管理、实验记录与写作沉淀一体化，配套统一环境一键复现、结构化笔记与可运行代码。

<p>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://github.com/readant/Sum"><img alt="License" src="https://img.shields.io/badge/license-MIT-4DABF7?style=flat-square"></a>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-RTX%2050%20Series-76B900?style=flat-square&logo=nvidia">
</p>

<p>
  <b>统一环境一键复现</b> ·
  <b>方法复现 + 单元测试</b> ·
  <b>可引文献库 (BibTeX)</b> ·
  <b>实验记录可追溯</b>
</p>

</div>

---

## 目录

- [研究主题与背景](#研究主题与背景)
- [研究问题 / 假设](#研究问题--假设)
- [当前进展](#当前进展)
- [仓库结构](#仓库结构)
- [复现指南](#复现指南)
- [科研日志与文献管理](#科研日志与文献管理)
- [使用与提交规范](#使用与提交规范)
- [免责声明](#免责声明)

---

## 研究主题与背景

> **TODO（待补充）**：用 2–3 句话说明本仓库聚焦的研究方向，例如「面向 XX 任务的深度学习注意力机制研究」。
> 本仓库用于系统沉淀方法复现、文献笔记、实验记录与写作草稿。

> 研究背景与动机可在此展开：为什么选这个方向、现有工作的不足、本工作的切入点。建议从导师给定的知识点清单出发，逐步收敛为明确的研究问题。

## 研究问题 / 假设

> **TODO（待补充）**：列出 1–3 个核心研究问题或假设，例如：
> - **RQ1**：……
> - **RQ2**：……
> - **假设 H**：……

> 研究问题应可验证、有边界，并能映射到一个或多个可复现实验（见 `experiments/`）。

## 当前进展

| 模块 | 位置 | 状态 | 说明 |
| :--- | :--- | :---: | :--- |
| 注意力机制 / 多头注意力 (MHA) | `02_deep_learning/01_代码/01_注意力机制/` | ✅ 已落地 | NumPy + PyTorch 实现、14 条单元测试、可视化 demo、文献笔记 |
| 神经网络从零实现（NNDL · MNIST） | `02_deep_learning/01_代码/02_神经网络从零实现_NNDL/` | ✅ 已落地 | 三层 MLP、7 条单元测试、合成数据离线 demo、批判性笔记 |
| 机器学习基础（NumPy / sklearn / 算法） | `01_machine_learning/` | ✅ 方法底座 | 作为研究的方法学基础 |
| 计算机视觉（OpenCV / YOLO） | `03_computer_vision/` | ✅ 方法底座 | 作为研究的方法学基础 |
| 姿态估计（MediaPipe 手/身/面） | `04_pose_estimation/` | ✅ 方法底座 | 作为研究的方法学基础 |
| 深度学习其余方法（张量/autograd、NN、CNN、RNN） | `02_deep_learning/` | 🟡 规划中 | 待按研究主线补齐 |
| 首个研究实验 | `experiments/` | ⬜ 待启动 | 由研究问题驱动 |

> 进展随 `research_log.md` 同步更新。

---

## 仓库结构

遵循「研究层与基础层分离、笔记代码分离、数据模型统一管理、实验可追溯」的原则：

```plaintext
ML-DL-CV-Lab/
├── README.md                       # 本文档（科研向）
├── research_log.md                 # 科研日志 / 实验室笔记（lab notebook）
├── references.bib                  # 统一可引文献库 (BibTeX)
├── environment.yml                 # main 统一环境定义（版本锁定）
├── config.py                       # 路径基建（统一定位根级 models/data）
├── pyproject.toml                  # 项目工具链配置（ruff 等）
│
├── literature/                     # 【研究层】文献管理（按主题：清单+批判性笔记）
├── experiments/                    # 【研究层】可复现实验（config+run+results）
├── docs/
│   └── meetings/                   # 【研究层】导师见面记录（模板见 _模板.md）
│
├── 00_prerequisites/               # 【基础层】前置：Python、数学
├── 01_machine_learning/            # 【基础层】机器学习（方法底座）
├── 02_deep_learning/               # 【基础层】深度学习（含注意力机制、神经网络从零实现）
├── 03_computer_vision/             # 【基础层】计算机视觉（YOLO）
├── 04_pose_estimation/             # 【基础层】姿态估计（MediaPipe）
├── 05_projects/                    # 【基础层】综合项目（竞赛 / 完整项目 / 作品集）
│
├── data/                           # 根级共享数据集（不入库）
├── models/                         # 根级共享预训练模型（不入库）
├── scripts/                        # 工具脚本（环境检查 / 通用工具）
└── .gitignore                      # Git 忽略规则
```

### 路径规范

- 数据与模型**统一放在根级 `data/`、`models/`**，各阶段不单独存放。
- `.py` 脚本通过 `pathlib.Path(__file__)` 上溯定位根级目录，不受运行目录影响。
- Notebook 因 `__file__` 不可用，请在首个 cell 运行各阶段 README 中的 bootstrap cell 定位仓库根目录后导入 `config`。

---

## 复现指南

研究内容依赖统一的 Conda `main` 环境，所有依赖（PyTorch、YOLO、MediaPipe 等）安装在同一环境中，通过 `environment.yml` 一键创建，保证环境可复现。

### 前置依赖

- 安装 [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)（推荐 Python 3.11+ 版本）。
- Windows 用户需提前更新 NVIDIA 显卡驱动至最新版（RTX 50 系显卡需驱动版本 ≥ 572.97）。

### 一键创建统一环境

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

### 快速开始（运行已有方法模块）

```bash
# 激活环境后，运行注意力机制单元测试作为冒烟测试
conda activate main
cd 02_deep_learning/01_代码/01_注意力机制
python -m unittest test_attention -v      # Ran 14 tests ... OK
python demo_attention.py                  # 打印注意力权重矩阵

# 神经网络从零实现（NNDL）：合成数据、离线、数秒跑完
cd ../02_神经网络从零实现_NNDL
python -m unittest test_network -v        # Ran 7 tests ... OK
python demo_network.py                    # 合成数据跑一个 epoch 看损失下降
python network.py                         # 真实 MNIST：首次需联网下载，约 94% 准确率
```

> 环境就绪后，可依次阅读 `00_prerequisites/` → `01_machine_learning/` → `02_deep_learning/` 作为方法底座。

---

## 科研日志与文献管理

本仓库面向科研复现与文献管理，约定如下工作流（详见各文件夹 README）：

| 用途 | 位置 | 说明 |
| :--- | :--- | :--- |
| 科研日志 | [`research_log.md`](research_log.md) | 每日/每周推进：做了什么、结论、卡点、下一步 |
| 统一文献库 | [`references.bib`](references.bib) | BibTeX 题录，便于引用与溯源 |
| 文献管理 | [`literature/`](literature/) | 按主题：文献清单 + 批判性笔记（四问法） |
| 实验记录 | [`experiments/`](experiments/) | 每个实验 = config + run + results + REPRODUCE.md |
| 导师见面 | [`docs/meetings/`](docs/meetings/) | 见面记录模板，沉淀导师意见与待办 |

**检索与选文**：优先 Papers with Code（代码+论文绑定）、arXiv、Google Scholar、Semantic Scholar、知网(CNKI)；先读综述锁定脉络，再按 venue / 引用数 / 官方实现筛选。

---

## 使用与提交规范

### 文件使用规范

1. 方法复现代码放对应阶段的 `01_代码/`，文献笔记放 `00_笔记/`；研究级文献统一在根 `literature/`。
2. 数据集、模型文件**统一存放于根级 `data/` 或 `models/`**，绝不与代码、笔记混放。
3. 每个方法模块配套 `test_*.py` 单元测试，用断言钉死易错细节；实验须可复现（固定种子 + 版本锁定）。
4. 整个仓库统一使用 `conda activate main` 激活环境，无需在多个环境间切换。

### Git 提交规范

提交信息严格遵循 `[类型]: 提交内容` 的格式，便于回溯研究历程：

| 类型 | 用途 |
| :--- | :--- |
| `feat` | 新增方法复现 / 实验 / 文献笔记 |
| `fix` | 修复代码 bug / 环境问题 |
| `docs` | 更新 README / 科研日志 / 文献 |
| `refactor` | 重构代码 / 优化结构 |
| `chore` | 配置更新 / 工具脚本优化 |

---

## 免责声明

1. 本仓库为个人科研学习记录，所有代码与笔记均为学习/研究过程产出，请勿用于商业用途。
2. 预训练模型来自 Ultralytics、Google MediaPipe 等官方开源项目，版权归原作者所有。
3. 环境配置方案仅为个人踩坑经验，因硬件/软件版本差异导致的问题需自行排查。
4. 文献题录与笔记仅供个人研究参考，引用请以原始出版物为准。
