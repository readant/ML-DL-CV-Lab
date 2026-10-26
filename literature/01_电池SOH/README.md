# 文献资料（Literature）

> 本目录收录与「注意力 + DeepHPM + 电池 SOH」方向相关的文献地图、文献清单与已获取的原文。
> 整理日期：2026-10-26

---

## 一、目录内容

| 文件 | 内容 |
| :--- | :--- |
| [文献清单.md](文献清单.md) | 结构化题录（题名/作者/年份/venue/DOI/被引）与检索记录 |
| [文献笔记.md](文献笔记.md) | 批判性笔记（四问法），逐篇随读随填 |
| [电池SOH-注意力与DeepHPM文献地图.html](电池SOH-注意力与DeepHPM文献地图.html) | **主文档**。DeepHPM 与 `PINN4SOH` 架构的对应关系、竞品解剖、可用选题空位 |
| [DeepHPM文献清单-深度隐藏物理模型.html](DeepHPM文献清单-深度隐藏物理模型.html) | DeepHPM 通用文献清单（非电池专项）：源头论文辨析、方法演进 |
| `pdfs/` | 已下载的原文 PDF（**本地存放，不入版本库**，见 `.gitignore` 的 `literature/**/pdfs/`） |

---

## 二、核心结论速览

**1. `PINN4SOH` 本身就是 DeepHPM 实例**，不是新框架：

| DeepHPM 概念 | 本仓库对应物 |
| :--- | :--- |
| 解代理网络 `F(x,t)` | `Solution_u` |
| 隐藏动力学网络 `G(u,λ)` | `dynamical_F`（输入 `concat(x,t,u,u_x,u_t)`）|
| PDE 残差 `u_t − N(u,λ)` | `loss2`（权重 α）|
| 数据拟合项 | `loss1` |
| 物理先验 | `loss3`（ReLU 单调性，权重 β）|

→ 导师所指的「注意力 + Deep HPM」，落点是**在现有模型上加注意力**，而非重建。

**2. 已有直接竞品占位**：Energies 2025 的 AM-MFF-PINN 已同时使用
多域特征（时域/频域/小波）、Pearson+MIC 特征筛选、注意力机制、
**动态损失平衡**，且数据集就是 XJTU + TJU（与本仓库重合两个）。

→ 因此原计划的「Optuna 自动搜 α/β」路线**不建议单独立题**。

**3. 仍空白的四个方向**：不确定性量化 / 注意力作用于 `dynamical_F` 的导数输入 /
系统化迁移学习 / Arrhenius 等物理机理约束。详见主文档第 04 节。

---

## 三、已获取原文（3 篇，合法开放获取）

存放于 `pdfs/`：

| 文件 | 出处 | DOI |
| :--- | :--- | :--- |
| `[2018] Raissi - Deep Hidden Physics Models (arXiv1801.06637).pdf` | arXiv / JMLR | `10.48550/arXiv.1801.06637` |
| `[2021] Chen - Physics-informed learning of governing equations (NatCommun).pdf` | Nature Communications | `10.1038/s41467-021-26434-1` |
| `[2025] Wu - Attention Multi-Feature Fusion PINN for SOH (Energies).pdf` | Energies (MDPI) | `10.3390/en18236270` |

> 基础论文（Wang et al., Nat. Commun. 2024）不在本目录重复存放，
> 见 nc-PINN 项目仓库：`E:\Projects0\Python\nc-PINN\nc-PINN.pdf`。

---

## 四、未能获取（订阅墙）

学校（河南科技学院）**未订阅** ScienceDirect（Elsevier）与 IEEE Xplore：

- ScienceDirect 文章页明确提示：`Henan Institute of Science and Technology does not subscribe to this content`
- IEEE 的 PDF 端点只返回登录页 HTML

| 文献 | 出处 | DOI |
| :--- | :--- | :--- |
| Hidden physics models: Machine learning of nonlinear PDEs | J. Comput. Phys. 2018 | `10.1016/j.jcp.2017.11.039` |
| A time series PINN framework for SOH estimation | J. Energy Storage 2025 | `10.1016/j.est.2025.119549` |
| Integrating transformers into PINNs for SOH prognostics | IJEPES 2025 | `10.1016/j.ijepes.2025.111173` |
| Physics-informed hierarchical perception modulation network | Energy 2025 | `10.1016/j.energy.2025.138229` |
| An attention-guided residual block-based PINN | Energy 2026 | `10.1016/j.energy.2026.142247` |
| PINN for PHM of Lithium-Ion Batteries | IEEE T-IV 2023 | `10.1109/tiv.2023.3315548` |
| A Robust SOH Estimation Method Based on Physics-Informed Enhancement | IEEE T-IE 2026 | `10.1109/tie.2026.3654776` |
| A learnable graph spatiotemporal attention PINN | Meas. Sci. Technol. 2026 | `10.1088/1361-6501/ae5fc0` |

**合法获取途径**：图书馆文献传递 / 馆际互借（CALIS、CASHL）、NSTL 按篇付费、
作者主页或机构仓库自存档、邮件向通讯作者索取。
本目录**不使用** sci-hub 等镜像站。

---

## 五、阅读顺序建议

1. `pdfs/[2025] Wu - AM-MFF-PINN` —— 唯一能立即拿到全文的直接竞品，先看它做到哪一步。
2. nc-PINN 仓库根 `nc-PINN.pdf` —— 项目基础论文，用于对照自身实现。
3. `pdfs/[2018] Raissi - DeepHPM` —— 方法学源头，写"为什么用双网络"时引这篇。

> **注意**：文献地图中所有"为什么重要""空位判断"均基于标题与摘要级信息，
> 除 AM-MFF-PINN 读过摘要外，其余未读正文。写综述前须核对原文。
