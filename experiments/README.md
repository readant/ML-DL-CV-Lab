# 实验记录 (experiments/)

每个独立实验一个子文件夹，命名 `YYYY-MM-DD-实验简称/`。

## 标准结构

```plaintext
YYYY-MM-DD-实验简称/
├── config.yaml      # 超参、随机种子、数据路径（复现必需）
├── run.py           # 入口脚本，读取 config 后运行
├── results/         # 输出：指标表、曲线图、日志
└── REPRODUCE.md     # 一句话复现命令 + 预期结果
```

## 复现命令（统一环境）

```bash
conda activate main
cd experiments/YYYY-MM-DD-实验简称
python run.py --config config.yaml
```

## 注意

- 固定随机种子，并记录所依赖的 `torch` / 框架具体版本（见根 `environment.yml`）。
- 实验结论先在 `research_log.md` 登记，再在此落地代码。
- 涉及对比的，结果表用 Markdown 表格呈现，含 baseline 对照。
