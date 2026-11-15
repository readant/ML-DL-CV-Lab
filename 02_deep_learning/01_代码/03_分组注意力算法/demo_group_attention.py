"""分组注意力的可观察演示（合成数据，离线，秒级）。

本脚本**不承担性能证明**，只回答四个可核对的问题：

1. 算法是否真的与配置无关 —— 同一份实现能否直接用于不同分组结构；
2. 恒等性成不成立 —— 初始输出与同权重原网络的最大逐元素差；
3. 门控会不会动、往哪动 —— 训练后 gᵢ 是否偏离 1，以及偏向哪一组；
4. 具体项目里怎么用 —— 只打印配置，不写死任何项目常量。

第 3 点用合成任务，**不能**据此推断「在真实任务上会更准」。

运行：
    cd 02_deep_learning/01_代码/03_分组注意力算法
    python demo_group_attention.py
"""
import torch
import torch.nn as nn

from group_attention import GroupAttention, GroupGatedNet

# 若干与领域无关的分组配置
CONFIGS = [
    ((3, 5, 2), "3 组 / 共 10 维"),
    ((4, 4), "2 组 / 共 8 维"),
    ((1, 1, 1), "3 组 / 共 3 维"),
    ((2, 3, 4, 5, 6), "5 组 / 共 20 维"),
]

# 一个真实用例：PINN4SOH 的 dynamical_F 输入端。
# 注意这里只是「一份配置」，算法本身不知道也不关心它的含义。
PINN4SOH_DIMS = (17, 1, 16, 1)
PINN4SOH_NAMES = ("xt 状态+循环索引", "u 解输出", "u_x 空间导数", "u_t 时间导数")


def _make_net(in_dim, hidden=64, dropout=0.2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def show_config_independence():
    print("=" * 76)
    print("1. 算法与配置分离：同一实现直接适用于任意分组")
    print("=" * 76)
    for dims, desc in CONFIGS:
        ga = GroupAttention(dims)
        groups = [torch.randn(6, d) for d in dims]
        gated, gate = ga(groups)
        out_dims = tuple(g.size(1) for g in gated)
        n = sum(p.numel() for p in ga.parameters())
        ok = "一致" if out_dims == dims else "不一致"
        print(f"{desc:<16} dims={dims}  →  输出 {out_dims} {ok}  "
              f"gate{tuple(gate.shape)}  门控参数 {n}")
    print("说明：组数、各组维度、总维度都可变，算法内部无任何硬编码分支。")
    print()


def show_identity():
    print("=" * 76)
    print("2. 恒等初始化（以 3 组 / 10 维为例）")
    print("=" * 76)
    dims = (3, 5, 2)
    torch.manual_seed(7)
    net_a, net_b = _make_net(sum(dims)), _make_net(sum(dims))
    net_b.load_state_dict(net_a.state_dict())
    gated = GroupGatedNet(net_a, dims)
    gated.eval()
    net_b.eval()

    z = torch.randn(64, sum(dims))
    with torch.no_grad():
        out_gated, out_plain = gated(z), net_b(z)
        gate = gated.gate_summary(z)

    max_diff = (out_gated - out_plain).abs().max().item()
    print(f"初始门控: {[round(float(v), 6) for v in gate[0]]}")
    print(f"gate 极值: min={gate.min():.6f}  max={gate.max():.6f}")
    print(f"与同权重原网络的最大逐元素差: {max_diff:.3e}  "
          f"→ {'恒等成立' if max_diff < 1e-6 else '恒等被破坏'}")
    print("注：eval 模式严格恒等；train 模式因 Dropout 随机掩码，单次前向不相等。")
    print()


def show_gate_learns():
    print("=" * 76)
    print("3. 训练后门控如何变化（合成任务，仅演示机制）")
    print("=" * 76)
    dims = (3, 5, 2)
    torch.manual_seed(11)
    gated = GroupGatedNet(_make_net(sum(dims)), dims)
    gated.train()

    # 合成任务：只有「第 2 组」带信息，其余两组为纯噪声。
    # 若门控机制有效，训练后第 2 组的 gate 应相对抬高。
    n, batch = 512, 128
    z_all = torch.randn(n, sum(dims))
    split = torch.split(z_all, dims, dim=-1)
    y_all = 3.0 * split[1][:, :1] + 0.5 * torch.randn(n, 1)

    opt = torch.optim.Adam(gated.parameters(), lr=3e-3)
    lossf = nn.MSELoss()
    before = gated.gate_summary(z_all[:batch]).mean(dim=0)

    for step in range(1, 601):
        idx = torch.randint(0, n, (batch,))
        opt.zero_grad()
        loss = lossf(gated(z_all[idx]), y_all[idx])
        loss.backward()
        opt.step()
        if step in (1, 100, 300, 600):
            g = gated.gate_summary(z_all[:batch]).mean(dim=0)
            print(f"step {step:4d}  loss={loss.item():8.4f}  "
                  f"gate={[round(float(v), 4) for v in g]}")

    after = gated.gate_summary(z_all[:batch]).mean(dim=0)
    gain = after - before
    print()
    print(f"门控变化: {[round(float(v), 4) for v in gain]}")
    print(f"最高组: 第 {int(after.argmax())} 组（合成任务中带信息的正是第 1 组）")
    print("注：这只是「机制在学」的证据，不是「更准」的证据。")
    print()


def show_attention_map():
    print("=" * 76)
    print("4. 组间注意力图（未训练，仅确认形状与归一化）")
    print("=" * 76)
    dims = (3, 5, 2)
    ga = GroupAttention(dims)
    ga.eval()
    with torch.no_grad():
        ga([torch.randn(256, d) for d in dims])
        w = ga.last_attn_w.mean(dim=0)

    print("query \\ key" + "".join(f"{'组'+str(j):>10}" for j in range(len(dims))))
    for i in range(len(dims)):
        print(f"{'组'+str(i):<12}" + "".join(f"{float(v):10.4f}" for v in w[i]))
    print("行和为 1。")
    print()


def show_project_config():
    print("=" * 76)
    print("5. 具体项目怎么用：这里只打印配置，算法代码不含任何项目常量")
    print("=" * 76)
    print(f"分组配置: {PINN4SOH_DIMS}  合计 {sum(PINN4SOH_DIMS)} 维")
    for name, d in zip(PINN4SOH_NAMES, PINN4SOH_DIMS):
        print(f"  - {name:<16} {d:>3} 维")
    print()
    print("用法（三行，不改动算法本体）:")
    print("    net    = 你的 dynamical_F 主干")
    print(f"    gated  = GroupGatedNet(net, {PINN4SOH_DIMS})")
    print("    output = gated(z)   # z 为拼接后的输入，宽度必须等于 sum(dims)")
    print()
    print("恒等初始化保证：替换后首步输出与原来一致（eval 模式逐元素相等），")
    print("因此「加注意力前后的差异」可完全归因于训练中学到的门控偏离。")
    print()


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    show_config_independence()
    show_identity()
    show_gate_learns()
    show_attention_map()
    show_project_config()
    print("演示结束。以上不构成性能结论。")


if __name__ == "__main__":
    main()
