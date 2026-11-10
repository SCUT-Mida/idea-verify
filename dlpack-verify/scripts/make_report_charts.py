# -*- coding: utf-8 -*-
"""
make_report_charts.py — 从真机运行数据生成验证报告图表
======================================================

数据来源: 910B 真机运行输出 (2026-11-10, CANN 8.3.RC1 / torch_npu 2.7.1 / mindspore 2.10.0)
输出: charts/ 目录下的 PNG (dpi=150), 由 验证报告.md 引用

生成 3 张图:
  1) loss_curves_npu.png   — A/B 臂 loss 曲线对照 (M1 证据, 两种规模)
  2) metrics_scorecard.png — M1~M4 底线评分卡 (2x2 面板, 通过带/阈值线)
  3) bench_npu.png         — 纯交换吞吐对比 (dlpack vs copy, --bench)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SCRIPT_DIR)
OUT = os.path.join(PROJECT, "charts")
LOG_DIR = os.path.join(PROJECT, "results", "npu")   # 真机逐步日志
os.makedirs(OUT, exist_ok=True)

C_A = "#7f7f7f"   # A 臂 灰
C_B = "#1f77b4"   # B 臂 蓝
C_OK = "#2ca02c"
C_BAD = "#d62728"
C_COPY = "#ff7f0e"

STEPS = [1, 50, 100, 150, 200, 250, 300]

# ---- 真机数据 (train_hybrid.py, 910B) ----
RUN1 = {  # hidden=64, batch=128, lr=0.05
    "label": "hidden=64, batch=128, lr=0.05",
    "csv": os.path.join(LOG_DIR, "training_log_dlpack_h64.csv"),   # 存在则用逐步 dense 数据
    "A": [1.531153e-01, 1.837008e-02, 1.249019e-02, 9.028116e-03,
          6.845919e-03, 5.431500e-03, 4.488275e-03],
    "B": [1.531153e-01, 1.824383e-02, 1.242429e-02, 8.990590e-03,
          6.822213e-03, 5.416146e-03, 4.477296e-03],
    "M1": 0.9976, "M2": 0.9989, "M3": 2.586, "M4": 2.611e-06,
    "lat_ms": (1.20, 3.11),
}
RUN2 = {  # hidden=1024, batch=1024, lr=0.005
    "label": "hidden=1024, batch=1024, lr=0.005",
    "csv": os.path.join(LOG_DIR, "training_log_dlpack_h1024.csv"),   # 存在则用逐步 dense 数据
    "A": [2.679930e-01, 1.636368e-02, 1.052756e-02, 7.674805e-03,
          6.122353e-03, 5.208739e-03, 4.639633e-03],
    "B": [2.679930e-01, 1.619514e-02, 1.044948e-02, 7.634023e-03,
          6.099036e-03, 5.194513e-03, 4.630510e-03],
    "M1": 0.9980, "M2": 0.9979, "M3": 2.762, "M4": 6.535e-06,
    "lat_ms": (1.20, 3.31),
}
BENCH = {"fwd": (14.8, 206.3), "bwd": (13.8, 120.5)}  # µs/次 (dlpack, copy)


def load_csv_curve(path):
    """读取 train_hybrid.py 落盘的逐步日志 (step, loss_a, loss_b, t_a_ms, t_b_ms)"""
    import csv
    steps, a, b = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            a.append(float(row["loss_a"]))
            b.append(float(row["loss_b"]))
    return steps, a, b


def run_curves(run):
    """优先读取逐步 CSV (dense); 不存在则回退到检查点数据"""
    csv_path = run.get("csv", "")
    if csv_path and os.path.exists(csv_path):
        return load_csv_curve(csv_path)
    return STEPS, run["A"], run["B"]


# ===========================================================================
# 图 1: loss 曲线对照 (M1 证据)
# ===========================================================================
def fig_loss_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    for col, run in zip(range(2), [RUN1, RUN2]):
        steps_, A_, B_ = run_curves(run)
        # 上排: loss 曲线
        ax = axes[0][col]
        ax.semilogy(steps_, A_, "s--", color=C_A, markerfacecolor="white",
                    label="A 臂: 单框架端到端基线 (PyTorch)")
        ax.semilogy(steps_, B_, "o-", color=C_B,
                    label="B 臂: 混合训练 (MS 前段 ─DLPack─ PT 后段)")
        ax.set_ylabel("训练 loss (MSE, log 刻度)")
        ax.set_title(run["label"], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        # 下排: 逐检查点 B/A 比 (重合度可视化)
        ax2 = axes[1][col]
        ratio = [b / a for b, a in zip(B_, A_)]
        ax2.plot(steps_, ratio, "o-", color=C_OK, markersize=3 if len(steps_) > 50 else 6)
        ax2.axhline(1.0, color="gray", ls="--", lw=1)
        ax2.set_ylim(0.985, 1.005)
        ax2.set_xlabel("训练步数 (step)")
        ax2.set_ylabel("逐检查点 B/A 比")
        ax2.set_title("两臂逐点重合度 (比值越接近 1.0 越等价)", fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.annotate(f"最终 B/A = {ratio[-1]:.4f}", xy=(steps_[-1], ratio[-1]),
                     xytext=(steps_[-1] - max(1, int(len(steps_) * 0.35)),
                             ratio[-1] + 0.004), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle("混合训练 vs 单框架基线 — loss 收敛曲线对照 (910B 真机, 同种子/同初始化/同超参)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "loss_curves_npu.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ===========================================================================
# 图 2: M1~M4 评分卡
# ===========================================================================
def fig_scorecard():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), dpi=150)
    names = ["h=64\nbatch=128", "h=1024\nbatch=1024"]

    # M1: 最终 loss 比 B/A, 通过带 [0.5, 2.0]
    ax = axes[0][0]
    ax.axhspan(0.5, 2.0, color=C_OK, alpha=0.13)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.bar(names, [RUN1["M1"], RUN2["M1"]], color=C_OK, width=0.5)
    for i, v in enumerate([RUN1["M1"], RUN2["M1"]]):
        ax.text(i, v + 0.04, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, 2.3)
    ax.set_title("M1 收敛等价性 (B/A 最终 loss 比) — PASS", color=C_OK, fontsize=10)
    ax.set_ylabel("比值")
    ax.grid(alpha=0.3, axis="y")

    # M2: held-out 评估比, 阈值 1.01
    ax = axes[0][1]
    ax.axhspan(0, 1.01, color=C_OK, alpha=0.13)
    ax.axhline(1.01, color=C_BAD, ls="--", lw=1)
    ax.bar(names, [RUN1["M2"], RUN2["M2"]], color=C_OK, width=0.5)
    for i, v in enumerate([RUN1["M2"], RUN2["M2"]]):
        ax.text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_title("M2 指标不恶化 (B/A held-out 评估 MSE 比) — PASS", color=C_OK, fontsize=10)
    ax.set_ylabel("比值")
    ax.grid(alpha=0.3, axis="y")

    # M3: 步时延比, 阈值 1.02 (当前未达, 玩具规模)
    ax = axes[1][0]
    ax.axhspan(0, 1.02, color=C_OK, alpha=0.13)
    ax.axhline(1.02, color=C_OK, ls="--", lw=1, label="阈值 1.02 (1+m3_tol)")
    ax.bar(names, [RUN1["M3"], RUN2["M3"]], color=C_BAD, width=0.5)
    for i, (run, v) in enumerate(zip([RUN1, RUN2], [RUN1["M3"], RUN2["M3"]])):
        a, b = run["lat_ms"]
        ax.text(i, v + 0.06, f"{v:.2f}x", ha="center", fontsize=9)
        ax.text(i, 0.08, f"A={a} ms\nB={b} ms", ha="center", fontsize=8,
                color="white", fontweight="bold")
    ax.set_ylim(0, 3.3)
    ax.set_title("M3 步时延 (B/A 比值) — 玩具规模未达", color=C_BAD, fontsize=10)
    ax.set_ylabel("比值")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    # M4: 边界梯度相对误差, 阈值 rtol=1e-3 (log)
    ax = axes[1][1]
    ax.axhline(1e-3, color=C_BAD, ls="--", lw=1, label="rtol 阈值 1e-3")
    ax.bar(names, [RUN1["M4"], RUN2["M4"]], color=C_OK, width=0.5)
    for i, v in enumerate([RUN1["M4"], RUN2["M4"]]):
        ax.text(i, v * 1.35, f"{v:.2e}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 3e-3)
    ax.set_title("M4 边界梯度相对误差 (桥接 vs 单框架) — PASS", color=C_OK, fontsize=10)
    ax.set_ylabel("最大相对误差 (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("四条底线指标评分卡 (M1~M4, 910B 真机)", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.text(0.01, 0.008,
             "M3 说明: B 臂每步比 A 臂多 ~1.9~2.1 ms, 主体为 MindSpore PYNATIVE 单算子调度常数 "
             "(B 臂每步 4 次 MS 算子调度), 并非交换开销 (dlpack 交换段仅 0.10~0.63 ms/步)。\n"
             "该常数为框架固有开销, 生产规模 (每步计算百 ms~秒级) 按占比摊薄; 且 MS 侧前段"
             "无论是否混合, 其自身调度开销都存在 —— 混合范式净增成本仅为交换段 (µs 级)。"
             "生产规模复测列入后续验证计划。",
             fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round", fc="#fff7e6", alpha=0.9))
    p = os.path.join(OUT, "metrics_scorecard.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ===========================================================================
# 图 3: 纯交换吞吐对比 (--bench)
# ===========================================================================
def fig_bench():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    groups = ["正向 MS→PT", "回传 PT→MS"]
    dl = [BENCH["fwd"][0], BENCH["bwd"][0]]
    cp = [BENCH["fwd"][1], BENCH["bwd"][1]]
    xs = np.arange(len(groups))
    w = 0.32
    b1 = ax.bar(xs - w / 2, dl, w, color=C_B, label="dlpack (零拷贝, 留在 NPU 显存)")
    b2 = ax.bar(xs + w / 2, cp, w, color=C_COPY, label="copy (NPU→host→NPU 往返)")
    ax.set_yscale("log")
    ax.set_ylim(5, 500)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel("单次交换耗时 (µs, log 刻度)")
    ax.bar_label(b1, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, fmt="%.1f", fontsize=9)
    for i, (d, c) in enumerate(zip(dl, cp)):
        ax.text(i, max(c, d) * 1.6, f"copy 慢 {c / d:.1f}x", ha="center",
                fontsize=10, color=C_BAD, fontweight="bold")
    ax.set_title("纯交换吞吐对比 (910B --bench, 同进程, N=200, 剥离框架算子开销)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    p = os.path.join(OUT, "bench_npu.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_loss_curves, fig_scorecard, fig_bench):
        print("生成:", fn())
    print("DONE")
