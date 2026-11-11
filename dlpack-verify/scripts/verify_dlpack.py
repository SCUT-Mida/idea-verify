# -*- coding: utf-8 -*-
"""
verify_dlpack.py — MindSpore ↔ PyTorch (DLPack) 验证脚本
=========================================================

验证内容 (对应验证方案的验收标准):
  STEP 1  MS→PT   基础交换: dtype/shape/设备正确
  STEP 2  PT→MS   基础交换: dtype/shape/设备正确
  STEP 3  零拷贝语义: 一侧原地修改, 另一侧同步变化
  STEP 4  PyTorch 局部计算图: DLPack 叶子 requires_grad_ 后 backward 正常
  STEP 5  梯度桥接: pt_input.grad → sens → GradOperation(sens_param=True)
  STEP 6  数值正确性: 桥接梯度 vs float64 解析解 (手算对照), 容差内一致
  STEP 7  负对照: 用 ones_like 代替真实 sens, 梯度应明显不同
  STEP 8  两侧参数均可更新 (PT optim.step / MS set_data)
  STEP 9  交换耗时统计: dlpack 路径 vs 拷贝路径
  STEP 10 图语义聚合对照 (GNN C1): 交换边界两侧各做一次 scatter 聚合, 结果必须一致
  STEP 11 稀疏 Embedding 梯度完整性 (GNN C3): 交换路径上未访问行必须零变化

流程图 (5 个关键动作):
  MS: x ─▶ ms_block ─▶ ms_out
              │ ①(导出) ②DLPack
              ▼
  PT:      pt_input ③requires_grad_(True) ─▶ pt_head ─▶ loss ─④ backward
                                                        │
                          pt_input.grad ──⑤──▶ MS sens ─▶ ms_block 参数梯度 ─▶ 更新

结构:
    main() 为入口; 每个 STEP 一个函数, 便于 PyCharm 单步调试。
    各 STEP 共享的状态放在 ctx 字典中 (模型/输入/梯度等)。

用法:
    python verify_dlpack.py                 # 自动探测 (有 torch_npu 用 NPU, 否则 CPU)
    python verify_dlpack.py --device cpu    # 强制 CPU
    python verify_dlpack.py --device npu    # 强制 NPU (910B)
"""
import argparse
import json
import os
import sys
import time
import warnings

# 静音两类已知无害的平台告警 (成因见 README §4.3), 按消息精确匹配, 其他警告仍可见:
#   - "owner does not match": CANN 由 root 安装而以普通用户运行, torch_npu 环境收集器的提示
#   - "smallest subnormal":   ARM 容器内核关闭非规格化浮点, numpy 提示 denormal 刷零
warnings.filterwarnings("ignore", message=".*owner does not match.*")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")

import numpy as np
import torch
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import GradOperation
from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack

LR = 0.01                      # 两侧共用的学习率 (最朴素 SGD)
results = []                   # 验证结果收集: (step, name, ok, detail)
MEAS = {}                      # 实测数值收集 (供 make_report.py 生成报告)

# 全局运行时配置, setup_frameworks() 里赋值
DEVICE = MS_TARGET = None
IS_NPU = False
RTOL = 1e-3


def check(step, name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] STEP {step}  {name}" + (f"  -- {detail}" if detail else ""))
    results.append((step, name, ok, detail))
    return ok


def to_np(t):
    """torch tensor -> numpy (统一走 host, 屏蔽 cpu/npu 差异)"""
    return t.detach().cpu().numpy()


def rel_err(a, b):
    """逐元素相对误差最大值"""
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))


# ---------------------------------------------------------------------------
# 环境初始化
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="MindSpore ↔ PyTorch DLPack 验证")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "npu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rtol", type=float, default=1e-3,
                    help="数值对照相对容差 (CPU 建议 1e-3, NPU 可放宽到 5e-3)")
    return ap.parse_args()


def resolve_device(requested: str):
    """返回 (device, device_target)。npu 路径必须先 import torch_npu。"""
    if requested == "cpu":
        return "cpu", "CPU"
    npu_ok = False
    try:
        import torch_npu  # noqa: F401  (必须导入, 补丁才生效 — 910B 上关键!)
        npu_ok = True
    except ImportError:
        pass
    if requested == "npu":
        if not npu_ok:
            print("[FATAL] --device npu 但 torch_npu 不可用, 请检查 910B 环境")
            sys.exit(2)
        return "npu:0", "Ascend"
    # auto
    if npu_ok:
        try:
            if torch.npu.is_available():
                return "npu:0", "Ascend"
        except Exception:  # noqa: BLE001
            pass
    return "cpu", "CPU"


def setup_frameworks(args):
    """设备探测 + MindSpore 上下文 + 随机种子 (910B 时在此导入 torch_npu)"""
    global DEVICE, MS_TARGET, IS_NPU, RTOL
    DEVICE, MS_TARGET = resolve_device(args.device)
    IS_NPU = DEVICE.startswith("npu")
    RTOL = args.rtol
    if IS_NPU:
        import torch_npu  # noqa: F401  -- npu 环境必须导入, 使 DLPack 补丁生效
    ms.set_context(device_target=MS_TARGET, mode=ms.PYNATIVE_MODE)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("=" * 78)
    print(f"DLPack 验证  device={DEVICE}  device_target={MS_TARGET}")
    print(f"numpy={np.__version__}  torch={torch.__version__}  "
          f"mindspore={ms.__version__}  mode=PYNATIVE")
    MEAS["env"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                   "torch": torch.__version__, "mindspore": ms.__version__,
                   "device": DEVICE}
    print("=" * 78)


class MsBlock(nn.Cell):
    """MindSpore 侧小模型 (4->4), 与验证方案一致"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(4, 4, weight_init="normal", bias_init="zeros")

    def construct(self, x):
        return self.fc(x)


def build_models_and_data():
    """构建两侧模型、固定输入/目标, 装入共享上下文 ctx"""
    ms_block = MsBlock()
    pt_head = torch.nn.Linear(4, 2).to(DEVICE)
    pt_optim = torch.optim.SGD(pt_head.parameters(), lr=LR)

    # 固定输入与监督目标 (float64 留作解析解, float32 供真实计算)
    x64 = np.random.randn(1, 4)
    t64 = np.array([[1.0, 0.0]])
    x_np = x64.astype(np.float32)
    target = torch.tensor(t64.astype(np.float32), device=DEVICE)
    loss_fn = torch.nn.MSELoss()
    return {
        "ms_block": ms_block, "pt_head": pt_head, "pt_optim": pt_optim,
        "x64": x64, "t64": t64, "x_np": x_np,
        "target": target, "loss_fn": loss_fn,
    }


# ---------------------------------------------------------------------------
# STEP 1 / 2: 双向基础交换
# ---------------------------------------------------------------------------
def step1_2_basic_exchange(ctx):
    print("\n--- STEP 1/2: 双向基础交换 (dtype/shape/设备) ---")
    ms_tensor = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    try:
        cap = ms_to_dlpack(ms_tensor)
        pt_from_ms = torch_from_dlpack(cap)
        ok = tuple(pt_from_ms.shape) == tuple(ms_tensor.shape)
        check(1, "MS→PT 交换", ok,
              f"shape={tuple(pt_from_ms.shape)} dtype={pt_from_ms.dtype} "
              f"device={pt_from_ms.device}")
    except Exception as e:  # noqa: BLE001
        check(1, "MS→PT 交换", False, f"异常: {type(e).__name__}: {e}")

    pt_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device=DEVICE)
    try:
        cap2 = torch_to_dlpack(pt_tensor)
        ms_from_pt = ms_from_dlpack(cap2)
        ok = tuple(ms_from_pt.shape) == (2, 2) and ms_from_pt.dtype == ms.float32
        check(2, "PT→MS 交换", ok, f"shape={ms_from_pt.shape} dtype={ms_from_pt.dtype}")
    except Exception as e:  # noqa: BLE001
        check(2, "PT→MS 交换", False, f"异常: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# STEP 3: 零拷贝语义 (torch 侧原地修改, 观察 MS 侧)
# ---------------------------------------------------------------------------
def step3_zero_copy(ctx):
    print("\n--- STEP 3: 零拷贝语义验证 ---")
    # 方向 A: MS→PT 交换后, torch 原地修改, MS 侧应同步
    try:
        src_ms = ms.Tensor(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
        view_pt = torch_from_dlpack(ms_to_dlpack(src_ms))
        view_pt.fill_(7.0)
        after = src_ms.asnumpy()
        ok = bool(np.all(after == 7.0))
        check(3, "MS→PT 零拷贝 (torch.fill_ 后 MS 侧同步)", ok,
              f"MS 侧读回={after.tolist()}")
    except Exception as e:  # noqa: BLE001
        check(3, "MS→PT 零拷贝", False, f"异常: {type(e).__name__}: {e}")

    # 方向 B: PT→MS 交换后, 修改 torch 源, MS 侧应同步
    try:
        src_pt = torch.ones(2, 2, device=DEVICE)
        view_ms = ms_from_dlpack(torch_to_dlpack(src_pt))
        src_pt.add_(3.0)
        after = view_ms.asnumpy()
        ok = bool(np.all(after == 4.0))
        check(3, "PT→MS 零拷贝 (torch.add_ 后 MS 侧同步)", ok,
              f"MS 侧读回={after.tolist()}")
    except Exception as e:  # noqa: BLE001
        check(3, "PT→MS 零拷贝", False, f"异常: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# STEP 4: PyTorch 局部计算图 (DLPack 叶子重建)
# ---------------------------------------------------------------------------
def step4_local_autograd(ctx):
    print("\n--- STEP 4: PyTorch 局部计算图 (DLPack 叶子重建) ---")
    ms_out = ctx["ms_block"](ms.Tensor(ctx["x_np"]))   # MS 前向 (PYNATIVE, 无梯度上下文)
    capsule = ms_to_dlpack(ms_out)                     # ② DLPack 传递 (导出无需 detach, 见 README)
    pt_input = torch_from_dlpack(capsule)              # 接到 PT
    pt_input.requires_grad_(True)                      # ③ 重立叶子

    pt_out = ctx["pt_head"](pt_input)
    loss = ctx["loss_fn"](pt_out, ctx["target"])
    loss.backward()                                    # ④ PT 反向

    g = to_np(pt_input.grad)
    check(4, "pt_input.grad 非空", g is not None and np.isfinite(g).all(),
          f"grad shape={g.shape if g is not None else None}")
    ok4 = True
    for _, p in ctx["pt_head"].named_parameters():
        ok4 &= p.grad is not None and torch.isfinite(p.grad).all()
    check(4, "pt_head 参数梯度非空", ok4)

    ctx.update({"ms_out": ms_out, "pt_input": pt_input, "loss": loss, "g": g})


# ---------------------------------------------------------------------------
# STEP 5: 梯度桥接 (pt_input.grad → MS sens)
# ---------------------------------------------------------------------------
def step5_gradient_bridge(ctx):
    print("\n--- STEP 5: 梯度桥接 (pt_input.grad → MS sens) ---")
    # ⑤ 把边界梯度通过 DLPack 送回 MindSpore 作为 sens
    sens = ms_from_dlpack(torch_to_dlpack(ctx["pt_input"].grad.detach()))
    grad_op = GradOperation(get_all=False, get_by_list=True, sens_param=True)
    ms_grad_fn = grad_op(ctx["ms_block"])
    param_grads = ms_grad_fn(ms.Tensor(ctx["x_np"]), sens)  # 返回 (w_grad, b_grad)

    ms_params = ctx["ms_block"].trainable_params()
    ok5 = len(param_grads) == len(ms_params) and all(
        gg is not None and np.isfinite(gg.asnumpy()).all() for gg in param_grads)
    check(5, "MS 参数梯度非空 (sens_param=True)", ok5,
          " | ".join(f"{p.name}:{tuple(gg.shape)}"
                     for p, gg in zip(ms_params, param_grads)))

    ctx.update({"sens": sens, "ms_grad_fn": ms_grad_fn,
                "param_grads": param_grads, "ms_params": ms_params})


# ---------------------------------------------------------------------------
# STEP 6: 数值正确性 (float64 解析解对照)
# ---------------------------------------------------------------------------
def step6_numeric_check(ctx):
    print("\n--- STEP 6: 数值正确性 (float64 解析解对照) ---")
    # 复合函数: loss = MSE( pt_head( ms_block(x) ), t ), 线性链, 可解析求导
    x64, t64 = ctx["x64"], ctx["t64"]
    W1 = ctx["ms_block"].fc.weight.asnumpy().astype(np.float64)
    b1 = ctx["ms_block"].fc.bias.asnumpy().astype(np.float64)
    W2 = ctx["pt_head"].weight.detach().cpu().numpy().astype(np.float64)
    b2 = ctx["pt_head"].bias.detach().cpu().numpy().astype(np.float64)

    y = x64 @ W1.T + b1          # ms_block 输出 (1,4)
    z = y @ W2.T + b2            # pt_head 输出 (1,2)
    loss_ref = float(np.mean((z - t64) ** 2))
    loss_torch = float(ctx["loss"].item())

    dz = 2.0 * (z - t64) / z.size      # (1,2)
    dy = dz @ W2                        # (1,4) == d(loss)/d(ms_out), 期望的 pt_input.grad
    dW1_ref = dy.T @ x64                # (4,4)
    db1_ref = dy.reshape(-1)            # (4,)

    g = ctx["g"]
    MEAS["loss_torch"] = loss_torch
    MEAS["loss_ref"] = loss_ref
    MEAS["rel_grad_pt"] = rel_err(g, dy)
    MEAS["rel_grad_w"] = rel_err(pg_w, dW1_ref) if "pg_w" in dir() else None
    check(6, "两框架前向 loss 一致", rel_err(loss_torch, loss_ref) < RTOL,
          f"torch={loss_torch:.8f} vs 解析={loss_ref:.8f}")
    check(6, "pt_input.grad == 解析解 d(loss)/d(ms_out)", rel_err(g, dy) < RTOL,
          f"rel_err={rel_err(g, dy):.3e}")
    pg_w = ctx["param_grads"][0].asnumpy().astype(np.float64)
    pg_b = ctx["param_grads"][1].asnumpy().astype(np.float64)
    MEAS["rel_grad_w"] = rel_err(pg_w, dW1_ref)
    MEAS["rel_grad_b"] = rel_err(pg_b, db1_ref)
    check(6, "MS weight 梯度 == 解析解 d(loss)/dW1", rel_err(pg_w, dW1_ref) < RTOL,
          f"rel_err={rel_err(pg_w, dW1_ref):.3e}")
    check(6, "MS bias 梯度 == 解析解 d(loss)/db1", rel_err(pg_b, db1_ref) < RTOL,
          f"rel_err={rel_err(pg_b, db1_ref):.3e}")

    ctx.update({"pg_w": pg_w, "pg_b": pg_b})


# ---------------------------------------------------------------------------
# STEP 7: 负对照 (ones_like 假 sens 应产生不同梯度)
# ---------------------------------------------------------------------------
def step7_negative_control(ctx):
    print("\n--- STEP 7: 负对照 (ones_like 假 sens 应产生不同梯度) ---")
    sens_fake = ops.ones_like(ctx["ms_out"])
    pg_fake = ctx["ms_grad_fn"](ms.Tensor(ctx["x_np"]), sens_fake)
    diff_w = float(np.max(np.abs(pg_fake[0].asnumpy() - ctx["pg_w"])))
    MEAS["negctrl_diff_w"] = diff_w
    check(7, "假 sens 梯度 ≠ 真实桥接梯度", diff_w > 1e-6, f"max|ΔW|={diff_w:.3e}")


# ---------------------------------------------------------------------------
# STEP 8: 两侧参数更新
# ---------------------------------------------------------------------------
def step8_update_params(ctx):
    print("\n--- STEP 8: 两侧参数更新 ---")
    pt_head, pt_optim = ctx["pt_head"], ctx["pt_optim"]
    ms_block, ms_params, param_grads = ctx["ms_block"], ctx["ms_params"], ctx["param_grads"]

    pt_w_before = to_np(pt_head.weight).copy()
    ms_w_before = ms_block.fc.weight.asnumpy().copy()
    pt_optim.step()
    pt_optim.zero_grad()
    for p, gg in zip(ms_params, param_grads):
        p.set_data(p - LR * gg)      # 最朴素 SGD: w = w - lr * grad (算术返回 Tensor, set_data 写回)
    pt_moved = float(np.max(np.abs(to_np(pt_head.weight) - pt_w_before)))
    ms_moved = float(np.max(np.abs(ms_block.fc.weight.asnumpy() - ms_w_before)))
    MEAS["pt_moved"] = pt_moved
    MEAS["ms_moved"] = ms_moved
    check(8, "PyTorch 参数已更新", pt_moved > 0, f"max|Δw|={pt_moved:.3e}")
    check(8, "MindSpore 参数已更新", ms_moved > 0, f"max|Δw|={ms_moved:.3e}")


# ---------------------------------------------------------------------------
# STEP 9: 交换耗时 (dlpack vs 拷贝路径)
# ---------------------------------------------------------------------------
def step9_benchmark(ctx):
    print("\n--- STEP 9: 交换耗时统计 (100 次均值) ---")
    src = ms.Tensor(ctx["x_np"])
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        _v = torch_from_dlpack(ms_to_dlpack(src))
    t_dlpack = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        _v = torch.from_numpy(src.asnumpy().copy())
    t_copy = (time.perf_counter() - t0) / N * 1e6
    MEAS["t_dlpack_us"] = t_dlpack
    MEAS["t_copy_us"] = t_copy
    print(f"  dlpack 交换: {t_dlpack:8.1f} us/次   |  "
          f"拷贝路径(asnumpy→torch): {t_copy:8.1f} us/次")


# ---------------------------------------------------------------------------
# STEP 10: 图语义聚合对照 (GNN C1: 边界聚合两侧一致)
# ---------------------------------------------------------------------------
def step10_graph_semantics(ctx):
    """GNN 正确性不能只比边界张量数值——聚合语义是否被破坏才是关键。
    对同一交换边界, 两侧各执行一次 scatter 聚合(MS: unsorted_segment_sum /
    PT: index_add_), 聚合结果必须一致 (≤1e-5)。"""
    print("\n--- STEP 10: 图语义聚合对照 (边界 scatter 聚合, 两侧一致) ---")
    h = np.random.randn(6, 4).astype(np.float32)
    seg = np.array([0, 1, 0, 2, 1, 2], dtype=np.int32)   # 3 个聚合目标节点
    try:
        h_ms = ms.Tensor(h)                              # MS 侧边界张量
        h_pt = torch_from_dlpack(ms_to_dlpack(h_ms))     # DLPack 交换 (同一内存)
        # MS 侧聚合 (GNN scatter 的 MS 等价算子)
        out_ms = ops.unsorted_segment_sum(h_ms, ms.Tensor(seg), 3)
        # PT 侧聚合 (GNN scatter 的 PT 等价算子), 张量上 device 后执行
        seg_pt = torch.tensor(seg, device=DEVICE)
        out_pt = torch.zeros(3, 4, device=DEVICE).index_add_(
            0, seg_pt, h_pt.to(DEVICE))
        err = rel_err(to_np(out_pt), out_ms.asnumpy())
        MEAS["rel_graph_agg"] = err
        check(10, "交换边界 scatter 聚合两侧一致 (图语义 C1)", err < 1e-5,
              f"rel_err={err:.3e}")
    except Exception as e:  # noqa: BLE001
        check(10, "交换边界 scatter 聚合两侧一致 (图语义 C1)", False,
              f"异常: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# STEP 11: 稀疏 Embedding 梯度完整性 (GNN C3)
# ---------------------------------------------------------------------------
def step11_sparse_embedding(ctx):
    """GNN 的 Embedding 梯度是稀疏的 (仅被采样节点有梯度)。验证跨框架交换
    路径上: 未访问行的梯度恒为零, 稀疏更新后未访问行数值零变化——否则说明
    交换引入了错误的内存共享或梯度扩散。"""
    print("\n--- STEP 11: 稀疏 Embedding 梯度完整性 (未访问行必须零变化) ---")
    V, D = 8, 4
    emb = torch.nn.Embedding(V, D).to(DEVICE)
    ids = torch.tensor([1, 5, 7], device=DEVICE)         # 仅 3 行被访问
    untouched_before = to_np(emb.weight.detach()).copy()
    mask = torch.ones(V, dtype=torch.bool)
    mask[ids] = False
    try:
        h = emb(ids)
        # 嵌入输出经 DLPack 交换到 MS 侧消费 (跨框架路径的一部分)
        h_ms = ms_from_dlpack(torch_to_dlpack(h.detach()))
        _read = float(np.sum(h_ms.asnumpy()))            # MS 侧读取 (模拟消费)
        loss = h.sum()
        loss.backward()
        g = to_np(emb.weight.grad)
        untouched_grad_max = float(np.max(np.abs(g[mask]))) if bool(mask.any()) else 0.0
        check(11, "稀疏梯度完整性: 未访问行梯度恒为零", untouched_grad_max == 0.0,
              f"max|g|_untouched={untouched_grad_max:.3e}")
        # 一次稀疏更新 (手写 SGD, 等价 optim.step 对未访问行的效果)
        with torch.no_grad():
            emb.weight.copy_(emb.weight - LR * emb.weight.grad)
        diff = float(np.max(np.abs(to_np(emb.weight.detach())[mask]
                                   - untouched_before[mask])))
        MEAS["emb_untouched_diff"] = diff
        check(11, "稀疏更新后未访问行零变化", diff == 0.0, f"max|Δ|={diff:.3e}")
    except Exception as e:  # noqa: BLE001
        check(11, "稀疏 Embedding 梯度完整性", False, f"异常: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def report():
    print("\n" + "=" * 78)
    print("验证报告")
    print("=" * 78)
    n_pass = sum(1 for r in results if r[2])
    for step, name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] STEP {step:<2} {name}"
              + (f"   ({detail})" if detail else ""))
    print("-" * 78)
    print(f"合计: {n_pass}/{len(results)} PASS")
    if n_pass == len(results):
        print("结论: 全部通过 — DLPack 交换/零拷贝/梯度桥接机制在该环境成立。")
    else:
        print("结论: 存在失败项, 见上方 FAIL 明细 (失败原因与移植建议见 README)。")
    dev = "npu" if DEVICE.startswith("npu") else "cpu"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", dev)
    os.makedirs(out_dir, exist_ok=True)
    summary = {"kind": "verify_dlpack", "device": DEVICE, "env": MEAS.get("env", {}),
               "n_pass": n_pass, "n_total": len(results),
               "checks": [{"step": s, "name": n, "ok": bool(ok), "detail": str(d)}
                          for s, n, ok, d in results],
               "meas": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                            float(v) if isinstance(v, (np.floating, np.integer))
                            else v)
                        for k, v in MEAS.items() if k != "env"}}
    with open(os.path.join(out_dir, "verify_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"机器可读摘要已写入: {os.path.abspath(os.path.join(out_dir, 'verify_summary.json'))}")
    return 0 if n_pass == len(results) else 1


def main():
    args = parse_args()
    setup_frameworks(args)
    ctx = build_models_and_data()

    step1_2_basic_exchange(ctx)
    step3_zero_copy(ctx)
    step4_local_autograd(ctx)
    step5_gradient_bridge(ctx)
    step6_numeric_check(ctx)
    step7_negative_control(ctx)
    step8_update_params(ctx)
    step9_benchmark(ctx)
    step10_graph_semantics(ctx)
    step11_sparse_embedding(ctx)

    return report()


if __name__ == "__main__":
    sys.exit(main())
