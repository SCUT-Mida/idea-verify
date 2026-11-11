# -*- coding: utf-8 -*-
"""
train_hybrid.py — 混合编程过渡态验证: 等价参照 + 交换机制主对照 + 形态对称性
==============================================================================

背景:
    客户模型当前为 MindSpore, 目标迁移至 PyTorch; 因人力有限且模型持续演进,
    选择 DLPack 混合编程作为迁移过渡态——未迁移模块留在 MS, 已迁移模块用 PT,
    边界两侧通过零拷贝交换 + 梯度桥接联合训练。本脚本验证该过渡态:

      MS 参照臂 (B0)  : 纯 MindSpore 端到端训练 (唯一基线) —— 切换等价性参照,
                       回答"切到混合态后训练效果与现在是否一致"(统计等价, 非逐位:
                       跨框架算子 ulp 差异经训练动力学混沌放大属正常)。
      E2 混合 (dlpack): 前段 MindSpore(Dense+ReLU) ──DLPack──▶ 后段 PyTorch(Linear),
                       每步完整梯度桥接 (①剪 ②传 ③立 ④反向 ⑤递)
      E3 混合 (copy) : 同 E2, 交换改为 host 拷贝 —— 唯一变量 = 交换机制 (主对照)
      E4 对称形态    : PT 前段 ──DLPack──▶ MS 后段 (--topology front_pt) ——
                       迁移过半时的真实形态, 证明切分方向不影响机制

    每次运行内置底线 (单次自检):
      收敛性          : loss 下降、无 NaN/Inf
      泛化(过拟合检查): held-out 验证损失随训练同步下降 (训练前/每 --eval-every 步/
                        训练后各评一次, 曲线落盘 CSV), 终点与训练损失同量级无分歧
      切换等价性      : 混合臂 vs MS 参照臂——最终 loss 比 ∈ (0.5, 2.0) 且双臂收敛,
                        held-out 验证 MSE 比 ≤ 1.01
      M4 边界数值正确 : 训练前同权重标定一次——桥接得到的边界梯度 vs torch 参考模型
                        同边界梯度 (不训练, 仅单次前向反向), 相对误差 < rtol
      初始一致性      : 混合前向初始 loss 与 torch 参考前向 loss 一致 (偏差 ≈ 0)

任务设计 (GNN 图代理, 单层消息传递网络):
    固定小图 (N=batch 节点, 每节点 k=4 邻居, seed 派生可复现) 上的节点级回归。
    聚合 (均值消息传递, 真实包含 scatter/gather 前向与反向):
        agg = segment_sum(gather(X, src), dst) / k        # GNN scatter 的标准形态
    教师 (固定随机权重, 与学生同构): y = ReLU(agg@W1.T+b1) @ W2.T + b2   y∈R^2
    学生混合切分点在聚合之后 (边界 = 节点表示, 真实 GNN 的典型切分位置):
         front_ms: MS(聚合+fc1+ReLU) ──▶ torch(fc2)   front_pt: torch(聚合+fc1+ReLU) ──▶ MS(fc2)
    聚合不改变特征维度 => 权重形状与初始化/拷贝/优化器代码完全复用; 训练链路真实包含
    图算子前向与梯度散布 (P1 回退检测采样所关心的正是这类算子)。
    torch 参考模型/MS 参照臂/混合臂由同一初始化拷贝权重值, 数据/更新/超参完全一致。
    held-out 验证集 (同一图上的独立节点特征/标签) 训练前/每 --eval-every 步/训练后
    各评一次 (验证曲线落盘逐步 CSV, 监测过拟合; 评估在计时段外, 不污染步时延)。

用法 (在项目根目录, 已激活 venv 与 CANN 环境):
    python scripts/train_hybrid.py --device npu --rtol 5e-3                # E2 dlpack
    python scripts/train_hybrid.py --device npu --exchange copy --rtol 5e-3  # E3 copy 对照
    python scripts/train_hybrid.py --device npu --topology front_pt --rtol 5e-3  # E4 对称形态
    python scripts/train_hybrid.py --device npu --bench                    # E5 纯交换吞吐 (尺寸扫描)
"""
import argparse
import sys
import time
import warnings

# 静音两类已知无害的平台告警 (成因见 README §7), 按消息精确匹配, 其他警告仍可见
warnings.filterwarnings("ignore", message=".*owner does not match.*")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")

import numpy as np
import torch
from mindspore import ops, vjp

CFG = {}


def _agg_ms(x, seg, src, n, k_f):
    """均值消息传递聚合 (MS): segment_sum(gather(x, src), dst) / k"""
    return ops.unsorted_segment_sum(ops.gather(x, src, 0), seg, n) / k_f


def _agg_pt(x, dst_pt, src_pt, k_f):
    """均值消息传递聚合 (PT): zeros.index_add_(0, dst, x[src]) / k"""
    return torch.zeros(x.shape[0], x.shape[1], device=x.device).index_add_(
        0, dst_pt, x[src_pt]) / k_f


def _back_fn(h, w2, b2, y):
    """对称形态的 MS 后段: fc2 + MSE, vjp 一次得 (边界梯度, fc2 梯度)。"""
    y_pred = ops.dense(h, w2, b2)
    return ops.mean((y_pred - y) ** 2)


# ---------------------------------------------------------------------------
# 环境初始化 (与 verify_dlpack.py 同一套约定)
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="MindSpore ↔ PyTorch 混合训练收敛 + 交换机制主对照 (E2 dlpack / E3 copy)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "npu"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-n", type=int, default=512, help="held-out 评估集样本数")
    ap.add_argument("--hidden", type=int, default=64, help="隐藏层宽度 (教师/学生同构)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--exchange", default="dlpack", choices=["dlpack", "copy"],
                    help="交换机制: dlpack=零拷贝(E2) / copy=host 拷贝对照(E3)")
    ap.add_argument("--topology", default="front_ms", choices=["front_ms", "front_pt"],
                    help="混合形态: front_ms=MS前段+PT后段(E2/E3) / front_pt=PT前段+MS后段(E4)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--result-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="每 N 步做一次 held-out 验证评估 (计时段外; 首末步必评)")
    ap.add_argument("--rtol", type=float, default=1e-3,
                    help="M4 边界梯度相对误差容差 (CPU 1e-3, NPU 建议 5e-3)")
    ap.add_argument("--bench", action="store_true",
                    help="纯交换吞吐 microbenchmark (同进程对比 dlpack/copy, 尺寸扫描)")
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--bench-sizes", default="128,4096,16384",
                    help="bench 尺寸扫描的 batch 列表 (逗号分隔; 边界张量 = batch×hidden×4B)")
    return ap.parse_args()


def resolve_device(requested: str):
    if requested == "cpu":
        return "cpu", "CPU"
    npu_ok = False
    try:
        import torch_npu  # noqa: F401
        npu_ok = True
    except ImportError:
        pass
    if requested == "npu":
        if not npu_ok:
            print("[FATAL] --device npu 但 torch_npu 不可用, 请检查 910B 环境")
            sys.exit(2)
        return "npu:0", "Ascend"
    if npu_ok:
        try:
            if torch.npu.is_available():
                return "npu:0", "Ascend"
        except Exception:  # noqa: BLE001
            pass
    return "cpu", "CPU"


def setup(args):
    device, target = resolve_device(args.device)
    is_npu = device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa: F401  -- npu 环境必须导入, 使 DLPack 补丁生效
    import mindspore as ms
    ms.set_context(device_target=target, mode=ms.PYNATIVE_MODE)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    CFG.update({"device": device, "target": target, "is_npu": is_npu, "ms": ms,
                "exchange": args.exchange, "topology": args.topology})
    print("=" * 78)
    print(f"混合训练主对照  device={device}  device_target={target}")
    print(f"numpy={np.__version__}  torch={torch.__version__}  mindspore={ms.__version__}  "
          f"hidden={args.hidden}  batch={args.batch}  exchange={args.exchange}")
    print("=" * 78)


def to_np(t):
    """torch/MS tensor -> numpy (双栈兼容)"""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t.asnumpy() if hasattr(t, "asnumpy") else np.asarray(t)


def result_dir():
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    out = _os.path.join(_here, "..", "results", "npu" if CFG["is_npu"] else "cpu")
    _os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# 数据 / 模型构建
# ---------------------------------------------------------------------------
def build_everything(args):
    from mindspore import Parameter, Tensor as MSTensor

    h = args.hidden

    # ---- 教师 (固定随机权重, 与学生同构的单层消息传递图网络) ----
    # W2 按 1/sqrt(h) 缩放 (Xavier 思想), 保证任意 hidden 下教师输出保持 O(1),
    # 避免 lr=0.05 在宽隐层下发散 (loss 初始 ~h*0.26 会随宽度爆炸)
    rng_t = np.random.default_rng(args.seed + 1000)
    W1t = (rng_t.standard_normal((h, 4)) * 0.6).astype(np.float32)
    b1t = (rng_t.standard_normal(h) * 0.1).astype(np.float32)
    W2t = (rng_t.standard_normal((2, h)) * (0.6 / np.sqrt(h))).astype(np.float32)
    b2t = (rng_t.standard_normal(2) * 0.1).astype(np.float32)

    # ---- 固定小图结构 (消息传递拓扑, seed 派生可复现; 训练图/验证图各一份) ----
    k = 4                                                        # 每节点入边邻居数
    def make_edges(n):
        dst = np.repeat(np.arange(n, dtype=np.int32), k)         # (n*k,) 目标节点
        src = rng_t.integers(0, n, size=n * k).astype(np.int32)  # (n*k,) 源节点
        return dst, src
    dst_tr, src_tr = make_edges(args.batch)
    dst_ev, src_ev = make_edges(args.eval_n)
    k_f = float(k)

    def teacher(x, dst, src):
        agg = np.zeros_like(x)
        np.add.at(agg, dst, x[src])                    # segment_sum (均值消息传递)
        agg /= k
        y = np.maximum(agg @ W1t.T + b1t, 0.0)         # ReLU
        return (y @ W2t.T + b2t).astype(np.float32)

    x = rng_t.standard_normal((args.batch, 4)).astype(np.float32)
    y = teacher(x, dst_tr, src_tr)
    x_eval = rng_t.standard_normal((args.eval_n, 4)).astype(np.float32)
    y_eval = teacher(x_eval, dst_ev, src_ev)
    y_t = torch.tensor(y, device=CFG["device"])
    y_eval_t = torch.tensor(y_eval, device=CFG["device"])
    loss_fn = torch.nn.MSELoss()

    # ---- torch 参考模型 (聚合+fc1+ReLU+fc2): 仅用于 M4 标定与初始一致性校验, 不训练 ----
    # torch.manual_seed 已定 => 初始化可复现; 混合学生的权重值从这里拷贝,
    # 因此 E2/E3 两跑 (及与旧版本) 的初始化完全一致。
    # 前向含与 MS 侧一致的均值聚合 (函数式, 见 calibrate_m4)
    fc1_ref = torch.nn.Linear(4, h).to(CFG["device"])
    fc2_ref = torch.nn.Linear(h, 2).to(CFG["device"])

    # ---- 图拓扑常量 (MS/PT 双侧; 训练图/验证图分开) ----
    seg_ms = MSTensor(dst_tr)
    src_ms = MSTensor(src_tr)
    seg_e_ms = MSTensor(dst_ev)
    src_e_ms = MSTensor(src_ev)
    dst_pt = torch.tensor(dst_tr, device=CFG["device"])
    src_pt = torch.tensor(src_tr, device=CFG["device"])
    dst_e_pt = torch.tensor(dst_ev, device=CFG["device"])
    src_e_pt = torch.tensor(src_ev, device=CFG["device"])

    # ---- 混合学生 front_ms 形态: MS 前段 (聚合+Dense+ReLU, vjp 闭包绑定训练图)
    #      + PT 后段; 权重值从参考模型拷贝。接真实模型时替换闭包内的聚合/前向即可
    w_ref = Parameter(MSTensor(to_np(fc1_ref.weight)), name="front_w")
    b_ref = Parameter(MSTensor(to_np(fc1_ref.bias)), name="front_b")
    pt_head = torch.nn.Linear(h, 2).to(CFG["device"])
    with torch.no_grad():
        pt_head.weight.copy_(fc2_ref.weight)
        pt_head.bias.copy_(fc2_ref.bias)
    opt_b = torch.optim.SGD(pt_head.parameters(), lr=args.lr)
    front = (lambda x, w, b: ops.relu(ops.dense(
        _agg_ms(x, seg_ms, src_ms, args.batch, k_f), w, b)))
    front_eval = (lambda x, w, b: ops.relu(ops.dense(
        _agg_ms(x, seg_e_ms, src_e_ms, args.eval_n, k_f), w, b)))

    # ---- MS 参照臂参数 (纯 MS 端到端, 基线 B0; 权重值同源拷贝) ----
    w1m = Parameter(MSTensor(to_np(fc1_ref.weight)), name="ref_w1")
    b1m = Parameter(MSTensor(to_np(fc1_ref.bias)), name="ref_b1")
    w2m = Parameter(MSTensor(to_np(fc2_ref.weight)), name="ref_w2")
    b2m = Parameter(MSTensor(to_np(fc2_ref.bias)), name="ref_b2")
    ref_fn = (lambda x, w1, b1, w2, b2, y: ops.mean(
        (ops.dense(ops.relu(ops.dense(
            _agg_ms(x, seg_ms, src_ms, args.batch, k_f), w1, b1)), w2, b2) - y) ** 2))

    # ---- 对称形态 front_pt: PT 前段 (聚合+fc1+ReLU, 值同源拷贝) + MS 后段 (独立参数,
    #      与参照臂解耦——各臂独立训练互不污染) ----
    w2s = Parameter(MSTensor(to_np(fc2_ref.weight)), name="sym_w2")
    b2s = Parameter(MSTensor(to_np(fc2_ref.bias)), name="sym_b2")
    pt_fc1 = torch.nn.Linear(4, h).to(CFG["device"])
    with torch.no_grad():
        pt_fc1.weight.copy_(fc1_ref.weight)
        pt_fc1.bias.copy_(fc1_ref.bias)
    opt_sym = torch.optim.SGD(pt_fc1.parameters(), lr=args.lr)

    return {
        "teacher": teacher, "x": x, "y_t": y_t, "loss_fn": loss_fn,
        "x_eval": x_eval, "y_eval_t": y_eval_t, "y_np": y, "y_eval_np": y_eval,
        "fc1_ref": fc1_ref, "fc2_ref": fc2_ref,
        "seg_ms": seg_ms, "src_ms": src_ms,
        "seg_e_ms": seg_e_ms, "src_e_ms": src_e_ms,
        "front": front, "front_eval": front_eval, "ref_fn": ref_fn,
        "w_ref": w_ref, "b_ref": b_ref,
        "pt_head": pt_head, "opt_b": opt_b,
        "w1m": w1m, "b1m": b1m, "w2m": w2m, "b2m": b2m,
        "w2s": w2s, "b2s": b2s,
        "pt_fc1": pt_fc1, "opt_sym": opt_sym,
        "dst_pt": dst_pt, "src_pt": src_pt,
        "dst_e_pt": dst_e_pt, "src_e_pt": src_e_pt,
        "k_f": k_f,
    }


# ---------------------------------------------------------------------------
# 跨框架交换 (仅交换段计时; copy 模式含 NPU<->host 往返)
# ---------------------------------------------------------------------------
def exchange_forward(ms_out):
    t0 = time.perf_counter()
    if CFG["exchange"] == "dlpack":
        from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
        from torch.utils.dlpack import from_dlpack as torch_from_dlpack
        pt_input = torch_from_dlpack(ms_to_dlpack(ms_out))
    else:
        pt_input = torch.from_numpy(np.array(ms_out.asnumpy()))
        if CFG["is_npu"]:
            pt_input = pt_input.to(CFG["device"])
    return pt_input, time.perf_counter() - t0


def exchange_backward(pt_input):
    from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack
    t0 = time.perf_counter()
    grad = pt_input.grad.detach()
    if CFG["exchange"] == "dlpack":
        sens = ms_from_dlpack(torch_to_dlpack(grad))
    else:
        sens = CFG["ms"].Tensor(to_np(grad))
    return sens, time.perf_counter() - t0


def exchange_activation(pt_act):
    """PT→MS 激活值交换 (对称形态正向; 仅交换段计时)。"""
    t0 = time.perf_counter()
    if CFG["exchange"] == "dlpack":
        from torch.utils.dlpack import to_dlpack as torch_to_dlpack
        from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
        h_ms = ms_from_dlpack(torch_to_dlpack(pt_act.detach()))
    else:
        h_ms = CFG["ms"].Tensor(to_np(pt_act))
    return h_ms, time.perf_counter() - t0


def exchange_grad_ms2pt(g_ms):
    """MS→PT 梯度交换 (对称形态回传; 仅交换段计时)。"""
    from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
    t0 = time.perf_counter()
    if CFG["exchange"] == "dlpack":
        from torch.utils.dlpack import from_dlpack as torch_from_dlpack
        g_pt = torch_from_dlpack(ms_to_dlpack(g_ms))
    else:
        g_pt = torch.tensor(to_np(g_ms), device=CFG["device"])
    return g_pt, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# M4: 边界梯度标定 (训练开始前, 两侧同权重, 桥接梯度 vs torch 参考梯度;
#     按拓扑取对应边界的参数梯度: front_ms→fc1 / front_pt→fc2)
# ---------------------------------------------------------------------------
def calibrate_m4(ctx, args):
    ms, x, y_t, loss_fn = CFG["ms"], ctx["x"], ctx["y_t"], ctx["loss_fn"]
    # torch 参考: 全模型单次前向反向 (含均值聚合; 该模型不参与训练), 取 fc1/fc2 梯度
    x_t = torch.tensor(x, device=CFG["device"])
    agg_ref = _agg_pt(x_t, ctx["dst_pt"], ctx["src_pt"], ctx["k_f"])
    ref_out = ctx["fc2_ref"](torch.relu(ctx["fc1_ref"](agg_ref)))
    ref_loss = float(loss_fn(ref_out, y_t).item())
    loss_fn(ref_out, y_t).backward()

    if CFG["topology"] == "front_ms":
        # form-1 桥接: MS 前向 -> 交换 -> PT fc2 -> backward -> sens -> vjp 反向
        # 官方用法: inputs 注册进 vjp(fn, *inputs), 返回的 bwd_fn 只接收种子向量 v
        ms_out, bwd_fn = vjp(ctx["front"], ms.Tensor(x), ctx["w_ref"], ctx["b_ref"])
        pt_input, _ = exchange_forward(ms_out)
        pt_input.requires_grad_(True)
        pt_out = ctx["pt_head"](pt_input)
        mix_loss = float(loss_fn(pt_out, y_t).item())
        loss_fn(pt_out, y_t).backward()
        sens, _ = exchange_backward(pt_input)
        _, grad_w, grad_b = bwd_fn(sens)
        ref_w = to_np(ctx["fc1_ref"].weight.grad)
        ref_b = to_np(ctx["fc1_ref"].bias.grad)
        boundary = "fc1"
    else:
        # form-2 桥接: PT 前段 -> 交换 -> MS fc2 + MSE -> vjp(标量 sens=1)
        with torch.no_grad():
            agg = _agg_pt(x_t, ctx["dst_pt"], ctx["src_pt"], ctx["k_f"])
            h_act = torch.relu(ctx["pt_fc1"](agg))
        h_ms, _ = exchange_activation(h_act)
        y_ms = ms.Tensor(ctx["y_np"])
        loss, bwd_fn = vjp(_back_fn, h_ms, ctx["w2s"], ctx["b2s"], y_ms)
        mix_loss = float(to_np(loss))
        grads = bwd_fn(ms.Tensor(1.0))
        grad_w, grad_b = to_np(grads[1]), to_np(grads[2])
        ref_w = to_np(ctx["fc2_ref"].weight.grad)
        ref_b = to_np(ctx["fc2_ref"].bias.grad)
        boundary = "fc2"

    rel = lambda a, b: float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))
    err = max(rel(to_np(grad_w), ref_w), rel(to_np(grad_b), ref_b))
    # B0 锚 (梯度等价性 C2 口径): 桥接梯度 vs 纯 MindSpore 端到端同权重单步梯度
    y_ms = ms.Tensor(ctx["y_np"])
    _, ref_bwd = vjp(ctx["ref_fn"], ms.Tensor(x), ctx["w1m"], ctx["b1m"],
                     ctx["w2m"], ctx["b2m"], y_ms)
    ref_g = ref_bwd(ms.Tensor(1.0))          # (dx, dw1, db1, dw2, db2, dy)
    if CFG["topology"] == "front_ms":
        ms_ref_err = max(rel(to_np(grad_w), to_np(ref_g[1])),
                         rel(to_np(grad_b), to_np(ref_g[2])))
    else:
        ms_ref_err = max(rel(to_np(grad_w), to_np(ref_g[3])),
                         rel(to_np(grad_b), to_np(ref_g[4])))
    ok = err < args.rtol and ms_ref_err < args.rtol
    print(f"  [M4] 边界梯度标定 ({boundary} 边界): 桥接梯度 vs torch 参考梯度  "
          f"max_rel_err={err:.3e} (rtol={args.rtol:g})  "
          f"[{'PASS' if err < args.rtol else 'FAIL'}]")
    print(f"  [M4] 双锚-B0/MS参照: 桥接梯度 vs 纯 MS 端到端梯度  "
          f"max_rel_err={ms_ref_err:.3e}  "
          f"[{'PASS' if ms_ref_err < args.rtol else 'FAIL'}]")
    # 清掉参考模型本次标定遗留的梯度, 不影响正式训练
    ctx["fc1_ref"].zero_grad(set_to_none=True)
    ctx["fc2_ref"].zero_grad(set_to_none=True)
    return ok, err, ms_ref_err, ref_loss, mix_loss


# ---------------------------------------------------------------------------
# 混合训练 (完整梯度桥接; exchange 由 CFG 决定 dlpack/copy)
# ---------------------------------------------------------------------------
def train_mixed(ctx, args):
    ms = CFG["ms"]
    front, w_ref, b_ref = ctx["front"], ctx["w_ref"], ctx["b_ref"]
    pt_head, opt_b = ctx["pt_head"], ctx["opt_b"]
    x, y_t, loss_fn = ctx["x"], ctx["y_t"], ctx["loss_fn"]
    x_ms = ms.Tensor(x)                                  # 常量输入张量只建一次

    hist, times = [], []
    val_curve = {}                                       # step -> held-out MSE (计时段外)
    v0 = eval_mse(ctx)                                   # 训练前初始验证
    val_curve[0] = v0
    print(f"  [验证] step     0  held-out MSE = {v0:.6e}")
    t_fwd = t_bwd = 0.0
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        # ---- MS 前向 (vjp: 前向仅执行一次, 返回反向闭包) ----
        ms_out, bwd_fn = vjp(front, x_ms, w_ref, b_ref)
        pt_input, dt_f = exchange_forward(ms_out)        # ② 交换 (MS→PT)
        t_fwd += dt_f
        pt_input.requires_grad_(True)                    # ③ 重立叶子
        loss = loss_fn(pt_head(pt_input), y_t)
        loss.backward()                                  # ④ PT 反向
        sens, dt_b = exchange_backward(pt_input)         # ⑤ 梯度回传 (PT→MS)
        t_bwd += dt_b
        _, dw, db = bwd_fn(sens)                         # 反向 (无重复前向)

        opt_b.step()
        opt_b.zero_grad()
        w_ref.set_data(w_ref - args.lr * dw)
        b_ref.set_data(b_ref - args.lr * db)

        times.append(time.perf_counter() - t0)
        v = float(loss.item())
        hist.append(v)
        if step % args.eval_every == 0 or step == args.steps:
            val_curve[step] = eval_mse(ctx)              # 计时段外, 不污染步时延
            print(f"  [验证] step {step:5d}  held-out MSE = {val_curve[step]:.6e}")
        if step == 1 or step % args.result_every == 0 or step == args.steps:
            print(f"  [混合] step {step:5d}  loss = {v:.6e}")
    return hist, times, t_fwd / args.steps * 1e6, t_bwd / args.steps * 1e6, val_curve


def eval_mse(ctx):
    """held-out 评估 (混合前向, front_ms 形态)。返回 MSE(float)。"""
    with torch.no_grad():
        ms_out = ctx["front_eval"](CFG["ms"].Tensor(ctx["x_eval"]),
                                   ctx["w_ref"], ctx["b_ref"])
        pt_input, _ = exchange_forward(ms_out)
        out = ctx["pt_head"](pt_input)
        return float(ctx["loss_fn"](out, ctx["y_eval_t"]).item())


def eval_ref_ms(ctx):
    """held-out 评估 (纯 MS 参照臂, 验证图)。"""
    ms = CFG["ms"]
    agg = _agg_ms(ms.Tensor(ctx["x_eval"]), ctx["seg_e_ms"], ctx["src_e_ms"],
                  len(ctx["x_eval"]), ctx["k_f"])
    out = ops.dense(ops.relu(ops.dense(agg, ctx["w1m"], ctx["b1m"])),
                    ctx["w2m"], ctx["b2m"])
    return float(to_np(ops.mean((out - ms.Tensor(ctx["y_eval_np"])) ** 2)))


def eval_sym(ctx):
    """held-out 评估 (对称形态: PT 前段聚合+fc1 ──▶ MS fc2, 验证图)。"""
    ms = CFG["ms"]
    with torch.no_grad():
        agg = _agg_pt(torch.tensor(ctx["x_eval"], device=CFG["device"]),
                      ctx["dst_e_pt"], ctx["src_e_pt"], ctx["k_f"])
        h_act = torch.relu(ctx["pt_fc1"](agg))
    h_ms, _ = exchange_activation(h_act)
    out = ops.dense(h_ms, ctx["w2s"], ctx["b2s"])
    return float(to_np(ops.mean((out - ms.Tensor(ctx["y_eval_np"])) ** 2)))


def train_ref_ms(ctx, args):
    """MS 参照臂 (B0): 纯 MindSpore 端到端训练 (唯一基线, 切换等价性参照)。"""
    ms = CFG["ms"]
    x_ms = ms.Tensor(ctx["x"])
    y_ms = ms.Tensor(ctx["y_np"])
    params = (ctx["w1m"], ctx["b1m"], ctx["w2m"], ctx["b2m"])

    hist, times = [], []
    val_curve = {}
    v0 = eval_ref_ms(ctx)
    val_curve[0] = v0
    print(f"  [验证] step     0  held-out MSE = {v0:.6e}")
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        loss, bwd_fn = vjp(ctx["ref_fn"], x_ms, *params, y_ms)
        grads = bwd_fn(ms.Tensor(1.0))          # 标量输出 -> 标量种子, 一次得全部梯度
        for p, g in zip(params, grads[1:5]):
            p.set_data(p - args.lr * g)
        times.append(time.perf_counter() - t0)
        v = float(to_np(loss))
        hist.append(v)
        if step % args.eval_every == 0 or step == args.steps:
            val_curve[step] = eval_ref_ms(ctx)
            print(f"  [验证] step {step:5d}  held-out MSE = {val_curve[step]:.6e}")
        if step == 1 or step % args.result_every == 0 or step == args.steps:
            print(f"  [MS参照] step {step:5d}  loss = {v:.6e}")
    return hist, times, val_curve


def train_sym(ctx, args):
    """对称形态混合训练: PT 前段(fc1+ReLU) ──交换──▶ MS 后段(fc2), 完整梯度桥接。"""
    ms = CFG["ms"]
    pt_fc1, opt_sym = ctx["pt_fc1"], ctx["opt_sym"]
    w2s, b2s = ctx["w2s"], ctx["b2s"]
    x_t = torch.tensor(ctx["x"], device=CFG["device"])
    y_ms = ms.Tensor(ctx["y_np"])

    hist, times = [], []
    val_curve = {}
    v0 = eval_sym(ctx)
    val_curve[0] = v0
    print(f"  [验证] step     0  held-out MSE = {v0:.6e}")
    t_fwd = t_bwd = 0.0
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        opt_sym.zero_grad()
        agg = _agg_pt(x_t, ctx["dst_pt"], ctx["src_pt"], ctx["k_f"])   # 消息传递聚合
        h_act = torch.relu(ctx["pt_fc1"](agg))                         # PT 前段 (在图内)
        h_ms, dt_f = exchange_activation(h_act)          # ① 交换 (PT→MS, 激活值)
        t_fwd += dt_f
        # ② MS 后段 fc2 + MSE: vjp(标量 sens=1) 一次得 边界梯度 + fc2 梯度
        loss, bwd_fn = vjp(_back_fn, h_ms, w2s, b2s, y_ms)
        grads = bwd_fn(ms.Tensor(1.0))
        g_ms, g_w2, g_b2 = grads[0], grads[1], grads[2]
        g_pt, dt_b = exchange_grad_ms2pt(g_ms)           # ③ 交换 (MS→PT, 边界梯度)
        t_bwd += dt_b
        h_act.backward(gradient=g_pt)                    # ④ 以外部梯度种子回传 PT 前段
        opt_sym.step()                                   # ⑤ 两侧更新
        opt_sym.zero_grad()
        w2s.set_data(w2s - args.lr * g_w2)
        b2s.set_data(b2s - args.lr * g_b2)

        times.append(time.perf_counter() - t0)
        v = float(to_np(loss))
        hist.append(v)
        if step % args.eval_every == 0 or step == args.steps:
            val_curve[step] = eval_sym(ctx)
            print(f"  [验证] step {step:5d}  held-out MSE = {val_curve[step]:.6e}")
        if step == 1 or step % args.result_every == 0 or step == args.steps:
            print(f"  [对称混合] step {step:5d}  loss = {v:.6e}")
    return hist, times, t_fwd / args.steps * 1e6, t_bwd / args.steps * 1e6, val_curve


# ---------------------------------------------------------------------------
# 报告: 单次运行底线 (收敛 + M4) + 机器可读摘要 v2
# ---------------------------------------------------------------------------
def report(args, hist, times, eval_v, fwd_us, bwd_us, m4, m4_err, m4_ms_err,
           ref_loss, mix_loss, val_curve, hist_r, times_r, val_r):
    hist = np.array(hist)
    if not np.isfinite(hist).all():
        print("[FATAL] 训练发散 (出现 NaN/Inf): 请降低 --lr 或减小 --hidden 后重试")
        return 1
    t_arr = np.array(times)[int(len(times) * 0.2):]   # 去掉前 20% 预热
    converged = bool(hist[-1] < hist[0])
    init_dev = abs(mix_loss - ref_loss) / max(abs(ref_loss), 1e-300)
    eval_init = val_curve.get(0, float("nan"))
    gen_gap = eval_v / max(hist[-1], 1e-300)
    val_synced = all(val_curve[s] < val_curve[0] for s in val_curve if s > 0)

    # ---- MS 参照臂 (基线 B0) 与切换等价性判定 ----
    ref_hist = np.array(hist_r)
    ref_converged = bool(ref_hist[-1] < ref_hist[0])
    ref_t = np.array(times_r)[int(len(times_r) * 0.2):]
    ref_eval = val_r[args.steps]
    ref_synced = all(val_r[s] < val_r[0] for s in val_r if s > 0)
    eq_final = float(hist[-1] / max(ref_hist[-1], 1e-300))
    eq_eval = float(eval_v / max(ref_eval, 1e-300))
    eq_pass = bool(0.5 < eq_final < 2.0 and eval_v <= ref_eval * 1.01
                   and converged and ref_converged)

    print("\n" + "=" * 78)
    print(f"混合过渡态报告 (topology={args.topology}, exchange={CFG['exchange']}) — 单次运行底线")
    print("-" * 78)
    print(f"  loss 曲线          : {hist[0]:.3e} -> {hist[-1]:.3e}  "
          f"[{'PASS' if converged else 'FAIL'} 收敛]")
    print(f"  held-out 验证 MSE  : {eval_init:.6e} -> {eval_v:.6e}  "
          f"[{'PASS' if val_synced and converged else 'FAIL'} 泛化(无过拟合分歧)]")
    print(f"  验证/训练终点比    : {gen_gap:.2f} (同量级即无病态过拟合)")
    print(f"  MS 参照臂 loss     : {ref_hist[0]:.3e} -> {ref_hist[-1]:.3e}  "
          f"[{'PASS' if ref_converged else 'FAIL'} 收敛]")
    print(f"  切换等价性(混合/参照): loss 比 {eq_final:.4f} (∈[0.5,2.0]) | "
          f"held-out 比 {eq_eval:.4f} (≤1.01)  [{'PASS' if eq_pass else 'FAIL'}]")
    print(f"  步时延(去预热)     : 混合 {np.mean(t_arr) * 1e3:.2f} ms | "
          f"MS参照 {np.mean(ref_t) * 1e3:.2f} ms (范式代价披露, 非 dlpack 判据)")
    print(f"  交换开销(含算子同步): 正向 {fwd_us:.1f} us/步, 回传 {bwd_us:.1f} us/步")
    exchange_ratio = float((fwd_us + bwd_us) / max(np.mean(t_arr) * 1e6, 1e-9) * 100)
    print(f"  交换信噪比(P2 口径) : 交换段占单步训练 {exchange_ratio:.2f}% (<1% 为佳)")
    print(f"  [sanity] 初始 loss 一致性: |混合-参考|/参考 = {init_dev:.2e} (应 ≈ 0)")
    print(f"  [{'PASS' if m4 else 'FAIL'}] M4 边界数值正确: 见上方标定输出")
    print("-" * 78)
    print("  说明: 数值一致性与耗时对照 (E2 vs E3) 由 scripts/make_report.py "
          "汇总两份摘要生成。")
    ok = converged and m4 and eq_pass

    import json
    import os as _os
    out_dir = result_dir()
    summary = {
        "kind": "train_hybrid", "version": 3, "device": CFG["device"],
        "hidden": args.hidden, "batch": args.batch, "lr": args.lr,
        "steps": args.steps, "seed": args.seed, "exchange": args.exchange,
        "topology": args.topology, "rtol": args.rtol,
        "results": {
            "initial_loss": float(hist[0]), "final_loss": float(hist[-1]),
            "eval_mse": float(eval_v), "eval_mse_initial": float(eval_init),
            "gen_gap": float(gen_gap), "val_synced": bool(val_synced),
            "lat_ms": float(np.mean(t_arr) * 1e3),
            "exchange_fwd_us": float(fwd_us), "exchange_bwd_us": float(bwd_us),
            "exchange_ratio_pct": exchange_ratio,
            "converged": converged, "m4": bool(m4)},
        "ms_ref": {"initial_loss": float(ref_hist[0]), "final_loss": float(ref_hist[-1]),
                    "eval_mse": float(ref_eval), "eval_mse_initial": float(val_r[0]),
                    "gen_gap": float(ref_eval / max(ref_hist[-1], 1e-300)),
                    "val_synced": bool(ref_synced),
                    "lat_ms": float(np.mean(ref_t) * 1e3),
                    "converged": ref_converged},
        "eq": {"final_ratio": eq_final, "eval_ratio": eq_eval, "pass": eq_pass},
        "m4_calibration": {"rel_err": float(m4_err), "ms_rel_err": float(m4_ms_err),
                            "rtol": args.rtol, "pass": bool(m4)},
        "initial_consistency_dev": float(init_dev),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    suffix = "_sym" if args.topology == "front_pt" else ""
    sname = f"train_summary_h{args.hidden}_{args.exchange}{suffix}.json"
    with open(_os.path.join(out_dir, sname), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"机器可读摘要已写入: {_os.path.abspath(_os.path.join(out_dir, sname))}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 纯交换吞吐 microbenchmark (剥离框架算子开销; 同进程对比 dlpack/copy)
# ---------------------------------------------------------------------------
def bench(args):
    """纯交换吞吐尺寸扫描: 每个 batch 档位测 dlpack vs copy 的正/回向交换
    (边界张量 = batch×hidden×4B; 值用 zeros, DLPack/拷贝耗时与数值无关)。"""
    from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
    from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack

    ctx = build_everything(args)
    ms = CFG["ms"]
    h = args.hidden
    sizes = [int(s) for s in str(args.bench_sizes).split(",") if s.strip()]
    N, WARM = args.bench_n, 10

    def timeit(fn):
        for _ in range(WARM):
            fn()
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        return (time.perf_counter() - t0) / N * 1e6

    def fmt_size(b):
        kb = b * h * 4 / 1024
        return f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"

    print("=" * 78)
    print(f"纯交换吞吐尺寸扫描  device={CFG['device']}  hidden={h} float32  "
          f"N={N} (预热 {WARM})")
    print("-" * 78)
    rows = []
    for b in sizes:
        # front: (B,4) 特征 -> 聚合 -> (B,h) 边界张量; 每档位建独立图, 计时段内不含前向
        z = np.zeros((b, 4), dtype=np.float32)
        dst_b = np.repeat(np.arange(b, dtype=np.int32), 4)
        src_b = np.random.default_rng(args.seed + 7).integers(0, b, size=b * 4).astype(np.int32)
        agg = _agg_ms(ms.Tensor(z), ms.Tensor(dst_b), ms.Tensor(src_b), b, 4.0)
        ms_out = ops.relu(ops.dense(agg, ctx["w_ref"], ctx["b_ref"]))
        grad = torch.zeros((b, h), device=CFG["device"])

        def fwd_dlpack():
            torch_from_dlpack(ms_to_dlpack(ms_out))

        def fwd_copy():
            t = torch.from_numpy(np.array(ms_out.asnumpy()))
            if CFG["is_npu"]:
                t = t.to(CFG["device"])

        def bwd_dlpack():
            ms_from_dlpack(torch_to_dlpack(grad))

        def bwd_copy():
            CFG["ms"].Tensor(to_np(grad))

        f_d = timeit(fwd_dlpack)
        f_c = timeit(fwd_copy)
        b_d = timeit(bwd_dlpack)
        b_c = timeit(bwd_copy)
        rows.append({"batch": b, "boundary": fmt_size(b),
                     "fwd_dlpack_us": f_d, "fwd_copy_us": f_c,
                     "bwd_dlpack_us": b_d, "bwd_copy_us": b_c})
        print(f"  边界张量 {fmt_size(b):>7} (batch={b:6d})")
        print(f"    正向 MS→PT : dlpack {f_d:9.1f} us/次 | copy {f_c:9.1f} us/次 "
              f"(copy 慢 {f_c / max(f_d, 1e-9):6.1f}x)")
        print(f"    回传 PT→MS : dlpack {b_d:9.1f} us/次 | copy {b_c:9.1f} us/次 "
              f"(copy 慢 {b_c / max(b_d, 1e-9):6.1f}x)")
    print("-" * 78)
    print("  说明: dlpack 全程留在 device 内存; copy 每次交换含 NPU<->host 往返与流同步;"
          "收益随边界张量增大线性放大(真实模型边界张量更大, 收益只会更显著)。")
    import json
    import os as _os
    out_dir = result_dir()
    with open(_os.path.join(out_dir, "bench_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"kind": "bench", "version": 2, "device": CFG["device"],
                   "hidden": h, "n": N, "sizes": rows}, f, ensure_ascii=False, indent=2)
    print(f"机器可读摘要已写入: {_os.path.abspath(_os.path.join(out_dir, 'bench_summary.json'))}")
    return 0


def main():
    args = parse_args()
    setup(args)
    if args.bench:
        return bench(args)

    ctx = build_everything(args)
    print("\n--- M4 边界梯度标定 (同权重单次对照, torch 参考模型不参与训练) ---")
    m4, m4_err, m4_ms_err, ref_loss, mix_loss = calibrate_m4(ctx, args)

    print(f"\n--- MS 参照臂 (基线 B0): 纯 MindSpore 端到端 ({args.steps} 步) ---")
    hist_r, times_r, val_r = train_ref_ms(ctx, args)

    sym = args.topology == "front_pt"
    tag = ("对称混合臂: PT前段 ──▶ MS后段" if sym
           else "混合臂: MS前段 ──▶ PT后段")
    print(f"\n--- {tag} ({args.steps} 步, exchange={args.exchange}) ---")
    if sym:
        hist, times, fwd_us, bwd_us, val_curve = train_sym(ctx, args)
    else:
        hist, times, fwd_us, bwd_us, val_curve = train_mixed(ctx, args)
    eval_v = val_curve[args.steps]                   # 末步必评, 与曲线终点一致

    # 逐步训练日志落盘 (E2/E3/E4 对照曲线图表的数据源): results/{cpu|npu}/
    # 文件名含 exchange 与形态后缀, 保证各跑日志互不覆盖
    import csv
    import os as _os
    suffix = "_sym" if sym else ""
    log_name = _os.path.join(
        result_dir(),
        f"training_log_{args.exchange}_h{args.hidden}{suffix}.csv")
    with open(log_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "val_loss", "ref_loss", "ref_val_loss", "t_ms"])
        for i in range(args.steps):
            step = i + 1
            w.writerow([step, hist[i],
                        f"{val_curve[step]:.10e}" if step in val_curve else "",
                        f"{hist_r[i]:.10e}",
                        f"{val_r[step]:.10e}" if step in val_r else "",
                        times[i] * 1e3])
        # step 0 的初始验证点不落 CSV (曲线从 step 1 起), 已进摘要 eval_mse_initial
    print(f"\n逐步训练日志已写入: {log_name}")

    return report(args, hist, times, eval_v, fwd_us, bwd_us, m4, m4_err, m4_ms_err,
                  ref_loss, mix_loss, val_curve, hist_r, times_r, val_r)


if __name__ == "__main__":
    sys.exit(main())
