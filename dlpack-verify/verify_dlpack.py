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

流程图 (5 个关键动作):
  MS: x ─▶ ms_block ─▶ ms_out
              │ ①(导出) ②DLPack
              ▼
  PT:      pt_input ③requires_grad_(True) ─▶ pt_head ─▶ loss ─④ backward
                                                        │
                          pt_input.grad ──⑤──▶ MS sens ─▶ ms_block 参数梯度 ─▶ 更新

用法:
    python verify_dlpack.py                 # 自动探测 (有 torch_npu 用 NPU, 否则 CPU)
    python verify_dlpack.py --device cpu    # 强制 CPU
    python verify_dlpack.py --device npu    # 强制 NPU (910B)
"""
import argparse
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 0. 设备探测与框架初始化
# ---------------------------------------------------------------------------
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
            import torch
            if torch.npu.is_available():
                return "npu:0", "Ascend"
        except Exception:  # noqa: BLE001
            pass
    return "cpu", "CPU"


ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "npu"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--rtol", type=float, default=1e-3,
                help="数值对照相对容差 (CPU 建议 1e-3, NPU 可放宽到 5e-3)")
args = ap.parse_args()

DEVICE, MS_TARGET = resolve_device(args.device)
IS_NPU = DEVICE.startswith("npu")

import torch
if IS_NPU:
    import torch_npu  # noqa: F401  -- npu 环境必须导入, 使 DLPack 补丁生效
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import GradOperation
from mindspore.utils.dlpack import to_dlpack as ms_to_dlpack
from mindspore.utils.dlpack import from_dlpack as ms_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack

ms.set_context(device_target=MS_TARGET, mode=ms.PYNATIVE_MODE)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("=" * 78)
print(f"DLPack 验证  device={DEVICE}  device_target={MS_TARGET}")
print(f"numpy={np.__version__}  torch={torch.__version__}  "
      f"mindspore={ms.__version__}  mode=PYNATIVE")
print("=" * 78)

results = []  # (step, name, ok, detail)


def check(step, name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] STEP {step}  {name}" + (f"  -- {detail}" if detail else ""))
    results.append((step, name, ok, detail))
    return ok


def to_np(t):
    """torch tensor -> numpy (统一走 host, 屏蔽 cpu/npu 差异)"""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 模型定义: MS Block (4->4) 与 PT Head (4->2), 与验证方案一致的小模型
# ---------------------------------------------------------------------------
class MsBlock(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(4, 4, weight_init="normal", bias_init="zeros")

    def construct(self, x):
        return self.fc(x)


def build_pt_head(device):
    head = torch.nn.Linear(4, 2)
    return head.to(device)


ms_block = MsBlock()
LR = 0.01
pt_head = build_pt_head(DEVICE)
pt_optim = torch.optim.SGD(pt_head.parameters(), lr=LR)

# 固定输入与监督目标 (float64 留作解析解, float32 供真实计算)
x64 = np.random.randn(1, 4)
t64 = np.array([[1.0, 0.0]])
x_np = x64.astype(np.float32)
target = torch.tensor(t64.astype(np.float32), device=DEVICE)
loss_fn = torch.nn.MSELoss()

# ---------------------------------------------------------------------------
# STEP 1 / 2: 双向基础交换
# ---------------------------------------------------------------------------
print("\n--- STEP 1/2: 双向基础交换 (dtype/shape/设备) ---")
ms_tensor = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
try:
    cap = ms_to_dlpack(ms_tensor)
    pt_from_ms = torch_from_dlpack(cap)
    ok = tuple(pt_from_ms.shape) == tuple(ms_tensor.shape)
    check(1, "MS→PT 交换", ok,
          f"shape={tuple(pt_from_ms.shape)} dtype={pt_from_ms.dtype} device={pt_from_ms.device}")
except Exception as e:  # noqa: BLE001
    pt_from_ms = None
    check(1, "MS→PT 交换", False, f"异常: {type(e).__name__}: {e}")

pt_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device=DEVICE)
try:
    cap2 = torch_to_dlpack(pt_tensor)
    ms_from_pt = ms_from_dlpack(cap2)
    ok = tuple(ms_from_pt.shape) == (2, 2) and ms_from_pt.dtype == ms.float32
    check(2, "PT→MS 交换", ok, f"shape={ms_from_pt.shape} dtype={ms_from_pt.dtype}")
except Exception as e:  # noqa: BLE001
    ms_from_pt = None
    check(2, "PT→MS 交换", False, f"异常: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# STEP 3: 零拷贝语义 (torch 侧原地修改, 观察 MS 侧)
# ---------------------------------------------------------------------------
print("\n--- STEP 3: 零拷贝语义验证 ---")
# 方向 A: MS→PT 交换后, torch 原地修改, MS 侧应同步
try:
    src_ms = ms.Tensor(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    view_pt = torch_from_dlpack(ms_to_dlpack(src_ms))
    view_pt.fill_(7.0)
    after = src_ms.asnumpy()
    ok = bool(np.all(after == 7.0))
    check(3, "MS→PT 零拷贝 (torch.fill_ 后 MS 侧同步)", ok, f"MS 侧读回={after.tolist()}")
except Exception as e:  # noqa: BLE001
    check(3, "MS→PT 零拷贝", False, f"异常: {type(e).__name__}: {e}")

# 方向 B: PT→MS 交换后, 修改 torch 源, MS 侧应同步
try:
    src_pt = torch.ones(2, 2, device=DEVICE)
    view_ms = ms_from_dlpack(torch_to_dlpack(src_pt))
    src_pt.add_(3.0)
    after = view_ms.asnumpy()
    ok = bool(np.all(after == 4.0))
    check(3, "PT→MS 零拷贝 (torch.add_ 后 MS 侧同步)", ok, f"MS 侧读回={after.tolist()}")
except Exception as e:  # noqa: BLE001
    check(3, "PT→MS 零拷贝", False, f"异常: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# STEP 4/5/6: 训练一步 — DLPack 交换 + 局部计算图 + 梯度桥接
# ---------------------------------------------------------------------------
print("\n--- STEP 4: PyTorch 局部计算图 (DLPack 叶子重建) ---")
ms_out = ms_block(ms.Tensor(x_np))                # MS 前向 (PYNATIVE, 无梯度上下文)
capsule = ms_to_dlpack(ms_out)                    # ② DLPack 传递 (导出无需 detach, 见 README)
pt_input = torch_from_dlpack(capsule)             # 接到 PT
pt_input.requires_grad_(True)                     # ③ 重立叶子

pt_out = pt_head(pt_input)
loss = loss_fn(pt_out, target)
loss.backward()                                   # ④ PT 反向

g = to_np(pt_input.grad)
check(4, "pt_input.grad 非空", g is not None and np.isfinite(g).all(),
      f"grad shape={g.shape if g is not None else None}")
ok4 = True
for n, p in pt_head.named_parameters():
    ok4 &= p.grad is not None and torch.isfinite(p.grad).all()
check(4, "pt_head 参数梯度非空", ok4)

print("\n--- STEP 5: 梯度桥接 (pt_input.grad → MS sens) ---")
# ⑤ 把边界梯度通过 DLPack 送回 MindSpore 作为 sens
sens = ms_from_dlpack(torch_to_dlpack(pt_input.grad.detach()))
grad_op = GradOperation(get_all=False, get_by_list=True, sens_param=True)
ms_grad_fn = grad_op(ms_block)
param_grads = ms_grad_fn(ms.Tensor(x_np), sens)   # 返回 (w_grad, b_grad), 顺序同 trainable_params

ms_params = ms_block.trainable_params()
ok5 = len(param_grads) == len(ms_params) and all(
    gg is not None and np.isfinite(gg.asnumpy()).all() for gg in param_grads)
check(5, "MS 参数梯度非空 (sens_param=True)", ok5,
      " | ".join(f"{p.name}:{tuple(gg.shape)}" for p, gg in zip(ms_params, param_grads)))

print("\n--- STEP 6: 数值正确性 (float64 解析解对照) ---")
# 复合函数: loss = MSE( pt_head( ms_block(x) ), t ), 线性链, 可解析求导
W1 = ms_block.fc.weight.asnumpy().astype(np.float64)
b1 = ms_block.fc.bias.asnumpy().astype(np.float64)
W2 = pt_head.weight.detach().cpu().numpy().astype(np.float64)
b2 = pt_head.bias.detach().cpu().numpy().astype(np.float64)

y = x64 @ W1.T + b1          # ms_block 输出 (1,4)
z = y @ W2.T + b2            # pt_head 输出 (1,2)
loss_ref = float(np.mean((z - t64) ** 2))
loss_torch = float(loss.item())

dz = 2.0 * (z - t64) / z.size      # (1,2)
dy = dz @ W2                        # (1,4)  == ∂loss/∂ms_out == 期望的 pt_input.grad
dW2_ref = dz.T @ y                  # (2,4)
db2_ref = dz.reshape(-1)            # (2,)
dW1_ref = dy.T @ x64                # (4,4)
db1_ref = dy.reshape(-1)            # (4,)

rel = lambda a, b: float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))

check(6, "两框架前向 loss 一致", rel(loss_torch, loss_ref) < args.rtol,
      f"torch={loss_torch:.8f} vs 解析={loss_ref:.8f}")
check(6, "pt_input.grad == 解析解 d(loss)/d(ms_out)", rel(g, dy) < args.rtol,
      f"rel_err={rel(g, dy):.3e}")
pg_w = param_grads[0].asnumpy().astype(np.float64)
pg_b = param_grads[1].asnumpy().astype(np.float64)
check(6, "MS weight 梯度 == 解析解 d(loss)/dW1", rel(pg_w, dW1_ref) < args.rtol,
      f"rel_err={rel(pg_w, dW1_ref):.3e}")
check(6, "MS bias 梯度 == 解析解 d(loss)/db1", rel(pg_b, db1_ref) < args.rtol,
      f"rel_err={rel(pg_b, db1_ref):.3e}")

print("\n--- STEP 7: 负对照 (ones_like 假 sens 应产生不同梯度) ---")
sens_fake = ops.ones_like(ms_out)
pg_fake = ms_grad_fn(ms.Tensor(x_np), sens_fake)
diff_w = float(np.max(np.abs(pg_fake[0].asnumpy() - pg_w)))
check(7, "假 sens 梯度 ≠ 真实桥接梯度", diff_w > 1e-6, f"max|ΔW|={diff_w:.3e}")

print("\n--- STEP 8: 两侧参数更新 ---")
pt_w_before = to_np(pt_head.weight).copy()
ms_w_before = ms_block.fc.weight.asnumpy().copy()
pt_optim.step()
pt_optim.zero_grad()
for p, gg in zip(ms_params, param_grads):
    p.set_data(p - LR * gg)          # 最朴素 SGD: w = w - lr * grad (算术返回 Tensor, set_data 写回)
pt_moved = float(np.max(np.abs(to_np(pt_head.weight) - pt_w_before)))
ms_moved = float(np.max(np.abs(ms_block.fc.weight.asnumpy() - ms_w_before)))
check(8, "PyTorch 参数已更新", pt_moved > 0, f"max|Δw|={pt_moved:.3e}")
check(8, "MindSpore 参数已更新", ms_moved > 0, f"max|Δw|={ms_moved:.3e}")

# ---------------------------------------------------------------------------
# STEP 9: 交换耗时 (dlpack vs 拷贝路径)
# ---------------------------------------------------------------------------
print("\n--- STEP 9: 交换耗时统计 (100 次均值) ---")
src = ms.Tensor(x_np)
N = 100
t0 = time.perf_counter()
for _ in range(N):
    _v = torch_from_dlpack(ms_to_dlpack(src))
t_dlpack = (time.perf_counter() - t0) / N * 1e6

t0 = time.perf_counter()
for _ in range(N):
    _v = torch.from_numpy(src.asnumpy().copy())
t_copy = (time.perf_counter() - t0) / N * 1e6
print(f"  dlpack 交换: {t_dlpack:8.1f} us/次   |  拷贝路径(asnumpy→torch): {t_copy:8.1f} us/次")

# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
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
sys.exit(0 if n_pass == len(results) else 1)

