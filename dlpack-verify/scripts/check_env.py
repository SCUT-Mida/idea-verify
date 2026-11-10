# -*- coding: utf-8 -*-
"""
check_env.py — DLPack 验证环境自检脚本 (Windows CPU / 昇腾 910B 通用)

用途:
    在开始验证前, 检查 Python / numpy / PyTorch / MindSpore 版本,
    以及 DLPack 相关能力是否就绪。在 910B 上运行时, 会额外探测
    torch_npu / NPU 设备 / CANN 版本。

用法:
    python check_env.py            # 当前激活环境
退出码:
    0 = 核心组件齐备; 1 = 有关键缺失(看输出定位)
"""
import platform
import subprocess
import sys
import importlib
import warnings

# 静音两类已知无害的平台告警 (成因见 README §4.3), 按消息精确匹配, 其他警告仍可见:
#   - "owner does not match": CANN 由 root 安装而以普通用户运行, torch_npu 环境收集器的提示
#   - "smallest subnormal":   ARM 容器内核关闭非规格化浮点, numpy 提示 denormal 刷零
warnings.filterwarnings("ignore", message=".*owner does not match.*")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")

REPORT = []
CRITICAL_MISSING = []
IS_WINDOWS = platform.system() == "Windows"


def item(name, ok, detail, optional_on_windows=False):
    """optional_on_windows=True 的项在 Windows 上缺失不算失败 (910B/Linux 上仍必需)"""
    if ok:
        flag = "[OK]  "
    elif optional_on_windows and IS_WINDOWS:
        flag = "[OPT] " + "  "
    else:
        flag = "[MISS]"
    line = f"{flag} {name:<28} {detail}"
    REPORT.append(line)
    if not ok and not (optional_on_windows and IS_WINDOWS):
        CRITICAL_MISSING.append(name)
    return ok


def try_import(modname):
    try:
        return importlib.import_module(modname), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    print("=" * 78)
    print("DLPack 验证环境自检")
    print("=" * 78)

    # ---------- 基础 ----------
    item("Python", True, sys.version.split()[0])
    item("平台", True, platform.platform())
    np, np_err = try_import("numpy")
    if np is not None:
        major_ok = int(np.__version__.split(".")[0]) < 2
        item("numpy", True, np.__version__ + ("" if major_ok else
             "  [警告] numpy>=2 与 mindspore wheel (numpy1.x ABI) 不兼容, 请降级 numpy<2"))
    else:
        item("numpy", False, np_err)

    # ---------- PyTorch ----------
    torch, torch_err = try_import("torch")
    if torch is not None:
        item("PyTorch", True, torch.__version__)
    else:
        item("PyTorch", False, torch_err)

    # ---------- torch_npu (仅昇腾环境需要) ----------
    torch_npu, tnpu_err = try_import("torch_npu")
    if torch_npu is not None:
        n = 0
        try:
            n = torch_npu.npu.device_count()
        except Exception:  # noqa: BLE001
            pass
        item("torch_npu", True, getattr(torch_npu, "__version__", "?") + f"  NPU设备数={n}")
    else:
        item("torch_npu", False, tnpu_err + "  (CPU 环境下正常, 910B 上必须可用)",
             optional_on_windows=True)

    # ---------- MindSpore ----------
    ms, ms_err = try_import("mindspore")
    if ms is not None:
        item("MindSpore", True, ms.__version__)
    else:
        item("MindSpore", False, ms_err)

    # ---------- MindSpore DLPack API ----------
    ms_dlpack, msdl_err = try_import("mindspore.utils.dlpack")
    if ms_dlpack is not None:
        has_to = hasattr(ms_dlpack, "to_dlpack")
        has_from = hasattr(ms_dlpack, "from_dlpack")
        item("mindspore.utils.dlpack", has_to and has_from,
             f"to_dlpack={has_to}, from_dlpack={has_from}")
    else:
        item("mindspore.utils.dlpack", False, msdl_err)

    # ---------- torch.utils.dlpack ----------
    if torch is not None:
        tdl, tdl_err = try_import("torch.utils.dlpack")
        if tdl is not None:
            item("torch.utils.dlpack", hasattr(tdl, "to_dlpack") and hasattr(tdl, "from_dlpack"),
                 f"to_dlpack/from_dlpack 均存在")
        else:
            item("torch.utils.dlpack", False, tdl_err)

    # ---------- CANN (仅昇腾) ----------
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
        first = (out.stdout or "").strip().splitlines()
        item("CANN/npu-smi", out.returncode == 0 and bool(first),
             (first[0] if first else "无输出")[:60])
    except Exception as e:  # noqa: BLE001
        item("CANN/npu-smi", False, f"不可用 ({type(e).__name__})  (CPU 环境下正常)",
             optional_on_windows=True)

    # ---------- 冒烟测试: 双向 dlpack capsule ----------
    print("-" * 78)
    print("冒烟测试 (smoke test):")
    if ms is not None and ms_dlpack is not None and torch is not None:
        try:
            import numpy as _np
            _t = ms.Tensor(_np.arange(6, dtype=_np.float32).reshape(2, 3))
            _cap = ms_dlpack.to_dlpack(_t)
            print(f"  [OK]  MindSpore -> to_dlpack  capsule 导出成功: {type(_cap)}")
            try:
                _tt = torch.utils.dlpack.from_dlpack(_cap)
                print(f"  [OK]  MindSpore -> PyTorch  from_dlpack 成功: shape={tuple(_tt.shape)}, "
                      f"device={_tt.device}, dtype={_tt.dtype}")
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] PyTorch 接收 MindSpore capsule 失败: {e}")
            try:
                _cap2 = torch.utils.dlpack.to_dlpack(torch.ones(2, 3))
                _t2 = ms_dlpack.from_dlpack(_cap2)
                print(f"  [OK]  PyTorch -> MindSpore from_dlpack 成功: shape={_t2.shape}, "
                      f"dtype={_t2.dtype}")
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] MindSpore 接收 PyTorch capsule 失败: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] MindSpore to_dlpack 导出失败: {e}")
    else:
        print("  [SKIP] 关键组件缺失, 无法冒烟")

    # ---------- 汇总 ----------
    print("=" * 78)
    print("\n".join(REPORT))
    print("=" * 78)
    if CRITICAL_MISSING:
        print("结论: 存在关键缺失项 ->", ", ".join(CRITICAL_MISSING))
        return 1
    print("结论: 环境齐备, 可以开始 DLPack 验证。(标注 [OPT] 的项为当前平台可选)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
