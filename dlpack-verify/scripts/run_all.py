# -*- coding: utf-8 -*-
"""
run_all.py — 一键全跑: 顺序执行 E0~E5 全部验证阶段, 输出实时回显并 tee 到
results/{dev}/run_all_<时间戳>.log (日志中每个阶段有明确横幅标注做了什么)。
任一阶段失败立即停止并以非 0 退出; 全部通过后导出时间戳版多 sheet 数据工作簿。

用法 (服务器上已激活 venv 与 CANN 环境, 项目根目录):
    python scripts/run_all.py                          # 全量默认 (npu, 300 步)
    python scripts/run_all.py --device cpu --steps 20  # 本地快速冒烟
"""
import argparse
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")   # 管道输出统一 utf-8, 防 GBK 乱码
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJECT, "scripts")
PY = sys.executable


def start_smi_sampler(dev, tag, out_dir, interval=1.0):
    """P1 回退检测采样 (仅 NPU): 后台循环采集 npu-smi 的 AI Core/HBM 利用率到 CSV。
    DeepSeek C/P 体系的 P1 一票否决门禁在真实 GNN 接入时执行; 本验证阶段做采样存证。
    返回 (stop_event, csv_path); cpu 环境返回 None。"""
    if dev != "npu":
        return None
    cmd = ["npu-smi", "info", "-t", "usage", "-i", "0"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"smi_{tag}_{ts}.csv")
    stop = threading.Event()

    def parse(text):
        aicore = hbm = ""
        for line in text.splitlines():
            low = line.lower()
            if "%" in line and ("aicore" in low or "hbm" in low):
                digits = "".join(ch for ch in line if ch.isdigit() or ch == ".")
                if "aicore" in low:
                    aicore = digits
                else:
                    hbm = digits
        return aicore, hbm, text.replace("\n", " ")[:200]

    def loop():
        with open(path, "w", encoding="utf-8") as f:
            f.write("unix_ts,aicore_pct,hbm_pct,ok,raw\n")
            while not stop.is_set():
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=10).stdout
                    a, h, raw = parse(out)
                    f.write(f"{time.time()},{a},{h},{int(bool(a))},\"{raw}\"\n")
                except Exception as e:  # noqa: BLE001
                    f.write(f"{time.time()},,,0,\"{type(e).__name__}: {e}\"\n")
                f.flush()
                stop.wait(interval)

    threading.Thread(target=loop, daemon=True).start()
    return stop, path


def build_stages(a):
    """阶段表: (编号, 说明, 命令参数列表)。参数按各脚本的接口透传。"""
    common = ["--device", a.device]
    train = (common + ["--steps", str(a.steps), "--hidden", str(a.hidden),
                       "--batch", str(a.batch), "--lr", str(a.lr),
                       "--rtol", str(a.rtol), "--eval-every", str(a.eval_every),
                       "--seed", str(a.seed)])
    return [
        ("E0", "环境自检 (torch_npu/NPU/CANN/双向 DLPack 冒烟)",
         [os.path.join(SCRIPTS, "check_env.py")]),
        ("E1", "机制验证: 9 STEP / 14 检查项 (零拷贝/桥接/解析解/负对照)",
         [os.path.join(SCRIPTS, "verify_dlpack.py"), "--device", a.device,
          "--rtol", str(a.rtol)]),
        ("E2", "混合训练 dlpack + MS 参照臂 (收敛/泛化/切换等价性/M4, 逐步 CSV)",
         [os.path.join(SCRIPTS, "train_hybrid.py")] + train),
        ("E3", "混合训练 copy 对照 + MS 参照臂 (唯一变量=交换机制)",
         [os.path.join(SCRIPTS, "train_hybrid.py")] + train
         + ["--exchange", "copy"]),
        ("E4", "形态对称性: PT前段+MS后段 + MS 参照臂 (迁移过半形态)",
         [os.path.join(SCRIPTS, "train_hybrid.py")] + train
         + ["--topology", "front_pt"]),
        ("E5", "纯交换吞吐尺寸扫描 (边界张量 32KB/1MB/4MB)",
         [os.path.join(SCRIPTS, "train_hybrid.py")] + common
         + ["--hidden", str(a.hidden), "--bench", "--bench-sizes", a.bench_sizes,
            "--bench-n", str(a.bench_n)]),
    ]


def main():
    ap = argparse.ArgumentParser(
        description="DLPack 混合过渡态验证: 一键顺序执行 E0~E5 并 tee 日志")
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rtol", default="5e-3", help="M4 容差 (默认 NPU 口径 5e-3)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bench-sizes", default="128,4096,16384")
    ap.add_argument("--bench-n", type=int, default=200)
    a = ap.parse_args()

    dev = "cpu" if a.device == "cpu" else "npu"
    out_dir = os.path.join(PROJECT, "results", dev)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir,
                            f"run_all_{time.strftime('%Y%m%d_%H%M%S')}.log")
    stages = build_stages(a)

    print("=" * 78)
    print(f"DLPack 混合过渡态验证 — 一键全跑 ({len(stages)} 个阶段)")
    print(f"device={a.device}  steps={a.steps}  hidden={a.hidden}  batch={a.batch}  "
          f"rtol={a.rtol}")
    print(f"日志: {log_path}")
    print("=" * 78)

    t_all = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        def tee(line):
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        for i, (tag, desc, cmd) in enumerate(stages, 1):
            tee("\n" + "#" * 78 + "\n")
            tee(f"### [{i}/{len(stages)}] {tag} — {desc}\n")
            tee(f"### cmd: {PY} {cmd[0]} {' '.join(cmd[1:])}\n")
            tee("#" * 78 + "\n")
            sampler = start_smi_sampler(a.device, tag, out_dir) \
                if tag in ("E2", "E3", "E4") else None
            if sampler:
                tee(f"### [P1] npu-smi 回退检测采样已启动: {sampler[1]}\n")
            t0 = time.perf_counter()
            # 子进程继承 PYTHONIOENCODING=utf-8 (本模块顶部已设), 管道按 utf-8 读
            proc = subprocess.Popen([PY] + cmd, cwd=PROJECT,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            for line in proc.stdout:
                tee(line)
            rc = proc.wait()
            if sampler:
                sampler[0].set()
                tee(f"### [P1] 采样已停止, 数据: {sampler[1]}\n")
            dt = time.perf_counter() - t0
            tee(f"### [{tag}] 退出码 {rc}  耗时 {dt:.1f} s\n")
            if rc != 0:
                tee("\n" + "!" * 78 + "\n")
                tee(f"!!! 一键全跑在阶段 {tag} 失败 (退出码 {rc}), 后续阶段未执行。\n")
                tee(f"!!! 日志: {log_path}\n")
                return rc if rc else 1

        tee("\n" + "#" * 78 + "\n")
        tee(f"### 全部 {len(stages)} 个阶段通过, 总耗时 "
            f"{(time.perf_counter() - t_all) / 60:.1f} 分钟\n")
        tee("#" * 78 + "\n")
        try:
            sys.path.insert(0, SCRIPTS)
            from export_xlsx import export_xlsx
            xlsx = export_xlsx(dev)
            tee(f"### 数据结果工作簿: {xlsx}\n")
        except ImportError:
            tee("### [提示] 未安装 openpyxl, 跳过 Excel 工作簿导出 "
                "(pip install openpyxl)\n")
        tee(f"### 完整日志: {log_path}\n")
        tee("### 下一步: 把 results/ 目录拷回 Windows, 运行 "
            "python scripts/make_report.py 生成最终验证报告\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
