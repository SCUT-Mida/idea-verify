# -*- coding: utf-8 -*-
"""
export_xlsx.py — 把 results/{dev}/ 下的机器可读摘要与逐步日志整合为
一个多 sheet Excel 工作簿 (文件名带时间戳), 避免零散同名文件搞混数据。

Sheet 规划:
    摘要总览      : 各摘要 JSON 的全部字段 (来源/指标/值 平铺)
    机制验证      : verify_summary 的 checks 明细 + meas 数值
    E2_训练曲线   : dlpack 逐步数据 (step/loss/val_loss/ref_loss/ref_val_loss/t_ms)
    E3_训练曲线   : copy 逐步数据 (同列)
    E4_对称形态   : front_pt 逐步数据 (同列)
    E5_bench扫描  : 尺寸扫描各行 (兼容 v1 单尺寸)

用法:
    from export_xlsx import export_xlsx
    export_xlsx("npu")        # -> results/npu/results_YYYYMMDD_HHMMSS.xlsx
    python scripts/export_xlsx.py [--device npu]   # 独立运行
"""
import csv
import json
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILES = [  # (sheet 名, 摘要文件, 逐步 CSV 文件)
    ("E2_训练曲线", "train_summary_h64_dlpack.json", "training_log_dlpack_h64.csv"),
    ("E3_训练曲线", "train_summary_h64_copy.json", "training_log_copy_h64.csv"),
    ("E4_对称形态", "train_summary_h64_dlpack_sym.json", "training_log_dlpack_h64_sym.csv"),
]


def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _flatten(obj, prefix=""):
    """嵌套 dict -> [(指标, 值)] 平铺 (一层段前缀); 列表值 JSON 字符串化。"""
    rows = []
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.extend(_flatten(v, key))
        elif isinstance(v, (list, tuple)):
            rows.append((key, json.dumps(v, ensure_ascii=False)))
        else:
            rows.append((key, v))
    return rows


def _write_table(ws, headers, rows, header_font):
    ws.append(headers)
    for c in ws[1]:
        c.font = header_font
    for r in rows:
        ws.append(list(r))


def export_xlsx(dev="npu", out_dir=None):
    """整合 results/{dev}/ 数据 -> xlsx, 返回输出路径。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    dev_dir = out_dir or os.path.join(PROJECT, "results", dev)
    wb = Workbook()
    bold = Font(bold=True)

    # ---- Sheet: 摘要总览 (全部摘要字段平铺) ----
    ws = wb.active
    ws.title = "摘要总览"
    summary_files = [
        ("verify_summary.json", "机制验证"),
        ("train_summary_h64_dlpack.json", "E2 混合训练 dlpack"),
        ("train_summary_h64_copy.json", "E3 混合训练 copy"),
        ("train_summary_h64_dlpack_sym.json", "E4 对称形态 front_pt"),
        ("bench_summary.json", "E5 bench 尺寸扫描"),
    ]
    _write_table(ws, ["来源", "指标", "值"], [], bold)
    any_summary = False
    for fname, tag in summary_files:
        data = _load_json(os.path.join(dev_dir, fname))
        if data is None:
            ws.append([tag, f"({fname} 不存在, 待复测)", ""])
            continue
        any_summary = True
        for k, v in _flatten(data):
            ws.append([tag, k, v])
    if not any_summary:
        ws.append(["提示", "results 目录下没有任何摘要, 请先运行实验", ""])

    # ---- Sheet: 机制验证 (checks 明细) ----
    verify = _load_json(os.path.join(dev_dir, "verify_summary.json"))
    ws = wb.create_sheet("机制验证")
    if verify:
        rows = [(c.get("step"), c.get("name"),
                 "PASS" if c.get("ok") else "FAIL", c.get("detail"))
                for c in verify.get("checks", [])]
        _write_table(ws, ["STEP", "检查项", "结果", "明细"], rows, bold)
        ws.append([])
        ws.append(["n_pass / n_total",
                   f"{verify.get('n_pass')} / {verify.get('n_total')}", "", ""])
        for k, v in sorted((verify.get("meas") or {}).items()):
            ws.append(["meas", k, v])
    else:
        _write_table(ws, ["提示"], [("verify_summary.json 不存在",)], bold)

    # ---- Sheets: 各训练跑逐步曲线 ----
    curve_headers = ["step", "loss", "val_loss", "ref_loss", "ref_val_loss", "t_ms"]
    for sheet, sname, cname in TRAIN_FILES:
        ws = wb.create_sheet(sheet)
        summary = _load_json(os.path.join(dev_dir, sname))
        if summary is None:
            _write_table(ws, ["提示"],
                         [(f"{sname} 不存在 — 用当前脚本重跑后生成",)], bold)
            continue
        params = [("exchange", summary.get("exchange")),
                  ("topology", summary.get("topology", "front_ms")),
                  ("hidden", summary.get("hidden")), ("batch", summary.get("batch")),
                  ("lr", summary.get("lr")), ("steps", summary.get("steps")),
                  ("seed", summary.get("seed")), ("rtol", summary.get("rtol"))]
        res = summary.get("results", {})
        params += [("final_loss", res.get("final_loss", res.get("final_loss_b"))),
                   ("eval_mse", res.get("eval_mse", res.get("eval_b"))),
                   ("lat_ms", res.get("lat_ms", res.get("lat_b_ms"))),
                   ("exchange_fwd_us", res.get("exchange_fwd_us")),
                   ("exchange_bwd_us", res.get("exchange_bwd_us")),
                   ("M4 rel_err", (summary.get("m4_calibration") or {}).get("rel_err"))]
        _write_table(ws, ["参数/指标", "值"], params, bold)
        ws.append([])
        cpath = os.path.join(dev_dir, cname)
        if os.path.exists(cpath):
            ws.append([f"逐步曲线 ({cname})"])
            with open(cpath, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    ws.append(row)
        else:
            ws.append([f"逐步曲线 ({cname}) 不存在 — 旧版运行无此日志, 重跑后生成"])

    # ---- Sheet: bench 尺寸扫描 ----
    ws = wb.create_sheet("E5_bench扫描")
    bench = _load_json(os.path.join(dev_dir, "bench_summary.json"))
    if bench is None:
        _write_table(ws, ["提示"], [("bench_summary.json 不存在",)], bold)
    elif isinstance(bench.get("sizes"), list):
        rows = [(r.get("batch"), r.get("boundary"),
                 r["fwd_dlpack_us"], r["fwd_copy_us"],
                 r["fwd_copy_us"] / max(r["fwd_dlpack_us"], 1e-9),
                 r["bwd_dlpack_us"], r["bwd_copy_us"],
                 r["bwd_copy_us"] / max(r["bwd_dlpack_us"], 1e-9))
                for r in bench["sizes"]]
        _write_table(ws, ["batch", "边界张量", "fwd_dlpack_us", "fwd_copy_us",
                          "fwd 慢倍数", "bwd_dlpack_us", "bwd_copy_us",
                          "bwd 慢倍数"], rows, bold)
    else:  # v1 单尺寸扁平格式
        _write_table(ws, ["方向", "dlpack_us", "copy_us", "copy 慢倍数"],
                     [("正向 MS->PT", bench["fwd_dlpack_us"], bench["fwd_copy_us"],
                       bench["fwd_copy_us"] / max(bench["fwd_dlpack_us"], 1e-9)),
                      ("回传 PT->MS", bench["bwd_dlpack_us"], bench["bwd_copy_us"],
                       bench["bwd_copy_us"] / max(bench["bwd_dlpack_us"], 1e-9))], bold)

    out = os.path.join(dev_dir,
                       f"results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
    wb.save(out)
    return out


def main():
    dev = "cpu"
    if "--device" in sys.argv:
        dev = sys.argv[sys.argv.index("--device") + 1]
    path = export_xlsx(dev)
    print(f"数据结果工作簿已生成: {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
