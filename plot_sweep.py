#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 run_sweep.py 产生的 JSONL,画出:
不同 KV 命中率下,输入吞吐 随 输入长度 的变化走势(输出固定为 1 token)。

输出两张图:
  - sweep_nominal_input_throughput.png  名义输入吞吐(含命中部分,sum_prompt/dur)
  - sweep_real_prefill_throughput.png   真实 prefill 吞吐(只算 miss 部分)
并导出汇总 CSV: sweep_summary.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 标签:有中文字体用中文,否则回退英文(避免方块乱码)
L = {
    "xlabel": "输入长度 (tokens)",
    "hit_legend": "KV 命中率",
    "nominal_title": "不同 KV 命中率下 名义输入吞吐 vs 输入长度 (输出=1)",
    "nominal_ylabel": "名义输入吞吐 (k tok/s)",
    "real_title": "不同 KV 命中率下 真实 prefill 吞吐 vs 输入长度 (输出=1)",
    "real_ylabel": "真实 prefill 吞吐 (k tok/s, 仅 miss 部分)",
}
L_EN = {
    "xlabel": "Input length (tokens)",
    "hit_legend": "KV hit rate",
    "nominal_title": "Nominal input throughput vs input length (output=1)",
    "nominal_ylabel": "Nominal input throughput (k tok/s)",
    "real_title": "Real prefill throughput vs input length (output=1)",
    "real_ylabel": "Real prefill throughput (k tok/s, miss only)",
}


def setup_font():
    """返回标签字典:装得上中文字体则用中文,否则回退英文。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ["PingFang SC", "Heiti SC", "Arial Unicode MS",
                "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
                "Microsoft YaHei", "STHeiti"]:
        if fam in available:
            matplotlib.rcParams["font.sans-serif"] = [fam]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return L
    return L_EN


def human(n):
    return f"{int(n) // 1024}k"


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_metric(rows, metric, title, ylabel, out_png, lab):
    # series[hit_rate] = {input_len: value}  (同组合若重复,取最后一次)
    series = defaultdict(dict)
    for r in rows:
        series[r["target_hit_rate"]][r["input_len"]] = r[metric]

    plt.figure(figsize=(9, 6))
    for hit in sorted(series):
        pts = sorted(series[hit].items())
        xs = [p[0] for p in pts]
        ys = [p[1] / 1000.0 for p in pts]  # 转成 k tok/s
        plt.plot(xs, ys, marker="o", label=f"hit={hit:.2f}")

    all_lens = sorted({r["input_len"] for r in rows})
    plt.xticks(all_lens, [human(x) for x in all_lens])
    plt.xlabel(lab["xlabel"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title=lab["hit_legend"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[ok] 已保存 {out_png}")


def export_csv(rows, out_csv):
    cols = ["input_len", "target_hit_rate", "measured_hit_rate",
            "num_prompts", "concurrency", "success", "fail", "duration_s",
            "sum_prompt_tokens", "sum_cached_tokens",
            "nominal_input_throughput", "real_prefill_throughput",
            "output_throughput", "req_throughput",
            "lat_mean_ms", "lat_p50_ms", "lat_p90_ms", "lat_p99_ms"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["input_len"], x["target_hit_rate"])):
            w.writerow(r)
    print(f"[ok] 已保存 {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL 结果文件")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    rows = load(args.inp)
    if not rows:
        print("没有数据可画")
        return
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.inp))

    lab = setup_font()

    export_csv(rows, os.path.join(outdir, "sweep_summary.csv"))
    plot_metric(rows, "nominal_input_throughput",
                lab["nominal_title"], lab["nominal_ylabel"],
                os.path.join(outdir, "sweep_nominal_input_throughput.png"), lab)
    plot_metric(rows, "real_prefill_throughput",
                lab["real_title"], lab["real_ylabel"],
                os.path.join(outdir, "sweep_real_prefill_throughput.png"), lab)


if __name__ == "__main__":
    main()
