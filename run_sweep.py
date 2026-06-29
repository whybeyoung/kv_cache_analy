#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化扫描:按顺序跑 输入长度 x 命中率 的笛卡尔积,逐个调用 hirate.py。
- 输入长度:128k / 256k / 512k / 768k
- 命中率:0.85 / 0.90 / 0.95 / 0.98 / 0.99
- num-prompts 和 concurrency 随输入变大而变小(见 PROFILE)
- 每次结果以 JSONL 追加保存,跑完自动出走势图。

用法:
  python run_sweep.py \
      --url http://26.5.37.7:30000 \
      --tokenizer /work/models \
      --out results.jsonl

  # 跳过实跑,只用已有 results.jsonl 出图:
  python run_sweep.py --out results.jsonl --plot-only
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HIRATE = os.path.join(HERE, "hirate.py")

# 每个输入长度对应的压测规模(随输入变大而变小)
# key: input_len(token);  value: (num_prompts, concurrency, warmup)
PROFILE = {
    131072: (300, 200, 2),   # 128k
    262144: (300, 200, 2),   # 256k
    524288: (150, 100, 2),   # 512k
    786432: (80,  48,  2),   # 768k
}

INPUT_LENS = [131072, 262144, 524288, 786432]
HIT_RATES = [0.85, 0.90, 0.95, 0.98, 0.99]


def human(n):
    return f"{n // 1024}k"


def run_one(args, input_len, hit_rate):
    num_prompts, concurrency, warmup = PROFILE[input_len]
    cmd = [
        sys.executable, HIRATE,
        "--url", args.url,
        "--input-len", str(input_len),
        "--hit-rate", str(hit_rate),
        "--num-prompts", str(num_prompts),
        "--concurrency", str(concurrency),
        "--max-new-tokens", "1",
        "--warmup", str(warmup),
        "--timeout", str(args.timeout),
        "--result-json", args.out,
    ]
    if args.vocab_size:
        cmd += ["--vocab-size", str(args.vocab_size)]
    else:
        cmd += ["--tokenizer", args.tokenizer]

    print(f"\n>>> 输入 {human(input_len)}  命中率 {hit_rate}  "
          f"(prompts={num_prompts}, conc={concurrency})", flush=True)
    print("    " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"<<< 用时 {time.time()-t0:.1f}s  退出码={rc}", flush=True)
    if rc != 0:
        print(f"[warn] 该组合失败(rc={rc}),继续下一个", file=sys.stderr)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://26.5.37.7:30000")
    ap.add_argument("--tokenizer", default="/work/models")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="给定则跳过 tokenizer 加载")
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_results.jsonl"))
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--plot-only", action="store_true",
                    help="不跑压测,仅用已有 --out 数据出图")
    ap.add_argument("--no-plot", action="store_true", help="跑完不出图")
    ap.add_argument("--fresh", action="store_true",
                    help="开跑前清空 --out 文件")
    args = ap.parse_args()

    if not args.plot_only:
        if args.fresh and os.path.exists(args.out):
            os.remove(args.out)
        total = len(INPUT_LENS) * len(HIT_RATES)
        i = 0
        for input_len in INPUT_LENS:           # 按输入长度顺序
            for hit_rate in HIT_RATES:         # 每个长度跑全部命中率
                i += 1
                print(f"\n===== [{i}/{total}] =====")
                run_one(args, input_len, hit_rate)
        print(f"\n全部完成,结果保存在 {args.out}")

    if not args.no_plot:
        plot_cmd = [sys.executable, os.path.join(HERE, "plot_sweep.py"),
                    "--in", args.out]
        subprocess.run(plot_cmd)


if __name__ == "__main__":
    main()
