#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可控命中率压测:每条请求 = 共享前缀(命中) + 唯一后缀(miss)。
命中率 = prefix_len / total_len。直打 SGLang /generate,token-in。
用服务端 meta_info.cached_tokens 报告【实测命中率】,与目标对照。

用法:
  python bench_hitrate.py \
      --url http://127.0.0.1:30000 \
      --tokenizer /work/models \
      --input-len 131072 --hit-rate 0.9 \
      --num-prompts 200 --concurrency 50 \
      --max-new-tokens 1 --warmup 2

  # 不想加载 tokenizer(规避 deepseek_v4 rope 报错):直接给词表大小
  python bench_hitrate.py --url ... --vocab-size 129280 \
      --input-len 131072 --hit-rate 0.9 --num-prompts 200 --concurrency 50

原理:
  - 共享前缀(prefix_len 个 token,所有请求相同)第一次被 prefill 后进 radix tree;
    之后每条请求都命中这段 -> 命中 token = prefix_len。
  - 唯一后缀(suffix_len 个随机 token,每条不同)必须真算 -> miss。
  - 预热阶段先发"仅前缀"请求把前缀写进树,保证并发时前缀已就绪。
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import aiohttp


def load_vocab_size(args):
    if args.vocab_size:
        return args.vocab_size
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] AutoTokenizer 失败: {type(e).__name__}: {e}", file=sys.stderr)
        tj = os.path.join(args.tokenizer, "tokenizer.json")
        from transformers import PreTrainedTokenizerFast
        print(f"[warn] 回退到 PreTrainedTokenizerFast({tj})", file=sys.stderr)
        tok = PreTrainedTokenizerFast(tokenizer_file=tj)
    return getattr(tok, "vocab_size", None) or len(tok.get_vocab())


def build_inputs(args, vocab):
    total = args.input_len
    prefix_len = int(round(total * args.hit_rate))
    suffix_len = total - prefix_len
    lo, hi = 10, max(11, vocab - 1)

    rng = random.Random(args.seed)
    shared_prefix = [rng.randint(lo, hi) for _ in range(prefix_len)]

    inputs = []
    for i in range(args.num_prompts):
        r = random.Random(args.seed * 1_000_003 + i)  # 每条独立、可复现
        suffix = [r.randint(lo, hi) for _ in range(suffix_len)]
        if suffix:
            suffix[0] = lo + (i % (hi - lo))  # 保证首 token 各异,干净分叉
        inputs.append(shared_prefix + suffix)
    return shared_prefix, inputs, prefix_len, suffix_len


async def one_request(session, url, input_ids, sp, stream):
    payload = {"input_ids": input_ids, "sampling_params": sp, "stream": stream}
    t0 = time.perf_counter()
    ttft = None
    n_out = n_prompt = n_cached = 0
    ok = False
    err = None
    try:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            if resp.status != 200:
                err = f"HTTP {resp.status}: {text[:300]}"
            else:
                data = json.loads(text)
                mi = data.get("meta_info", {}) if isinstance(data, dict) else {}
                if isinstance(data, dict) and "output_ids" in data:
                    n_out = len(data["output_ids"])
                else:
                    n_out = mi.get("completion_tokens", 0)
                n_prompt = mi.get("prompt_tokens", 0)
                n_cached = mi.get("cached_tokens", 0)
                ok = True
        ttft = (time.perf_counter() - t0) * 1000
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    lat = (time.perf_counter() - t0) * 1000
    return {"ok": ok, "lat": lat, "ttft": ttft, "n_out": n_out,
            "n_prompt": n_prompt, "n_cached": n_cached, "err": err}


async def run(args, shared_prefix, inputs):
    url = args.url.rstrip("/") + "/generate"
    sp = {"max_new_tokens": args.max_new_tokens,
          "temperature": args.temperature, "ignore_eos": True}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # 预热:仅发共享前缀,把它写进 radix tree
        for k in range(args.warmup):
            r = await one_request(session, url, shared_prefix, sp, False)
            tag = r["err"] if r["err"] else f"OK lat={r['lat']:.0f}ms"
            print(f"[warmup] 前缀预热 {k+1}/{args.warmup}: {tag}", file=sys.stderr)

        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(ids):
            async with sem:
                return await one_request(session, url, ids, sp, False)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(ids) for ids in inputs])
        dur = time.perf_counter() - t0
    return results, dur


def report(args, prefix_len, suffix_len, results, dur):
    ok = [r for r in results if r["ok"]]
    fail = len(results) - len(ok)
    lats = sorted(r["lat"] for r in ok)
    tot_out = sum(r["n_out"] for r in ok)
    sum_prompt = sum(r["n_prompt"] for r in ok)
    sum_cached = sum(r["n_cached"] for r in ok)
    have_meta = sum_prompt > 0
    total_len = prefix_len + suffix_len
    if not have_meta:
        sum_prompt = len(ok) * total_len
    real_input = max(sum_prompt - sum_cached, 0)
    hit = (sum_cached / sum_prompt * 100) if sum_prompt else 0.0

    def pct(xs, p):
        if not xs:
            return 0.0
        k = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
        return xs[k]

    print("\n========= 可控命中率 Benchmark =========")
    print(f"目标命中率:            {args.hit_rate*100:.1f}%")
    print(f"前缀/后缀/总:          {prefix_len} / {suffix_len} / {total_len} tokens")
    print(f"并发:                  {args.concurrency}")
    print(f"成功/失败:             {len(ok)} / {fail}")
    print(f"Duration (s):          {dur:.2f}")
    print(f"Request throughput:    {len(ok)/dur:.2f} req/s")
    print("--- 命中核对(来自服务端 meta_info)---")
    print(f"  meta_info 可用:      {'是' if have_meta else '否(回退估算)'}")
    print(f"  总 prompt tokens:    {sum_prompt:,}")
    print(f"  命中(cached):        {sum_cached:,}")
    print(f"  实测命中率:          {hit:.1f}%   (目标 {args.hit_rate*100:.1f}%)")
    print(f"  名义输入吞吐:        {sum_prompt/dur:,.0f} tok/s")
    print(f"  真实prefill吞吐:     {real_input/dur:,.0f} tok/s  (只算后缀那部分)")
    print(f"Output token throughput:{tot_out/dur:,.2f} tok/s")
    if lats:
        print("--- E2E Latency (ms) ---")
        print(f"  Mean: {statistics.mean(lats):.2f}  P50: {pct(lats,50):.2f}  "
              f"P90: {pct(lats,90):.2f}  P99: {pct(lats,99):.2f}")
    print("========================================")
    if fail:
        for e in [r["err"] for r in results if not r["ok"] and r["err"]][:3]:
            print(f"[失败] {e}", file=sys.stderr)
    # 命中率偏差提醒
    if have_meta and abs(hit - args.hit_rate * 100) > 3:
        print(f"[注意] 实测命中率与目标偏差 >3%。可能原因:并发过高导致前缀被驱逐("
              f"调低并发或加大显存),或 page 对齐误差。", file=sys.stderr)

    return {
        "target_hit_rate": args.hit_rate,
        "input_len": total_len,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "success": len(ok),
        "fail": fail,
        "duration_s": dur,
        "req_throughput": (len(ok) / dur) if dur else 0.0,
        "meta_available": have_meta,
        "sum_prompt_tokens": sum_prompt,
        "sum_cached_tokens": sum_cached,
        "measured_hit_rate": hit / 100.0,
        "nominal_input_throughput": (sum_prompt / dur) if dur else 0.0,
        "real_prefill_throughput": (real_input / dur) if dur else 0.0,
        "output_throughput": (tot_out / dur) if dur else 0.0,
        "lat_mean_ms": statistics.mean(lats) if lats else 0.0,
        "lat_p50_ms": pct(lats, 50),
        "lat_p90_ms": pct(lats, 90),
        "lat_p99_ms": pct(lats, 99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tokenizer", default=None, help="本地模型目录(用于取词表大小)")
    ap.add_argument("--vocab-size", type=int, default=None, help="直接给词表大小,跳过 tokenizer")
    ap.add_argument("--input-len", type=int, default=131072, help="单条总输入 token 数")
    ap.add_argument("--hit-rate", type=float, default=0.9, help="目标命中率 0~1")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=2, help="前缀预热次数")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--result-json", default=None,
                    help="把本次结果以 JSON 写入(或追加到 JSONL)该文件")
    args = ap.parse_args()

    if not (args.vocab_size or args.tokenizer):
        ap.error("需指定 --tokenizer 或 --vocab-size")
    if not (0.0 < args.hit_rate < 1.0):
        ap.error("--hit-rate 必须在 (0,1) 之间")

    vocab = load_vocab_size(args)
    print(f"[info] vocab_size = {vocab}", file=sys.stderr)
    shared_prefix, inputs, prefix_len, suffix_len = build_inputs(args, vocab)
    print(f"[info] prefix_len={prefix_len}, suffix_len={suffix_len}, "
          f"共 {len(inputs)} 条", file=sys.stderr)

    results, dur = asyncio.run(run(args, shared_prefix, inputs))
    summary = report(args, prefix_len, suffix_len, results, dur)
    if args.result_json:
        with open(args.result_json, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"[info] 结果已追加到 {args.result_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
