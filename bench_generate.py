#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接打 SGLang /generate 接口的并发压测,token-in-token-out。

目的:绕开 bench_serving 客户端的 tokenize / 输出 retokenize 计数开销,
测纯引擎吞吐。固定一条 input_ids 重复并发 -> 100% 前缀命中。

两层"绕过 token 处理":
  1) 输入:发 input_ids(本脚本只 tokenize 一次,不进并发循环)-> 跳过服务端输入 tokenize。
  2) 输出:服务端用 `--skip-tokenizer-init` 启动时,返回 output_ids、不做 detokenize。
     不带该 flag 时,输出仍会 detokenize,但 max_new_tokens=1 时这点开销可忽略。

用法:
  # A. 从 sharegpt_128k.json 取那条 prompt,tokenize 一次后并发打
  python bench_generate.py \
      --url http://127.0.0.1:8000 \
      --tokenizer /work/models \
      --input-file ./sharegpt_128k.json \
      --num-prompts 200 --concurrency 50 \
      --max-new-tokens 1 --warmup 1

  # B. 完全不用 tokenizer:直接喂一个 input_ids 文件(json 数组)
  python bench_generate.py --url http://127.0.0.1:8000 \
      --input-ids-file ids.json \
      --num-prompts 200 --concurrency 50 --max-new-tokens 1 --warmup 1

  # C. 不依赖任何文件:随机造 N 个 token id(注意随机 id 命中后是同一条)
  python bench_generate.py --url http://127.0.0.1:8000 \
      --tokenizer /work/models --input-len 131072 \
      --num-prompts 200 --concurrency 50 --max-new-tokens 1 --warmup 1

判断 100% 命中:并发跑的 Mean Latency 应远低于单条冷 prefill(几十 ms 量级)。
若想测真实 prefill(无命中),server 端加 --disable-radix-cache 或每条 input_ids 不同。
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


def load_tokenizer(path):
    """加载 tokenizer,带回退(规避新版 transformers 对某些 config 的 rope bug)。"""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] AutoTokenizer 失败: {type(e).__name__}: {e}", file=sys.stderr)
        tj = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else None
        if tj and os.path.isfile(tj):
            from transformers import PreTrainedTokenizerFast
            print(f"[warn] 回退到 PreTrainedTokenizerFast({tj})", file=sys.stderr)
            return PreTrainedTokenizerFast(tokenizer_file=tj)
        raise


def build_input_ids(args):
    """构造固定的一条 input_ids(所有并发请求复用 -> 命中)。"""
    if args.input_ids_file:
        ids = json.load(open(args.input_ids_file))
        assert isinstance(ids, list) and ids and isinstance(ids[0], int)
        return ids

    tok = load_tokenizer(args.tokenizer)
    if args.input_file:
        data = json.load(open(args.input_file))
        # 取第一条 human 段作为 prompt
        text = ""
        for turn in data[0]["conversations"]:
            if turn.get("from") == "human":
                text = turn["value"]
                break
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        # 随机造 token id(避开前若干特殊 token)
        vocab = getattr(tok, "vocab_size", 32000)
        n = args.input_len or 131071
        ids = [random.randint(10, vocab - 1) for _ in range(n)]

    if args.input_len:
        if len(ids) > args.input_len:
            ids = ids[: args.input_len]
        elif len(ids) < args.input_len:
            ids = ids + ids[: args.input_len - len(ids)]
            while len(ids) < args.input_len:
                ids += ids[: args.input_len - len(ids)]
    return ids


async def one_request(session, url, input_ids, sp, stream):
    payload = {"input_ids": input_ids, "sampling_params": sp, "stream": stream}
    t0 = time.perf_counter()
    ttft = None
    n_out = 0
    ok = False
    try:
        if stream:
            async with session.post(url, json=payload) as resp:
                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    try:
                        data = json.loads(body)
                        mi = data.get("meta_info", {})
                        n_out = mi.get("completion_tokens", n_out)
                    except Exception:
                        pass
                ok = True
        else:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                mi = data.get("meta_info", {}) if isinstance(data, dict) else {}
                if "output_ids" in data:                 # skip-tokenizer-init 模式
                    n_out = len(data["output_ids"])
                else:
                    n_out = mi.get("completion_tokens", 0)
                ok = resp.status == 200
            ttft = (time.perf_counter() - t0) * 1000   # 非流式 TTFT≈E2E
    except Exception as e:
        print(f"[req-error] {type(e).__name__}: {e}", file=sys.stderr)
    lat = (time.perf_counter() - t0) * 1000
    return {"ok": ok, "lat": lat, "ttft": ttft, "n_out": n_out}


async def run(args, input_ids):
    url = args.url.rstrip("/") + "/generate"
    sp = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=0)  # 不限连接数,由信号量控并发
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # 串行预热:把这条 128k 前缀写进 radix tree
        for _ in range(args.warmup):
            await one_request(session, url, input_ids, sp, args.stream)
        if args.warmup:
            print(f"[warmup] 完成 {args.warmup} 条预热", file=sys.stderr)

        sem = asyncio.Semaphore(args.concurrency)

        async def guarded():
            async with sem:
                return await one_request(session, url, input_ids, sp, args.stream)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[guarded() for _ in range(args.num_prompts)])
        dur = time.perf_counter() - t_start
    return results, dur


def report(args, input_ids, results, dur):
    n_in = len(input_ids)
    ok = [r for r in results if r["ok"]]
    fail = len(results) - len(ok)
    lats = sorted(r["lat"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    tot_out = sum(r["n_out"] for r in ok)

    def pct(xs, p):
        if not xs:
            return 0.0
        k = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
        return xs[k]

    print("\n============ /generate Benchmark ============")
    print(f"URL:                    {args.url}/generate")
    print(f"Stream:                 {args.stream}")
    print(f"Input len (tokens):     {n_in}")
    print(f"Max new tokens:         {args.max_new_tokens}")
    print(f"Concurrency:            {args.concurrency}")
    print(f"Successful requests:    {len(ok)}  (failed: {fail})")
    print(f"Duration (s):           {dur:.2f}")
    print(f"Request throughput:     {len(ok)/dur:.2f} req/s")
    print(f"Input token throughput: {len(ok)*n_in/dur:,.0f} tok/s")
    print(f"Output token throughput:{tot_out/dur:,.2f} tok/s")
    if lats:
        print(f"--- E2E Latency (ms) ---")
        print(f"  Mean:  {statistics.mean(lats):.2f}")
        print(f"  P50:   {pct(lats,50):.2f}")
        print(f"  P90:   {pct(lats,90):.2f}")
        print(f"  P99:   {pct(lats,99):.2f}")
    if ttfts and args.stream:
        print(f"--- TTFT (ms) ---")
        print(f"  Mean:  {statistics.mean(ttfts):.2f}")
        print(f"  P99:   {pct(ttfts,99):.2f}")
    print("=============================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000",
                    help="server 基址,如 http://127.0.0.1:8000")
    ap.add_argument("--tokenizer", default=None, help="本地模型目录(用于 tokenize 输入)")
    ap.add_argument("--input-file", default=None, help="sharegpt 格式 json,取首条 human 段")
    ap.add_argument("--input-ids-file", default=None, help="直接给 input_ids(json 数组),不用 tokenizer")
    ap.add_argument("--input-len", type=int, default=None, help="目标输入 token 数(裁剪/补齐或随机生成)")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--stream", action="store_true", help="开流式以单独测 TTFT")
    ap.add_argument("--warmup", type=int, default=1, help="串行预热条数(焐热前缀 -> 命中)")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    if not (args.input_file or args.input_ids_file or (args.tokenizer and args.input_len)):
        ap.error("需指定 --input-file,或 --input-ids-file,或 (--tokenizer + --input-len)")

    input_ids = build_input_ids(args)
    print(f"[info] input_ids 长度 = {len(input_ids)} tokens", file=sys.stderr)

    results, dur = asyncio.run(run(args, input_ids))
    report(args, input_ids, results, dur)


if __name__ == "__main__":
    main()
