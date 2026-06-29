#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 ShareGPT_V3 同格式、单条精确 N token 的数据集。

用法:
    # 纯填充(不依赖原始文件)
    python gen_sharegpt_128k.py --target-tokens 131072 \
        --tokenizer /path/to/model --out sharegpt_128k.json

    # 从原始 ShareGPT 抽真实文本拼接到 128k(分布更真实)
    python gen_sharegpt_128k.py --target-tokens 131072 \
        --tokenizer /path/to/model \
        --src ShareGPT_V3_unfiltered_cleaned_split.json \
        --out sharegpt_128k.json

输出格式与 ShareGPT_V3_unfiltered_cleaned_split.json 一致:
[
  {"id": "...", "conversations": [
      {"from": "human", "value": "<约 N token 的 prompt>"},
      {"from": "gpt",   "value": "<output>"}
  ]}
]
注意:bench_serving 用 sharegpt 模式时,human 段算 prompt(input),
gpt 段算期望 output 长度。所以把长文本放 human 段。
"""
import argparse
import json
import os
import random
import sys


def load_tokenizer(path):
    """加载 tokenizer。

    新版 transformers 在解析某些模型 config 的 rope 参数时有回归 bug
    (AttributeError: 'PreTrainedConfig' object has no attribute
    'max_position_embeddings'),AutoTokenizer 会先加载 config 因而失败。
    这里回退到直接读 tokenizer.json,完全绕过 config。
    """
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] AutoTokenizer 失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        tj = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else None
        if tj and os.path.isfile(tj):
            from transformers import PreTrainedTokenizerFast
            print(f"[warn] 回退到 PreTrainedTokenizerFast({tj})", file=sys.stderr)
            return PreTrainedTokenizerFast(tokenizer_file=tj)
        # 再退一步:试 sentencepiece 的 tokenizer.model
        tm = os.path.join(path, "tokenizer.model") if os.path.isdir(path) else None
        if tm and os.path.isfile(tm):
            try:
                from transformers import LlamaTokenizer
                print(f"[warn] 回退到 LlamaTokenizer({tm})", file=sys.stderr)
                return LlamaTokenizer(vocab_file=tm)
            except Exception as e2:
                print(f"[warn] sentencepiece 回退也失败: {e2}", file=sys.stderr)
        raise RuntimeError(
            f"无法加载 tokenizer: {path}\n"
            f"请确认该目录下有 tokenizer.json,或换成与模型兼容的 transformers 版本。"
        )


def collect_text_from_sharegpt(path, tokenizer, need_tokens):
    """从原始 ShareGPT 文件里顺序抽取 human/gpt 文本,直到 token 数够用。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    buf, total = [], 0
    for item in data:
        for turn in item.get("conversations", []):
            v = turn.get("value", "")
            if not v:
                continue
            buf.append(v)
            total += len(tokenizer(v, add_special_tokens=False)["input_ids"])
            if total >= need_tokens:
                return "\n\n".join(buf)
    # 原始文本不够,循环复用
    text = "\n\n".join(buf) if buf else "The quick brown fox jumps over the lazy dog. "
    while total < need_tokens:
        buf.append(text)
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return "\n\n".join(buf)


def filler_text():
    # 中英混合填充,避免分词器把重复 token 折叠得过狠
    base = (
        "In a distributed inference system, prefill and decode stages are "
        "disaggregated to maximize GPU utilization and throughput. "
        "在大规模推理服务中,KV cache 的传输、调度与 backpressure 控制是核心难点。 "
    )
    return base


def pad_to_exact(text, tokenizer, target):
    """把 text 精确裁剪/填充到 target 个 token。"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) >= target:
        ids = ids[:target]
    else:
        f_ids = tokenizer(filler_text(), add_special_tokens=False)["input_ids"]
        while len(ids) < target:
            ids += f_ids
        ids = ids[:target]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=131072,
                    help="单条 prompt 的目标 token 数,默认 128k=131072")
    ap.add_argument("--tokenizer", required=True,
                    help="模型本地路径(建议本地目录,含 tokenizer.json)")
    ap.add_argument("--src", default=None,
                    help="可选:原始 ShareGPT json,用于抽真实文本")
    ap.add_argument("--out", default="sharegpt_128k.json")
    ap.add_argument("--num", type=int, default=1, help="生成几条")
    ap.add_argument("--output-tokens", type=int, default=512,
                    help="gpt 段(期望输出)token 数")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)

    dataset = []
    for i in range(args.num):
        if args.src:
            raw = collect_text_from_sharegpt(args.src, tok, args.target_tokens)
        else:
            raw = filler_text() * (args.target_tokens // 10 + 1)
        prompt, n_in = pad_to_exact(raw, tok, args.target_tokens)
        gpt_txt, n_out = pad_to_exact(filler_text(), tok, args.output_tokens)
        dataset.append({
            "id": f"long128k_{i}",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": gpt_txt},
            ],
        })
        print(f"[{i}] prompt={n_in} tok, output={n_out} tok", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    print(f"已写出 {args.out},共 {len(dataset)} 条", file=sys.stderr)


if __name__ == "__main__":
    main()

