#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 sglang.bench_serving 加一个 --repeat-to 开关:
把加载到的数据集(尤其是只有 1 条的 sharegpt 文件)循环复制到 N 条,
用于"固定一条 prompt 重复并发跑"(100% 前缀命中)的场景。

用法(--repeat-to 之后的所有参数原样透传给 bench_serving):
    python bench_repeat.py --repeat-to 200 -- \
        --backend sglang \
        --host 127.0.0.1 --port 8000 \
        --model /work/models --tokenizer /work/models \
        --dataset-name sharegpt \
        --dataset-path ./sharegpt_128k.json \
        --sharegpt-output-len 512 \
        --num-prompts 200 \
        --max-concurrency 40 \
        --request-rate inf

要点:
- --repeat-to 一般设成和 --num-prompts 相同(或更大)。
- 复制出来的条目内容完全一致 -> 共享同一 128k 前缀 -> 命中。
- 不改 sglang 任何源码,升级不受影响。
"""
import runpy
import sys


def _split_argv(argv):
    """从 argv 里抽出 --repeat-to N,其余原样返回给 bench_serving。
    支持 `--repeat-to 200 -- <rest>` 或 `--repeat-to 200 <rest>` 两种写法。
    """
    repeat_to = None
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--repeat-to":
            repeat_to = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--repeat-to="):
            repeat_to = int(a.split("=", 1)[1])
            i += 1
            continue
        if a == "--":           # 分隔符,跳过
            i += 1
            continue
        rest.append(a)
        i += 1
    return repeat_to, rest


def _install_patch(repeat_to):
    """包装 get_dataset:返回的行数不足 repeat_to 时循环补齐。"""
    import itertools

    import sglang.benchmark.datasets as ds

    orig_get_dataset = ds.get_dataset

    def _cycle_rows(rows, n):
        rows = list(rows)
        if not rows:
            print("[repeat] 警告:数据集为空,无法复制", file=sys.stderr)
            return rows
        if len(rows) >= n:
            return rows
        out = list(itertools.islice(itertools.cycle(rows), n))
        print(f"[repeat] 数据集从 {len(rows)} 条循环复制到 {len(out)} 条",
              file=sys.stderr)
        return out

    def patched_get_dataset(*args, **kwargs):
        result = orig_get_dataset(*args, **kwargs)
        # get_dataset 可能直接返回 list[DatasetRow],也可能返回 (rows, ...) 元组
        if isinstance(result, list):
            return _cycle_rows(result, repeat_to)
        if isinstance(result, tuple) and result and isinstance(result[0], list):
            return (_cycle_rows(result[0], repeat_to), *result[1:])
        print("[repeat] 警告:get_dataset 返回类型未识别,未做复制",
              file=sys.stderr)
        return result

    ds.get_dataset = patched_get_dataset


def main():
    repeat_to, rest = _split_argv(sys.argv[1:])
    if repeat_to is None:
        print("用法: python bench_repeat.py --repeat-to N -- <bench_serving 参数...>",
              file=sys.stderr)
        sys.exit(2)

    # 必须在 import/运行 bench_serving 之前打补丁,
    # 这样 bench_serving 里的 `from ... import get_dataset` 拿到的是包装版本。
    _install_patch(repeat_to)

    # 透传参数给 bench_serving 的 __main__
    sys.argv = ["sglang.bench_serving"] + rest
    runpy.run_module("sglang.bench_serving", run_name="__main__")


if __name__ == "__main__":
    main()
