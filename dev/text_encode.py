"""Load the real quantized Qwen conditioner and encode one raw prompt."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_h3 import loading, memory, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        help="Text to encode. No prompt content is stored by this utility.",
    )
    parser.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    parser.add_argument(
        "--tokenizer",
        default="weights/tokenizer/tokenizer.json",
    )
    parser.add_argument(
        "--text-encoder",
        default="weights/mlx-8bit/te_qwen3vl_a8g32.safetensors",
    )
    args = parser.parse_args()

    tok = tokenizer.QwenTokenizer.from_file(args.tokenizer)
    token_ids = tok.encode_prompt(args.prompt)
    memory.configure(args.budget)
    guard = memory.Guard("text_encode", args.budget)
    print(f"  prompt tokens {len(token_ids)}")
    print(memory.report("start        "))

    started = time.perf_counter()
    model = loading.load_text_encoder(args.text_encoder)
    guard.check("loaded")
    print(memory.report(f"loaded {time.perf_counter() - started:5.1f}s  "))

    started = time.perf_counter()
    states = model(mx.array([token_ids], dtype=mx.int32))
    mx.eval(states)
    elapsed = time.perf_counter() - started
    guard.check("encoded")
    print(memory.report("encoded      "))
    finite = mx.isfinite(states).all().item()
    print(f"  states {states.shape} {states.dtype} in {elapsed:.2f}s  finite: {finite}")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
