"""Quantize the bf16 MiniMax-H3 release to MLX affine weights.

Reads one tensor at a time by seeking into the safetensors payload, so the 61.7 GiB
DiT never lands in RAM whole: peak is the largest single tensor plus the output
being accumulated.

What stays dense, and why it is not a name whitelist:

  * F32 tensors. The checkpoint marks its own precision-sensitive tensors as F32 --
    patch projections, time embedder, final output heads, rope inv_freq. Filtering
    by dtype is exact; a name list copied from another framework is not, because
    this repack uses ComfyUI naming.
  * Anything under 2 dimensions, or whose last axis is not divisible by group_size.
  * Gathered lookup tables (embed_tokens, pos_embed). A shape test happily packs
    these, and the gather then returns uint32 garbage -- shape cannot decide this
    case, so it is an explicit list.

Usage:
    uv run python dev/quantize.py weights/bf16/<file>.safetensors weights/mlx-8bit/<name>.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

GROUP_SIZE = 32
BITS = 8

#: Read with take_axis at runtime; packing them yields garbage after the gather.
DENSE_SUBSTRINGS = ("embed_tokens", "pos_embed", "embedding")

_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I32": np.int32}


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def read_tensor(f, base: int, meta: dict) -> mx.array:
    start, stop = meta["data_offsets"]
    f.seek(base + start)
    raw = f.read(stop - start)
    dtype = meta["dtype"]
    arr = np.frombuffer(raw, dtype=_NP[dtype]).reshape(meta["shape"])
    out = mx.array(arr)
    return out.view(mx.bfloat16) if dtype == "BF16" else out


def should_quantize(name: str, meta: dict) -> tuple[bool, str]:
    if meta["dtype"] != "BF16":
        return False, f"dtype {meta['dtype']}"
    if not name.endswith(".weight"):
        # nn.QuantizedLinear names its packed parameter `weight` and puts the
        # scales beside it; anything else has nowhere to hang them.
        return False, "not a .weight"
    if any(s in name for s in DENSE_SUBSTRINGS):
        return False, "lookup table"
    shape = meta["shape"]
    if len(shape) != 2:
        return False, f"rank {len(shape)}"
    if shape[-1] % GROUP_SIZE:
        return False, f"last dim {shape[-1]} % {GROUP_SIZE}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--bits", type=int, default=BITS)
    args = ap.parse_args()

    header, base = read_header(args.src)
    order = sorted(header, key=lambda k: header[k]["data_offsets"][0])
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    out: dict[str, mx.array] = {}
    n_q = n_d = 0
    bytes_in = bytes_out = 0
    skipped: dict[str, int] = {}
    t0 = time.perf_counter()

    with args.src.open("rb") as f:
        for i, name in enumerate(order):
            meta = header[name]
            w = read_tensor(f, base, meta)
            bytes_in += w.nbytes

            ok, why = should_quantize(name, meta)
            if ok:
                q, s, b = mx.quantize(w, group_size=args.group_size, bits=args.bits)
                mx.eval(q, s, b)
                # Sibling names, not children of `weight`: that is where
                # nn.QuantizedLinear looks for them.
                module = name.removesuffix(".weight")
                out[name] = q
                out[f"{module}.scales"] = s
                out[f"{module}.biases"] = b
                bytes_out += q.nbytes + s.nbytes + b.nbytes
                n_q += 1
            else:
                mx.eval(w)
                out[name] = w
                bytes_out += w.nbytes
                n_d += 1
                skipped[why] = skipped.get(why, 0) + 1

            if (i + 1) % 100 == 0 or i + 1 == len(order):
                print(
                    f"\r  {i + 1}/{len(order)} tensors  "
                    f"{bytes_in / 2**30:6.1f} -> {bytes_out / 2**30:6.1f} GiB  "
                    f"{time.perf_counter() - t0:5.0f}s",
                    end="",
                    flush=True,
                )

    print(f"\n  quantized {n_q}, kept dense {n_d}")
    for why, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"    dense: {count:4d}  {why}")

    meta_out = {
        "quantization.mode": "affine",
        "quantization.bits": str(args.bits),
        "quantization.group_size": str(args.group_size),
        "source": args.src.name,
    }
    mx.save_safetensors(str(args.dst), out, metadata=meta_out)
    size = args.dst.stat().st_size
    print(
        f"  wrote {args.dst}  {size / 2**30:.1f} GiB  "
        f"({size / bytes_in * 100:.1f}% of source)  {time.perf_counter() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
