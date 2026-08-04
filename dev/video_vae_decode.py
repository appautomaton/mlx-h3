"""Load the real visual VAE decoder and run a bounded latent decode."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_h3 import loading, memory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, choices=(5, 22, 56, 124), default=5)
    parser.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    parser.add_argument(
        "--vae",
        default="weights/bf16/vae/minimax_h3_video_vae_fp16.safetensors",
    )
    args = parser.parse_args()
    if args.width % 16 or args.height % 16:
        parser.error("--width and --height must be divisible by 16")

    latent_t = 2 if args.frames == 5 else ((args.frames - 5) // 17) * 5 + 2
    latent_height, latent_width = args.height // 16, args.width // 16
    memory.configure(args.budget)
    guard = memory.Guard("video_vae_decode", args.budget)
    print(memory.report("start        "))

    started = time.perf_counter()
    vae = loading.load_video_vae(args.vae)
    guard.check("loaded")
    print(memory.report(f"loaded {time.perf_counter() - started:5.1f}s  "))

    latent = mx.random.normal((1, 24, latent_t, latent_height, latent_width))
    mx.eval(latent)
    started = time.perf_counter()
    frames = vae(latent)
    mx.eval(frames)
    elapsed = time.perf_counter() - started
    guard.check("decoded")
    print(memory.report("decoded      "))
    print(
        f"  latent {latent.shape} -> frames {frames.shape} {frames.dtype} "
        f"in {elapsed:.2f}s"
    )
    finite = mx.isfinite(frames).all().item()
    print(
        f"  finite: {finite}  range [{mx.min(frames).item():.4f}, "
        f"{mx.max(frames).item():.4f}]"
    )
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
