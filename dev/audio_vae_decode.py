"""Load the real BigVGAN decoder and run a bounded stereo latent decode."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_h3 import layout, loading, memory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    parser.add_argument(
        "--vae",
        default="weights/bf16/vae/minimax_h3_audio_vae_fp32.safetensors",
    )
    args = parser.parse_args()

    frame_count, _, audio_t = layout.temporal_shape(args.frames)
    memory.configure(args.budget)
    guard = memory.Guard("audio_vae_decode", args.budget)
    print(memory.report("start        "))

    started = time.perf_counter()
    vae = loading.load_audio_vae(args.vae)
    guard.check("loaded")
    print(memory.report(f"loaded {time.perf_counter() - started:5.1f}s  "))

    latent = mx.random.normal((1, 32, 2, audio_t))
    mx.eval(latent)
    started = time.perf_counter()
    waveform = vae(latent)
    mx.eval(waveform)
    elapsed = time.perf_counter() - started
    guard.check("decoded")
    print(memory.report("decoded      "))
    print(
        f"  {frame_count} video frames -> latent {latent.shape} -> waveform "
        f"{waveform.shape} {waveform.dtype} in {elapsed:.2f}s"
    )
    finite = mx.isfinite(waveform).all().item()
    rms = mx.sqrt(mx.mean(waveform * waveform)).item()
    print(
        f"  finite: {finite}  range [{mx.min(waveform).item():.4f}, "
        f"{mx.max(waveform).item():.4f}]  rms {rms:.4f}"
    )
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
