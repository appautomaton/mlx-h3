"""Load the real quantized DiT and run one denoising step under the budget.

Not a correctness check -- there is no fixture for the assembled model. It
answers three questions the unit tests cannot: does the checkpoint load, does a
step stay inside the memory budget without paging, and how long does a block
take at the shapes we intend to serve.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from mlx_h3 import layout, loading, memory, model as h3_model, sampler

SHAPES = {
    # 864x480, 56 frames -- the iteration config
    "dev": dict(width=864, height=480, frames=56),
    # 1344x768, 124 frames -- on spec
    "spec": dict(width=1344, height=768, frames=124),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=list(SHAPES), default="dev")
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-velocity",
        type=Path,
        help="optional untracked safetensors output for numerical A/B checks",
    )
    ap.add_argument("--dit", default="weights/mlx-8bit/dit_fl2va_a8g32.safetensors")
    ap.add_argument("--turbo-lora")
    ap.add_argument(
        "--nax-group-size",
        type=int,
        choices=(64, 256, 448, 896),
        help="experimental M5 W8A8 group size; default keeps MLX W8A16",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="schedule steps (default: 20 base, 6 with Turbo LoRA)",
    )
    args = ap.parse_args()
    sampling = (
        sampler.TURBO_PROFILE
        if args.turbo_lora is not None
        else sampler.BASE_PROFILE
    )
    steps = sampling.resolve_steps(args.steps)

    memory.configure(args.budget)
    print(memory.report("start        "))

    spec = SHAPES[args.shape]
    frame_count, latent_t, audio_t = layout.temporal_shape(spec["frames"])
    latent_h, latent_w = layout.latent_canvas(spec["width"], spec["height"])
    packed = layout.pack(
        text_len=args.text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        frame_count=frame_count,
    )
    print(
        f"  {args.shape}: {frame_count} frames {spec['width']}x{spec['height']} -> "
        f"latent {latent_t}x{latent_h}x{latent_w}, audio_t {audio_t}, "
        f"seq_len {packed.seq_len}"
    )

    guard = memory.Guard(f"one_step/{args.shape}", args.budget)
    sigmas = sampler.schedule(steps)
    plans = tuple(
        h3_model.plan(packed, sigma_video, sigma_audio=sigma_audio)
        for sigma_video, sigma_audio in zip(
            sigmas.video[:-1], sigmas.audio[:-1], strict=True
        )
    )
    t0 = time.perf_counter()
    dit = loading.load_dit(
        args.dit,
        plans=plans,
        modulation_dtype=mx.bfloat16,
        adapter_path=args.turbo_lora,
        nax_group_size=args.nax_group_size,
    )
    guard.check("after load")
    print(memory.report(f"loaded {time.perf_counter() - t0:5.1f}s  "))

    cfg = dit.config
    mx.random.seed(args.seed)
    video = mx.random.normal((1, cfg.latents_dim, latent_t, latent_h, latent_w))
    audio = mx.random.normal((1, cfg.audio_latents_dim, 2, audio_t))
    text = mx.random.normal((args.text_len, cfg.hidden_size), dtype=mx.bfloat16)
    mx.eval(video, audio, text)

    marks: list[float] = []

    def on_block(i: int) -> None:
        marks.append(time.perf_counter())
        if i % 10 == 0 or i == cfg.num_layers - 1:
            guard.check(f"block {i}")

    t0 = time.perf_counter()
    marks.append(t0)
    video_velocity, audio_velocity = dit(
        video,
        audio,
        text,
        packed,
        sigma_video=sigmas.video[0],
        sigma_audio=sigmas.audio[0],
        step_index=0,
        on_block=on_block,
    )
    mx.eval(video_velocity, audio_velocity)
    total = time.perf_counter() - t0

    per_block = [b - a for a, b in zip(marks, marks[1:], strict=False)]
    print(memory.report("after step   "))
    print(
        f"  step {total:6.2f}s   block min {min(per_block) * 1e3:6.0f} ms "
        f"median {sorted(per_block)[len(per_block) // 2] * 1e3:6.0f} ms"
    )
    model_evaluations = steps
    print(f"  -> {model_evaluations} model evaluations would be {total * model_evaluations / 60:.1f} min")
    print(
        f"  velocity video {video_velocity.shape} {video_velocity.dtype}, "
        f"audio {audio_velocity.shape} {audio_velocity.dtype}"
    )
    finite = (
        mx.isfinite(video_velocity).all().item()
        and mx.isfinite(audio_velocity).all().item()
    )
    print(f"  finite: {finite}   |v| max {mx.abs(video_velocity).max().item():.4f}")
    if args.save_velocity is not None:
        args.save_velocity.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(args.save_velocity),
            {"video": video_velocity, "audio": audio_velocity},
            metadata={
                "seed": str(args.seed),
                "nax_group_size": str(args.nax_group_size),
            },
        )
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
