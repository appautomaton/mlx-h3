"""Command-line entry point for staged MiniMax-H3 generation."""

from __future__ import annotations

import argparse
import time

from . import memory, output, pipeline, sampler


def _gib(value: int) -> float:
    return value / memory.GIB


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mlx-h3", description="Generate synchronized video and audio with MiniMax-H3."
    )
    parser.add_argument("prompt")
    parser.add_argument("--output", default="outputs/minimax-h3.mp4")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=sampler.DEFAULT_STEPS)
    parser.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    parser.add_argument("--tokenizer", default=pipeline.ModelPaths.tokenizer)
    parser.add_argument("--text-encoder", default=pipeline.ModelPaths.text_encoder)
    parser.add_argument("--dit", default=pipeline.ModelPaths.dit)
    parser.add_argument("--video-vae", default=pipeline.ModelPaths.video_vae)
    parser.add_argument("--audio-vae", default=pipeline.ModelPaths.audio_vae)
    args = parser.parse_args()

    config = pipeline.GenerationConfig(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        seed=args.seed,
        steps=args.steps,
    )
    paths = pipeline.ModelPaths(
        tokenizer=args.tokenizer,
        text_encoder=args.text_encoder,
        dit=args.dit,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
    )
    memory.configure(args.budget)
    guard = memory.Guard("generate", args.budget)
    started = time.perf_counter()
    print(memory.report("start        "), flush=True)

    def report(item: pipeline.PhaseReport) -> None:
        print(
            f"{item.label:12} load {item.load_seconds:6.1f}s  run {item.run_seconds:7.1f}s  "
            f"release {item.release_seconds:4.1f}s  active {_gib(item.active_after_run):4.1f}  "
            f"released {_gib(item.active_after_release):4.1f} GiB",
            flush=True,
        )

    step_started = time.perf_counter()

    def progress(done: int, total: int, sigma_video: float, sigma_audio: float) -> None:
        nonlocal step_started
        now = time.perf_counter()
        print(
            f"  step {done:2}/{total}  {now - step_started:6.1f}s  "
            f"sigma video {sigma_video:.5f} audio {sigma_audio:.5f}  "
            f"{memory.report()}",
            flush=True,
        )
        step_started = now

    media = pipeline.generate(
        config, paths, guard, on_step=progress, on_report=report
    )
    destination = output.mux_mp4(
        args.output,
        media.frames,
        media.audio,
        fps=media.fps,
        sample_rate=media.sample_rate,
    )
    guard.check("output written")
    print(
        f"wrote {destination}  tokens {media.prompt_tokens}  sequence {media.sequence_length}  "
        f"elapsed {(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
