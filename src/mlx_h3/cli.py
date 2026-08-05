"""Command-line entry point for staged MiniMax-H3 generation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import memory, output, pipeline, sampler


class _ReferenceAction(argparse.Action):
    """Append one typed reference while preserving cross-option CLI order."""

    def __init__(self, *args, reference_kind: str, **kwargs):
        self.reference_kind = reference_kind
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        references = list(getattr(namespace, self.dest) or ())
        if self.reference_kind == "image":
            reference = pipeline.Reference(image=values)
        elif self.reference_kind == "video":
            reference = pipeline.Reference(video=values)
        elif self.reference_kind == "silent_video":
            reference = pipeline.Reference(video=values, include_video_audio=False)
        elif self.reference_kind == "video_audio":
            reference = pipeline.Reference(video=values[0], audio=values[1])
        else:
            reference = pipeline.Reference(audio=values)
        references.append(reference)
        setattr(namespace, self.dest, references)


def _gib(value: int) -> float:
    return value / memory.GIB


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-h3", description="Generate synchronized video and audio with MiniMax-H3."
    )
    parser.add_argument("prompt", nargs="?", help="generation prompt")
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="read the generation prompt from a UTF-8 file",
    )
    parser.add_argument("--output", default="outputs/minimax-h3.mp4")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=sampler.DEFAULT_STEPS)
    parser.add_argument("--first-frame")
    parser.add_argument("--last-frame")
    parser.add_argument(
        "--ref-image",
        dest="references",
        action=_ReferenceAction,
        reference_kind="image",
        metavar="PATH",
        help="append a Ref2VA image at this point in presentation order",
    )
    parser.add_argument(
        "--ref-image-size",
        choices=("match", "max"),
        default="match",
        help="down-only reference sizing policy (default: match output pixel area)",
    )
    parser.add_argument(
        "--ref-video",
        dest="references",
        action=_ReferenceAction,
        reference_kind="video",
        metavar="PATH",
        help="append a video and its embedded soundtrack when present",
    )
    parser.add_argument(
        "--ref-video-silent",
        dest="references",
        action=_ReferenceAction,
        reference_kind="silent_video",
        metavar="PATH",
        help="append a video without conditioning on its embedded soundtrack",
    )
    parser.add_argument(
        "--ref-video-with-audio",
        dest="references",
        action=_ReferenceAction,
        reference_kind="video_audio",
        nargs=2,
        metavar=("VIDEO", "AUDIO"),
        help="append a video with an explicit soundtrack override",
    )
    parser.add_argument(
        "--ref-audio",
        dest="references",
        action=_ReferenceAction,
        reference_kind="audio",
        metavar="PATH",
        help="append a standalone audio reference at this point",
    )
    parser.add_argument("--budget", type=int, default=memory.BUDGET_GIB)
    parser.add_argument("--tokenizer", default=pipeline.ModelPaths.tokenizer)
    parser.add_argument("--text-encoder", default=pipeline.ModelPaths.text_encoder)
    parser.add_argument("--dit", default=pipeline.ModelPaths.dit)
    parser.add_argument(
        "--ref-dit",
        default=pipeline.ModelPaths.ref_dit,
        help="dedicated Ref2VA DiT checkpoint",
    )
    parser.add_argument("--video-vae", default=pipeline.ModelPaths.video_vae)
    parser.add_argument("--audio-vae", default=pipeline.ModelPaths.audio_vae)
    return parser


def _resolve_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if (prompt is None) == (prompt_file is None):
        raise ValueError("provide exactly one of PROMPT or --prompt-file")
    if prompt_file is not None:
        source = Path(prompt_file).expanduser()
        if not source.is_file():
            raise ValueError(f"prompt file does not exist: {source}")
        try:
            prompt = source.read_text(encoding="utf-8").rstrip("\r\n")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"could not read prompt file {source}: {error}") from error
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    return prompt


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        prompt = _resolve_prompt(args.prompt, args.prompt_file)
    except ValueError as error:
        parser.error(str(error))

    config = pipeline.GenerationConfig(
        prompt=prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        seed=args.seed,
        steps=args.steps,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        references=tuple(args.references or ()),
        ref_image_size=args.ref_image_size,
    )
    paths = pipeline.ModelPaths(
        tokenizer=args.tokenizer,
        text_encoder=args.text_encoder,
        dit=args.dit,
        ref_dit=args.ref_dit,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
    )
    try:
        paths.validate(ref2va=bool(config.references))
    except FileNotFoundError as error:
        parser.error(str(error))
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
