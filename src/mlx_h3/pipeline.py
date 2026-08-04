"""Staged model residency for the MiniMax-H3 inference pipeline.

The phase primitive enforces the one invariant the full pipeline cannot
retrofit: a model is loaded, used, and released before the next model is loaded.

Safety checks always run. Optional instrumentation receives scalar-only reports
and disappears from the release path when no callback is supplied.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import mlx.core as mx

from . import layout, loading, memory, sampler, tokenizer

ModelT = TypeVar("ModelT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class PhaseReport:
    """Scalar-only optional telemetry from one fully released model phase."""

    label: str
    load_seconds: float
    run_seconds: float
    release_seconds: float
    active_after_load: int
    active_after_run: int
    active_after_release: int
    peak: int


@dataclass(frozen=True)
class ModelPaths:
    tokenizer: str | Path = "weights/tokenizer/tokenizer.json"
    text_encoder: str | Path = "weights/mlx-8bit/te_qwen3vl_a8g32.safetensors"
    dit: str | Path = "weights/mlx-8bit/dit_fl2va_a8g32.safetensors"
    video_vae: str | Path = (
        "weights/bf16/vae/minimax_h3_video_vae_fp16.safetensors"
    )
    audio_vae: str | Path = (
        "weights/bf16/vae/minimax_h3_audio_vae_fp32.safetensors"
    )


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    width: int = 864
    height: int = 480
    frames: int = 56
    seed: int = 42
    steps: int = sampler.DEFAULT_STEPS
    max_prompt_tokens: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if self.width < 32 or self.height < 32:
            raise ValueError("width and height must be at least 32")
        if self.width % layout.CANVAS_MULTIPLE or self.height % layout.CANVAS_MULTIPLE:
            raise ValueError(
                f"width and height must be multiples of {layout.CANVAS_MULTIPLE}"
            )
        if self.width * self.height > layout.MAX_PIXELS:
            raise ValueError(
                f"canvas {self.width}x{self.height} exceeds the {layout.MAX_PIXELS} pixel limit"
            )
        if self.frames < 5:
            raise ValueError("frames must be at least 5")
        if layout.align_frame_count(self.frames) > 362:
            raise ValueError("aligned frame count exceeds the released 15 second limit")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.steps < 1 or self.steps > 1000:
            raise ValueError("steps must be in [1, 1000]")
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")


@dataclass(frozen=True)
class GeneratedMedia:
    frames: mx.array
    audio: mx.array
    fps: int
    sample_rate: int
    seed: int
    prompt_tokens: int
    sequence_length: int


def run_phase(
    label: str,
    load: Callable[[], ModelT],
    execute: Callable[[ModelT], OutputT],
    guard: memory.Guard,
    *,
    on_report: Callable[[PhaseReport], None] | None = None,
) -> OutputT:
    """Load one model, materialize its output, then release it completely.

    The returned value must contain only artifacts needed by later phases, never
    the model itself. ``execute`` receives the model explicitly so it need not
    capture it in a closure.
    """
    started = time.perf_counter() if on_report is not None else 0.0
    model: ModelT | None = None
    try:
        model = load()
        after_load = guard.check(f"{label} / loaded")
        loaded = time.perf_counter() if on_report is not None else 0.0

        output = execute(model)
        mx.eval(output)
        after_run = guard.check(f"{label} / complete")
        completed = time.perf_counter() if on_report is not None else 0.0
    finally:
        model = None
        gc.collect()
        memory.release()

    after_release = guard.check(f"{label} / released")
    if on_report is not None:
        released = time.perf_counter()
        on_report(
            PhaseReport(
                label=label,
                load_seconds=loaded - started,
                run_seconds=completed - loaded,
                release_seconds=released - completed,
                active_after_load=after_load.active,
                active_after_run=after_run.active,
                active_after_release=after_release.active,
                peak=max(after_load.peak, after_run.peak, after_release.peak),
            )
        )
    return output


def generate(
    config: GenerationConfig,
    paths: ModelPaths,
    guard: memory.Guard,
    *,
    on_step: Callable[[int, int, float, float], None] | None = None,
    on_report: Callable[[PhaseReport], None] | None = None,
) -> GeneratedMedia:
    """Run text-to-video-and-audio with exactly one resident model per phase."""
    tok = tokenizer.QwenTokenizer.from_file(paths.tokenizer)
    token_ids = tok.encode_prompt(config.prompt)
    if len(token_ids) > config.max_prompt_tokens:
        raise ValueError(
            f"prompt has {len(token_ids)} tokens, limit is {config.max_prompt_tokens}"
        )

    text_states = run_phase(
        "text encoder",
        lambda: loading.load_text_encoder(paths.text_encoder),
        lambda model: model(mx.array([token_ids], dtype=mx.int32))[0],
        guard,
        on_report=on_report,
    )

    frame_count, latent_t, audio_t = layout.temporal_shape(config.frames)
    latent_h, latent_w = layout.latent_canvas(config.width, config.height)
    packed = layout.pack(
        text_len=len(token_ids),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        frame_count=frame_count,
    )

    # One native MLX RNG stream, in the reference's draw order.
    mx.random.seed(config.seed)
    video_noise = mx.random.normal((1, 24, latent_t, latent_h, latent_w))
    audio_noise = mx.random.normal((1, 32, 2, audio_t))
    mx.eval(video_noise, audio_noise)
    sigmas = sampler.schedule(config.steps)

    def run_dit(model):
        refined_text = model.refine_text(text_states)
        mx.eval(refined_text)
        return sampler.denoise(
            model,
            video_noise,
            audio_noise,
            refined_text,
            packed,
            sigmas,
            guard=guard,
            on_step=on_step,
        )

    video_latent, audio_latent = run_phase(
        "DiT",
        lambda: loading.load_dit(paths.dit),
        run_dit,
        guard,
        on_report=on_report,
    )
    text_states = video_noise = audio_noise = None
    gc.collect()
    memory.release()

    frames = run_phase(
        "video VAE",
        lambda: loading.load_video_vae(paths.video_vae),
        lambda model: model(video_latent),
        guard,
        on_report=on_report,
    )
    video_latent = None
    gc.collect()
    memory.release()

    audio = run_phase(
        "audio VAE",
        lambda: loading.load_audio_vae(paths.audio_vae),
        lambda model: model(audio_latent),
        guard,
        on_report=on_report,
    )
    audio_latent = None
    gc.collect()
    memory.release()
    guard.check("generation complete")
    if not mx.isfinite(frames).all().item():
        raise ValueError("video VAE produced non-finite frames")
    if not mx.isfinite(audio).all().item():
        raise ValueError("audio VAE produced a non-finite waveform")

    return GeneratedMedia(
        frames=frames,
        audio=audio,
        fps=layout.FPS,
        sample_rate=32_000,
        seed=config.seed,
        prompt_tokens=len(token_ids),
        sequence_length=packed.seq_len,
    )
