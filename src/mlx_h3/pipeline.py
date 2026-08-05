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

from . import (
    layout,
    loading,
    media,
    memory,
    model as h3_model,
    sampler,
    text_encoder,
    tokenizer,
)

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
    ref_dit: str | Path = "weights/mlx-8bit/dit_ref2va_a8g32.safetensors"
    video_vae: str | Path = (
        "weights/bf16/vae/minimax_h3_video_vae_fp16.safetensors"
    )
    audio_vae: str | Path = (
        "weights/bf16/vae/minimax_h3_audio_vae_fp32.safetensors"
    )

    def validate(self, *, ref2va: bool) -> None:
        """Fail before model loading when a required local asset is absent."""
        required = {
            "tokenizer": self.tokenizer,
            "text encoder": self.text_encoder,
            "Ref2VA DiT" if ref2va else "FL2VA DiT": (
                self.ref_dit if ref2va else self.dit
            ),
            "Video VAE": self.video_vae,
            "Audio VAE": self.audio_vae,
        }
        missing = [
            f"{label}: {Path(path).expanduser()}"
            for label, path in required.items()
            if not Path(path).expanduser().is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing required model files:\n  " + "\n  ".join(missing)
            )


@dataclass(frozen=True)
class Reference:
    """One Ref2VA input; request order is part of the model contract."""

    image: str | Path | None = None
    video: str | Path | None = None
    audio: str | Path | None = None
    include_video_audio: bool = True

    def __post_init__(self) -> None:
        present = tuple(
            name
            for name in ("image", "video", "audio")
            if getattr(self, name) is not None
        )
        if present not in (("image",), ("video",), ("audio",), ("video", "audio")):
            raise ValueError(
                "a reference must contain image, video, audio, or video plus audio"
            )
        if not isinstance(self.include_video_audio, bool):
            raise ValueError("include_video_audio must be a boolean")

    @property
    def kind(self) -> str:
        if self.image is not None:
            return "image"
        return "video" if self.video is not None else "audio"


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    width: int = 864
    height: int = 480
    frames: int = 56
    seed: int = 42
    steps: int = sampler.DEFAULT_STEPS
    max_prompt_tokens: int = 4096
    first_frame: str | Path | None = None
    last_frame: str | Path | None = None
    references: tuple[Reference, ...] = ()
    ref_image_size: media.ReferenceImageSize = "match"

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
        if not all(isinstance(reference, Reference) for reference in self.references):
            raise ValueError("references must contain Reference values")
        image_count = sum(reference.kind == "image" for reference in self.references)
        video_count = sum(reference.kind == "video" for reference in self.references)
        explicit_audio_count = sum(
            reference.audio is not None for reference in self.references
        )
        if image_count > 9:
            raise ValueError("Ref2VA supports at most 9 reference images")
        if video_count > 3:
            raise ValueError("Ref2VA supports at most 3 reference videos")
        if explicit_audio_count > 3:
            raise ValueError("Ref2VA supports at most 3 audio references")
        if self.ref_image_size not in ("match", "max"):
            raise ValueError(
                f"unsupported reference image size policy: {self.ref_image_size}"
            )
        if self.references and (
            self.first_frame is not None or self.last_frame is not None
        ):
            raise ValueError("references cannot be combined with frame anchors")
        if self.references and all(
            reference.kind == "audio" for reference in self.references
        ):
            raise ValueError("reference audio requires an image or video reference")
        if len(self.references) > 12:
            raise ValueError("Ref2VA supports at most 12 references")


@dataclass(frozen=True)
class GeneratedMedia:
    frames: mx.array
    audio: mx.array
    fps: int
    sample_rate: int
    seed: int
    prompt_tokens: int
    sequence_length: int


@dataclass(frozen=True)
class _PreparedReference:
    kind: str
    pixels: mx.array | None
    waveform: mx.array | None


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
    """Run generation with exactly one resident model per phase."""
    ref2va = bool(config.references)
    paths.validate(ref2va=ref2va)
    dit_path = paths.ref_dit if ref2va else paths.dit

    tok = tokenizer.QwenTokenizer.from_file(paths.tokenizer)
    raw_token_ids = tok.encode_prompt(config.prompt)
    if len(raw_token_ids) > config.max_prompt_tokens:
        raise ValueError(
            f"prompt has {len(raw_token_ids)} tokens, limit is {config.max_prompt_tokens}"
        )

    frame_count, latent_t, audio_t = layout.temporal_shape(config.frames)
    latent_h, latent_w = layout.latent_canvas(config.width, config.height)
    keyframes = []
    keyframe_pixels = []
    if config.first_frame is not None:
        keyframe_pixels.append(
            media.load_rgb_image(
                config.first_frame,
                width=config.width,
                height=config.height,
                fit="stretch",
            )
        )
        keyframes.append(layout.Keyframe(0))
    if config.last_frame is not None:
        keyframe_pixels.append(
            media.load_rgb_image(
                config.last_frame,
                width=config.width,
                height=config.height,
                fit="cover",
            )
        )
        keyframes.append(layout.Keyframe(frame_count - 1))

    refs = []
    prepared_references = []
    for reference in config.references:
        pixels = waveform = None
        if reference.kind == "image":
            ref_width, ref_height = media.reference_image_canvas(
                reference.image,
                target_width=config.width,
                target_height=config.height,
                size=config.ref_image_size,
            )
            pixels = media.load_rgb_image(
                reference.image,
                width=ref_width,
                height=ref_height,
                fit="stretch",
            )
        elif reference.kind == "video":
            ref_width, ref_height = media.reference_video_canvas(reference.video)
            pixels = media.load_rgb_video(
                reference.video,
                width=ref_width,
                height=ref_height,
                max_frames=frame_count,
            )
            soundtrack = reference.audio
            if (
                soundtrack is None
                and reference.include_video_audio
                and media.has_audio_stream(reference.video)
            ):
                soundtrack = reference.video
            if soundtrack is not None:
                waveform = media.load_stereo_audio(
                    soundtrack,
                    max_seconds=pixels.shape[2] / layout.FPS,
                )
        else:
            waveform = media.load_stereo_audio(reference.audio)
        prepared_references.append(
            _PreparedReference(reference.kind, pixels, waveform)
        )

    ref_audio_waveforms = tuple(
        reference.waveform
        for reference in prepared_references
        if reference.waveform is not None
    )
    if len(ref_audio_waveforms) > 3:
        raise ValueError("Ref2VA supports at most 3 audio references")
    total_audio_seconds = sum(
        waveform.shape[-1] / media.AUDIO_SAMPLE_RATE
        for waveform in ref_audio_waveforms
    )
    if total_audio_seconds > media.REFERENCE_AUDIO_MAX_SECONDS:
        raise ValueError(
            "total reference audio duration exceeds "
            f"{media.REFERENCE_AUDIO_MAX_SECONDS:g} seconds"
        )

    cond_video_latents = ()
    cond_audio_latents = ()
    if ref2va:
        condition_pixels = tuple(
            reference.pixels
            for reference in prepared_references
            if reference.pixels is not None
        )
    else:
        condition_pixels = tuple(keyframe_pixels)
    if condition_pixels:
        cond_video_latents = run_phase(
            "video VAE encoder",
            lambda: loading.load_video_vae_encoder(paths.video_vae),
            lambda encoder: tuple(encoder(image) for image in condition_pixels),
            guard,
            on_report=on_report,
        )
        if ref2va:
            for latent in cond_video_latents:
                if latent.ndim != 5 or latent.shape[:2] != (1, 24):
                    raise ValueError(
                        "reference visual latent must have shape [1,24,T,H,W]"
                    )

        if ref_audio_waveforms:
            cond_audio_latents = run_phase(
                "audio VAE encoder",
                lambda: loading.load_audio_vae_encoder(paths.audio_vae),
                lambda encoder: tuple(
                    encoder(waveform) for waveform in ref_audio_waveforms
                ),
                guard,
                on_report=on_report,
            )
            for latent in cond_audio_latents:
                if latent.ndim != 4 or latent.shape[:3] != (1, 32, 2):
                    raise ValueError(
                        "reference audio latent must have shape [1,32,2,T]"
                    )

        refs = []
        presentations = []
        visual_latents = iter(cond_video_latents)
        audio_latents = iter(cond_audio_latents)
        for reference in prepared_references:
            visual_latent = (
                next(visual_latents) if reference.pixels is not None else None
            )
            audio_latent = (
                next(audio_latents) if reference.waveform is not None else None
            )
            presentations.append(
                text_encoder.ReferencePresentation(
                    reference.kind,
                    reference.pixels,
                    has_audio=audio_latent is not None,
                )
            )
            if reference.kind == "image":
                if visual_latent.shape[-3] != 1:
                    raise ValueError("reference image latent must have T=1")
                refs.append(
                    layout.RefImage(visual_latent.shape[-2], visual_latent.shape[-1])
                )
            elif reference.kind == "video":
                refs.append(
                    layout.RefVideo(
                        visual_latent.shape[-3],
                        visual_latent.shape[-2],
                        visual_latent.shape[-1],
                        audio_t=0 if audio_latent is None else audio_latent.shape[-1],
                    )
                )
            else:
                refs.append(layout.RefAudio(audio_latent.shape[-1]))

        def encode_multimodal(encoder):
            if ref2va:
                return encoder.encode_ref_references(
                    tok,
                    config.prompt,
                    tuple(presentations),
                )
            return encoder.encode_fl2va(
                tok,
                config.prompt,
                tuple(condition_pixels),
            )

        text_states, text_tag_array = run_phase(
            "text/vision encoder",
            lambda: loading.load_multimodal_text_encoder(paths.text_encoder),
            encode_multimodal,
            guard,
            on_report=on_report,
        )
        text_tags = tuple(int(tag) for tag in text_tag_array.tolist())
        keyframe_pixels = condition_pixels = prepared_references = None
        presentations = ref_audio_waveforms = None
        text_tag_array = None
        if text_states.shape[0] > config.max_prompt_tokens:
            raise ValueError(
                f"multimodal presentation has {text_states.shape[0]} tokens, "
                f"limit is {config.max_prompt_tokens}"
            )
    else:
        text_states = run_phase(
            "text encoder",
            lambda: loading.load_text_encoder(paths.text_encoder),
            lambda encoder: encoder(
                mx.array([raw_token_ids], dtype=mx.int32)
            )[0],
            guard,
            on_report=on_report,
        )
        text_tags = None

    text_length = text_states.shape[-2]
    packed = layout.pack(
        text_len=text_length,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        keyframes=tuple(keyframes),
        refs=tuple(refs),
        frame_count=frame_count,
    )

    # One native MLX RNG stream, in the reference's draw order.
    mx.random.seed(config.seed)
    video_noise = mx.random.normal((1, 24, latent_t, latent_h, latent_w))
    audio_noise = mx.random.normal((1, 32, 2, audio_t))
    mx.eval(video_noise, audio_noise)
    cond_video_rows = None
    if cond_video_latents:
        rows = []
        for latent in cond_video_latents:
            value = h3_model.patchify_video(
                latent.astype(mx.float32),
                h3_model.H3Config().patch_size,
            )
            noise = mx.random.normal(value.shape, key=mx.random.key(config.seed))
            aug = layout.VISUAL_COND_TIMESTEP
            value = aug * value + (1.0 - aug) * noise
            mx.eval(value)
            rows.append(value)
        cond_video_rows = mx.concatenate(rows, axis=0)
        mx.eval(cond_video_rows)
    cond_audio_rows = None
    if cond_audio_latents:
        cond_audio_rows = mx.concatenate(
            [
                h3_model.pack_audio(latent.astype(mx.float32))
                for latent in cond_audio_latents
            ],
            axis=0,
        )
        mx.eval(cond_audio_rows)
    sigmas = sampler.schedule(config.steps)
    step_plans = tuple(
        h3_model.plan(
            packed,
            sigma_video,
            sigma_audio=sigma_audio,
            text_tags=text_tags,
        )
        for sigma_video, sigma_audio in zip(
            sigmas.video[:-1], sigmas.audio[:-1], strict=True
        )
    )

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
            cond_video_rows=cond_video_rows,
            cond_audio_rows=cond_audio_rows,
            text_tags=text_tags,
            guard=guard,
            on_step=on_step,
        )

    video_latent, audio_latent = run_phase(
        "DiT",
        lambda: loading.load_dit(
            dit_path,
            plans=step_plans,
            modulation_dtype=text_states.dtype,
        ),
        run_dit,
        guard,
        on_report=on_report,
    )
    text_states = video_noise = audio_noise = cond_video_rows = cond_audio_rows = None
    cond_video_latents = cond_audio_latents = None
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
        prompt_tokens=text_length,
        sequence_length=packed.seq_len,
    )
