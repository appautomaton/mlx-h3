"""Deterministic dual-schedule RES multistep sampling for MiniMax-H3.

VALIDATION. The schedule and solver are transcribed from Comfy-Org's official
MiniMax-H3 workflow (``simple``, 20 steps, ``res_multistep``) and current sampler
source. The reference is read statically, never executed.

MiniMax-H3 predicts a data-ward velocity for video and audio in one transformer
call. The official Comfy adapter maps the audio ODE onto the video sigma grid by
scaling its velocity with ``d(sigma_audio) / d(sigma_video)``. RES then uses
Euler for the first and last updates and a second-order multistep formula
between them. Eta is zero: it never injects noise after initialization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from . import layout, memory
from .layout import PackedLayout

DEFAULT_STEPS = 20


@dataclass(frozen=True)
class SigmaSchedule:
    """Paired video/audio sigma grids, including the terminal zero."""

    video: tuple[float, ...]
    audio: tuple[float, ...]
    video_shift: float = 1.0
    audio_shift: float = 1.0

    def __post_init__(self) -> None:
        if len(self.video) != len(self.audio) or len(self.video) < 2:
            raise ValueError("video and audio schedules must have the same length >= 2")
        for name, values in (("video", self.video), ("audio", self.audio)):
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} schedule must be finite")
            if values[0] != 1.0 or values[-1] != 0.0:
                raise ValueError(f"{name} schedule must start at 1 and end at 0")
            if not all(current > following for current, following in zip(values, values[1:])):
                raise ValueError(f"{name} schedule must be strictly decreasing")
        for name, value in (
            ("video_shift", self.video_shift),
            ("audio_shift", self.audio_shift),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def steps(self) -> int:
        """Number of model evaluations; terminal sigma zero has no forward."""
        return len(self.video) - 1


def _shifted_sigmas(num_steps: int, shift: float) -> tuple[float, ...]:
    if num_steps < 1 or num_steps > 1000:
        raise ValueError(f"num_steps must be in [1, 1000], got {num_steps}")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"shift must be positive and finite, got {shift}")

    # ComfyUI BasicScheduler("simple") indexes the model's 1000-point flow
    # schedule at floor(i * 1000 / steps), then appends terminal zero.
    base = mx.array(
        [
            (1000 - int(index * 1000 / num_steps)) / 1000
            for index in range(num_steps)
        ]
        + [0.0],
        dtype=mx.float32,
    )
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    mx.eval(shifted)
    return tuple(float(value) for value in shifted.tolist())


def schedule(
    num_steps: int = DEFAULT_STEPS,
    *,
    video_shift: float = layout.SIGMA_SHIFT_VIDEO,
    audio_shift: float = layout.SIGMA_SHIFT_AUDIO,
) -> SigmaSchedule:
    """Build the official paired ``simple`` schedule, including terminal zero."""
    return SigmaSchedule(
        video=_shifted_sigmas(num_steps, video_shift),
        audio=_shifted_sigmas(num_steps, audio_shift),
        video_shift=video_shift,
        audio_shift=audio_shift,
    )


def _euler_step(
    sample: mx.array,
    velocity: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    x = sample.astype(mx.float32)
    v = velocity.astype(mx.float32)
    # The reference forms x0 from the timestep seen by the transformer, while
    # the Euler ratio comes directly from the sigma grid. Preserve that f32
    # round trip: below sigma=0.5, 1 - (1 - sigma) need not equal sigma exactly.
    timestep = mx.array(1.0 - sigma, dtype=mx.float32)
    sigma_from_timestep = 1.0 - timestep
    ratio = mx.array(sigma_next / sigma, dtype=mx.float32)
    denoised = x + sigma_from_timestep * v
    return (ratio * x + (1.0 - ratio) * denoised).astype(sample.dtype)


def euler_step(
    sample: mx.array,
    velocity: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """Take one eta=0 Euler step from a raw data-ward velocity."""
    if sample.shape != velocity.shape:
        raise ValueError(f"sample and velocity shapes differ: {sample.shape} != {velocity.shape}")
    if not math.isfinite(sigma) or not math.isfinite(sigma_next):
        raise ValueError("sigmas must be finite")
    if sigma <= 0.0 or sigma_next < 0.0 or sigma_next >= sigma:
        raise ValueError(f"expected sigma > sigma_next >= 0, got {sigma}, {sigma_next}")
    return _euler_step(sample, velocity, sigma, sigma_next)


def _denoised(
    sample: mx.array,
    velocity: mx.array,
    sigma: float,
    *,
    velocity_scale: float = 1.0,
) -> mx.array:
    x = sample.astype(mx.float32)
    timestep = mx.array(1.0 - sigma, dtype=mx.float32)
    sigma_from_timestep = 1.0 - timestep
    return x + sigma_from_timestep * velocity.astype(mx.float32) * velocity_scale


def _phi1(value: float) -> float:
    return math.expm1(value) / value


def _res_step(
    sample: mx.array,
    denoised: mx.array,
    values: tuple[float, ...],
    index: int,
    old_denoised: mx.array | None,
) -> mx.array:
    sigma = values[index]
    sigma_next = values[index + 1]
    x = sample.astype(mx.float32)
    if old_denoised is None or sigma_next == 0.0:
        derivative = (x - denoised) / sigma
        return (x + derivative * (sigma_next - sigma)).astype(sample.dtype)

    t = -math.log(sigma)
    t_next = -math.log(sigma_next)
    t_prev = -math.log(values[index - 1])
    h = t_next - t
    c2 = (t_prev - t) / h
    phi1 = _phi1(-h)
    phi2 = (phi1 - 1.0) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2
    out = math.exp(-h) * x + h * (
        b1 * denoised + b2 * old_denoised.astype(mx.float32)
    )
    return out.astype(sample.dtype)


def denoise(
    model: Callable[..., tuple[mx.array, mx.array]],
    video_latent: mx.array,
    audio_latent: mx.array,
    text_embed: mx.array,
    packed: PackedLayout,
    sigmas: SigmaSchedule,
    *,
    guard: memory.Guard | None = None,
    on_step: Callable[[int, int, float, float], None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Denoise video and audio together while stepping their own schedules.

    ``on_step`` receives only scalar progress values so instrumentation cannot
    accidentally retain a latent or model reference across iterations.
    """
    video, audio = video_latent, audio_latent
    old_video_denoised = None
    old_audio_denoised = None
    for index in range(sigmas.steps):
        sigma_video = sigmas.video[index]
        sigma_audio = sigmas.audio[index]
        video_velocity, audio_velocity = model(
            video,
            audio,
            text_embed,
            packed,
            sigma_video=sigma_video,
            sigma_audio=sigma_audio,
        )
        video_denoised = _denoised(video, video_velocity, sigma_video)
        audio_slope = layout.time_shift_slope(
            sigma_video, sigmas.video_shift, sigmas.audio_shift
        )
        audio_denoised = _denoised(
            audio,
            audio_velocity,
            sigma_video,
            velocity_scale=audio_slope,
        )
        video = _res_step(
            video, video_denoised, sigmas.video, index, old_video_denoised
        )
        audio = _res_step(
            audio, audio_denoised, sigmas.video, index, old_audio_denoised
        )
        mx.eval(video, audio, video_denoised, audio_denoised)
        old_video_denoised = video_denoised
        old_audio_denoised = audio_denoised

        completed = index + 1
        if guard is not None:
            guard.check(f"step {completed}/{sigmas.steps}")
        if on_step is not None:
            on_step(completed, sigmas.steps, sigma_video, sigma_audio)
    return video, audio
