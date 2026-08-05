"""Contract tests for deterministic MiniMax-H3 RES multistep sampling.

VALIDATION TIER. The formulas are transcribed from two local reference
implementations and are not backed by an executed Torch fixture. Closed-form
constant-velocity tests make schedule or sign mistakes directly observable.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_h3 import sampler


def test_official_simple_schedule_has_20_model_evaluations():
    sigmas = sampler.schedule()
    assert len(sigmas.video) == sampler.DEFAULT_STEPS + 1
    assert sigmas.steps == 20
    assert sigmas.video[0] == sigmas.audio[0] == 1.0
    assert sigmas.video[-1] == sigmas.audio[-1] == 0.0

    # BasicScheduler("simple") walks base sigmas 1.00, 0.95, ..., 0.05.
    assert sigmas.video[1] == pytest.approx(228 / 229, rel=1e-6)
    assert sigmas.audio[1] == pytest.approx(57 / 58, rel=1e-6)
    assert sigmas.video[-2] == pytest.approx(12 / 31, rel=1e-6)
    assert sigmas.audio[-2] == pytest.approx(3 / 22, rel=1e-6)


def test_schedule_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="num_steps"):
        sampler.schedule(0)
    with pytest.raises(ValueError, match="num_steps"):
        sampler.schedule(1001)
    with pytest.raises(ValueError, match="positive"):
        sampler.schedule(video_shift=0.0)
    with pytest.raises(ValueError, match="same length"):
        sampler.SigmaSchedule((1.0, 0.0), (1.0, 0.5, 0.0))


def test_euler_step_matches_the_constant_velocity_solution():
    x = mx.array([[-2.0, 3.0]], dtype=mx.bfloat16)
    velocity = mx.array([[0.5, -1.5]], dtype=mx.bfloat16)
    got = sampler.euler_step(x, velocity, sigma=0.8, sigma_next=0.3)
    want = x.astype(mx.float32) + 0.5 * velocity.astype(mx.float32)
    assert got.dtype == mx.bfloat16
    assert mx.allclose(got.astype(mx.float32), want, rtol=1e-2, atol=1e-2)


def test_full_shifted_schedules_integrate_constant_velocity_exactly():
    sigmas = sampler.schedule()
    video = mx.array([1.0], dtype=mx.float32)
    audio = mx.array([-2.0], dtype=mx.float32)
    video_velocity = mx.array([0.25], dtype=mx.float32)
    audio_velocity = mx.array([0.75], dtype=mx.float32)

    for current, following in zip(sigmas.video, sigmas.video[1:]):
        video = sampler.euler_step(video, video_velocity, current, following)
    for current, following in zip(sigmas.audio, sigmas.audio[1:]):
        audio = sampler.euler_step(audio, audio_velocity, current, following)

    assert video.item() == pytest.approx(1.25, abs=2e-6)
    assert audio.item() == pytest.approx(-1.25, abs=2e-6)


def test_res_multistep_uses_second_order_updates_between_euler_endpoints():
    sigmas = sampler.SigmaSchedule(
        (1.0, 0.75, 0.25, 0.0), (1.0, 0.75, 0.25, 0.0)
    )

    def sigma_velocity(
        video, audio, text, packed, *, sigma_video, sigma_audio, step_index
    ):
        del text, packed
        assert step_index in (0, 1, 2)
        return mx.full_like(video, sigma_video), mx.full_like(audio, sigma_audio)

    video, audio = sampler.denoise(
        sigma_velocity,
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1, 1)),
        object(),
        sigmas,
    )
    euler_result = 0.6875
    exact_integral = 0.5
    assert video.item() == pytest.approx(audio.item(), abs=1e-6)
    assert video.item() != pytest.approx(euler_result, abs=1e-3)
    assert abs(video.item() - exact_integral) < abs(euler_result - exact_integral)


def test_audio_velocity_is_mapped_onto_the_video_res_grid():
    sigmas = sampler.schedule(2, video_shift=2.0, audio_shift=1.0)

    def constant_model(
        video, audio, text, packed, *, sigma_video, sigma_audio, step_index
    ):
        del text, packed, sigma_video, sigma_audio, step_index
        return mx.ones_like(video), mx.ones_like(audio)

    video, audio = sampler.denoise(
        constant_model,
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((1, 1)),
        object(),
        sigmas,
    )

    # Comfy integrates both streams on video sigmas [1, 2/3, 0]. The audio
    # velocity scales are 2 and 9/8, giving 2/3 after the first Euler update
    # and 17/12 after the terminal Euler update. Independent audio stepping
    # would incorrectly produce 1 here.
    assert video.item() == pytest.approx(1.0, abs=1e-6)
    assert audio.item() == pytest.approx(17 / 12, abs=1e-6)


def test_denoise_passes_paired_sigmas_and_checks_each_step():
    sigmas = sampler.schedule(3)
    calls = []
    progress = []

    class FakeGuard:
        def __init__(self):
            self.notes = []

        def check(self, note):
            self.notes.append(note)

    def constant_model(
        video, audio, text, packed, *, sigma_video, sigma_audio, step_index
    ):
        del text, packed
        calls.append((step_index, sigma_video, sigma_audio))
        return mx.ones_like(video), -mx.ones_like(audio)

    guard = FakeGuard()
    video, audio = sampler.denoise(
        constant_model,
        mx.zeros((1, 2), dtype=mx.float32),
        mx.zeros((1, 3), dtype=mx.float32),
        mx.zeros((1, 1), dtype=mx.float32),
        object(),
        sigmas,
        guard=guard,
        on_step=lambda *args: progress.append(args),
    )

    assert calls == [
        (index, sigma_video, sigma_audio)
        for index, (sigma_video, sigma_audio) in enumerate(
            zip(sigmas.video[:-1], sigmas.audio[:-1])
        )
    ]
    assert guard.notes == ["step 1/3", "step 2/3", "step 3/3"]
    assert progress == [
        (1, 3, sigmas.video[0], sigmas.audio[0]),
        (2, 3, sigmas.video[1], sigmas.audio[1]),
        (3, 3, sigmas.video[2], sigmas.audio[2]),
    ]
    assert mx.allclose(video, mx.ones_like(video))
    assert bool(mx.all(mx.isfinite(audio)).item())
