"""Lifecycle tests for staged model residency.

These tests use tiny stand-ins. Real phase memory is checked by runtime scripts;
the unit contract here is reference release, materialization, check cadence and
scalar-only optional telemetry.
"""

from __future__ import annotations

import weakref
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import memory, pipeline


class FakeGuard:
    def __init__(self):
        self.notes = []

    def check(self, note):
        self.notes.append(note)
        value = len(self.notes)
        return memory.Sample(value, value + 10, 0, 0, 0, 0, 100)


def test_run_phase_materializes_output_and_releases_model(monkeypatch):
    released = []
    model_refs = []
    reports = []

    class Model:
        pass

    def load():
        model = Model()
        model_refs.append(weakref.ref(model))
        return model

    def execute(model):
        assert model_refs[0]() is model
        return mx.arange(4, dtype=mx.float32) + 1

    monkeypatch.setattr(pipeline.memory, "release", lambda: released.append(True))
    guard = FakeGuard()
    output = pipeline.run_phase(
        "dit", load, execute, guard, on_report=reports.append
    )

    assert output.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert model_refs[0]() is None
    assert released == [True]
    assert guard.notes == [
        "dit / loaded",
        "dit / complete",
        "dit / released",
    ]
    assert len(reports) == 1
    assert reports[0].label == "dit"
    assert reports[0].active_after_load == 1
    assert reports[0].active_after_run == 2
    assert reports[0].active_after_release == 3
    assert reports[0].peak == 13


def test_release_path_builds_no_report(monkeypatch):
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)
    guard = FakeGuard()
    output = pipeline.run_phase(
        "text", lambda: object(), lambda _: mx.array([7]), guard
    )
    assert output.item() == 7
    assert guard.notes[-1] == "text / released"


def test_failed_phase_still_releases_the_model(monkeypatch):
    released = []
    model_refs = []

    class Model:
        pass

    def load():
        model = Model()
        model_refs.append(weakref.ref(model))
        return model

    def fail(_):
        raise RuntimeError("phase failed")

    monkeypatch.setattr(pipeline.memory, "release", lambda: released.append(True))
    with pytest.raises(RuntimeError, match="phase failed"):
        pipeline.run_phase("vae", load, fail, FakeGuard())

    assert model_refs[0]() is None
    assert released == [True]


def test_generation_config_rejects_only_real_resource_boundaries():
    pipeline.GenerationConfig("input", width=32, height=32, frames=5)
    with pytest.raises(ValueError, match="multiples"):
        pipeline.GenerationConfig("input", width=33, height=32)
    with pytest.raises(ValueError, match="pixel limit"):
        pipeline.GenerationConfig("input", width=2048, height=1024)
    with pytest.raises(ValueError, match="15 second"):
        pipeline.GenerationConfig("input", frames=363)
    with pytest.raises(ValueError, match="steps"):
        pipeline.GenerationConfig("input", steps=0)


def test_generate_runs_all_models_in_separate_phases(monkeypatch, tmp_path: Path):
    loaded = []

    class FakeTokenizer:
        def encode_prompt(self, prompt):
            assert prompt == "test input"
            return [4, 5, 6]

    class FakeTextEncoder:
        def __call__(self, token_ids):
            return mx.zeros((1, token_ids.shape[1], 5120), dtype=mx.bfloat16)

    class FakeDiT:
        def refine_text(self, states):
            return mx.zeros((states.shape[0], 5376), dtype=mx.bfloat16)

        def __call__(
            self,
            video,
            audio,
            text,
            packed,
            *,
            sigma_video,
            sigma_audio,
        ):
            assert text.shape == (3, 5376)
            assert packed.seq_len > 3
            assert sigma_video > 0 and sigma_audio > 0
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoVAE:
        def __call__(self, latent):
            assert latent.shape == (1, 24, 2, 2, 2)
            return mx.zeros((1, 3, 5, 32, 32))

    class FakeAudioVAE:
        def __call__(self, latent):
            assert latent.shape == (1, 32, 2, 8)
            return mx.zeros((1, 2, 6400))

    def loader(label, value):
        def load(_):
            loaded.append(label)
            return value()

        return load

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.loading, "load_text_encoder", loader("text", FakeTextEncoder)
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", loader("dit", FakeDiT))
    monkeypatch.setattr(
        pipeline.loading, "load_video_vae", loader("video", FakeVideoVAE)
    )
    monkeypatch.setattr(
        pipeline.loading, "load_audio_vae", loader("audio", FakeAudioVAE)
    )
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    result = pipeline.generate(
        pipeline.GenerationConfig(
            "test input", width=32, height=32, frames=5, steps=3
        ),
        pipeline.ModelPaths(tokenizer=tmp_path / "tokenizer.json"),
        FakeGuard(),
    )
    assert loaded == ["text", "dit", "video", "audio"]
    assert result.frames.shape == (1, 3, 5, 32, 32)
    assert result.audio.shape == (1, 2, 6400)
    assert result.prompt_tokens == 3
    assert result.seed == 42


def test_generate_rejects_non_finite_decoded_media(monkeypatch, tmp_path: Path):
    class FakeTokenizer:
        def encode_prompt(self, _):
            return [4]

    class FakeTextEncoder:
        def __call__(self, token_ids):
            return mx.zeros((1, token_ids.shape[1], 5120), dtype=mx.bfloat16)

    class FakeDiT:
        def refine_text(self, states):
            return mx.zeros((states.shape[0], 5376), dtype=mx.bfloat16)

        def __call__(self, video, audio, *args, **kwargs):
            return mx.zeros_like(video), mx.zeros_like(audio)

    class BadVideoVAE:
        def __call__(self, _):
            return mx.full((1, 3, 5, 32, 32), float("nan"))

    class FakeAudioVAE:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.loading, "load_text_encoder", lambda _: FakeTextEncoder()
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", lambda _: FakeDiT())
    monkeypatch.setattr(pipeline.loading, "load_video_vae", lambda _: BadVideoVAE())
    monkeypatch.setattr(pipeline.loading, "load_audio_vae", lambda _: FakeAudioVAE())
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    with pytest.raises(ValueError, match="non-finite frames"):
        pipeline.generate(
            pipeline.GenerationConfig(
                "test", width=32, height=32, frames=5, steps=2
            ),
            pipeline.ModelPaths(tokenizer=tmp_path / "tokenizer.json"),
            FakeGuard(),
        )
