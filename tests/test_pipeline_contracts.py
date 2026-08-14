"""Contracts for pipeline configuration, phase lifetime, and failure boundaries."""

from __future__ import annotations

import weakref
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import pipeline
from tests._support.pipeline import RecordingGuard, model_paths


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
    guard = RecordingGuard()
    output = pipeline.run_phase(
        "dit", load, execute, guard, on_report=reports.append
    )

    assert output.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert model_refs[0]() is None
    assert released == [True]
    assert guard.notes == ["dit / loaded", "dit / complete", "dit / released"]
    assert len(reports) == 1
    assert reports[0].label == "dit"
    assert reports[0].active_after_load == 1
    assert reports[0].active_after_run == 2
    assert reports[0].active_after_release == 3
    assert reports[0].peak == 13


def test_release_path_builds_no_report(monkeypatch):
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)
    guard = RecordingGuard()

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
        pipeline.run_phase("vae", load, fail, RecordingGuard())

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
    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.GenerationConfig(
            "input",
            first_frame="first.png",
            references=(pipeline.Reference(image="ref.png"),),
        )
    with pytest.raises(ValueError, match="requires an image or video"):
        pipeline.GenerationConfig(
            "input", references=(pipeline.Reference(audio="ref.wav"),)
        )
    with pytest.raises(ValueError, match="must contain"):
        pipeline.Reference(image="ref.png", audio="ref.wav")


def test_model_paths_report_all_missing_files_before_loading(tmp_path: Path):
    paths = pipeline.ModelPaths(
        tokenizer=tmp_path / "tokenizer.json",
        text_encoder=tmp_path / "text.safetensors",
        dit=tmp_path / "fl2va.safetensors",
        ref_dit=tmp_path / "ref2va.safetensors",
        video_vae=tmp_path / "video.safetensors",
        audio_vae=tmp_path / "audio.safetensors",
    )

    with pytest.raises(FileNotFoundError) as error:
        paths.validate(ref2va=True)

    message = str(error.value)
    assert "tokenizer:" in message
    assert "text encoder:" in message
    assert "Ref2VA DiT:" in message
    assert "FL2VA DiT:" not in message
    assert "Video VAE:" in message
    assert "Audio VAE:" in message


def test_model_paths_select_sampling_profile_from_adapter_presence(tmp_path: Path):
    base = model_paths(tmp_path)
    turbo = replace(base, turbo_lora=tmp_path / "turbo.safetensors")

    assert base.sampling_profile is pipeline.sampler.BASE_PROFILE
    assert turbo.sampling_profile is pipeline.sampler.TURBO_PROFILE


def test_turbo_adapter_is_required_and_enforces_four_to_eight_steps(tmp_path: Path):
    missing = tmp_path / "missing-turbo.safetensors"
    paths = replace(model_paths(tmp_path), turbo_lora=missing)
    with pytest.raises(FileNotFoundError, match="Turbo LoRA"):
        paths.validate(ref2va=False)

    missing.touch()
    with pytest.raises(ValueError, match=r"\[4, 8\].*3"):
        pipeline.generate(
            pipeline.GenerationConfig("input", width=32, height=32, frames=5, steps=3),
            paths,
            RecordingGuard(),
        )
    with pytest.raises(ValueError, match=r"\[4, 8\].*9"):
        pipeline.generate(
            pipeline.GenerationConfig("input", width=32, height=32, frames=5, steps=9),
            paths,
            RecordingGuard(),
        )
