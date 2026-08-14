"""Focused CLI input-boundary tests."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import cli, pipeline, sampler
from tests._support.pipeline import model_paths


def test_prompt_file_preserves_structure_and_requires_one_source(tmp_path: Path):
    prompt_file = tmp_path / "request.txt"
    prompt_file.write_text("first line\n\nsecond line\n", encoding="utf-8")

    assert cli._resolve_prompt(None, str(prompt_file)) == "first line\n\nsecond line"
    assert cli._resolve_prompt("direct", None) == "direct"
    with pytest.raises(ValueError, match="exactly one"):
        cli._resolve_prompt(None, None)
    with pytest.raises(ValueError, match="exactly one"):
        cli._resolve_prompt("direct", str(prompt_file))


def test_reference_flags_preserve_cross_modality_cli_order():
    args = cli._build_parser().parse_args(
        [
            "request",
            "--ref-audio",
            "voice.wav",
            "--ref-image",
            "subject.png",
            "--ref-video-silent",
            "motion.mp4",
        ]
    )

    assert [reference.kind for reference in args.references] == [
        "audio",
        "image",
        "video",
    ]
    assert args.references[-1].include_video_audio is False


def test_turbo_lora_path_is_explicit():
    args = cli._build_parser().parse_args(
        ["request", "--turbo-lora", "weights/adapters/turbo.safetensors", "--steps", "6"]
    )

    assert args.turbo_lora == "weights/adapters/turbo.safetensors"
    assert args.steps == 6


def test_nax_group_size_is_explicit():
    args = cli._build_parser().parse_args(
        ["input", "--nax-group-size", "896"]
    )
    assert args.nax_group_size == 896


def test_step_default_is_resolved_from_adapter_presence():
    parser = cli._build_parser()
    base = parser.parse_args(["request"])
    turbo = parser.parse_args(
        ["request", "--turbo-lora", "weights/adapters/turbo.safetensors"]
    )

    assert base.steps is None
    assert turbo.steps is None
    assert sampler.BASE_PROFILE.resolve_steps(base.steps) == 20
    assert sampler.TURBO_PROFILE.resolve_steps(turbo.steps) == 6


def test_main_wires_cli_configuration_through_generation_and_mux(
    monkeypatch, tmp_path: Path, capsys
):
    paths = model_paths(tmp_path)
    destination = tmp_path / "result.mp4"
    calls = {}

    class Guard:
        def __init__(self, label, budget):
            calls["guard"] = (label, budget)

        def check(self, note):
            calls.setdefault("checks", []).append(note)

    def generate(config, actual_paths, guard, **kwargs):
        calls["generate"] = (config, actual_paths, guard, kwargs)
        return pipeline.GeneratedMedia(
            frames=mx.zeros((1, 3, 5, 32, 32)),
            audio=mx.zeros((1, 2, 6400)),
            fps=24,
            sample_rate=32_000,
            seed=config.seed,
            prompt_tokens=3,
            sequence_length=17,
        )

    def mux(path, frames, audio, *, fps, sample_rate):
        calls["mux"] = (path, frames.shape, audio.shape, fps, sample_rate)
        destination.touch()
        return destination

    monkeypatch.setattr(cli.memory, "configure", lambda budget: calls.setdefault("budget", budget))
    monkeypatch.setattr(cli.memory, "Guard", Guard)
    monkeypatch.setattr(cli.memory, "report", lambda label="": label + "memory")
    monkeypatch.setattr(cli.pipeline, "generate", generate)
    monkeypatch.setattr(cli.output, "mux_mp4", mux)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-h3",
            "request",
            "--width",
            "32",
            "--height",
            "32",
            "--frames",
            "5",
            "--steps",
            "7",
            "--seed",
            "9",
            "--budget",
            "70",
            "--nax-group-size",
            "64",
            "--tokenizer",
            str(paths.tokenizer),
            "--text-encoder",
            str(paths.text_encoder),
            "--dit",
            str(paths.dit),
            "--ref-dit",
            str(paths.ref_dit),
            "--video-vae",
            str(paths.video_vae),
            "--audio-vae",
            str(paths.audio_vae),
            "--output",
            str(destination),
        ],
    )

    assert cli.main() == 0

    config, actual_paths, guard, kwargs = calls["generate"]
    assert config == pipeline.GenerationConfig(
        "request", width=32, height=32, frames=5, steps=7, seed=9
    )
    for field in (
        "tokenizer",
        "text_encoder",
        "dit",
        "ref_dit",
        "video_vae",
        "audio_vae",
    ):
        assert Path(getattr(actual_paths, field)) == Path(getattr(paths, field))
    assert actual_paths.turbo_lora is None
    assert kwargs["nax_group_size"] == 64
    assert callable(kwargs["on_step"])
    assert callable(kwargs["on_report"])
    assert calls["budget"] == 70
    assert calls["guard"] == ("generate", 70)
    assert calls["mux"] == (
        str(destination),
        (1, 3, 5, 32, 32),
        (1, 2, 6400),
        24,
        32_000,
    )
    assert calls["checks"] == ["output written"]
    assert "wrote" in capsys.readouterr().out
