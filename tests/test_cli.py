"""Focused CLI input-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_h3 import cli, sampler


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
