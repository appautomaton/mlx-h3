"""Tests for the FFmpeg MP4 boundary."""

from __future__ import annotations

import shutil
import subprocess

import mlx.core as mx
import pytest

from mlx_h3 import output


def test_output_shape_checks_run_before_ffmpeg():
    with pytest.raises(ValueError, match="frames"):
        output.mux_mp4("unused.mp4", mx.zeros((3, 8, 8)), mx.zeros((1, 2, 10)))
    with pytest.raises(ValueError, match="audio"):
        output.mux_mp4(
            "unused.mp4", mx.zeros((1, 3, 2, 8, 8)), mx.zeros((2, 10))
        )


@pytest.mark.runtime
def test_ffmpeg_writes_a_tiny_mp4(tmp_path):
    ffprobe = shutil.which("ffprobe")
    if shutil.which("ffmpeg") is None or ffprobe is None:
        pytest.skip("ffmpeg or ffprobe absent")
    frames = mx.linspace(0.0, 1.0, 5 * 32 * 32 * 3).reshape(1, 5, 32, 32, 3)
    frames = mx.transpose(frames, (0, 4, 1, 2, 3))
    audio = mx.zeros((1, 2, 8000), dtype=mx.float32)
    path = output.mux_mp4(tmp_path / "tiny.mp4", frames, audio)
    assert path.exists()
    assert path.stat().st_size > 1000
    with path.open("rb") as file:
        assert b"ftyp" in file.read(32)
    frame_count = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert frame_count == "5"
