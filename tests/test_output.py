"""Tests for the FFmpeg MP4 boundary."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import output


class ByteSink:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, value):
        self.data.extend(value)

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, return_code=0, stderr=b""):
        self.stdin = ByteSink()
        self.stderr = io.BytesIO(stderr)
        self.return_code = return_code
        self.killed = False

    def wait(self):
        return self.return_code

    def poll(self):
        return self.return_code

    def kill(self):
        self.killed = True


def test_output_shape_checks_run_before_ffmpeg():
    with pytest.raises(ValueError, match="frames"):
        output.mux_mp4("unused.mp4", mx.zeros((3, 8, 8)), mx.zeros((1, 2, 10)))
    with pytest.raises(ValueError, match="audio"):
        output.mux_mp4(
            "unused.mp4", mx.zeros((1, 3, 2, 8, 8)), mx.zeros((2, 10))
        )


def test_mux_streams_frames_and_atomically_replaces_destination(
    monkeypatch, tmp_path
):
    destination = tmp_path / "nested" / "result.mp4"
    frames = mx.linspace(0.0, 1.0, 2 * 2 * 2 * 3).reshape(1, 3, 2, 2, 2)
    audio = mx.zeros((1, 2, 16), dtype=mx.float32)
    captured = {}
    process = FakeProcess()

    monkeypatch.setattr(output.shutil, "which", lambda _: "/tools/ffmpeg")

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raw_audio = Path(command[command.index("f32le") + 6])
        captured["audio_bytes"] = raw_audio.read_bytes()
        captured["temporary"] = Path(command[-1])
        return process

    monkeypatch.setattr(output.subprocess, "Popen", popen)

    result = output.mux_mp4(destination, frames, audio, fps=2, sample_rate=8)

    assert result == destination.resolve()
    assert destination.exists()
    assert process.stdin.closed
    assert len(process.stdin.data) == 2 * 2 * 2 * 3
    assert len(captured["audio_bytes"]) == 1 * 2 * 16 * 4
    assert captured["command"][0] == "/tools/ffmpeg"
    assert captured["command"][captured["command"].index("-s:v") + 1] == "2x2"
    assert captured["command"][captured["command"].index("-t") + 1] == "1.000000000"
    assert captured["kwargs"] == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
    }


def test_mux_removes_temporary_output_when_ffmpeg_fails(monkeypatch, tmp_path):
    destination = tmp_path / "result.mp4"
    frames = mx.zeros((1, 3, 1, 2, 2))
    audio = mx.zeros((1, 2, 16))
    process = FakeProcess(return_code=2, stderr=b"encoder failed")
    captured = {}
    monkeypatch.setattr(output.shutil, "which", lambda _: "/tools/ffmpeg")

    def popen(command, **_):
        captured["temporary"] = Path(command[-1])
        return process

    monkeypatch.setattr(output.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="exit code 2: encoder failed"):
        output.mux_mp4(destination, frames, audio)

    assert not destination.exists()
    assert not captured["temporary"].exists()


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
