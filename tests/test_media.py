"""Deterministic media-boundary tests with strict FFmpeg process stubs."""

from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest

from mlx_h3 import media


def process(*, stdout, stderr=b"", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def install_binary_stub(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/tools/{name}")


def test_image_size_uses_the_first_visual_stream(monkeypatch, tmp_path):
    source = tmp_path / "image.png"
    source.touch()
    install_binary_stub(monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return process(stdout="640x480\n", stderr="")

    monkeypatch.setattr(media.subprocess, "run", run)

    assert media.image_size(source) == (640, 480)
    assert calls[0][0][0] == "/tools/ffprobe"
    assert calls[0][0][-1] == str(source)
    assert calls[0][1] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }


@pytest.mark.parametrize(("stdout", "expected"), [("0\n", True), ("", False)])
def test_has_audio_stream_uses_ffprobe_presence(stdout, expected, monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    install_binary_stub(monkeypatch)
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_, **__: process(stdout=stdout, stderr=""),
    )

    assert media.has_audio_stream(source) is expected


def test_reference_canvases_are_down_only_and_aligned(monkeypatch):
    monkeypatch.setattr(media, "image_size", lambda _: (640, 320))

    assert media.reference_image_canvas(
        "ignored.png", target_width=320, target_height=320
    ) == (448, 224)
    assert media.reference_image_canvas(
        "ignored.png", target_width=320, target_height=320, size="max"
    ) == (640, 320)
    assert media.reference_video_canvas("ignored.mp4") == (640, 320)


def test_load_rgb_image_preserves_rgb_channel_order(monkeypatch, tmp_path):
    source = tmp_path / "image.png"
    source.touch()
    install_binary_stub(monkeypatch)
    payload = bytes((0, 127, 255, 255, 0, 127))
    commands = []

    def run(command, **_):
        commands.append(command)
        return process(stdout=payload)

    monkeypatch.setattr(media.subprocess, "run", run)

    pixels = media.load_rgb_image(source, width=2, height=1, fit="cover")

    assert pixels.shape == (1, 3, 1, 2)
    assert pixels[0, 0, 0].tolist() == pytest.approx([0.0, 1.0])
    assert pixels[0, 1, 0].tolist() == pytest.approx([127 / 255, 0.0])
    assert pixels[0, 2, 0].tolist() == pytest.approx([1.0, 127 / 255])
    video_filter = commands[0][commands[0].index("-vf") + 1]
    assert "force_original_aspect_ratio=increase" in video_filter
    assert "crop=2:1" in video_filter


def test_load_rgb_video_truncates_to_released_frame_alignment(
    monkeypatch, tmp_path
):
    source = tmp_path / "video.mp4"
    source.touch()
    install_binary_stub(monkeypatch)
    payload = bytes(range(69))
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_, **__: process(stdout=payload),
    )

    pixels = media.load_rgb_video(source, width=1, height=1, max_frames=23)

    assert pixels.shape == (1, 3, 22, 1, 1)
    assert pixels[0, 0, 0, 0, 0].item() == 0.0
    assert pixels[0, 2, -1, 0, 0].item() == pytest.approx(65 / 255)


def test_load_stereo_audio_deinterleaves_channels(monkeypatch, tmp_path):
    source = tmp_path / "audio.wav"
    source.touch()
    install_binary_stub(monkeypatch)
    sample_count = 2 * media.AUDIO_SAMPLE_RATE
    payload = (array("f", (0.25, -0.5)) * sample_count).tobytes()
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_, **__: process(stdout=payload),
    )

    waveform = media.load_stereo_audio(source, max_seconds=2.0)

    assert waveform.shape == (1, 2, sample_count)
    assert waveform[0, 0, :3].tolist() == [0.25, 0.25, 0.25]
    assert waveform[0, 1, :3].tolist() == [-0.5, -0.5, -0.5]


def test_media_errors_include_the_failed_boundary(monkeypatch, tmp_path):
    source = tmp_path / "broken.mp4"
    source.touch()
    install_binary_stub(monkeypatch)
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_, **__: process(stdout=b"", stderr=b"bad stream", returncode=2),
    )

    with pytest.raises(RuntimeError, match=r"ffmpeg could not decode.*exit 2.*bad stream"):
        media.load_rgb_video(source, width=32, height=32, max_frames=5)
