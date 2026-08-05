"""Deterministic media input decoding through the existing FFmpeg boundary."""

from __future__ import annotations

import math
import shutil
import subprocess
from array import array
from pathlib import Path
from typing import Literal

import mlx.core as mx

from . import layout

ImageFit = Literal["stretch", "cover"]
ReferenceImageSize = Literal["match", "max"]

REFERENCE_IMAGE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
REFERENCE_AUDIO_MIN_SECONDS = 2.0
REFERENCE_AUDIO_MAX_SECONDS = 15.0
AUDIO_SAMPLE_RATE = 32_000


def has_audio_stream(path: str | Path) -> bool:
    """Return whether a local media file contains at least one audio stream."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"media does not exist: {source}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to inspect media inputs")
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.strip()
        raise RuntimeError(
            f"ffprobe could not inspect {source} (exit {process.returncode}): {message}"
        )
    return bool(process.stdout.strip())


def image_size(path: str | Path) -> tuple[int, int]:
    """Return `(width, height)` for the first visual stream."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"image does not exist: {source}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to inspect image inputs")
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.strip()
        raise RuntimeError(
            f"ffprobe could not inspect {source} (exit {process.returncode}): {message}"
        )
    try:
        width, height = (int(value) for value in process.stdout.strip().split("x"))
    except ValueError as error:
        raise RuntimeError(
            f"ffprobe returned an invalid image size for {source}: {process.stdout!r}"
        ) from error
    if width < 1 or height < 1:
        raise RuntimeError(f"image has invalid dimensions {width}x{height}: {source}")
    return width, height


def reference_image_canvas(
    path: str | Path,
    *,
    target_width: int,
    target_height: int,
    size: ReferenceImageSize = "match",
) -> tuple[int, int]:
    """Resolve the released Ref2VA down-only, aspect-preserving image canvas."""
    if size not in ("match", "max"):
        raise ValueError(f"unsupported reference image size policy: {size}")
    width, height = image_size(path)
    if size == "match":
        scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    else:
        scale = min(1.0, REFERENCE_IMAGE_SHORT_EDGE / min(width, height))
    resized_width = max(
        CANVAS_MULTIPLE,
        round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    resized_height = max(
        CANVAS_MULTIPLE,
        round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return resized_width, resized_height


def reference_video_canvas(path: str | Path) -> tuple[int, int]:
    """Resolve the released target-style video canvas without large upscaling."""
    width, height = image_size(path)
    target_width, target_height = layout.adapt_canvas(width, height)
    if width * height < target_width * target_height:
        target_width = max(
            CANVAS_MULTIPLE,
            round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )
        target_height = max(
            CANVAS_MULTIPLE,
            round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )
    return target_width, target_height


def load_rgb_image(
    path: str | Path,
    *,
    width: int,
    height: int,
    fit: ImageFit,
) -> mx.array:
    """Decode one image to `[1,3,H,W]` float32 RGB in `[0,1]`.

    `stretch` is the first-frame geometry anchor. `cover` preserves aspect ratio
    and center-crops the last-frame follower, matching the released workflow.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"image does not exist: {source}")
    if width < 1 or height < 1:
        raise ValueError("image width and height must be positive")
    if fit not in ("stretch", "cover"):
        raise ValueError(f"unsupported image fit: {fit}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode image inputs")

    if fit == "stretch":
        video_filter = f"scale={width}:{height}:flags=lanczos"
    else:
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height}"
        )
    process = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            video_filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode {source} (exit {process.returncode}): {message}"
        )
    expected = width * height * 3
    if len(process.stdout) != expected:
        raise RuntimeError(
            f"decoded image has {len(process.stdout)} bytes, expected {expected}"
        )

    pixels = mx.array(memoryview(process.stdout), dtype=mx.uint8).reshape(
        height, width, 3
    )
    return mx.transpose(pixels, (2, 0, 1))[None].astype(mx.float32) / 255.0


def load_rgb_video(
    path: str | Path,
    *,
    width: int,
    height: int,
    max_frames: int,
) -> mx.array:
    """Decode a bounded 24 fps reference to `[1,3,T,H,W]` RGB in `[0,1]`."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    if width < 1 or height < 1:
        raise ValueError("video width and height must be positive")
    if max_frames < 5:
        raise ValueError("reference video frame limit must be at least 5")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode video inputs")

    process = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={layout.FPS},scale={width}:{height}:flags=lanczos,setsar=1",
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode {source} (exit {process.returncode}): {message}"
        )
    frame_bytes = width * height * 3
    if len(process.stdout) % frame_bytes:
        raise RuntimeError(
            f"decoded video has a partial {width}x{height} RGB frame"
        )
    frame_count = len(process.stdout) // frame_bytes
    if frame_count < 5:
        raise ValueError("reference videos require at least 5 decoded frames")
    while frame_count % 17 != 5:
        frame_count -= 1
    payload = memoryview(process.stdout)[: frame_count * frame_bytes]
    pixels = mx.array(payload, dtype=mx.uint8).reshape(
        frame_count, height, width, 3
    )
    return mx.transpose(pixels, (3, 0, 1, 2))[None].astype(mx.float32) / 255.0


def load_stereo_audio(
    path: str | Path,
    *,
    max_seconds: float | None = None,
) -> mx.array:
    """Decode one bounded reference to `[1,2,L]` float32 PCM at 32 kHz."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"audio does not exist: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode audio inputs")
    limit = (
        REFERENCE_AUDIO_MAX_SECONDS + 0.1
        if max_seconds is None
        else float(max_seconds)
    )
    if not math.isfinite(limit) or limit < REFERENCE_AUDIO_MIN_SECONDS:
        raise ValueError(
            f"reference audio window must be at least {REFERENCE_AUDIO_MIN_SECONDS:g} seconds"
        )
    process = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-t",
            f"{limit:.9g}",
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode {source} (exit {process.returncode}): {message}"
        )
    samples = array("f")
    samples.frombytes(process.stdout)
    if len(samples) % 2:
        raise RuntimeError("decoded audio ends with a partial stereo sample")
    sample_count = len(samples) // 2
    duration = sample_count / AUDIO_SAMPLE_RATE
    if not REFERENCE_AUDIO_MIN_SECONDS <= duration <= REFERENCE_AUDIO_MAX_SECONDS:
        raise ValueError(
            "reference audio duration must be in "
            f"[{REFERENCE_AUDIO_MIN_SECONDS:g}, {REFERENCE_AUDIO_MAX_SECONDS:g}] "
            f"seconds, got {duration:.3f}"
        )
    waveform = mx.array(samples, dtype=mx.float32).reshape(sample_count, 2)
    return mx.transpose(waveform, (1, 0))[None]
