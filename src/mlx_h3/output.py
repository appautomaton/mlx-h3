"""FFmpeg output boundary for generated RGB frames and stereo audio."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx


def mux_mp4(
    path: str | Path,
    frames: mx.array,
    audio: mx.array,
    *,
    fps: int = 24,
    sample_rate: int = 32000,
    crf: int = 18,
) -> Path:
    """Encode ``[1,3,F,H,W]`` RGB and ``[1,2,S]`` audio into H.264/AAC."""
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] != 3:
        raise ValueError(f"expected frames [1,3,F,H,W], got {frames.shape}")
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2:
        raise ValueError(f"expected audio [1,2,S], got {audio.shape}")
    if fps < 1 or sample_rate < 1:
        raise ValueError("fps and sample_rate must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 output")

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, _, frame_count, height, width = frames.shape

    pixels = mx.contiguous(
        mx.clip(
            mx.transpose(frames[0], (1, 2, 3, 0)).astype(mx.float32) * 255.0
            + 0.5,
            0.0,
            255.0,
        ).astype(mx.uint8)
    )
    interleaved_audio = mx.contiguous(
        mx.transpose(audio[0].astype(mx.float32), (1, 0))
    )
    mx.eval(pixels, interleaved_audio)

    raw_audio = tempfile.NamedTemporaryFile(suffix=".f32")
    raw_audio.write(memoryview(interleaved_audio).cast("B"))
    raw_audio.flush()
    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.stem}-", suffix=".mp4", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-i",
        raw_audio.name,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-af",
        "apad",
        "-t",
        f"{frame_count / fps:.9f}",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for index in range(frame_count):
            process.stdin.write(memoryview(pixels[index]).cast("B"))
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {message}")
        os.replace(temporary_path, destination)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        raw_audio.close()
    return destination
