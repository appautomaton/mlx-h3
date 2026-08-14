"""Small pipeline test doubles that preserve runtime boundary contracts."""

from __future__ import annotations

from pathlib import Path

from mlx_h3 import memory, pipeline


class RecordingGuard:
    """Record every memory boundary while returning distinct scalar samples."""

    def __init__(self):
        self.notes: list[str] = []

    def check(self, note: str) -> memory.Sample:
        self.notes.append(note)
        value = len(self.notes)
        return memory.Sample(value, value + 10, 0, 0, 0, 0, 100)


def model_paths(
    tmp_path: Path, *, ref_dit: Path | None = None
) -> pipeline.ModelPaths:
    """Create every path required by the selected weightless pipeline mode."""
    files = {
        "tokenizer": tmp_path / "tokenizer.json",
        "text_encoder": tmp_path / "text-encoder.safetensors",
        "dit": tmp_path / "fl2va-dit.safetensors",
        "ref_dit": ref_dit or tmp_path / "ref2va-dit.safetensors",
        "video_vae": tmp_path / "video-vae.safetensors",
        "audio_vae": tmp_path / "audio-vae.safetensors",
    }
    for path in files.values():
        path.touch(exist_ok=True)
    return pipeline.ModelPaths(**files)
