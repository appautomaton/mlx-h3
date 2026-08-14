"""Weightless Ref2VA orchestration and ordering contracts."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import pipeline
from tests._support.pipeline import RecordingGuard, model_paths


def test_generate_ref2va_aligns_reference_images_and_selects_checkpoint(
    monkeypatch, tmp_path: Path
):
    loaded = []
    ref_dit = tmp_path / "ref-dit.safetensors"
    ref_dit.touch()

    class FakeTokenizer:
        def encode(self, _):
            return [4, 5]

        def encode_prompt(self, _):
            return [4, 5]

    class FakeVideoEncoder:
        def __call__(self, image):
            width = image.shape[-1]
            if width == 64:
                return mx.ones((1, 24, 1, 2, 4), dtype=mx.float16)
            assert width == 32
            return mx.full((1, 24, 1, 2, 2), 2, dtype=mx.float16)

    class FakeMultimodalEncoder:
        def encode_ref_references(self, tokenizer, prompt, references):
            assert tokenizer.encode(prompt) == [4, 5]
            assert [reference.kind for reference in references] == ["image", "image"]
            assert [reference.pixels.shape for reference in references] == [
                (1, 3, 32, 64),
                (1, 3, 32, 32),
            ]
            return (
                mx.zeros((5, 5120), dtype=mx.bfloat16),
                mx.array([1, 0, 0, 1, 1], dtype=mx.int32),
            )

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
            cond_video_rows,
            text_tags,
            **_,
        ):
            assert text.shape == (5, 5376)
            assert cond_video_rows.shape == (3, 96)
            assert mx.mean(cond_video_rows[:2]).item() == pytest.approx(1, abs=0.01)
            assert mx.mean(cond_video_rows[2:]).item() == pytest.approx(2, abs=0.01)
            assert tuple(text_tags) == (1, 0, 0, 1, 1)
            assert [segment.kind for segment in packed.segments] == [
                "text",
                "ref_img",
                "ref_img",
                "audio",
                "video",
            ]
            audio_segment = packed.kind_slice("audio")[0]
            assert packed.positions[audio_segment.start][0] == 7.0
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoDecoder:
        def __call__(self, _):
            return mx.zeros((1, 3, 5, 32, 32))

    class FakeAudioDecoder:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    def load(label, value):
        def loader(path, **__):
            if label == "dit":
                assert Path(path) == ref_dit
            loaded.append(label)
            return value()

        return loader

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.media,
        "reference_image_canvas",
        lambda path, **_: (64, 32) if path == "wide.png" else (32, 32),
    )
    monkeypatch.setattr(
        pipeline.media,
        "load_rgb_image",
        lambda _, *, width, height, **__: mx.zeros((1, 3, height, width)),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_video_vae_encoder",
        load("video encoder", FakeVideoEncoder),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_multimodal_text_encoder",
        load("text/vision", FakeMultimodalEncoder),
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", load("dit", FakeDiT))
    monkeypatch.setattr(
        pipeline.loading, "load_video_vae", load("video decoder", FakeVideoDecoder)
    )
    monkeypatch.setattr(
        pipeline.loading, "load_audio_vae", load("audio decoder", FakeAudioDecoder)
    )
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    result = pipeline.generate(
        pipeline.GenerationConfig(
            "test input",
            width=32,
            height=32,
            frames=5,
            steps=2,
            references=(
                pipeline.Reference(image="wide.png"),
                pipeline.Reference(image="square.png"),
            ),
        ),
        model_paths(tmp_path, ref_dit=ref_dit),
        RecordingGuard(),
    )

    assert loaded == [
        "video encoder",
        "text/vision",
        "dit",
        "video decoder",
        "audio decoder",
    ]
    assert result.prompt_tokens == 5


def test_generate_ref2va_aligns_reference_video_timeline(monkeypatch, tmp_path: Path):
    loaded = []
    ref_dit = tmp_path / "ref-dit.safetensors"
    ref_dit.touch()

    class FakeTokenizer:
        def encode(self, _):
            return [4, 5]

        def encode_prompt(self, _):
            return [4, 5]

    class FakeVideoEncoder:
        def __call__(self, video):
            assert video.shape == (1, 3, 22, 32, 32)
            return mx.ones((1, 24, 7, 2, 2), dtype=mx.float16)

    class FakeAudioEncoder:
        def __call__(self, waveform):
            assert waveform.shape == (1, 2, 29_334)
            return mx.ones((1, 32, 2, 3), dtype=mx.float32)

    class FakeMultimodalEncoder:
        def encode_ref_references(self, tokenizer, prompt, references):
            assert tokenizer.encode(prompt) == [4, 5]
            assert [reference.kind for reference in references] == ["video"]
            assert references[0].has_audio is True
            assert references[0].pixels.shape == (1, 3, 22, 32, 32)
            return (
                mx.zeros((4, 5120), dtype=mx.bfloat16),
                mx.array([1, 0, 0, 1], dtype=mx.int32),
            )

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
            cond_video_rows,
            cond_audio_rows,
            text_tags,
            **_,
        ):
            assert cond_video_rows.shape == (7, 96)
            assert cond_audio_rows.shape == (6, 32)
            assert mx.mean(cond_video_rows).item() == pytest.approx(1, abs=0.01)
            assert tuple(text_tags) == (1, 0, 0, 1)
            assert [segment.kind for segment in packed.segments] == [
                "text",
                "ref_audio",
                "ref_img",
                "audio",
                "video",
            ]
            reference = packed.kind_slice("ref_img")[0]
            assert len(reference) == 7
            reference_audio = packed.kind_slice("ref_audio")[0]
            assert packed.positions[reference_audio.start][0] == 4.0
            assert packed.positions[reference.start][0] == 4.0
            audio_segment = packed.kind_slice("audio")[0]
            expected_origin = 4 + sum(pipeline.layout.video_t_spans(7))
            assert packed.positions[audio_segment.start][0] == pytest.approx(
                expected_origin
            )
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoDecoder:
        def __call__(self, _):
            return mx.zeros((1, 3, 22, 32, 32))

    class FakeAudioDecoder:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    def load(label, value):
        def loader(path, **__):
            if label == "dit":
                assert Path(path) == ref_dit
            loaded.append(label)
            return value()

        return loader

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.media, "reference_video_canvas", lambda _: (32, 32)
    )
    monkeypatch.setattr(pipeline.media, "has_audio_stream", lambda _: True)

    def load_video(_, *, width, height, max_frames):
        assert (width, height, max_frames) == (32, 32, 22)
        return mx.zeros((1, 3, 22, 32, 32))

    monkeypatch.setattr(pipeline.media, "load_rgb_video", load_video)
    monkeypatch.setattr(
        pipeline.media,
        "load_stereo_audio",
        lambda _, *, max_seconds: mx.zeros((1, 2, 29_334))
        if max_seconds == pytest.approx(22 / 24)
        else None,
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_video_vae_encoder",
        load("video encoder", FakeVideoEncoder),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_audio_vae_encoder",
        load("audio encoder", FakeAudioEncoder),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_multimodal_text_encoder",
        load("text/vision", FakeMultimodalEncoder),
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", load("dit", FakeDiT))
    monkeypatch.setattr(
        pipeline.loading, "load_video_vae", load("video decoder", FakeVideoDecoder)
    )
    monkeypatch.setattr(
        pipeline.loading, "load_audio_vae", load("audio decoder", FakeAudioDecoder)
    )
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    result = pipeline.generate(
        pipeline.GenerationConfig(
            "test input",
            width=32,
            height=32,
            frames=22,
            steps=2,
            references=(pipeline.Reference(video="reference.mp4"),),
        ),
        model_paths(tmp_path, ref_dit=ref_dit),
        RecordingGuard(),
    )

    assert loaded == [
        "video encoder",
        "audio encoder",
        "text/vision",
        "dit",
        "video decoder",
        "audio decoder",
    ]
    assert result.prompt_tokens == 4


def test_generate_ref2va_preserves_cross_modality_reference_order(
    monkeypatch, tmp_path: Path
):
    loaded = []
    ref_dit = tmp_path / "ref-dit.safetensors"
    ref_dit.touch()

    class FakeTokenizer:
        def encode(self, _):
            return [4]

        def encode_prompt(self, _):
            return [4]

    class FakeVideoEncoder:
        def __call__(self, pixels):
            if pixels.ndim == 4:
                assert pixels.shape == (1, 3, 32, 32)
                return mx.ones((1, 24, 1, 2, 2), dtype=mx.float16)
            assert pixels.shape == (1, 3, 22, 32, 32)
            return mx.ones((1, 24, 7, 2, 2), dtype=mx.float16)

    class FakeAudioEncoder:
        def __call__(self, waveform):
            if waveform.shape[-1] == 29_334:
                return mx.ones((1, 32, 2, 3), dtype=mx.float32)
            assert waveform.shape == (1, 2, 64_000)
            return mx.full((1, 32, 2, 4), 2, dtype=mx.float32)

    class FakeMultimodalEncoder:
        def encode_ref_references(self, tokenizer, prompt, references):
            assert tokenizer.encode(prompt) == [4]
            assert [reference.kind for reference in references] == [
                "audio",
                "image",
                "video",
            ]
            assert [reference.has_audio for reference in references] == [
                True,
                False,
                True,
            ]
            return (
                mx.zeros((4, 5120), dtype=mx.bfloat16),
                mx.array([1, 0, 0, 1], dtype=mx.int32),
            )

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
            cond_video_rows,
            cond_audio_rows,
            **_,
        ):
            assert cond_video_rows.shape == (8, 96)
            assert cond_audio_rows.shape == (14, 32)
            assert mx.mean(cond_audio_rows[:8]).item() == pytest.approx(2)
            assert mx.mean(cond_audio_rows[8:]).item() == pytest.approx(1)
            assert [segment.kind for segment in packed.segments] == [
                "text",
                "ref_audio",
                "ref_img",
                "ref_audio",
                "ref_img",
                "audio",
                "video",
            ]
            audio_segment = packed.kind_slice("audio")[0]
            expected_origin = 9 + sum(pipeline.layout.video_t_spans(7))
            assert packed.positions[audio_segment.start][0] == pytest.approx(
                expected_origin
            )
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoDecoder:
        def __call__(self, _):
            return mx.zeros((1, 3, 22, 32, 32))

    class FakeAudioDecoder:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    def load(label, value):
        def loader(path, **__):
            if label == "dit":
                assert Path(path) == ref_dit
            loaded.append(label)
            return value()

        return loader

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.media, "reference_image_canvas", lambda *_, **__: (32, 32)
    )
    monkeypatch.setattr(
        pipeline.media,
        "load_rgb_image",
        lambda *_, **__: mx.zeros((1, 3, 32, 32)),
    )
    monkeypatch.setattr(
        pipeline.media, "reference_video_canvas", lambda _: (32, 32)
    )
    monkeypatch.setattr(
        pipeline.media,
        "load_rgb_video",
        lambda *_, **__: mx.zeros((1, 3, 22, 32, 32)),
    )

    def load_audio(path, *, max_seconds=None):
        if max_seconds is not None:
            assert max_seconds == pytest.approx(22 / 24)
            return mx.zeros((1, 2, 29_334))
        assert path == "standalone.wav"
        return mx.zeros((1, 2, 64_000))

    monkeypatch.setattr(
        pipeline.media,
        "load_stereo_audio",
        load_audio,
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_video_vae_encoder",
        load("video encoder", FakeVideoEncoder),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_audio_vae_encoder",
        load("audio encoder", FakeAudioEncoder),
    )
    monkeypatch.setattr(
        pipeline.loading,
        "load_multimodal_text_encoder",
        load("text/vision", FakeMultimodalEncoder),
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", load("dit", FakeDiT))
    monkeypatch.setattr(
        pipeline.loading, "load_video_vae", load("video decoder", FakeVideoDecoder)
    )
    monkeypatch.setattr(
        pipeline.loading, "load_audio_vae", load("audio decoder", FakeAudioDecoder)
    )
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    result = pipeline.generate(
        pipeline.GenerationConfig(
            "test input",
            width=32,
            height=32,
            frames=22,
            steps=2,
            references=(
                pipeline.Reference(audio="standalone.wav"),
                pipeline.Reference(image="reference.png"),
                pipeline.Reference(video="reference.mp4", audio="soundtrack.wav"),
            ),
        ),
        model_paths(tmp_path, ref_dit=ref_dit),
        RecordingGuard(),
    )

    assert loaded == [
        "video encoder",
        "audio encoder",
        "text/vision",
        "dit",
        "video decoder",
        "audio decoder",
    ]
    assert result.prompt_tokens == 4
