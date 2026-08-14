"""Weightless end-to-end tests for staged generation.

Tiny stand-ins preserve tensor and call-order contracts without loading checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import pipeline
from tests._support.pipeline import RecordingGuard, model_paths


def test_generate_runs_all_models_in_separate_phases(monkeypatch, tmp_path: Path):
    loaded = []

    class FakeTokenizer:
        def encode_prompt(self, prompt):
            assert prompt == "test input"
            return [4, 5, 6]

    class FakeTextEncoder:
        def __call__(self, token_ids):
            return mx.zeros((1, token_ids.shape[1], 5120), dtype=mx.bfloat16)

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
            sigma_video,
            sigma_audio,
            step_index,
        ):
            assert text.shape == (3, 5376)
            assert packed.seq_len > 3
            assert sigma_video > 0 and sigma_audio > 0
            assert step_index in (0, 1, 2)
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoVAE:
        def __call__(self, latent):
            assert latent.shape == (1, 24, 2, 2, 2)
            return mx.zeros((1, 3, 5, 32, 32))

    class FakeAudioVAE:
        def __call__(self, latent):
            assert latent.shape == (1, 32, 2, 8)
            return mx.zeros((1, 2, 6400))

    def loader(label, value):
        def load(_, **kwargs):
            if label == "dit":
                assert len(kwargs["plans"]) == 3
                assert kwargs["modulation_dtype"] == mx.bfloat16
                assert kwargs["nax_group_size"] == 896
            loaded.append(label)
            return value()

        return load

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.loading, "load_text_encoder", loader("text", FakeTextEncoder)
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", loader("dit", FakeDiT))
    monkeypatch.setattr(
        pipeline.loading, "load_video_vae", loader("video", FakeVideoVAE)
    )
    monkeypatch.setattr(
        pipeline.loading, "load_audio_vae", loader("audio", FakeAudioVAE)
    )
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    result = pipeline.generate(
        pipeline.GenerationConfig(
            "test input", width=32, height=32, frames=5, steps=3
        ),
        model_paths(tmp_path),
        RecordingGuard(),
        nax_group_size=896,
    )
    assert loaded == ["text", "dit", "video", "audio"]
    assert result.frames.shape == (1, 3, 5, 32, 32)
    assert result.audio.shape == (1, 2, 6400)
    assert result.prompt_tokens == 3
    assert result.seed == 42


def test_generate_rejects_non_finite_decoded_media(monkeypatch, tmp_path: Path):
    class FakeTokenizer:
        def encode_prompt(self, _):
            return [4]

    class FakeTextEncoder:
        def __call__(self, token_ids):
            return mx.zeros((1, token_ids.shape[1], 5120), dtype=mx.bfloat16)

    class FakeDiT:
        def refine_text(self, states):
            return mx.zeros((states.shape[0], 5376), dtype=mx.bfloat16)

        def __call__(self, video, audio, *args, **kwargs):
            return mx.zeros_like(video), mx.zeros_like(audio)

    class BadVideoVAE:
        def __call__(self, _):
            return mx.full((1, 3, 5, 32, 32), float("nan"))

    class FakeAudioVAE:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    monkeypatch.setattr(
        pipeline.tokenizer.QwenTokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline.loading, "load_text_encoder", lambda _: FakeTextEncoder()
    )
    monkeypatch.setattr(pipeline.loading, "load_dit", lambda _, **__: FakeDiT())
    monkeypatch.setattr(pipeline.loading, "load_video_vae", lambda _: BadVideoVAE())
    monkeypatch.setattr(pipeline.loading, "load_audio_vae", lambda _: FakeAudioVAE())
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)

    with pytest.raises(ValueError, match="non-finite frames"):
        pipeline.generate(
            pipeline.GenerationConfig(
                "test", width=32, height=32, frames=5, steps=2
            ),
            model_paths(tmp_path),
            RecordingGuard(),
        )


def test_generate_fl2va_keeps_image_models_in_separate_phases(
    monkeypatch, tmp_path: Path
):
    loaded = []

    class FakeTokenizer:
        def encode(self, _):
            return [4, 5]

        def encode_prompt(self, _):
            return [4, 5]

    class FakeVideoEncoder:
        def __call__(self, image):
            assert image.shape == (1, 3, 32, 32)
            return mx.zeros((1, 24, 1, 2, 2), dtype=mx.float16)

    class FakeMultimodalEncoder:
        def encode_fl2va(self, tokenizer, prompt, images):
            assert tokenizer.encode(prompt) == [4, 5]
            assert len(images) == 1
            return (
                mx.zeros((6, 5120), dtype=mx.bfloat16),
                mx.array([1, 0, 0, 0, 1, 1], dtype=mx.int32),
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
            assert text.shape == (6, 5376)
            assert cond_video_rows.shape == (1, 96)
            assert tuple(text_tags) == (1, 0, 0, 0, 1, 1)
            assert [segment.kind for segment in packed.segments] == [
                "text",
                "cond",
                "audio",
                "video",
            ]
            return mx.zeros_like(video), mx.zeros_like(audio)

    class FakeVideoDecoder:
        def __call__(self, _):
            return mx.zeros((1, 3, 5, 32, 32))

    class FakeAudioDecoder:
        def __call__(self, _):
            return mx.zeros((1, 2, 6400))

    def load(label, value):
        def loader(*_, **__):
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
        "load_rgb_image",
        lambda *_, **__: mx.zeros((1, 3, 32, 32)),
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
            first_frame="ignored.png",
        ),
        model_paths(tmp_path),
        RecordingGuard(),
    )

    assert loaded == [
        "video encoder",
        "text/vision",
        "dit",
        "video decoder",
        "audio decoder",
    ]
    assert result.prompt_tokens == 6
    assert result.frames.shape == (1, 3, 5, 32, 32)
