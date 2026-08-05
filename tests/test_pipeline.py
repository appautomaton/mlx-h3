"""Lifecycle tests for staged model residency.

These tests use tiny stand-ins. Real phase memory is checked by runtime scripts;
the unit contract here is reference release, materialization, check cadence and
scalar-only optional telemetry.
"""

from __future__ import annotations

import weakref
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import memory, pipeline


class FakeGuard:
    def __init__(self):
        self.notes = []

    def check(self, note):
        self.notes.append(note)
        value = len(self.notes)
        return memory.Sample(value, value + 10, 0, 0, 0, 0, 100)


def fake_model_paths(
    tmp_path: Path, *, ref_dit: Path | None = None
) -> pipeline.ModelPaths:
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


def test_run_phase_materializes_output_and_releases_model(monkeypatch):
    released = []
    model_refs = []
    reports = []

    class Model:
        pass

    def load():
        model = Model()
        model_refs.append(weakref.ref(model))
        return model

    def execute(model):
        assert model_refs[0]() is model
        return mx.arange(4, dtype=mx.float32) + 1

    monkeypatch.setattr(pipeline.memory, "release", lambda: released.append(True))
    guard = FakeGuard()
    output = pipeline.run_phase(
        "dit", load, execute, guard, on_report=reports.append
    )

    assert output.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert model_refs[0]() is None
    assert released == [True]
    assert guard.notes == [
        "dit / loaded",
        "dit / complete",
        "dit / released",
    ]
    assert len(reports) == 1
    assert reports[0].label == "dit"
    assert reports[0].active_after_load == 1
    assert reports[0].active_after_run == 2
    assert reports[0].active_after_release == 3
    assert reports[0].peak == 13


def test_release_path_builds_no_report(monkeypatch):
    monkeypatch.setattr(pipeline.memory, "release", lambda: None)
    guard = FakeGuard()
    output = pipeline.run_phase(
        "text", lambda: object(), lambda _: mx.array([7]), guard
    )
    assert output.item() == 7
    assert guard.notes[-1] == "text / released"


def test_failed_phase_still_releases_the_model(monkeypatch):
    released = []
    model_refs = []

    class Model:
        pass

    def load():
        model = Model()
        model_refs.append(weakref.ref(model))
        return model

    def fail(_):
        raise RuntimeError("phase failed")

    monkeypatch.setattr(pipeline.memory, "release", lambda: released.append(True))
    with pytest.raises(RuntimeError, match="phase failed"):
        pipeline.run_phase("vae", load, fail, FakeGuard())

    assert model_refs[0]() is None
    assert released == [True]


def test_generation_config_rejects_only_real_resource_boundaries():
    pipeline.GenerationConfig("input", width=32, height=32, frames=5)
    with pytest.raises(ValueError, match="multiples"):
        pipeline.GenerationConfig("input", width=33, height=32)
    with pytest.raises(ValueError, match="pixel limit"):
        pipeline.GenerationConfig("input", width=2048, height=1024)
    with pytest.raises(ValueError, match="15 second"):
        pipeline.GenerationConfig("input", frames=363)
    with pytest.raises(ValueError, match="steps"):
        pipeline.GenerationConfig("input", steps=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.GenerationConfig(
            "input",
            first_frame="first.png",
            references=(pipeline.Reference(image="ref.png"),),
        )
    with pytest.raises(ValueError, match="requires an image or video"):
        pipeline.GenerationConfig(
            "input", references=(pipeline.Reference(audio="ref.wav"),)
        )
    with pytest.raises(ValueError, match="must contain"):
        pipeline.Reference(image="ref.png", audio="ref.wav")


def test_model_paths_report_all_missing_files_before_loading(tmp_path: Path):
    paths = pipeline.ModelPaths(
        tokenizer=tmp_path / "tokenizer.json",
        text_encoder=tmp_path / "text.safetensors",
        dit=tmp_path / "fl2va.safetensors",
        ref_dit=tmp_path / "ref2va.safetensors",
        video_vae=tmp_path / "video.safetensors",
        audio_vae=tmp_path / "audio.safetensors",
    )

    with pytest.raises(FileNotFoundError) as error:
        paths.validate(ref2va=True)

    message = str(error.value)
    assert "tokenizer:" in message
    assert "text encoder:" in message
    assert "Ref2VA DiT:" in message
    assert "FL2VA DiT:" not in message
    assert "Video VAE:" in message
    assert "Audio VAE:" in message


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
        fake_model_paths(tmp_path),
        FakeGuard(),
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
            fake_model_paths(tmp_path),
            FakeGuard(),
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
        fake_model_paths(tmp_path),
        FakeGuard(),
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
        fake_model_paths(tmp_path, ref_dit=ref_dit),
        FakeGuard(),
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
        fake_model_paths(tmp_path, ref_dit=ref_dit),
        FakeGuard(),
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
        fake_model_paths(tmp_path, ref_dit=ref_dit),
        FakeGuard(),
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
