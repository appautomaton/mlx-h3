"""Direct contracts for safetensors inspection and MLX model loading."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_h3 import loading, model, text_encoder


class TinyLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.packed = nn.Linear(64, 16)
        self.dense = nn.Linear(64, 16)


class CapturingModel:
    """Minimal loader boundary that records the exact transformed tensor list."""

    def __init__(self, parameters, *, config=None):
        self._parameters = parameters
        self.config = config
        self.loaded = None

    def parameters(self):
        return self._parameters

    def load_weights(self, weights):
        self.loaded = dict(weights)


def install_loader_double(
    monkeypatch,
    constructor: str,
    parameters: dict,
    checkpoint: dict,
    *,
    extra: dict | None = None,
    config=None,
):
    model_double = CapturingModel(parameters, config=config)
    stored = checkpoint | (extra or {})
    header = {key: {} for key in stored}
    monkeypatch.setattr(loading, constructor, lambda _: model_double)
    monkeypatch.setattr(loading, "read_header", lambda _: (header, {}))
    monkeypatch.setattr(loading.mx, "load", lambda _: stored)
    return model_double


def tiny_text_config() -> text_encoder.TextEncoderConfig:
    return text_encoder.TextEncoderConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10_000.0,
    )


def tiny_dit_config() -> model.H3Config:
    return model.H3Config(
        hidden_size=64,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=32,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=16,
        timestep_input_dim=16,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=4,
    )


def test_read_header_separates_metadata_without_loading_payload(tmp_path):
    path = tmp_path / "tiny.safetensors"
    mx.save_safetensors(
        str(path),
        {"value": mx.arange(8, dtype=mx.float32)},
        metadata={"origin": "unit-test"},
    )

    header, metadata = loading.read_header(path)

    assert set(header) == {"value"}
    assert header["value"]["shape"] == [8]
    assert metadata == {"origin": "unit-test"}


def test_read_header_rejects_non_object_metadata(tmp_path):
    path = tmp_path / "invalid-metadata.safetensors"
    encoded = json.dumps({"__metadata__": []}).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)

    with pytest.raises(ValueError, match="metadata must be an object"):
        loading.read_header(path)


def test_quantization_metadata_defaults_and_explicit_values():
    assert loading.quantization({}) == {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
    }
    assert loading.quantization(
        {
            "quantization.group_size": "64",
            "quantization.bits": "4",
            "quantization.mode": "mxfp4",
        }
    ) == {"group_size": 64, "bits": 4, "mode": "mxfp4"}


def test_prepare_quantizes_only_modules_declared_by_checkpoint():
    model = TinyLinears()
    header = {"packed.scales": {}}

    result = loading.prepare(
        model,
        header,
        {
            "quantization.group_size": "32",
            "quantization.bits": "8",
            "quantization.mode": "affine",
        },
    )

    assert result is model
    assert isinstance(model.packed, nn.QuantizedLinear)
    assert isinstance(model.dense, nn.Linear)
    assert loading.quantized_modules(header) == {"packed"}


def test_check_reports_missing_and_unexpected_tensor_names():
    model = TinyLinears()
    stored = {key for key, _ in tree_flatten(model.parameters())}
    removed = min(stored)
    header = {key: {} for key in stored - {removed}}
    header["unexpected.weight"] = {}

    missing, unexpected = loading.check(model, header)

    assert missing == {removed}
    assert unexpected == {"unexpected.weight"}


def test_tiny_text_encoder_round_trips_through_loader(tmp_path):
    mx.random.seed(7)
    config = tiny_text_config()
    source = text_encoder.TextEncoder(config)
    weights = dict(tree_flatten(source.parameters()))
    mx.eval(weights)
    path = tmp_path / "text.safetensors"
    mx.save_safetensors(str(path), weights)

    loaded = loading.load_text_encoder(path, config)
    token_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = source(token_ids)
    actual = loaded(token_ids)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_tiny_dit_round_trips_through_loader(tmp_path):
    mx.random.seed(11)
    config = tiny_dit_config()
    source = model.MiniMaxH3(config)
    expected = dict(tree_flatten(source.parameters()))
    mx.eval(expected)
    path = tmp_path / "dit.safetensors"
    mx.save_safetensors(str(path), expected)

    loaded = loading.load_dit(path, config)
    actual = dict(tree_flatten(loaded.parameters()))
    mx.eval(actual)

    assert actual.keys() == expected.keys()
    assert all(
        mx.array_equal(actual[key], expected[key])
        for key in expected
    )


def test_text_loader_rejects_a_mismatched_tree_before_payload_load(tmp_path):
    path = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(path), {"model.unexpected": mx.zeros((1,))})

    with pytest.raises(ValueError, match="does not match the decoder tree"):
        loading.load_text_encoder(path, tiny_text_config())


def test_video_decoder_loader_normalizes_statistics_and_projection(
    monkeypatch
):
    parameters = {
        "latents_mean": mx.zeros((2,)),
        "latents_std": mx.zeros((2,)),
        "post_quant_conv": {"weight": mx.zeros((2, 2))},
    }
    checkpoint = {
        "latents_mean": mx.ones((2,), dtype=mx.float16),
        "latents_std": mx.ones((2,), dtype=mx.float16),
        "post_quant_conv.weight": mx.arange(4).reshape(2, 2, 1, 1),
    }
    model_double = install_loader_double(
        monkeypatch,
        "VideoVAE",
        parameters,
        checkpoint,
        extra={"encoder.unused": mx.zeros((1,))},
        config=SimpleNamespace(latent_channels=2),
    )

    assert loading.load_video_vae("ignored") is model_double

    assert model_double.loaded["latents_mean"].dtype == mx.float32
    assert model_double.loaded["latents_std"].dtype == mx.float32
    assert model_double.loaded["post_quant_conv.weight"].shape == (2, 2)
    assert "encoder.unused" not in model_double.loaded


def test_video_encoder_loader_transposes_torch_conv3d(monkeypatch):
    parameters = {
        "latents_mean": mx.zeros((2,)),
        "latents_std": mx.zeros((2,)),
        "encoder": {"conv": {"weight": mx.zeros((2, 4, 5, 6, 3))}},
    }
    source_weight = mx.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    checkpoint = {
        "latents_mean": mx.ones((2,), dtype=mx.float16),
        "latents_std": mx.ones((2,), dtype=mx.float16),
        "encoder.conv.weight": source_weight,
    }
    model_double = install_loader_double(
        monkeypatch,
        "VideoVAEEncoder",
        parameters,
        checkpoint,
        extra={"decoder.unused": mx.zeros((1,))},
    )

    assert loading.load_video_vae_encoder("ignored") is model_double

    transformed = model_double.loaded["encoder.conv.weight"]
    assert transformed.shape == (2, 4, 5, 6, 3)
    assert mx.array_equal(transformed, mx.transpose(source_weight, (0, 2, 3, 4, 1)))
    assert model_double.loaded["latents_mean"].dtype == mx.float32


def test_audio_decoder_loader_distinguishes_conv_and_transposed_conv(
    monkeypatch
):
    parameters = {
        "decoder": {
            "conv": {"weight": mx.zeros((2, 5, 3))},
            "ups": {"0": {"weight": mx.zeros((4, 5, 6))}},
        }
    }
    conv = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    transposed = mx.arange(6 * 4 * 5).reshape(6, 4, 5)
    checkpoint = {
        "decoder.conv.weight": conv,
        "decoder.ups.0.weight": transposed,
    }
    model_double = install_loader_double(
        monkeypatch,
        "AudioVAE",
        parameters,
        checkpoint,
        extra={"encoder.unused": mx.zeros((1,))},
    )

    assert loading.load_audio_vae("ignored") is model_double

    assert mx.array_equal(
        model_double.loaded["decoder.conv.weight"], mx.transpose(conv, (0, 2, 1))
    )
    assert mx.array_equal(
        model_double.loaded["decoder.ups.0.weight"],
        mx.transpose(transposed, (1, 2, 0)),
    )


def test_audio_encoder_loader_normalizes_statistics_and_conv_layout(
    monkeypatch
):
    parameters = {
        "latents_mean": mx.zeros((2,)),
        "latents_std": mx.zeros((2,)),
        "encoder": {"conv": {"weight": mx.zeros((2, 5, 3))}},
    }
    conv = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    checkpoint = {
        "latents_mean": mx.ones((2,), dtype=mx.float16),
        "latents_std": mx.ones((2,), dtype=mx.float16),
        "encoder.conv.weight": conv,
    }
    model_double = install_loader_double(
        monkeypatch,
        "AudioVAEEncoder",
        parameters,
        checkpoint,
        extra={"decoder.unused": mx.zeros((1,))},
    )

    assert loading.load_audio_vae_encoder("ignored") is model_double

    assert model_double.loaded["latents_mean"].dtype == mx.float32
    assert mx.array_equal(
        model_double.loaded["encoder.conv.weight"], mx.transpose(conv, (0, 2, 1))
    )


def test_multimodal_loader_flattens_the_torch_patch_projection(monkeypatch):
    parameters = {
        "model": {"weight": mx.zeros((2, 2))},
        "visual": {"patch_embed": {"proj": {"weight": mx.zeros((2, 24))}}},
    }
    patch_weight = mx.arange(2 * 3 * 2 * 2 * 2).reshape(2, 3, 2, 2, 2)
    checkpoint = {
        "model.weight": mx.ones((2, 2)),
        "visual.patch_embed.proj.weight": patch_weight,
    }
    model_double = CapturingModel(parameters)
    monkeypatch.setattr(
        loading, "MultimodalTextEncoder", lambda _: model_double
    )
    monkeypatch.setattr(
        loading, "read_header", lambda _: ({key: {} for key in checkpoint}, {})
    )
    monkeypatch.setattr(loading, "prepare", lambda model, *_: model)
    monkeypatch.setattr(loading.mx, "load", lambda _: checkpoint)

    assert loading.load_multimodal_text_encoder("ignored") is model_double

    assert model_double.loaded["visual.patch_embed.proj.weight"].shape == (2, 24)
    assert mx.array_equal(
        model_double.loaded["visual.patch_embed.proj.weight"],
        patch_weight.reshape(2, -1),
    )
