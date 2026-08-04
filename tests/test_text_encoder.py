"""Structural and causal tests for the truncated Qwen3-VL text decoder."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_h3 import loading, text_encoder


def tiny_config() -> text_encoder.TextEncoderConfig:
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


def test_tiny_encoder_returns_finite_raw_hidden_states():
    model = text_encoder.TextEncoder(tiny_config())
    output = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(output)
    assert output.shape == (1, 3, 16)
    assert mx.isfinite(output).all().item()
    keys = {key for key, _ in tree_flatten(model.parameters())}
    assert "model.norm.weight" not in keys
    assert not any(key.startswith("lm_head.") for key in keys)


def test_attention_is_causal():
    model = text_encoder.TextEncoder(tiny_config())
    prefix = model(mx.array([[3, 5]], dtype=mx.int32))
    extended = model(mx.array([[3, 5, 7]], dtype=mx.int32))
    mx.eval(prefix, extended)
    assert mx.allclose(prefix, extended[:, :2], rtol=1e-5, atol=1e-5)


def test_invalid_geometry_and_empty_sequence_fail_at_the_boundary():
    with pytest.raises(ValueError, match="divisible"):
        text_encoder.TextEncoderConfig(num_attention_heads=7, num_key_value_heads=2)
    model = text_encoder.TextEncoder(tiny_config())
    with pytest.raises(ValueError, match="non-empty"):
        model(mx.array([[]], dtype=mx.int32))


@pytest.mark.checkpoint
def test_checkpoint_text_subtree_is_exactly_the_truncated_decoder():
    path = (
        Path(__file__).resolve().parents[1]
        / "weights/mlx-8bit/te_qwen3vl_a8g32.safetensors"
    )
    if not path.exists():
        pytest.skip("quantized text encoder checkpoint absent")
    header, metadata = loading.read_header(path)
    text_keys = {key for key in header if key.startswith("model.")}
    assert len(text_keys) == 1251
    assert sum(key.endswith(".scales") for key in text_keys) == 350
    assert {
        int(key.split(".")[2])
        for key in text_keys
        if key.startswith("model.layers.")
    } == set(range(50))
    assert "model.norm.weight" not in text_keys
    assert not any(key.startswith("lm_head.") for key in text_keys)
    assert metadata["quantization.bits"] == "8"
