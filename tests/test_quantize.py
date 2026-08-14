"""Small deterministic contracts for the development quantizer."""

from __future__ import annotations

import sys

import mlx.core as mx
import pytest

from dev import quantize
from mlx_h3 import loading


def metadata(dtype="BF16", shape=(2, 64)):
    return {"dtype": dtype, "shape": list(shape)}


@pytest.mark.parametrize(
    ("name", "meta", "expected", "reason"),
    [
        ("block.weight", metadata(), True, ""),
        ("block.bias", metadata(), False, "not a .weight"),
        ("embed_tokens.weight", metadata(), False, "lookup table"),
        ("block.weight", metadata("F32"), False, "dtype F32"),
        ("block.weight", metadata(shape=(64,)), False, "rank 1"),
    ],
)
def test_should_quantize_explains_every_selection(name, meta, expected, reason):
    assert quantize.should_quantize(name, meta) == (expected, reason)


def test_should_quantize_uses_the_requested_group_size():
    meta = metadata(shape=(2, 32))

    assert quantize.should_quantize("block.weight", meta, group_size=32) == (
        True,
        "",
    )
    assert quantize.should_quantize("block.weight", meta, group_size=64) == (
        False,
        "last dim 32 % 64",
    )


def test_main_writes_only_group_compatible_quantized_modules(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "quantized.safetensors"
    tensors = {
        "wide.weight": mx.ones((2, 64), dtype=mx.bfloat16),
        "narrow.weight": mx.ones((2, 32), dtype=mx.bfloat16),
        "embed_tokens.weight": mx.ones((4, 64), dtype=mx.bfloat16),
        "head.weight": mx.ones((2, 64), dtype=mx.float32),
    }
    mx.save_safetensors(str(source), tensors)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize.py",
            str(source),
            str(destination),
            "--group-size",
            "64",
        ],
    )

    assert quantize.main() == 0

    header, output_metadata = loading.read_header(destination)
    assert loading.quantized_modules(header) == {"wide"}
    assert "narrow.scales" not in header
    assert header["narrow.weight"]["dtype"] == "BF16"
    assert header["embed_tokens.weight"]["dtype"] == "BF16"
    assert header["head.weight"]["dtype"] == "F32"
    assert output_metadata == {
        "quantization.mode": "affine",
        "quantization.bits": "8",
        "quantization.group_size": "64",
        "source": source.name,
    }
