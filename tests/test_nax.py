import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_h3 import nax


def quantized_linear(input_dims: int = 64, output_dims: int = 64):
    dense = nn.Linear(input_dims, output_dims, bias=False)
    return dense.to_quantized(group_size=32, bits=8)


def test_grouped_w8a8_requantizes_without_dense_source_weights():
    layer = nax.GroupedW8A8Linear(quantized_linear(), group_size=64)
    assert layer.weight.shape == (64, 64)
    assert layer.weight.dtype == mx.int8
    assert layer.scales.shape == (1, 64)
    assert layer.scales.dtype == mx.bfloat16


def test_grouped_w8a8_rejects_unsupported_geometry():
    with pytest.raises(ValueError, match="not divisible"):
        nax.GroupedW8A8Linear(
            quantized_linear(input_dims=64, output_dims=65),
            group_size=64,
        )


@pytest.mark.runtime
def test_grouped_w8a8_extension_matches_quantized_linear():
    extension = pytest.importorskip("mlx_nax_int")
    if not all(
        hasattr(extension, name)
        for name in ("grouped_quantize", "grouped_matmul")
    ):
        pytest.skip("mlx_nax_int lacks fused grouped W8A8 operations")

    base = quantized_linear()
    layer = nax.GroupedW8A8Linear(base, group_size=64)
    x = mx.random.normal((64, 64), dtype=mx.bfloat16)
    reference = base(x).astype(mx.float32)
    candidate = layer(x).astype(mx.float32)
    relative = mx.sqrt(mx.mean((candidate - reference) ** 2)) / mx.sqrt(
        mx.mean(reference**2)
    )
    assert relative.item() < 0.05
