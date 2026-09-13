import numpy as np

import mlx.core as mx
from mlx_nax_int import (
    affine_matmul,
    grouped_matmul,
    grouped_quantize,
    matmul,
    scaled_matmul,
)


def pack_int4(weight: np.ndarray) -> np.ndarray:
    assert weight.dtype == np.int8
    assert weight.ndim == 2 and weight.shape[1] % 2 == 0
    low = weight[:, 0::2].astype(np.uint8) & 0x0F
    high = (weight[:, 1::2].astype(np.uint8) & 0x0F) << 4
    return low | high


def check(bits: int, shape: tuple[int, int, int]) -> None:
    m, n, k = shape
    rng = np.random.default_rng(42 + bits + m + n + k)
    x = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    weight = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    encoded_weight = weight if bits == 8 else pack_int4(weight)
    expected = x.astype(np.int32) @ weight.astype(np.int32)

    actual = matmul(mx.array(x), mx.array(encoded_weight), bits)
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual), expected)
    print(f"W{bits}A8 {m}x{k} @ {k}x{n}: PASS")


def check_fused() -> None:
    rng = np.random.default_rng(91)
    x = rng.integers(-4, 5, size=(64, 64), dtype=np.int8)
    weight = rng.integers(-4, 5, size=(64, 64), dtype=np.int8)
    expected = x.astype(np.int32) @ weight.astype(np.int32)
    ones = mx.ones((64,), dtype=mx.bfloat16)

    scaled = scaled_matmul(mx.array(x), mx.array(weight), ones, ones)
    affine = affine_matmul(
        mx.array(x),
        mx.array(weight),
        ones,
        ones,
        mx.sum(mx.array(x).astype(mx.int32), axis=-1),
        mx.zeros((64,), dtype=mx.bfloat16),
    )
    mx.eval(scaled, affine)
    np.testing.assert_allclose(
        np.array(scaled.astype(mx.float32)), expected, rtol=8e-3, atol=4
    )
    np.testing.assert_allclose(
        np.array(affine.astype(mx.float32)), expected, rtol=8e-3, atol=4
    )
    print("fused scale/affine: PASS")


def check_grouped(group_size: int) -> None:
    rng = np.random.default_rng(100 + group_size)
    k = group_size * 2
    x = rng.integers(-2, 3, size=(64, k), dtype=np.int8)
    weight = rng.integers(-2, 3, size=(k, 64), dtype=np.int8)
    x_scale = rng.choice((0.5, 1.0), size=(64, 2)).astype(np.float32)
    weight_scale = rng.choice((0.5, 1.0), size=(2, 64)).astype(np.float32)
    expected = np.zeros((64, 64), dtype=np.float32)
    for group in range(2):
        group_slice = slice(group * group_size, (group + 1) * group_size)
        product = (
            x[:, group_slice].astype(np.int32)
            @ weight[group_slice].astype(np.int32)
        )
        expected += product * x_scale[:, group, None] * weight_scale[group]
    actual = grouped_matmul(
        mx.array(x),
        mx.array(weight),
        mx.array(x_scale).astype(mx.bfloat16),
        mx.array(weight_scale).astype(mx.bfloat16),
        group_size,
    )
    mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, rtol=8e-3, atol=4
    )
    print(f"group-{group_size}: PASS")


def check_quantize(group_size: int) -> None:
    rng = np.random.default_rng(200 + group_size)
    x = rng.normal(size=(64, group_size * 2)).astype(np.float32)
    source = mx.array(x).astype(mx.bfloat16)
    q, scales = grouped_quantize(source, group_size)
    grouped = source.reshape(64, 2, group_size)
    reference_scales = mx.maximum(
        mx.max(mx.abs(grouped), axis=-1, keepdims=True) / 127.0,
        1e-8,
    ).astype(mx.bfloat16)
    reference_q = mx.clip(
        mx.round(grouped / reference_scales), -127, 127
    ).astype(mx.int8)
    assert mx.array_equal(q.reshape(reference_q.shape), reference_q).item()
    assert mx.array_equal(scales, reference_scales[..., 0]).item()
    restored = q.astype(mx.float32).reshape(64, 2, group_size) * scales.astype(
        mx.float32
    )[..., None]
    reference = source.astype(mx.float32)
    error = mx.sqrt(mx.mean((restored.reshape(reference.shape) - reference) ** 2))
    relative = error / mx.sqrt(mx.mean(reference**2))
    mx.eval(relative)
    assert relative.item() < 0.01
    print(f"quantize-group-{group_size}: PASS")


if __name__ == "__main__":
    for test_shape in ((64, 64, 64), (64, 128, 128), (128, 64, 128)):
        check(8, test_shape)
        check(4, test_shape)
    check_fused()
    for size in (64, 256, 448, 896):
        check_grouped(size)
        check_quantize(size)
