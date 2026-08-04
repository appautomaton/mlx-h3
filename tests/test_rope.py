"""Parity for the rotary tables against both committed fixtures.

minimax_h3_layout.json  -> real hyperparameters (16 freqs, rot_dim 96), 3 probe
                           positions chosen to exercise integer, fractional and
                           large coordinates.
minimax_h3_dit.safetensors -> a toy block (4 freqs, rot_dim 24) carrying the
                           cos/sin actually fed to its attention, so the table
                           shape and axis order are pinned end to end.
"""

from __future__ import annotations

import json
import mlx.core as mx
import pytest

from mlx_h3 import rope

pytestmark = pytest.mark.fixture


@pytest.fixture(scope="module")
def golden(local_file) -> dict:
    path = local_file("MLX_H3_LAYOUT_FIXTURE")
    return json.loads(path.read_text())["rope_freqs"]


@pytest.fixture(scope="module")
def dit(local_file) -> dict:
    return mx.load(str(local_file("MLX_H3_DIT_FIXTURE")))


def maxabs(a: mx.array, b: mx.array) -> float:
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


def test_constants(golden):
    assert golden["rot_dim"] == rope.ROT_DIM
    assert len(golden["inv_freq"]) == rope.INV_FREQ_LEN


def test_angles(golden):
    positions = mx.array(golden["positions"], dtype=mx.float32)
    inv_freq = mx.array(golden["inv_freq"], dtype=mx.float32)
    want = mx.array(golden["angles"], dtype=mx.float32)
    half = rope.ROT_DIM // 2

    got = rope.angles(positions, inv_freq)
    assert got.shape == (positions.shape[0], half)
    assert maxabs(got, want[:, :half]) == 0.0

    # The reference's [S, 96] is its half duplicated; that is why we build one.
    assert maxabs(want[:, :half], want[:, half:]) == 0.0


def test_angle_axis_order(golden):
    """t occupies frequencies 0..15, h 16..31, w 32..47 -- not interleaved."""
    inv_freq = mx.array(golden["inv_freq"], dtype=mx.float32)
    n = rope.INV_FREQ_LEN
    got = rope.angles(mx.array([[2.0, 3.0, 5.0]], dtype=mx.float32), inv_freq)[0]
    for axis, coord in enumerate((2.0, 3.0, 5.0)):
        assert maxabs(got[axis * n : (axis + 1) * n], coord * inv_freq) == 0.0


def test_tables_match_dit_fixture(dit):
    """Toy block: positions -> cos/sin exactly as its attention consumed them."""
    got = rope.angles(dit["x.positions"], dit["rope.inv_freq"])
    cos, sin = rope.tables(got)
    assert cos.shape == (dit["x.positions"].shape[0], 1, dit["x.rope_cos"].shape[-1])
    assert maxabs(cos[:, 0], dit["x.rope_cos"]) < 1e-6
    assert maxabs(sin[:, 0], dit["x.rope_sin"]) < 1e-6


def test_apply_is_a_rotation(golden):
    """Split-half rotation preserves each pair's norm and leaves the tail alone."""
    inv_freq = mx.array(golden["inv_freq"], dtype=mx.float32)
    s, heads, head_dim = 7, 3, 128
    positions = mx.random.uniform(-40.0, 700.0, (s, 3))
    cos, sin = rope.tables(rope.angles(positions, inv_freq))
    x = mx.random.normal((s, heads, head_dim))
    got = rope.apply(x, cos, sin)

    assert got.shape == x.shape
    assert maxabs(got[..., rope.ROT_DIM :], x[..., rope.ROT_DIM :]) == 0.0

    half = rope.ROT_DIM // 2
    for lo, hi in ((slice(0, half), slice(half, rope.ROT_DIM)),):
        before = x[..., lo] ** 2 + x[..., hi] ** 2
        after = got[..., lo] ** 2 + got[..., hi] ** 2
        assert maxabs(before, after) < 1e-4


def test_apply_zero_angle_is_identity(golden):
    inv_freq = mx.array(golden["inv_freq"], dtype=mx.float32)
    cos, sin = rope.tables(rope.angles(mx.zeros((5, 3)), inv_freq))
    x = mx.random.normal((5, 2, 128))
    assert maxabs(rope.apply(x, cos, sin), x) == 0.0


def test_apply_partial_rotary_boundary(golden):
    """A 128-dim head rotates 96 dims: pairs are (i, i+48), tail is 96..127."""
    inv_freq = mx.array(golden["inv_freq"], dtype=mx.float32)
    positions = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    cos, sin = rope.tables(rope.angles(positions, inv_freq))
    half = rope.ROT_DIM // 2

    x = mx.zeros((1, 1, 128))
    x[0, 0, 0] = 1.0  # frequency 0 of the t axis, angle = 1.0 rad
    got = rope.apply(x, cos, sin)
    assert got[0, 0, 0].item() == pytest.approx(mx.cos(mx.array(1.0)).item(), abs=1e-6)
    assert got[0, 0, half].item() == pytest.approx(
        mx.sin(mx.array(1.0)).item(), abs=1e-6
    )
    assert mx.abs(got[0, 0, 1:half]).max().item() == 0.0
