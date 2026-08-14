"""Tiny-model contracts for Qwen vision conditioning."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_h3 import vision


def tiny_config() -> vision.VisionConfig:
    return vision.VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        depth=2,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=6,
        deepstack_indexes=(0,),
    )


def test_config_derives_patch_head_and_merge_dimensions():
    config = tiny_config()

    assert config.head_dim == 4
    assert config.patch_dim == 24
    assert config.merge_dim == 32


def test_split_rope_identity_and_quarter_turn():
    value = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    identity = vision._apply_split_rope(
        value, mx.ones_like(value), mx.zeros_like(value)
    )
    quarter = vision._apply_split_rope(
        value, mx.zeros_like(value), mx.ones_like(value)
    )

    assert mx.array_equal(identity, value)
    assert quarter.tolist() == [[[-3.0, -4.0, 1.0, 2.0]]]


@pytest.mark.parametrize("frames", [1, 2])
def test_tiny_model_accepts_an_image_or_video_pair(frames):
    model = vision.QwenVisionModel(tiny_config())
    pixels = mx.random.uniform(shape=(frames, 3, 8, 8))

    output = model(pixels)

    assert output.grid_h == 4
    assert output.grid_w == 4
    assert output.merged.shape == (4, 6)
    assert len(output.deepstack) == 1
    assert output.deepstack[0].shape == (4, 6)
    assert mx.isfinite(output.merged).all().item()
    assert mx.isfinite(output.deepstack[0]).all().item()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 8, 8),
        (1, 1, 8, 8),
        (3, 3, 8, 8),
        (1, 3, 6, 8),
    ],
)
def test_patch_boundary_rejects_invalid_media_geometry(shape):
    model = vision.QwenVisionModel(tiny_config())

    with pytest.raises(ValueError):
        model._patches(mx.zeros(shape))
