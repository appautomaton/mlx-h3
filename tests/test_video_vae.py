"""Structural tests for the MiniMax-H3 visual VAE encoder and decoder.

VALIDATION TIER. Temporal and tile arithmetic is independently testable and
agrees with the published geometry. Decoder tensor values have no executed fixture;
the tiny model tests shape, ordering and data flow only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import video_vae


def tiny_config() -> video_vae.VideoVAEConfig:
    return video_vae.VideoVAEConfig(
        latent_channels=4,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=32,
        num_register_tokens=1,
        patch_size=2,
        tile_size=32,
        tile_overlap=8,
    )


def test_released_geometry_is_derived():
    config = video_vae.VideoVAEConfig()
    assert config.hidden_size == 2048
    assert config.rope_dim == 48
    assert config.output_patch_dim == 3072
    assert config.tokens_chunk_size == 5
    assert config.frame_pre_padding == 3
    assert config.token_overlap == 2
    assert config.frame_overlap == 5


@pytest.mark.parametrize(
    ("frames", "latent_t"),
    [(5, 2), (22, 7), (56, 17), (124, 37), (362, 107)],
)
def test_temporal_plan_recovers_the_frame_count(frames, latent_t):
    plan = video_vae.temporal_plan(latent_t)
    assert plan.output_frames == frames
    assert plan.num_chunks >= 1
    assert plan.padded_length == latent_t + plan.pad_tokens


def test_shortest_clip_pads_to_one_real_chunk():
    plan = video_vae.temporal_plan(2)
    assert plan.num_chunks == 1
    assert plan.pad_tokens >= video_vae.VideoVAEConfig().tokens_chunk_size
    assert plan.output_frames == 5


@pytest.mark.parametrize("extent", [256, 288, 480, 512, 768, 864, 1024, 1344])
def test_tile_plan_covers_the_axis_exactly(extent):
    plan = video_vae.split_tiles(extent)
    kept = sum(
        plan.length - plan.overlaps[index]
        if index < len(plan.overlaps)
        else plan.length
        for index in range(len(plan.starts))
    )
    assert kept == extent
    assert plan.starts[-1] + plan.length == extent
    assert all(start % 16 == 0 for start in plan.starts)
    assert all(overlap % 16 == 0 for overlap in plan.overlaps)


def test_blend_uses_the_reference_half_open_ramp():
    previous = mx.ones((1, 1, 1, 1, 4))
    current = mx.zeros((1, 1, 1, 1, 4))
    got = video_vae._blend(previous, current, 4, -1)
    assert mx.allclose(got, mx.array([[[[[1.0, 0.75, 0.5, 0.25]]]]]))


def test_tiny_decoder_unpacks_voxels_to_pixel_blocks():
    config = tiny_config()
    decoder = video_vae.VideoDecoder(config)
    latent = mx.random.normal((1, config.latent_channels, 2, 2, 3))
    output = decoder(latent)
    mx.eval(output)
    assert output.shape == (1, 3, 8, 4, 6)
    assert mx.isfinite(output).all().item()


def test_tiny_vae_decodes_the_shortest_temporal_shape():
    config = tiny_config()
    vae = video_vae.VideoVAE(config)
    latent = mx.random.normal((1, config.latent_channels, 2, 2, 3))
    output = vae(latent)
    mx.eval(output)
    assert output.shape == (1, 3, 5, 4, 6)
    assert mx.isfinite(output).all().item()
    assert mx.min(output).item() >= 0.0
    assert mx.max(output).item() <= 1.0


def test_tiny_encoder_returns_a_normalized_single_frame_mean():
    config = video_vae.VideoVAEConfig(
        latent_channels=4,
        encoder_base_channels=4,
        encoder_channel_multipliers=(1, 2),
        encoder_num_res_blocks=1,
        encoder_space_down=(2, 2),
        encoder_time_down=(1, 2),
        encoder_norm_groups=1,
        tile_size=32,
        tile_overlap=8,
    )
    encoder = video_vae.VideoVAEEncoder(config)
    pixels = mx.random.uniform(shape=(1, 3, 8, 12))
    latent = encoder(pixels)
    mx.eval(latent)
    assert latent.shape == (1, 4, 1, 2, 3)
    assert mx.isfinite(latent).all().item()


def test_decoder_checkpoint_names_are_fully_accounted_for():
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "weights/bf16/vae/minimax_h3_video_vae_fp16.safetensors"
    )
    if not checkpoint.exists():
        pytest.skip("video VAE checkpoint absent")
    with checkpoint.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    header.pop("__metadata__", None)
    stored = {
        key
        for key in header
        if not key.startswith(("encoder.", "quant_conv."))
    }

    expected = {
        "latents_mean",
        "latents_std",
        "post_quant_conv.weight",
        "post_quant_conv.bias",
        "decoder.mask_token",
        "decoder.x_embedder.weight",
        "decoder.x_embedder.bias",
        "decoder.register_tokens",
        "decoder.norm_out.weight",
        "decoder.norm_out.bias",
        "decoder.proj_out.weight",
        "decoder.proj_out.bias",
    }
    suffixes = (
        "attn.to_qkv.weight",
        "attn.to_qkv.bias",
        "attn.to_out.weight",
        "attn.to_out.bias",
        "ff.w1.weight",
        "ff.w1.bias",
        "ff.w2.weight",
        "ff.w2.bias",
        "norm1.weight",
        "norm2.weight",
        "scale1",
        "scale2",
    )
    for index in range(video_vae.VideoVAEConfig().num_layers):
        expected.update(
            f"decoder.transformer_blocks.{index}.{suffix}" for suffix in suffixes
        )
    assert stored == expected
