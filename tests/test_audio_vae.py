"""Structural tests for the MiniMax-H3 BigVGAN audio decoder."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import audio_vae


def tiny_config() -> audio_vae.AudioVAEConfig:
    return audio_vae.AudioVAEConfig(
        latent_channels=4,
        latent_dim=8,
        decoder_dim=16,
        upsample_rates=(2, 2),
        upsample_kernels=(4, 4),
        resblock_kernels=(3,),
        resblock_dilations=((1,),),
        sample_rate=16,
    )


def test_released_rates_map_40_hz_to_32_khz():
    config = audio_vae.AudioVAEConfig()
    assert config.hop_length == 800
    assert config.sample_rate // config.hop_length == 40
    assert len(config.upsample_rates) == 7


@pytest.mark.parametrize("frames", [5, 22, 56, 124, 362])
def test_audio_and_video_durations_stay_within_one_latent_frame(frames):
    audio_t = round(frames / 24 * 40)
    audio_seconds = audio_t * audio_vae.AudioVAEConfig().hop_length / 32000
    video_seconds = frames / 24
    assert abs(audio_seconds - video_seconds) < 0.026


def test_alias_free_activation_preserves_length():
    activation = audio_vae.Activation1d(3)
    x = mx.random.normal((2, 11, 3))
    y = activation(x)
    mx.eval(y)
    assert y.shape == x.shape
    assert mx.isfinite(y).all().item()


def test_tiny_decoder_outputs_independent_stereo_waveforms():
    config = tiny_config()
    vae = audio_vae.AudioVAE(config)
    latent = mx.zeros((1, config.latent_channels, 2, 5))
    latent[:, :, 1, :] = 1.0
    waveform = vae(latent)
    mx.eval(waveform)
    assert waveform.shape == (1, 2, 5 * config.hop_length)
    assert mx.isfinite(waveform).all().item()
    assert mx.max(mx.abs(waveform)).item() <= 1.0
    assert not mx.allclose(waveform[:, 0], waveform[:, 1])


def test_tiny_encoder_returns_normalized_stereo_latents():
    config = audio_vae.AudioVAEConfig(
        latent_channels=4,
        latent_dim=32,
        encoder_dim=1,
        downsample_rates=(2, 2),
        encoder_attention_heads=1,
        decoder_dim=16,
        upsample_rates=(2, 2),
        upsample_kernels=(4, 4),
        resblock_kernels=(3,),
        resblock_dilations=((1,),),
        sample_rate=16,
    )
    vae = audio_vae.AudioVAEEncoder(config)
    latent = vae(mx.random.normal((1, 2, 13)))
    mx.eval(latent)
    assert latent.shape == (1, 4, 2, 4)
    assert mx.isfinite(latent).all().item()


def test_checkpoint_has_the_released_decoder_geometry():
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "weights/bf16/vae/minimax_h3_audio_vae_fp32.safetensors"
    )
    if not checkpoint.exists():
        pytest.skip("audio VAE checkpoint absent")
    with checkpoint.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    header.pop("__metadata__", None)
    assert header["dec_in_proj.weight"]["shape"] == [2048, 32, 1]
    assert header["decoder.conv_pre.weight"]["shape"] == [1024, 2048, 7]
    assert header["decoder.conv_post.weight"]["shape"] == [1, 8, 7]
    assert header["encoder.block.0.weight"]["shape"] == [64, 1, 7]
    assert header["encoder.block.7.weight"]["shape"] == [2048, 2048, 3]
    assert header["pre_block.attn.qkv.weight"]["shape"] == [6144, 2048]
    assert header["mean_proj.weight"]["shape"] == [32, 32, 1]
    assert {
        int(key.split(".")[2])
        for key in header
        if key.startswith("decoder.ups.")
    } == set(range(7))
    assert {
        int(key.split(".")[2])
        for key in header
        if key.startswith("decoder.resblocks.")
    } == set(range(21))
    assert math.prod(audio_vae.AudioVAEConfig().upsample_rates) == 800
