"""MLX waveform encoder and BigVGAN decoder for the MiniMax-H3 audio VAE.

The diffusion latent is stereo, but the released decoder is mono: left and
right channels fold into the batch axis and are decoded independently. The
decoder upsamples 40 Hz latents by 800x to a 32 kHz waveform.

VALIDATION. The model tree is checked against the real checkpoint and a live
decode checks length, range, memory and finite output. No executed waveform
fixture exists, so this does not claim sample-level parity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class AudioVAEConfig:
    latent_channels: int = 32
    latent_dim: int = 2048
    encoder_dim: int = 64
    downsample_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    encoder_attention_heads: int = 8
    decoder_dim: int = 1024
    upsample_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    upsample_kernels: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernels: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    sample_rate: int = 32000

    @property
    def hop_length(self) -> int:
        return math.prod(self.upsample_rates)

    @property
    def encoder_hop_length(self) -> int:
        return math.prod(self.downsample_rates)


def _replicate_pad(x: mx.array, left: int, right: int) -> mx.array:
    pieces = []
    if left:
        pieces.append(mx.repeat(x[:, :1], left, axis=1))
    pieces.append(x)
    if right:
        pieces.append(mx.repeat(x[:, -1:], right, axis=1))
    return mx.concatenate(pieces, axis=1)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha).reshape(1, 1, -1)
        beta = mx.exp(self.beta).reshape(1, 1, -1)
        return x + mx.sin(alpha * x) ** 2 / (beta + 1e-9)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.transpose(self.alpha, (0, 2, 1)).astype(x.dtype)
        return x + mx.sin(alpha * x) ** 2 / (alpha + 1e-9)


class EncoderResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = [
            Snake1d(channels),
            nn.Conv1d(
                channels,
                channels,
                7,
                padding=3 * dilation,
                dilation=dilation,
            ),
            Snake1d(channels),
            nn.Conv1d(channels, channels, 1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.block:
            x = layer(x)
        crop = (residual.shape[1] - x.shape[1]) // 2
        if crop > 0:
            residual = residual[:, crop:-crop]
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        inputs = channels // 2
        self.block = [
            EncoderResidualUnit(inputs, 1),
            EncoderResidualUnit(inputs, 3),
            EncoderResidualUnit(inputs, 9),
            Snake1d(inputs),
            nn.Conv1d(
                inputs,
                channels,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        channels = config.encoder_dim
        block: list[nn.Module] = [nn.Conv1d(1, channels, 7, padding=3)]
        for stride in config.downsample_rates:
            channels *= 2
            block.append(EncoderBlock(channels, stride))
        block.extend(
            [
                Snake1d(channels),
                nn.Conv1d(channels, config.latent_dim, 3, padding=1),
            ]
        )
        self.block = block

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
            mx.eval(x)
        return x


class GeGluMLP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5)
        self.w0 = nn.Linear(channels, channels * 2)
        self.w1 = nn.Linear(channels, channels * 2)
        self.w2 = nn.Linear(channels * 2, channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        return self.w2(nn.gelu_approx(self.w0(x)) * self.w1(x))


class CausalAttentionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int):
        super().__init__()
        if in_dim % heads or in_dim // heads % out_dim:
            raise ValueError("audio attention dimensions do not divide evenly")
        self.heads = heads
        self.head_dim = in_dim // heads
        self.out_dim = out_dim
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = mx.zeros((in_dim,), dtype=mx.float32)
        self.v_bias = mx.zeros((in_dim,), dtype=mx.float32)
        self.zero_k_bias = mx.zeros((in_dim,), dtype=mx.float32)
        self.proj = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, channels = x.shape
        bias = mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias])
        qkv = self.qkv(x) + bias.astype(x.dtype)
        q, k, v = mx.split(
            qkv.reshape(batch, length, 3, self.heads, self.head_dim),
            3,
            axis=2,
        )
        q, k, v = (
            mx.transpose(value.squeeze(2), (0, 2, 1, 3))
            for value in (q, k, v)
        )
        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask="causal" if length > 1 else None,
        )
        attended = mx.mean(attended, axis=1)
        pool = self.head_dim // self.out_dim
        attended = mx.mean(
            attended.reshape(batch, length, self.out_dim, pool), axis=-1
        )
        return self.proj(attended)


class AttnProjection(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.latent_dim, eps=1e-5)
        self.attn = CausalAttentionProjection(
            config.latent_dim,
            config.latent_channels,
            config.encoder_attention_heads,
        )
        self.proj = nn.Linear(config.latent_dim, config.latent_channels)
        self.norm3 = nn.LayerNorm(config.latent_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.latent_channels, eps=1e-5)
        self.mlp = GeGluMLP(config.latent_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UpSample1d(nn.Module):
    def __init__(self, kernel_size: int = 12, ratio: int = 2):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        initial = [0.0] * kernel_size
        initial[kernel_size // 2 - 1] = 0.5
        initial[kernel_size // 2] = 0.5
        self.filter = mx.array(initial, dtype=mx.float32).reshape(1, 1, -1)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        weight = mx.broadcast_to(
            self.filter.reshape(1, self.kernel_size, 1),
            (channels, self.kernel_size, 1),
        )
        x = _replicate_pad(x, self.pad, self.pad)
        x = self.ratio * mx.conv_transpose1d(
            x, weight, stride=self.ratio, groups=channels
        )
        return x[:, self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, kernel_size: int = 12, ratio: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.ratio = ratio
        initial = [0.0] * kernel_size
        initial[kernel_size // 2 - 1] = 0.5
        initial[kernel_size // 2] = 0.5
        self.filter = mx.array(initial, dtype=mx.float32).reshape(1, 1, -1)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        weight = mx.broadcast_to(
            self.filter.reshape(1, self.kernel_size, 1),
            (channels, self.kernel_size, 1),
        )
        x = _replicate_pad(
            x, self.kernel_size // 2 - 1, self.kernel_size // 2
        )
        return mx.conv1d(x, weight, stride=self.ratio, groups=channels)


class Activation1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d()
        self.downsample = nn.Module()
        self.downsample.lowpass = LowPassFilter1d()

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample.lowpass(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilations: tuple[int, ...]
    ):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
            for dilation in dilations
        ]
        self.convs2 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilations
        ]
        # The checkpoint list is interleaved: pre-conv1, pre-conv2, repeated.
        self.activations = [Activation1d(channels) for _ in range(2 * len(dilations))]

    def __call__(self, x: mx.array) -> mx.array:
        for index, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            residual = conv1(self.activations[2 * index](x))
            residual = conv2(self.activations[2 * index + 1](residual))
            x = x + residual
        return x


class BigVGANDecoder(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config
        self.conv_pre = nn.Conv1d(config.latent_dim, config.decoder_dim, 7, padding=3)
        self.ups = []
        self.resblocks = []
        for stage, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernels)
        ):
            in_channels = config.decoder_dim // (2**stage)
            out_channels = config.decoder_dim // (2 ** (stage + 1))
            self.ups.append(
                [
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride=rate,
                        padding=(kernel - rate) // 2,
                    )
                ]
            )
            self.resblocks.extend(
                AMPBlock(out_channels, res_kernel, tuple(dilations))
                for res_kernel, dilations in zip(
                    config.resblock_kernels, config.resblock_dilations
                )
            )
        final_channels = config.decoder_dim // (2 ** len(config.upsample_rates))
        self.activation_post = Activation1d(final_channels)
        self.conv_post = nn.Conv1d(final_channels, 1, 7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        kernels = len(self.config.resblock_kernels)
        for stage, upsampler in enumerate(self.ups):
            x = upsampler[0](x)
            outputs = [
                self.resblocks[stage * kernels + index](x)
                for index in range(kernels)
            ]
            x = sum(outputs[1:], outputs[0]) / kernels
            # Seven stages form a long lazy graph and hold every high-rate
            # activation unless each stage is materialized.
            mx.eval(x)
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0)


class AudioVAEEncoder(nn.Module):
    """Encode 32 kHz stereo waveforms to normalized posterior means."""

    def __init__(self, config: AudioVAEConfig | None = None):
        super().__init__()
        self.config = config or AudioVAEConfig()
        cfg = self.config
        if cfg.encoder_hop_length != cfg.hop_length:
            raise ValueError("audio encoder and decoder hop lengths differ")
        self.latents_mean = mx.zeros((cfg.latent_channels,), dtype=mx.float32)
        self.latents_std = mx.ones((cfg.latent_channels,), dtype=mx.float32)
        self.encoder = AudioEncoder(cfg)
        self.pre_block = AttnProjection(cfg)
        self.mean_proj = nn.Conv1d(cfg.latent_channels, cfg.latent_channels, 1)

    def __call__(self, waveform: mx.array) -> mx.array:
        """Encode `[B,2,L]` waveforms to `[B,32,2,ceil(L/800)]`."""
        cfg = self.config
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError(f"expected [B,2,L] waveform, got {waveform.shape}")
        batch, stereo, length = waveform.shape
        right_pad = math.ceil(length / cfg.encoder_hop_length) * cfg.encoder_hop_length
        right_pad -= length
        if right_pad:
            waveform = mx.pad(waveform, ((0, 0), (0, 0), (0, right_pad)))
        hidden = mx.transpose(waveform, (0, 2, 1)).reshape(
            batch * stereo, waveform.shape[-1], 1
        )
        hidden = self.encoder(hidden.astype(self.encoder.block[0].weight.dtype))
        hidden = self.pre_block(hidden)
        latent = self.mean_proj(hidden)
        latent = (
            latent.astype(mx.float32) - self.latents_mean.reshape(1, 1, -1)
        ) / self.latents_std.reshape(1, 1, -1)
        return mx.transpose(
            latent.reshape(batch, stereo, latent.shape[1], cfg.latent_channels),
            (0, 3, 1, 2),
        )


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEConfig | None = None):
        super().__init__()
        self.config = config or AudioVAEConfig()
        cfg = self.config
        if len(cfg.upsample_rates) != len(cfg.upsample_kernels):
            raise ValueError("upsample rate/kernel lengths differ")
        if len(cfg.resblock_kernels) != len(cfg.resblock_dilations):
            raise ValueError("resblock kernel/dilation lengths differ")
        self.latents_mean = mx.zeros((cfg.latent_channels,), dtype=mx.float32)
        self.latents_std = mx.ones((cfg.latent_channels,), dtype=mx.float32)
        self.dec_in_proj = nn.Conv1d(cfg.latent_channels, cfg.latent_dim, 1)
        self.decoder = BigVGANDecoder(cfg)

    def __call__(self, normalized_latent: mx.array) -> mx.array:
        """Decode ``[B,32,2,T]`` latents to ``[B,2,T*800]`` waveform."""
        cfg = self.config
        if normalized_latent.ndim != 4 or normalized_latent.shape[1] != cfg.latent_channels:
            raise ValueError(
                f"expected [B,{cfg.latent_channels},stereo,T], got {normalized_latent.shape}"
            )
        batch, _, stereo, length = normalized_latent.shape
        latent = mx.transpose(normalized_latent, (0, 2, 3, 1)).reshape(
            batch * stereo, length, cfg.latent_channels
        )
        latent = (
            latent.astype(mx.float32) * self.latents_std.reshape(1, 1, -1)
            + self.latents_mean.reshape(1, 1, -1)
        )
        hidden = self.dec_in_proj(latent.astype(self.dec_in_proj.weight.dtype))
        waveform = self.decoder(hidden).astype(mx.float32)
        return waveform.reshape(batch, stereo, -1)
