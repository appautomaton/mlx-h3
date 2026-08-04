"""MLX BigVGAN decoder for the MiniMax-H3 audio VAE.

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
