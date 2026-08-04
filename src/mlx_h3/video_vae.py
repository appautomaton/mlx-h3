"""MLX decoder for the MiniMax-H3 visual VAE.

The released VAE is asymmetric: its encoder is a causal 3D CNN, while the
decoder is a 36-block non-causal Vision Transformer. This module ports only the
decoder needed by text-to-video-and-audio inference.

VALIDATION. Temporal and spatial plans are checked against the local reference
implementations, and the module tree is checked against the real checkpoint.
There is no executed decoder fixture, so a finite real-weight decode proves
assembly and runtime viability, not pixel parity.

Temporal chunking and spatial tiling are semantic model behavior. Each spatial
tile normalizes its own rotary coordinates, and each temporal chunk overlaps
the next. Neither path may be replaced by one untiled full-canvas pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoVAEConfig:
    latent_channels: int = 24
    out_channels: int = 3
    hidden_size: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ffn_hidden_size: int = 8192
    num_register_tokens: int = 4
    patch_size: int = 16
    patch_size_t: int = 4
    rope_theta: float = 100.0
    rope_dim_ratio: float = 0.75
    norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    tile_size: int = 256
    tile_overlap: int = 64

    @property
    def rope_dim(self) -> int:
        return int(self.attention_head_dim * self.rope_dim_ratio)

    @property
    def output_patch_dim(self) -> int:
        return self.out_channels * self.patch_size_t * self.patch_size**2

    @property
    def tokens_chunk_size(self) -> int:
        return math.ceil(self.clip_length / self.patch_size_t)

    @property
    def frame_pre_padding(self) -> int:
        return (-self.clip_length) % self.patch_size_t

    @property
    def token_overlap(self) -> int:
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_overlap(self) -> int:
        return max(
            self.token_overlap * self.patch_size_t - self.frame_pre_padding, 0
        )


@dataclass(frozen=True)
class TemporalPlan:
    pad_tokens: int
    num_chunks: int
    output_frames: int
    padded_length: int


def _pad_frames(config: VideoVAEConfig, padded_length: int, pad_tokens: int) -> int:
    if pad_tokens == 0:
        return 0
    intra_tail = config.clip_length % config.patch_size_t
    if intra_tail == 0:
        return pad_tokens * config.patch_size_t
    before = padded_length - pad_tokens
    return sum(
        intra_tail
        if (before + index) % config.tokens_chunk_size == 0
        else config.patch_size_t
        for index in range(pad_tokens)
    )


def temporal_plan(latent_t: int, config: VideoVAEConfig | None = None) -> TemporalPlan:
    """Plan the trained temporal chunk walk for one latent length."""
    if latent_t < 1:
        raise ValueError(f"latent_t must be positive, got {latent_t}")
    cfg = config or VideoVAEConfig()
    pseudo_length = latent_t + cfg.token_drop
    pad_tokens = (-pseudo_length) % cfg.tokens_chunk_size
    pseudo_length += pad_tokens
    num_chunks = pseudo_length // cfg.tokens_chunk_size - int(cfg.token_drop > 0)
    if num_chunks < 1:
        # The 5-frame case has only two real latent tokens. The released Python
        # path would otherwise form zero chunks; the local MLX reference pads a
        # full extra chunk and recovers the intended five frames.
        pad_tokens += cfg.tokens_chunk_size
        num_chunks += 1

    padded_length = latent_t + pad_tokens
    chunk_frames = cfg.tokens_chunk_size * cfg.patch_size_t
    total_frames = 0
    final_overlap = 0
    for index in range(num_chunks):
        start = index * cfg.tokens_chunk_size
        stop = start + cfg.tokens_chunk_size + cfg.token_overlap
        clip_tokens = max(
            0, min(stop, padded_length) - min(start, padded_length)
        )
        clip_frames = clip_tokens * cfg.patch_size_t
        for split in range(int(cfg.token_drop > 0) + 1):
            frame_start = split * chunk_frames
            frame_stop = min(frame_start + chunk_frames, clip_frames)
            frames = max(0, frame_stop - frame_start - cfg.frame_pre_padding)
            if split == 0:
                total_frames += frames
            else:
                final_overlap = frames
    total_frames += final_overlap
    output_frames = total_frames - _pad_frames(cfg, padded_length, pad_tokens)
    return TemporalPlan(pad_tokens, num_chunks, output_frames, padded_length)


@dataclass(frozen=True)
class TilePlan:
    starts: tuple[int, ...]
    overlaps: tuple[int, ...]
    length: int


def split_tiles(
    extent: int,
    *,
    tile_size: int = 256,
    min_overlap: int = 64,
    spatial_ratio: int = 16,
) -> TilePlan:
    """Cover one pixel axis with latent-aligned overlapping tiles."""
    if extent < 1 or extent % spatial_ratio:
        raise ValueError(f"extent must be a positive multiple of {spatial_ratio}")
    if tile_size % spatial_ratio or min_overlap % spatial_ratio:
        raise ValueError("tile size and overlap must be latent-aligned")
    if min_overlap >= tile_size:
        raise ValueError("tile overlap must be smaller than tile size")
    if tile_size >= extent:
        return TilePlan((0,), (), extent)

    num_tiles = math.ceil(extent / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) < extent:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - extent
    for index in range(remaining // spatial_ratio):
        overlaps[index % len(overlaps)] += spatial_ratio
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return TilePlan(tuple(starts), tuple(overlaps), tile_size)


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(
        x.astype(mx.float32), weight.astype(mx.float32), eps
    ).astype(x.dtype)


def _rms_norm_no_weight(x: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(
        x.dtype
    )


def _layer_norm(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float
) -> mx.array:
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    centered = xf - mean
    norm = centered * mx.rsqrt(
        mx.mean(centered * centered, axis=-1, keepdims=True) + eps
    )
    return (norm * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(
        x.dtype
    )


def _rope_tables(
    temporal: int,
    height: int,
    width: int,
    suffix: int,
    config: VideoVAEConfig,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    axes = [
        [2.0 * ((index + 0.5) / size) - 1.0 for index in range(size)]
        for size in (temporal, height, width)
    ]
    positions = [
        (t_pos, h_pos, w_pos)
        for t_pos in axes[0]
        for h_pos in axes[1]
        for w_pos in axes[2]
    ]
    positions.extend([(0.0, 0.0, 0.0)] * suffix)
    pos = mx.array(positions, dtype=mx.float32)

    step = 6.0 / config.rope_dim
    num_freqs = math.ceil(1.0 / step)
    inv_freq = 1.0 / (
        config.rope_theta ** (mx.arange(num_freqs, dtype=mx.float32) * step)
    )
    angles = (2.0 * math.pi * pos[..., None] * inv_freq).reshape(
        pos.shape[0], -1
    )
    return (
        mx.cos(angles).astype(dtype)[None, :, None, :],
        mx.sin(angles).astype(dtype)[None, :, None, :],
    )


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = cos.shape[-1]
    first = x[..., :half]
    second = x[..., half : 2 * half]
    return mx.concatenate(
        [
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., 2 * half :],
        ],
        axis=-1,
    )


class VideoAttention(nn.Module):
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.eps = config.norm_eps
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        batch, sequence, _ = x.shape
        # The checkpoint interleaves q/k/v inside every head. Splitting the
        # linear output into three contiguous thirds is a silent layout bug.
        qkv = self.to_qkv(x).reshape(
            batch, sequence, self.heads, 3 * self.head_dim
        )
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = _apply_rope(_rms_norm_no_weight(q, self.eps), cos, sin)
        k = _apply_rope(_rms_norm_no_weight(k, self.eps), cos, sin)
        q, k, v = (
            mx.transpose(value, (0, 2, 1, 3)) for value in (q, k, v)
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(
            batch, sequence, -1
        )
        return self.to_out(out)


class VideoFeedForward(nn.Module):
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2)
        self.w2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class VideoTransformerBlock(nn.Module):
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = VideoAttention(config)
        self.scale1 = mx.zeros((config.hidden_size,))
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = VideoFeedForward(config)
        self.scale2 = mx.zeros((config.hidden_size,))

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        x = x + self.attn(_rms_norm(x, self.norm1.weight, self.eps), cos, sin) * self.scale1
        return x + self.ff(_rms_norm(x, self.norm2.weight, self.eps)) * self.scale2


class VideoDecoder(nn.Module):
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        self.mask_token = mx.zeros((1, 1, config.hidden_size))
        self.x_embedder = nn.Linear(config.latent_channels, config.hidden_size)
        self.register_tokens = mx.zeros(
            (1, config.num_register_tokens, config.hidden_size)
        )
        self.transformer_blocks = [
            VideoTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.proj_out = nn.Linear(config.hidden_size, config.output_patch_dim)

    def __call__(self, latent: mx.array) -> mx.array:
        cfg = self.config
        batch, channels, temporal, height, width = latent.shape
        hidden = mx.transpose(latent, (0, 2, 3, 4, 1)).reshape(
            batch, temporal * height * width, channels
        )
        hidden = self.x_embedder(hidden.astype(self.x_embedder.weight.dtype))
        num_patches = hidden.shape[1]
        registers = mx.broadcast_to(
            self.register_tokens,
            (batch, cfg.num_register_tokens, cfg.hidden_size),
        )
        hidden = mx.concatenate(
            [hidden, registers, mx.zeros_like(hidden[:, :1])], axis=1
        )
        cos, sin = _rope_tables(
            temporal,
            height,
            width,
            cfg.num_register_tokens + 1,
            cfg,
            hidden.dtype,
        )
        for block in self.transformer_blocks:
            hidden = block(hidden, cos, sin)
            mx.eval(hidden)

        hidden = _layer_norm(
            hidden, self.norm_out.weight, self.norm_out.bias, cfg.norm_eps
        )
        output = self.proj_out(hidden.astype(self.proj_out.weight.dtype))
        output = output[:, :num_patches]
        output = output.reshape(
            batch,
            temporal,
            height,
            width,
            cfg.out_channels,
            cfg.patch_size_t,
            cfg.patch_size,
            cfg.patch_size,
        )
        output = mx.transpose(output, (0, 4, 1, 5, 2, 6, 3, 7))
        return output.reshape(
            batch,
            cfg.out_channels,
            temporal * cfg.patch_size_t,
            height * cfg.patch_size,
            width * cfg.patch_size,
        )


def _blend(
    previous: mx.array, current: mx.array, extent: int, axis: int
) -> mx.array:
    axis %= current.ndim
    extent = min(previous.shape[axis], current.shape[axis], extent)
    if extent == 0:
        return current
    shape = [1] * current.ndim
    shape[axis] = extent
    weight = (mx.arange(extent, dtype=mx.float32) / extent).reshape(shape)
    previous_slice = [slice(None)] * current.ndim
    previous_slice[axis] = slice(-extent, None)
    current_slice = [slice(None)] * current.ndim
    current_slice[axis] = slice(0, extent)
    mixed = (
        previous[tuple(previous_slice)].astype(mx.float32) * (1.0 - weight)
        + current[tuple(current_slice)].astype(mx.float32) * weight
    ).astype(current.dtype)
    if extent == current.shape[axis]:
        return mixed
    rest = [slice(None)] * current.ndim
    rest[axis] = slice(extent, None)
    return mx.concatenate([mixed, current[tuple(rest)]], axis=axis)


class VideoVAE(nn.Module):
    def __init__(self, config: VideoVAEConfig | None = None):
        super().__init__()
        self.config = config or VideoVAEConfig()
        cfg = self.config
        self.latents_mean = mx.zeros((cfg.latent_channels,), dtype=mx.float32)
        self.latents_std = mx.ones((cfg.latent_channels,), dtype=mx.float32)
        # A 1x1x1 convolution is exactly a channel-axis linear projection.
        self.post_quant_conv = nn.Linear(cfg.latent_channels, cfg.latent_channels)
        self.decoder = VideoDecoder(cfg)

    def _decode_spatial(self, latent: mx.array) -> mx.array:
        cfg = self.config
        pixel_height = latent.shape[-2] * cfg.patch_size
        pixel_width = latent.shape[-1] * cfg.patch_size
        y_plan = split_tiles(
            pixel_height,
            tile_size=cfg.tile_size,
            min_overlap=cfg.tile_overlap,
            spatial_ratio=cfg.patch_size,
        )
        x_plan = split_tiles(
            pixel_width,
            tile_size=cfg.tile_size,
            min_overlap=cfg.tile_overlap,
            spatial_ratio=cfg.patch_size,
        )
        ratio = cfg.patch_size
        rows: list[list[mx.array]] = []
        for y_start in y_plan.starts:
            row = []
            for x_start in x_plan.starts:
                tile = latent[
                    ...,
                    y_start // ratio : (y_start + y_plan.length) // ratio,
                    x_start // ratio : (x_start + x_plan.length) // ratio,
                ]
                decoded = self.decoder(tile)
                mx.eval(decoded)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = _blend(
                        rows[row_index - 1][column_index],
                        tile,
                        y_plan.overlaps[row_index - 1],
                        -2,
                    )
                if column_index > 0:
                    tile = _blend(
                        row[column_index - 1],
                        tile,
                        x_plan.overlaps[column_index - 1],
                        -1,
                    )
                if row_index < len(rows) - 1:
                    tile = tile[..., : -y_plan.overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -x_plan.overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=-1))
        return mx.concatenate(result_rows, axis=-2)

    def _decode_temporal(self, latent: mx.array) -> mx.array:
        cfg = self.config
        plan = temporal_plan(latent.shape[2], cfg)
        if plan.pad_tokens:
            latent = mx.concatenate(
                [latent]
                + [latent[:, :, -1:]] * plan.pad_tokens,
                axis=2,
            )

        chunk_frames = cfg.tokens_chunk_size * cfg.patch_size_t
        pieces = []
        overlap = None
        for index in range(plan.num_chunks):
            start = index * cfg.tokens_chunk_size
            stop = min(
                start + cfg.tokens_chunk_size + cfg.token_overlap,
                plan.padded_length,
            )
            decoded = self._decode_spatial(latent[:, :, start:stop])
            # Do not let the lazy graph span temporal chunks. At the released
            # canvas each chunk already contains dozens of spatial tile passes.
            mx.eval(decoded)
            for split in range(int(cfg.token_drop > 0) + 1):
                frame_start = split * chunk_frames
                frame_stop = min(frame_start + chunk_frames, decoded.shape[2])
                if frame_stop - frame_start <= cfg.frame_pre_padding:
                    continue
                part = decoded[
                    :,
                    :,
                    frame_start + cfg.frame_pre_padding : frame_stop,
                ]
                if split == 0:
                    if overlap is not None:
                        part = _blend(overlap, part, cfg.frame_overlap, 2)
                        overlap = None
                    pieces.append(part)
                else:
                    overlap = part
            if index == plan.num_chunks - 1 and overlap is not None:
                pieces.append(overlap)
                overlap = None
        if not pieces:
            raise RuntimeError("temporal decode produced no frames")
        output = mx.concatenate(pieces, axis=2)
        return output[:, :, : plan.output_frames]

    def __call__(self, normalized_latent: mx.array) -> mx.array:
        """Decode normalized ``[B,24,T,H,W]`` latents to RGB in ``[0,1]``."""
        cfg = self.config
        if normalized_latent.ndim != 5 or normalized_latent.shape[1] != cfg.latent_channels:
            raise ValueError(
                f"expected [B,{cfg.latent_channels},T,H,W], got {normalized_latent.shape}"
            )
        mean = self.latents_mean.reshape(1, -1, 1, 1, 1)
        std = self.latents_std.reshape(1, -1, 1, 1, 1)
        latent = normalized_latent.astype(mx.float32) * std + mean
        latent = mx.transpose(latent, (0, 2, 3, 4, 1))
        latent = self.post_quant_conv(
            latent.astype(self.post_quant_conv.weight.dtype)
        )
        latent = mx.transpose(latent, (0, 4, 1, 2, 3))
        decoded = self._decode_temporal(latent).astype(mx.float32)
        pixel_mean = mx.array(IMAGENET_MEAN, dtype=mx.float32).reshape(
            1, 3, 1, 1, 1
        )
        pixel_std = mx.array(IMAGENET_STD, dtype=mx.float32).reshape(
            1, 3, 1, 1, 1
        )
        return mx.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
