"""Qwen3-VL image and video conditioning used by MiniMax-H3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    depth: int = 27
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 5120
    deepstack_indexes: tuple[int, ...] = (8, 16, 24)
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size**2

    @property
    def merge_dim(self) -> int:
        return self.hidden_size * self.spatial_merge_size**2


@dataclass(frozen=True)
class VisionOutput:
    merged: mx.array
    deepstack: tuple[mx.array, ...]
    grid_h: int
    grid_w: int


def _layer_norm(x: mx.array, norm: nn.LayerNorm, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    centered = xf - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return (
        centered * mx.rsqrt(variance + eps) * norm.weight.astype(mx.float32)
        + norm.bias.astype(mx.float32)
    ).astype(x.dtype)


def _apply_split_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return mx.concatenate(
        [first * cos[..., :half] - second * sin[..., :half],
         second * cos[..., half:] + first * sin[..., half:]],
        axis=-1,
    )


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        # The source Conv3d consumes one already-flattened 2x16x16 patch. A
        # Linear has the identical contraction and avoids a reshape-only Conv3d.
        self.proj = nn.Linear(config.patch_dim, config.hidden_size)

    def __call__(self, patches: mx.array) -> mx.array:
        return self.proj(patches.astype(self.proj.weight.dtype))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        sequence = x.shape[0]
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        shape = (sequence, self.heads, self.head_dim)
        q = _apply_split_rope(q.reshape(shape), cos, sin)
        k = _apply_split_rope(k.reshape(shape), cos, sin)
        v = v.reshape(shape)
        q, k, v = (
            mx.transpose(value, (1, 0, 2))[None] for value in (q, k, v)
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None
        )
        out = mx.transpose(out[0], (1, 0, 2)).reshape(sequence, -1)
        return self.proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(_layer_norm(x, self.norm1, self.eps), cos, sin)
        return x + self.mlp(_layer_norm(x, self.norm2, self.eps))


class VisionMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = _layer_norm(x, self.norm, self.eps).reshape(-1, self.merge_dim)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class DeepstackMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.eps = config.norm_eps
        self.merge_dim = config.merge_dim
        self.norm = nn.LayerNorm(config.merge_dim, eps=config.norm_eps)
        self.linear_fc1 = nn.Linear(config.merge_dim, config.merge_dim)
        self.linear_fc2 = nn.Linear(config.merge_dim, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(-1, self.merge_dim)
        x = _layer_norm(x, self.norm, self.eps)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class QwenVisionModel(nn.Module):
    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        self.config = config or VisionConfig()
        cfg = self.config
        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = [VisionBlock(cfg) for _ in range(cfg.depth)]
        self.merger = VisionMerger(cfg)
        self.deepstack_merger_list = [
            DeepstackMerger(cfg) for _ in cfg.deepstack_indexes
        ]

    def _patches(self, pixels: mx.array) -> tuple[mx.array, int, int]:
        cfg = self.config
        if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[0] not in (1, 2):
            raise ValueError(
                f"expected one image or one video pair [1|2,3,H,W], got {pixels.shape}"
            )
        height, width = pixels.shape[-2:]
        factor = cfg.patch_size * cfg.spatial_merge_size
        if height % factor or width % factor:
            raise ValueError(f"vision image axes must be multiples of {factor}")
        grid_h, grid_w = height // cfg.patch_size, width // cfg.patch_size
        image = (pixels.astype(mx.float32) - 0.5) / 0.5
        if image.shape[0] == 1:
            image = mx.broadcast_to(
                image,
                (cfg.temporal_patch_size, 3, height, width),
            )
        patches = image.reshape(
            1,
            cfg.temporal_patch_size,
            3,
            grid_h // cfg.spatial_merge_size,
            cfg.spatial_merge_size,
            cfg.patch_size,
            grid_w // cfg.spatial_merge_size,
            cfg.spatial_merge_size,
            cfg.patch_size,
        )
        patches = mx.transpose(patches, (0, 3, 6, 4, 7, 2, 1, 5, 8))
        return patches.reshape(grid_h * grid_w, cfg.patch_dim), grid_h, grid_w

    def _position_embedding(self, grid_h: int, grid_w: int) -> mx.array:
        cfg = self.config
        side = math.isqrt(cfg.num_position_embeddings)
        h_values = [index * (side - 1) / (grid_h - 1) for index in range(grid_h)]
        w_values = [index * (side - 1) / (grid_w - 1) for index in range(grid_w)]
        indices = [[], [], [], []]
        weights = [[], [], [], []]
        for h_value in h_values:
            h_floor = int(h_value)
            h_ceil = min(h_floor + 1, side - 1)
            dh = h_value - h_floor
            for w_value in w_values:
                w_floor = int(w_value)
                w_ceil = min(w_floor + 1, side - 1)
                dw = w_value - w_floor
                indices[0].append(h_floor * side + w_floor)
                indices[1].append(h_floor * side + w_ceil)
                indices[2].append(h_ceil * side + w_floor)
                indices[3].append(h_ceil * side + w_ceil)
                weights[0].append((1.0 - dh) * (1.0 - dw))
                weights[1].append((1.0 - dh) * dw)
                weights[2].append(dh * (1.0 - dw))
                weights[3].append(dh * dw)
        index = mx.array(indices, dtype=mx.int32)
        weight = mx.array(weights, dtype=self.pos_embed.weight.dtype)
        values = self.pos_embed(index) * weight[..., None]
        pos = mx.sum(values, axis=0)
        merge = cfg.spatial_merge_size
        return mx.transpose(
            pos.reshape(grid_h // merge, merge, grid_w // merge, merge, -1),
            (0, 2, 1, 3, 4),
        ).reshape(grid_h * grid_w, -1)

    def _rope(self, grid_h: int, grid_w: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        cfg = self.config
        merge = cfg.spatial_merge_size
        coords = []
        for block_h in range(grid_h // merge):
            for block_w in range(grid_w // merge):
                for inner_h in range(merge):
                    for inner_w in range(merge):
                        coords.append((block_h * merge + inner_h, block_w * merge + inner_w))
        positions = mx.array(coords, dtype=mx.float32)
        dim = cfg.head_dim // 2
        inv_freq = 1.0 / (
            cfg.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        row = positions[:, :1] * inv_freq
        column = positions[:, 1:] * inv_freq
        angles = mx.concatenate([row, column], axis=-1)
        angles = mx.concatenate([angles, angles], axis=-1)[:, None]
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)

    def __call__(self, pixels: mx.array) -> VisionOutput:
        patches, grid_h, grid_w = self._patches(pixels)
        hidden = self.patch_embed(patches)
        hidden = hidden + self._position_embedding(grid_h, grid_w).astype(hidden.dtype)
        cos, sin = self._rope(grid_h, grid_w, hidden.dtype)
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            mx.eval(hidden)
            if index in self.config.deepstack_indexes:
                merger_index = self.config.deepstack_indexes.index(index)
                value = self.deepstack_merger_list[merger_index](hidden)
                mx.eval(value)
                deepstack.append(value)
        merged = self.merger(hidden)
        mx.eval(merged)
        return VisionOutput(merged, tuple(deepstack), grid_h, grid_w)
