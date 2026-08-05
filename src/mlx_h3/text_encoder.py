"""Qwen3-VL text and image conditioning for MiniMax-H3.

H3 consumes the unnormalized hidden state after decoder layer 50. Text-only
inference loads only that decoder; image-conditioned modes additionally load the
checkpoint's vision tower and construct the released `<Picture n>` presentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import layout, vision

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
VIDEO_PAD = 151656


@dataclass(frozen=True)
class TextEncoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_layers: int = 50
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_sections: tuple[int, int, int] = (24, 20, 20)

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        ) < 1:
            raise ValueError("text encoder dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")


def _rms_norm(x: mx.array, norm: nn.RMSNorm, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, norm.weight, eps)


def _apply_split_rope(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return mx.concatenate(
        [
            first * cos[..., :half] - second * sin[..., :half],
            second * cos[..., half:] + first * sin[..., half:],
        ],
        axis=-1,
    )


def _mrope_tables(
    position_ids: mx.array,
    config: TextEncoderConfig,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    half = config.head_dim // 2
    inv_freq = 1.0 / (
        config.rope_theta
        ** (mx.arange(half, dtype=mx.float32) * 2.0 / config.head_dim)
    )
    frequencies = position_ids.astype(mx.float32)[..., None] * inv_freq
    interleaved = mx.stack(
        [
            frequencies[
                1
                if index < config.mrope_sections[1] * 3 and index % 3 == 1
                else 2
                if index < config.mrope_sections[2] * 3 and index % 3 == 2
                else 0,
                :,
                index,
            ]
            for index in range(half)
        ],
        axis=-1,
    )
    angles = mx.concatenate([interleaved, interleaved], axis=-1)[None, None]
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


class TextAttention(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        q = mx.transpose(_rms_norm(q, self.q_norm, self.eps), (0, 2, 1, 3))
        k = mx.transpose(_rms_norm(k, self.k_norm, self.eps), (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        if rope is None:
            q = mx.fast.rope(
                q,
                dims=self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=0,
            )
            k = mx.fast.rope(
                k,
                dims=self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=0,
            )
        else:
            q = _apply_split_rope(q, *rope)
            k = _apply_split_rope(k, *rope)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal" if length > 1 else None
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, length, -1)
        return self.o_proj(out)


class TextMLP(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        x = x + self.self_attn(
            _rms_norm(x, self.input_layernorm, self.eps), rope
        )
        return x + self.mlp(_rms_norm(x, self.post_attention_layernorm, self.eps))


class TextModel(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TextDecoderLayer(config) for _ in range(config.num_layers)]

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.forward_embeddings(self.embed_tokens(token_ids))

    def forward_embeddings(
        self,
        embeddings: mx.array,
        *,
        position_ids: mx.array | None = None,
        deepstack: tuple[mx.array, ...] = (),
    ) -> mx.array:
        h = embeddings
        rope = (
            None
            if position_ids is None
            else _mrope_tables(position_ids, self.config, h.dtype)
        )
        for index, layer in enumerate(self.layers):
            h = layer(h, rope)
            if index < len(deepstack):
                h = h + deepstack[index].astype(h.dtype)
            # Prevent the lazy graph from retaining all 50 layers at once.
            mx.eval(h)
        return h


class TextEncoder(nn.Module):
    """The checkpoint-compatible ``model.*`` tree and raw layer-50 output."""

    def __init__(self, config: TextEncoderConfig | None = None):
        super().__init__()
        self.config = config or TextEncoderConfig()
        self.model = TextModel(self.config)

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape [batch, non-empty sequence]")
        return self.model(token_ids)


@dataclass(frozen=True)
class _ImageSpan:
    index: int
    size: int
    output: vision.VisionOutput


@dataclass(frozen=True)
class ReferencePresentation:
    """One prepared Ref2VA entry in request order."""

    kind: str
    pixels: mx.array | None = None
    has_audio: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError(f"unsupported reference kind: {self.kind}")
        if (self.kind == "audio") != (self.pixels is None):
            raise ValueError("only audio references omit vision pixels")
        if self.kind == "audio" and not self.has_audio:
            raise ValueError("audio references must carry audio")
        if self.kind == "image" and self.has_audio:
            raise ValueError("image references cannot carry audio")


def _mrope_position_ids(
    sequence: int,
    spans: tuple[_ImageSpan, ...],
) -> mx.array:
    positions = [[0] * sequence for _ in range(3)]
    offset = 0
    initialized = False
    for span in spans:
        start = span.index
        end = start + span.size
        if not initialized:
            for axis in range(3):
                positions[axis][:start] = list(range(start))
            initialized = True
        length_max = max(span.output.grid_h, span.output.grid_w) // 2
        start_next = length_max + start
        for axis in range(3):
            positions[axis][end:] = list(
                range(
                    start_next + offset,
                    start_next + (sequence - end) + offset,
                )
            )
        positions[0][start:end] = [start + offset] * span.size
        merged_h = span.output.grid_h // 2
        merged_w = span.output.grid_w // 2
        positions[1][start:end] = [
            start + offset + row
            for row in range(merged_h)
            for _ in range(merged_w)
        ]
        positions[2][start:end] = [
            start + offset + column
            for _ in range(merged_h)
            for column in range(merged_w)
        ]
        offset += length_max - span.size
    return mx.array(positions, dtype=mx.int32)


class MultimodalTextEncoder(TextEncoder):
    """The text decoder plus the Qwen3-VL tower used by image presentations."""

    def __init__(
        self,
        config: TextEncoderConfig | None = None,
        vision_config: vision.VisionConfig | None = None,
    ):
        super().__init__(config)
        self.visual = vision.QwenVisionModel(vision_config)

    def encode_fl2va(
        self,
        tokenizer,
        prompt: str,
        images: tuple[mx.array, ...],
    ) -> tuple[mx.array, mx.array]:
        return self._encode_images(tokenizer, prompt, images, mode="FL2VA")

    def encode_ref_references(
        self,
        tokenizer,
        prompt: str,
        references: tuple[ReferencePresentation, ...],
    ) -> tuple[mx.array, mx.array]:
        """Encode Ref2VA labels and vision blocks in exact request order."""
        if not references:
            raise ValueError("Ref2VA requires at least one reference")
        if not any(reference.kind != "audio" for reference in references):
            raise ValueError("Ref2VA audio requires an image or video reference")

        entries = []
        pending_tokens: list[int] = []
        image_ordinal = video_ordinal = audio_ordinal = 0
        for reference in references:
            if reference.has_audio:
                audio_ordinal += 1
                pending_tokens.extend(
                    tokenizer.encode(f"<Audio {audio_ordinal}>: ")
                )
            if reference.kind == "audio":
                continue
            if reference.kind == "image":
                image_ordinal += 1
                pending_tokens.extend(
                    tokenizer.encode(f"<Picture {image_ordinal}>: ")
                )
                entries.append(
                    (
                        tuple(pending_tokens),
                        IMAGE_PAD,
                        self.visual(reference.pixels),
                    )
                )
                pending_tokens = []
                continue

            video_ordinal += 1
            video_entries = self._video_entries(
                tokenizer,
                reference.pixels,
                ordinal=video_ordinal,
                leading_tokens=tuple(pending_tokens),
            )
            entries.extend(video_entries)
            pending_tokens = []

        return self._encode_vision_entries(
            tokenizer,
            prompt,
            tuple(entries),
            trailing_tokens=tuple(pending_tokens),
        )

    def _encode_images(
        self,
        tokenizer,
        prompt: str,
        images: tuple[mx.array, ...],
        *,
        mode: str,
    ) -> tuple[mx.array, mx.array]:
        if not images:
            raise ValueError(f"{mode} requires at least one image")
        outputs = tuple(self.visual(image) for image in images)

        entries = tuple(
            (
                tuple(tokenizer.encode(f"<Picture {ordinal}>: ")),
                IMAGE_PAD,
                output,
            )
            for ordinal, output in enumerate(outputs, start=1)
        )
        return self._encode_vision_entries(
            tokenizer,
            prompt,
            entries,
        )

    def _video_entries(
        self,
        tokenizer,
        video: mx.array,
        *,
        ordinal: int,
        leading_tokens: tuple[int, ...] = (),
    ) -> list[tuple[tuple[int, ...], int, vision.VisionOutput]]:
        entries = []
        if video.ndim != 5 or video.shape[:2] != (1, 3):
            raise ValueError(
                f"reference video must have shape [1,3,T,H,W], got {video.shape}"
            )
        sampled = mx.transpose(video[0, :, :: layout.FPS // 2], (1, 0, 2, 3))
        timestamps = [index / 2.0 for index in range(sampled.shape[0])]
        if sampled.shape[0] % 2:
            sampled = mx.concatenate([sampled, sampled[-1:]], axis=0)
            timestamps.append(timestamps[-1])
        for block in range(0, sampled.shape[0], 2):
            prefix = list(leading_tokens) if block == 0 else []
            if block == 0:
                prefix.extend(tokenizer.encode(f"<Video {ordinal}>: "))
            timestamp = (timestamps[block] + timestamps[block + 1]) / 2.0
            prefix.extend(tokenizer.encode(f"<{timestamp:.1f} seconds>"))
            entries.append(
                (
                    tuple(prefix),
                    VIDEO_PAD,
                    self.visual(sampled[block : block + 2]),
                )
            )
        return entries

    def _encode_vision_entries(
        self,
        tokenizer,
        prompt: str,
        entries: tuple[tuple[tuple[int, ...], int, vision.VisionOutput], ...],
        *,
        trailing_tokens: tuple[int, ...] = (),
    ) -> tuple[mx.array, mx.array]:
        outputs = tuple(entry[2] for entry in entries)

        tokens: list[int] = []
        spans = []
        for prefix, pad_token, output in entries:
            tokens.extend(prefix)
            tokens.append(VISION_START)
            index = len(tokens)
            tokens.extend([pad_token] * output.merged.shape[0])
            spans.append(_ImageSpan(index, output.merged.shape[0], output))
            tokens.append(VISION_END)
        tokens.extend(trailing_tokens)
        tokens.extend(tokenizer.encode(prompt))

        token_ids = mx.array([tokens], dtype=mx.int32)
        base = self.model.embed_tokens(token_ids)
        parts = []
        cursor = 0
        for span in spans:
            parts.append(base[:, cursor : span.index])
            parts.append(span.output.merged[None].astype(base.dtype))
            cursor = span.index + span.size
        parts.append(base[:, cursor:])
        embeddings = mx.concatenate(parts, axis=1)

        deepstack = []
        for layer in range(len(outputs[0].deepstack)):
            parts = []
            cursor = 0
            for span in spans:
                parts.append(mx.zeros_like(base[:, cursor : span.index]))
                parts.append(span.output.deepstack[layer][None].astype(base.dtype))
                cursor = span.index + span.size
            parts.append(mx.zeros_like(base[:, cursor:]))
            deepstack.append(mx.concatenate(parts, axis=1))

        tags = [1] * len(tokens)
        for span in spans:
            tags[span.index - 1 : span.index + span.size + 1] = [0] * (
                span.size + 2
            )
        states = self.model.forward_embeddings(
            embeddings,
            position_ids=_mrope_position_ids(len(tokens), tuple(spans)),
            deepstack=tuple(deepstack),
        )
        return states[0], mx.array(tags, dtype=mx.int32)
