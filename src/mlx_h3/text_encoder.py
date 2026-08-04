"""Text-only Qwen3-VL conditioner for MiniMax-H3.

H3 consumes the unnormalized hidden state after decoder layer 50.  Text-to-video
does not use the vision tower, final RMS norm, or language-model head, so none of
those modules are represented here or loaded at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


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

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        q = mx.transpose(_rms_norm(q, self.q_norm, self.eps), (0, 2, 1, 3))
        k = mx.transpose(_rms_norm(k, self.k_norm, self.eps), (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

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

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(_rms_norm(x, self.input_layernorm, self.eps))
        return x + self.mlp(_rms_norm(x, self.post_attention_layernorm, self.eps))


class TextModel(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TextDecoderLayer(config) for _ in range(config.num_layers)]

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed_tokens(token_ids)
        for layer in self.layers:
            h = layer(h)
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
