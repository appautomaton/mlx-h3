"""The MiniMax-H3 transformer block.

Modulation is what makes this block unusual. AdaLN normally emits one shift /
scale / gate per token; here it emits one per *(timestep, modality)* pair, and
the whole 38222-token sequence draws from at most 12 of them. `layout.pack`
already guarantees the segments are contiguous, so modulation is applied by
slicing rather than by broadcasting a [S, hidden] parameter tensor -- which at
spec size would be 411 MiB per parameter, three of them, per block.

The reference does this with in-place `mul_`/`addcmul_` on segment views. MLX has
no in-place, so each segment produces its own slice and they concatenate; peak is
one output buffer either way.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from . import rope

#: (start, stop, modulation row) covering the packed sequence contiguously.
Runs = list[tuple[int, int, int]]

#: adaLN modality tags. Vision conditioning rides the video tag; text has its own
#: except where vision pads are spliced into the presentation, which is why the
#: text span can carry mixed tags and is split into runs at forward time.
SEG_TAG = {
    "text": 1,
    "video": 0,
    "audio": 2,
    "cond": 0,
    "ref_img": 0,
    "ref_audio": 2,
}

#: shift/scale/gate for attention, then the same three for the MLP.
ADALN_EXPAND = 6
ADALN_MODALITIES = 3


class TimeEmbedder(nn.Module):
    """Sinusoidal timestep embedding. fp32 throughout, cosine before sine."""

    def __init__(self, freq_dim: int, hidden: int, out: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden)
        self.proj_out = nn.Linear(hidden, out)

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs
        return self.proj_out(nn.silu(self.proj_in(mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1))))


class AdalnProj(nn.Module):
    """``[M, t_dim]`` timesteps -> ``expand`` tensors of ``[M * modalities, hidden]``.

    The output is laid out (timestep, modality, expand, hidden), so a run's row is
    ``timestep_row * modalities + tag``. Cheap in FLOPs and enormous in weight:
    at the real config this linear is [96768, 2688] per block, 260M parameters
    evaluated against 2 to 4 rows.
    """

    def __init__(
        self,
        t_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        apply_silu: bool = True,
    ):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities)

    def __call__(self, t_emb: mx.array) -> list[mx.array]:
        x = self.linear(nn.silu(t_emb) if self.apply_silu else t_emb)
        x = x.reshape(x.shape[0] * self.modalities, self.expand * self.hidden)
        return mx.split(x, self.expand, axis=-1)


def modulate(
    x: mx.array,
    weight: mx.array,
    eps: float,
    shift: mx.array,
    scale: mx.array,
    runs: Runs,
) -> mx.array:
    """RMSNorm, then a per-run affine, assembled by slices.

    Norm and modulation are fused per run so only one full-width buffer exists at
    a time rather than a normalized one plus a modulated one.
    """
    return mx.concatenate(
        [
            mx.fast.rms_norm(x[a:b], weight, eps) * (1.0 + scale[r].astype(x.dtype))
            + shift[r].astype(x.dtype)
            for a, b, r in runs
        ],
        axis=0,
    )


def gate(x: mx.array, g: mx.array, other: mx.array, runs: Runs) -> mx.array:
    """Accumulate a gated residual: ``x + other * gate[row]`` per run."""
    return mx.concatenate(
        [x[a:b] + other[a:b] * g[r].astype(x.dtype) for a, b, r in runs], axis=0
    )


class Attention(nn.Module):
    """Full self-attention over the packed sequence -- no mask, no causality.

    Video, audio and text all see each other; that joint attention is the whole
    mechanism by which the two modalities stay in sync.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.eps = eps
        self.scale = head_dim**-0.5
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(
        self, x: mx.array, cos: mx.array | None = None, sin: mx.array | None = None
    ) -> mx.array:
        s = x.shape[0]
        q, k, v = mx.split(self.qkv_proj(x), 3, axis=-1)
        shape = (s, self.heads, self.head_dim)

        # The norm weights are held in nn.RMSNorm for loading, but applied through
        # mx.fast.rms_norm directly: the module would normalize the last axis of a
        # [S, hidden] view, and these normalize per head.
        q = mx.fast.rms_norm(q.reshape(shape), self.q_norm.weight, self.eps)
        k = mx.fast.rms_norm(k.reshape(shape), self.k_norm.weight, self.eps)
        if cos is not None:
            q = rope.apply(q, cos, sin)
            k = rope.apply(k, cos, sin)

        q, k, v = (mx.transpose(t, (1, 0, 2))[None] for t in (q, k, v.reshape(shape)))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self.out_proj(mx.transpose(out[0], (1, 0, 2)).reshape(s, -1))


class MLP(nn.Module):
    """SwiGLU. fc1 emits gate and up fused; the first half is the gate."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        g, up = mx.split(self.fc1(x), 2, axis=-1)
        return self.fc2(nn.silu(g) * up)


class RefinerBlock(nn.Module):
    """Pre-norm block with no modulation and no rope, used by the token refiner."""

    def __init__(
        self, hidden: int, heads: int, head_dim: int, ffn: int, eps: float, qk_eps: float
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class TokenRefiner(nn.Module):
    """Two blocks that condition the projected text states before packing."""

    def __init__(
        self,
        num_layers: int,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        eps: float,
        qk_eps: float,
        final_eps: float,
    ):
        super().__init__()
        self.blocks = [
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        t_dim: int,
        eps: float,
        qk_eps: float,
        apply_silu: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(
            t_dim, hidden, ADALN_EXPAND, ADALN_MODALITIES, apply_silu=apply_silu
        )

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        runs: Runs,
        cos: mx.array | None = None,
        sin: mx.array | None = None,
    ) -> mx.array:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_proj(t_emb)
        h = modulate(x, self.norm1.weight, self.eps, shift_a, scale_a, runs)
        x = gate(x, gate_a, self.attn(h, cos, sin), runs)
        h = modulate(x, self.norm2.weight, self.eps, shift_m, scale_m, runs)
        return gate(x, gate_m, self.mlp(h), runs)


class FinalLayer(nn.Module):
    """Project the two target streams back to latent space.

    Only the target video and audio segments are read; conditioning and text
    rows have served their purpose by now and are dropped. Its adaLN has one
    modality rather than three, so a run's row is just its timestep row.

    The output heads are the checkpoint's fp32 island and are left dense.
    """

    def __init__(
        self, hidden: int, t_dim: int, video_dim: int, audio_dim: int, eps: float,
        apply_silu: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim)
        self.audio_out = nn.Linear(hidden, audio_dim)

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        video_seg: tuple[int, int, int],
        audio_seg: tuple[int, int, int],
    ) -> tuple[mx.array, mx.array]:
        shift, scale = self.adaln_proj(t_emb)

        def head(seg, out):
            a, b, row = seg
            h = mx.fast.rms_norm(x[a:b], self.norm.weight, self.eps)
            h = h * (1.0 + scale[row].astype(x.dtype)) + shift[row].astype(x.dtype)
            return out(h.astype(mx.float32))

        return head(video_seg, self.video_out), head(audio_seg, self.audio_out)
