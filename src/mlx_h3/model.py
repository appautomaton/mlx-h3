"""Assembling one denoising step over the packed video+audio sequence.

Everything below is bookkeeping around `dit.DiTBlock`: turn latents into rows,
decide which of the handful of modulation vectors each row draws from, run the
stack, and project the two target streams back out.

VALIDATION. Unlike layout, rope and the block itself, nothing here has a
committed fixture -- the upstream VAE/assembly dump was never checked in. The
tests that accompany this file are self-consistency (patchify round-trips,
runs partition the sequence, shapes follow the layout) plus agreement with the
reference read statically. That is a weaker tier and is labelled as such.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import dit, layout
from .layout import PackedLayout
from .rope import angles as rope_angles
from .rope import tables as rope_tables


@dataclass(frozen=True)
class H3Config:
    """The shipped geometry. Every field is checked against the checkpoint."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw


# --- latents <-> rows -----------------------------------------------------


def patchify_video(latent: mx.array, patch_size: tuple[int, int, int]) -> mx.array:
    """``[1, C, T, H, W]`` -> ``[t*h*w, C*pt*ph*pw]``, row-major over (t, h, w).

    Row order has to match `layout.video_grid`, which walks frames outermost and
    patch rows within a frame -- a transposed unpack here shuffles the video in
    space and still produces a well-formed tensor.
    """
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = mx.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7))
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(
    rows: mx.array, t: int, h: int, w: int, c: int, patch_size: tuple[int, int, int]
) -> mx.array:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = mx.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7))
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent: mx.array) -> mx.array:
    """``[1, C, 2, T]`` -> ``[2*T, C]``, channel-major to match `layout.audio_grid`."""
    _, c, ch, t = latent.shape
    return mx.transpose(latent[0], (1, 2, 0)).reshape(ch * t, c)


def unpack_audio(rows: mx.array, channels: int = 2) -> mx.array:
    t = rows.shape[0] // channels
    return mx.transpose(rows.reshape(channels, t, rows.shape[-1]), (2, 0, 1))[None]


# --- per-step modulation plan ---------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Which distinct timesteps a step needs, and where each row draws from."""

    t_vals: tuple[float, ...]
    runs: dit.Runs
    video_seg: tuple[int, int, int]
    audio_seg: tuple[int, int, int]


@dataclass(frozen=True)
class AdalnSchedule:
    """Materialized AdaLN values, stored block-major and addressed by step."""

    t_values: tuple[tuple[float, ...], ...]
    blocks: tuple[tuple[dit.BlockModulation, ...], ...]
    final: tuple[dit.FinalModulation, ...]


def plan(
    packed: PackedLayout,
    sigma_video: float,
    *,
    sigma_audio: float | None = None,
    visual_cond_noise_aug: float = layout.VISUAL_COND_TIMESTEP,
    audio_cond_noise_aug: float = layout.AUDIO_COND_TIMESTEP,
    text_tags: Sequence[int] | None = None,
) -> Plan:
    """Resolve the step's timesteps and per-run modulation rows.

    Video and audio ride different shifted schedules, and conditioning rows pin
    near t=1 regardless. That is at most four distinct timesteps for the whole
    sequence -- which is why adaLN can be applied by segment instead of by token.
    """
    t_v = 1.0 - sigma_video
    t_a = 1.0 - (
        layout.audio_sigma(sigma_video) if sigma_audio is None else sigma_audio
    )
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, visual_cond_noise_aug),
        "ref_img": max(t_v, visual_cond_noise_aug),
        "ref_audio": max(t_a, audio_cond_noise_aug),
    }

    kinds = {s.kind for s in packed.segments}
    present = {t_v, t_a}
    if kinds & {"cond", "ref_img"}:
        present.add(seg_t["cond"])
    if "ref_audio" in kinds:
        present.add(seg_t["ref_audio"])
    t_vals = tuple(sorted(present))
    t_row = {t: i for i, t in enumerate(t_vals)}

    runs: dit.Runs = []
    for seg in packed.segments:
        base = t_row[seg_t[seg.kind]] * dit.ADALN_MODALITIES
        if seg.kind == "text" and text_tags is not None:
            # Vision pads spliced into the presentation carry the video tag, so
            # the text span is not uniform and splits into constant-tag runs.
            n = len(seg)
            if len(text_tags) != n:
                raise ValueError(f"{len(text_tags)} text tags for {n} text rows")
            start = 0
            for i in range(1, n + 1):
                if i == n or text_tags[i] != text_tags[start]:
                    runs.append((seg.start + start, seg.start + i, base + int(text_tags[start])))
                    start = i
        else:
            runs.append((seg.start, seg.stop, base + dit.SEG_TAG[seg.kind]))

    def target(kind: str) -> tuple[int, int, int]:
        seg = next(s for s in packed.segments if s.kind == kind)
        # The final layer's adaLN has one modality, so the row is the timestep row.
        return seg.start, seg.stop, t_row[seg_t[kind]]

    return Plan(t_vals, runs, target("video"), target("audio"))


class MiniMaxH3(nn.Module):
    def __init__(self, config: H3Config | None = None):
        super().__init__()
        cfg = config or H3Config()
        self.config = cfg

        self.video_patch_proj = nn.Linear(cfg.video_patch_dim, cfg.hidden_size)
        self.audio_patch_proj = nn.Linear(cfg.audio_latents_dim, cfg.hidden_size)
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.time_embedder = dit.TimeEmbedder(
            cfg.timestep_input_dim, cfg.time_embed_hidden_size, cfg.time_embed_dim
        )
        # A checkpoint buffer, not a computed constant: read it, do not derive it.
        self.rope = nn.Module()
        self.rope.inv_freq = mx.zeros((cfg.rope_inv_freq_len,))
        self.token_refiner = dit.TokenRefiner(
            cfg.token_refiner_num_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.ffn_hidden_size,
            cfg.norm_eps,
            cfg.qk_norm_eps,
            cfg.final_norm_eps,
        )
        self.blocks = [
            dit.DiTBlock(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.ffn_hidden_size,
                cfg.time_embed_dim,
                cfg.norm_eps,
                cfg.qk_norm_eps,
            )
            for _ in range(cfg.num_layers)
        ]
        self.final_layer = dit.FinalLayer(
            cfg.hidden_size,
            cfg.time_embed_dim,
            cfg.video_patch_dim,
            cfg.audio_latents_dim,
            cfg.final_norm_eps,
        )
        self._adaln_schedule: AdalnSchedule | None = None

    @property
    def has_precomputed_adaln(self) -> bool:
        return self._adaln_schedule is not None

    def precompute_adaln(
        self,
        plans: Sequence[Plan],
        *,
        dtype: mx.Dtype,
    ) -> None:
        """Materialize one AdaLN table per step, then release its projections."""
        steps = tuple(plans)
        if not steps:
            raise ValueError("at least one step plan is required")
        if self._adaln_schedule is not None:
            raise RuntimeError("AdaLN has already been precomputed")
        if self.time_embedder is None:
            raise RuntimeError("timestep embedding weights have already been released")

        embeddings = []
        for step in steps:
            value = self.time_embedder(
                mx.array(step.t_vals, dtype=mx.float32)
            ).astype(dtype)
            mx.eval(value)
            embeddings.append(value)
        self.time_embedder = None
        gc.collect()
        mx.clear_cache()

        block_tables = []
        for block in self.blocks:
            if block.adaln_proj is None:
                raise RuntimeError("block AdaLN weights have already been released")
            step_tables = []
            for embedding in embeddings:
                values = tuple(block.adaln_proj(embedding))
                mx.eval(*values)
                step_tables.append(values)
            block.adaln_proj = None
            block_tables.append(tuple(step_tables))
            gc.collect()
            mx.clear_cache()

        if self.final_layer.adaln_proj is None:
            raise RuntimeError("final AdaLN weights have already been released")
        final_tables = []
        for embedding in embeddings:
            values = tuple(self.final_layer.adaln_proj(embedding))
            mx.eval(*values)
            final_tables.append(values)
        self.final_layer.adaln_proj = None
        embeddings = None
        gc.collect()
        mx.clear_cache()

        self._adaln_schedule = AdalnSchedule(
            t_values=tuple(step.t_vals for step in steps),
            blocks=tuple(block_tables),
            final=tuple(final_tables),
        )

    def refine_text(self, text_states: mx.array) -> mx.array:
        """``[L, text_dim]`` Qwen states -> ``[L, hidden]``, or pass through."""
        if text_states.shape[-1] == self.config.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states))

    def embed(
        self,
        packed: PackedLayout,
        text_embed: mx.array,
        video_rows: mx.array,
        audio_rows: mx.array,
    ) -> mx.array:
        """Assemble the packed sequence from the three row sources.

        Segments are contiguous and the row sources are already in segment order,
        so this is a walk of slices; the reference's scatter into a preallocated
        buffer is the same thing written for in-place tensors.
        """
        video_embed = self.video_patch_proj(video_rows).astype(text_embed.dtype)
        audio_embed = self.audio_patch_proj(audio_rows).astype(text_embed.dtype)

        parts, voff, aoff = [], 0, 0
        for seg in packed.segments:
            n = len(seg)
            if seg.kind == "text":
                parts.append(text_embed)
            elif seg.kind in ("cond", "ref_img", "video"):
                parts.append(video_embed[voff : voff + n])
                voff += n
            else:
                parts.append(audio_embed[aoff : aoff + n])
                aoff += n
        if voff != video_embed.shape[0] or aoff != audio_embed.shape[0]:
            raise ValueError(
                f"row sources do not fill the layout: video {voff}/{video_embed.shape[0]}, "
                f"audio {aoff}/{audio_embed.shape[0]}"
            )
        return mx.concatenate(parts, axis=0)

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        text_embed: mx.array,
        packed: PackedLayout,
        sigma_video: float,
        *,
        sigma_audio: float | None = None,
        cond_video_rows: mx.array | None = None,
        cond_audio_rows: mx.array | None = None,
        text_tags: Sequence[int] | None = None,
        visual_cond_noise_aug: float = layout.VISUAL_COND_TIMESTEP,
        audio_cond_noise_aug: float = layout.AUDIO_COND_TIMESTEP,
        step_index: int | None = None,
        on_block: Callable[[int], None] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """One denoising step. Returns raw data-ward video and audio velocities.

        ``text_embed`` is already refined; the text encoder has been released from
        memory long before this runs, so refinement happens once at setup rather
        than inside the step loop. The sampler owns solver signs and maps the raw
        audio velocity onto the video integration grid; those concerns do not
        belong in the model output contract.
        """
        cfg = self.config
        step = plan(
            packed,
            sigma_video,
            sigma_audio=sigma_audio,
            visual_cond_noise_aug=visual_cond_noise_aug,
            audio_cond_noise_aug=audio_cond_noise_aug,
            text_tags=text_tags,
        )

        schedule = self._adaln_schedule
        if schedule is None:
            if self.time_embedder is None:
                raise RuntimeError("timestep embedding weights are unavailable")
            t_emb = self.time_embedder(
                mx.array(step.t_vals, dtype=mx.float32)
            ).astype(text_embed.dtype)
        else:
            if step_index is None or not 0 <= step_index < len(schedule.t_values):
                raise ValueError("a valid step_index is required for precomputed AdaLN")
            if step.t_vals != schedule.t_values[step_index]:
                raise ValueError("sampling step does not match the precomputed AdaLN schedule")
            t_emb = None
        cos, sin = rope_tables(
            rope_angles(layout.to_mlx(packed.positions), self.rope.inv_freq),
            text_embed.dtype,
        )

        # Conditioning rows precede target rows in each stream, so the reference's
        # boolean scatter is a concatenation here.
        video_rows = patchify_video(video_latent.astype(mx.float32), cfg.patch_size)
        audio_rows = pack_audio(audio_latent.astype(mx.float32))
        if cond_video_rows is not None:
            video_rows = mx.concatenate([cond_video_rows, video_rows], axis=0)
        if cond_audio_rows is not None:
            audio_rows = mx.concatenate([cond_audio_rows, audio_rows], axis=0)

        h = self.embed(packed, text_embed, video_rows, audio_rows)
        for i, block in enumerate(self.blocks):
            modulation = None if schedule is None else schedule.blocks[i][step_index]
            h = block(
                h,
                t_emb,
                step.runs,
                cos,
                sin,
                modulation=modulation,
            )
            # Force materialization per block: otherwise the lazy graph holds all
            # 50 blocks' intermediates alive at once, which at spec size is tens
            # of GiB that never needed to exist simultaneously.
            mx.eval(h)
            if on_block is not None:
                on_block(i)

        final_modulation = None if schedule is None else schedule.final[step_index]
        v, a = self.final_layer(
            h,
            t_emb,
            step.video_seg,
            step.audio_seg,
            modulation=final_modulation,
        )

        _, _, latent_t, latent_h, latent_w = video_latent.shape
        pt, ph, pw = cfg.patch_size
        video_out = unpatchify_video(
            v, latent_t // pt, latent_h // ph, latent_w // pw, cfg.latents_dim, cfg.patch_size
        )
        audio_out = unpack_audio(a)

        return (
            video_out.astype(video_latent.dtype),
            audio_out.astype(audio_latent.dtype),
        )
