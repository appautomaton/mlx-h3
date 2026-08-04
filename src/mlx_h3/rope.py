"""Rotary position encoding over the packed 3-axis position grid.

H3 rotates 96 of each head's 128 dimensions; the remaining 32 pass through
untouched. The 96 split three ways -- 16 frequencies each for the (t, h, w) axes
of `layout.pack` -- and pair split-half, element ``i`` with element ``i + 48``.

``mx.fast.rope`` cannot serve this: it derives positions from an integer offset
along the sequence, while these are arbitrary floats (a latent frame sits at
t=598.667, a stereo channel at w=-5.466). So the angles are built explicitly.

They are built exactly once per generation. Positions depend only on shape, not
on the denoising step, so one [S, 48] table is shared by all 50 blocks across all
50 steps -- 2500 applications off one 7 MB pair of tables.
"""

from __future__ import annotations

import mlx.core as mx

#: Rotated dimensions per head; the rest of head_dim passes through.
ROT_DIM = 96

#: Frequencies per position axis. 3 axes * 16 = ROT_DIM / 2 angles.
INV_FREQ_LEN = 16


def angles(positions: mx.array, inv_freq: mx.array) -> mx.array:
    """``[S, 3]`` positions x ``[16]`` frequencies -> ``[S, 48]`` angles.

    Axis-major: ``t*inv | h*inv | w*inv``. The reference then concatenates this
    with itself to [S, 96] and the consumer immediately slices the first half
    back off, so only the half is built here.
    """
    return (positions.astype(mx.float32)[..., None] * inv_freq).reshape(
        positions.shape[0], -1
    )


def tables(angle: mx.array, dtype: mx.Dtype = mx.float32) -> tuple[mx.array, mx.array]:
    """``[S, 48]`` angles -> broadcastable ``(cos, sin)`` of ``[S, 1, 48]``.

    The head axis is inserted here so the tables broadcast against ``[S, H, D]``
    without a reshape on the hot path.
    """
    return (
        mx.cos(angle).astype(dtype)[:, None, :],
        mx.sin(angle).astype(dtype)[:, None, :],
    )


@mx.compile
def _rotate(x: mx.array, cos: mx.array, sin: mx.array, half: int, rot: int) -> mx.array:
    lo = x[..., :half]
    hi = x[..., half:rot]
    return mx.concatenate(
        [lo * cos - hi * sin, hi * cos + lo * sin, x[..., rot:]], axis=-1
    )


def apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading ``2 * cos.shape[-1]`` dims of ``x``'s last axis.

    ``x`` is ``[S, H, D]``; ``cos``/``sin`` are ``[S, 1, D_rot/2]``. Compiled so
    the six elementwise ops fuse instead of materializing a temporary each.

    A hand-written ``mx.fast.metal_kernel`` doing this in one pass measured 3.4x
    faster than this at [38222, 56, 128] (2.9 ms vs 5.8 ms). It is not used: rope
    runs 5000 times per generation for ~15 s total, against ~67% of block time
    spent in attention alone, so the whole line item is under 1% of the run.
    """
    half = cos.shape[-1]
    return _rotate(x, cos, sin, half, 2 * half)
