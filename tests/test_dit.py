"""Stage-by-stage parity for one DiT block against minimax_h3_dit.safetensors.

WHAT THIS PROVES. The fixture's own header is explicit that it is a
TRANSCRIPTION of `comfy/ldm/minimax/model.py`, not the reference executing --
the reference block calls comfy_kitchen CUDA kernels that will not run on a Mac.
So a green test means this port agrees with an independently written
implementation of the same spec. It catches transcription slips (wrong axis,
wrong split order, rope over the whole head); it cannot catch a misreading the
two implementations share. Only the layout fixture executes the reference.

The weights are randomly initialized (`real: 0`), which is the right target:
every way this port fails silently is arithmetic, not weight values.

WHY THESE RUN ON THE CPU. MLX's float32 matmul on this Metal device rounds its
inputs -- measured 7.5e-4 relative error, independent of K, exact on integer
matrices, while elementwise ops and reductions stay at 1e-7. That is far above
the 1e-7 a structural check needs, so the parity math runs on the CPU stream
where fp32 is IEEE. `test_gpu_agrees_with_cpu` pins the GPU's actual tolerance
separately rather than pretending it does not exist.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_h3 import dit, rope

pytestmark = pytest.mark.fixture

HIDDEN, HEADS, HEAD_DIM, FFN = 256, 4, 32, 128
T_DIM, TIMESTEP_DIM, TIME_HIDDEN = 32, 32, 256
EPS = 1e-5

#: IEEE fp32 through two matmuls; anything looser would hide a real slip.
TOL = 1e-6

#: What this Metal device actually delivers for fp32 matmul.
GPU_TOL = 5e-3


@pytest.fixture(scope="module", autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def golden(local_file) -> dict:
    return mx.load(str(local_file("MLX_H3_DIT_FIXTURE")))


@pytest.fixture(scope="module")
def block(golden) -> dit.DiTBlock:
    b = dit.DiTBlock(HIDDEN, HEADS, HEAD_DIM, FFN, T_DIM, EPS, EPS)
    b.load_weights(
        [(k, v) for k, v in golden.items() if not k.startswith(("x.", "time_", "rope."))]
    )
    return b


@pytest.fixture(scope="module")
def embedder(golden) -> dit.TimeEmbedder:
    e = dit.TimeEmbedder(TIMESTEP_DIM, TIME_HIDDEN, T_DIM)
    e.load_weights(
        [
            (k.removeprefix("time_embedder."), v)
            for k, v in golden.items()
            if k.startswith("time_embedder.")
        ]
    )
    return e


@pytest.fixture(scope="module")
def runs(golden) -> dit.Runs:
    return [(int(a), int(b), int(r)) for a, b, r in golden["x.runs"].tolist()]


@pytest.fixture(scope="module")
def rope_tables(golden) -> tuple[mx.array, mx.array]:
    return rope.tables(rope.angles(golden["x.positions"], golden["rope.inv_freq"]))


def rel(got: mx.array, want: mx.array) -> float:
    return (mx.abs(got - want).max() / max(mx.abs(want).max().item(), 1e-12)).item()


def test_weights_fully_consumed(golden, block, embedder):
    """Nothing in the fixture's weight set is left unloaded or renamed away."""
    named = {k for k, _ in tree_flatten(block.parameters())}
    named |= {f"time_embedder.{k}" for k, _ in tree_flatten(embedder.parameters())}
    assert {k for k in golden if not k.startswith(("x.", "rope."))} == named


def test_time_embedder(golden, embedder):
    """cos before sin -- swapping the halves is silent and survives training."""
    assert rel(embedder(golden["x.t_vals"]), golden["x.t_emb"]) < TOL


def test_rope_tables(golden, rope_tables):
    cos, sin = rope_tables
    assert rel(cos[:, 0], golden["x.rope_cos"]) < TOL
    assert rel(sin[:, 0], golden["x.rope_sin"]) < TOL


def test_adaln_row_layout(golden, block, runs):
    """Row = timestep_row * 3 + modality tag, so 2 timesteps give 6 rows.

    A transposed view here silently swaps which stream each modulation lands on.
    """
    parts = block.adaln_proj(golden["x.t_emb"])
    assert len(parts) == dit.ADALN_EXPAND
    n_rows = golden["x.t_vals"].size * dit.ADALN_MODALITIES
    for p in parts:
        assert p.shape == (n_rows, HIDDEN)
    assert max(r for _, _, r in runs) < n_rows

    # The chunks partition the linear's output columns in order.
    flat = block.adaln_proj.linear(nn.silu(golden["x.t_emb"])).reshape(n_rows, -1)
    assert rel(mx.concatenate(parts, axis=-1), flat) == 0.0


def test_attn_in_is_plain_norm1(golden, block):
    """The fixture probes stages independently: attn_in carries no modulation."""
    got = mx.fast.rms_norm(golden["x.h_in"], block.norm1.weight, EPS)
    assert rel(got, golden["x.attn_in"]) < TOL


def test_attn_out(golden, block, rope_tables):
    """qkv split, per-head q/k norm BEFORE rope, partial rope, no mask."""
    cos, sin = rope_tables
    assert rel(block.attn(golden["x.attn_in"], cos, sin), golden["x.attn_out"]) < TOL


def test_mlp_out(golden, block):
    assert rel(block.mlp(golden["x.attn_in"]), golden["x.mlp_out"]) < TOL


def test_block_end_to_end(golden, block, runs, rope_tables):
    """Full block: adaLN modulation, gated residuals, both sublayers."""
    cos, sin = rope_tables
    got = block(golden["x.h_in"], golden["x.t_emb"], runs, cos, sin)
    assert rel(got, golden["x.h_out"]) < TOL


def test_rope_tail_is_unrotated(golden, block, rope_tables):
    """rot 24 of head_dim 32: the top 8 dims must pass through untouched.

    Rotating the whole head is the classic partial-rope slip and still runs.
    """
    cos, sin = rope_tables
    rot = 2 * cos.shape[-1]
    assert rot < HEAD_DIM
    x = mx.random.normal((golden["x.h_in"].shape[0], HEADS, HEAD_DIM))
    assert rel(rope.apply(x, cos, sin)[..., rot:], x[..., rot:]) == 0.0


def test_swiglu_halves_are_not_swapped(block):
    """silu(first half) * second half. Swapping is silent and plausible."""
    x = mx.random.normal((4, HIDDEN))
    g, up = mx.split(block.mlp.fc1(x), 2, axis=-1)
    assert rel(block.mlp(x), block.mlp.fc2(nn.silu(g) * up)) == 0.0
    assert rel(block.mlp(x), block.mlp.fc2(nn.silu(up) * g)) > 1e-3


def test_modulation_runs_cover_the_sequence(golden, runs):
    assert runs[0][0] == 0
    assert runs[-1][1] == golden["x.h_in"].shape[0]
    assert len({r for _, _, r in runs}) == len(runs), "runs must exercise distinct rows"
    for prev, nxt in zip(runs, runs[1:], strict=False):
        assert prev[1] == nxt[0]


def test_modulation_lands_on_the_right_run(golden, block, runs):
    """Perturbing one modulation row must move only that run's rows."""
    shift, scale = block.adaln_proj(golden["x.t_emb"])[:2]
    base = dit.modulate(golden["x.h_in"], block.norm1.weight, EPS, shift, scale, runs)

    target = runs[1]
    bumped = mx.array(shift)
    bumped[target[2]] = bumped[target[2]] + 1.0
    got = dit.modulate(golden["x.h_in"], block.norm1.weight, EPS, bumped, scale, runs)

    moved = (mx.abs(got - base) > 1e-6).any(axis=-1)
    assert moved[target[0] : target[1]].all().item()
    assert not moved[: target[0]].any().item()
    assert not moved[target[1] :].any().item()


def test_gpu_agrees_with_cpu(golden, block, runs, rope_tables):
    """Documents this device's fp32 matmul precision instead of assuming IEEE."""
    cos, sin = rope_tables
    want = block(golden["x.h_in"], golden["x.t_emb"], runs, cos, sin)
    with mx.stream(mx.gpu):
        got = block(golden["x.h_in"], golden["x.t_emb"], runs, cos, sin)
        mx.eval(got)
    assert TOL < rel(got, want) < GPU_TOL
