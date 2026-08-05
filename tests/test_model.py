"""Self-consistency for the sequence assembly.

VALIDATION TIER. Weaker than the rest of the suite on purpose, and worth saying
plainly: no fixture exists for this layer, so nothing here compares against
values produced elsewhere. What it does check is that the pieces agree with each
other and with `layout` -- round-trips invert, row orders line up, runs partition
the sequence, shapes follow from the layout rather than from constants typed in
twice. A misreading shared with the reference would survive all of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_h3 import dit, layout, model

PATCH = (1, 2, 2)

#: Small enough to run on CPU in a test, same structure as the shipped geometry.
TINY = model.H3Config(
    hidden_size=64,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=32,
    ffn_hidden_size=32,
    latents_dim=4,
    audio_latents_dim=8,
    text_dim=16,
    timestep_input_dim=16,
    time_embed_hidden_size=32,
    time_embed_dim=16,
    rope_inv_freq_len=4,
)

TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T = 8, 2, 4, 6, 3


@pytest.fixture(scope="module")
def packed() -> layout.PackedLayout:
    return layout.pack(
        text_len=TEXT_LEN,
        latent_t=LATENT_T,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=AUDIO_T,
    )


@pytest.fixture(scope="module")
def tiny() -> model.MiniMaxH3:
    m = model.MiniMaxH3(TINY)
    m.rope.inv_freq = mx.array([1.0, 0.1, 0.01, 0.001])
    return m


# --- rows -----------------------------------------------------------------


def test_patchify_round_trip():
    x = mx.random.normal((1, 4, LATENT_T, LATENT_H, LATENT_W))
    rows = model.patchify_video(x, PATCH)
    assert rows.shape == (LATENT_T * (LATENT_H // 2) * (LATENT_W // 2), 4 * 1 * 2 * 2)
    back = model.unpatchify_video(
        rows, LATENT_T, LATENT_H // 2, LATENT_W // 2, 4, PATCH
    )
    assert mx.array_equal(back, x)


def test_patchify_row_order_matches_the_position_grid(packed):
    """Rows walk (t, h, w) outermost-first, exactly as `layout.video_grid` does.

    A transposed unpack shuffles the video in space and still yields a valid
    tensor, so this order is pinned against the grid rather than assumed.
    """
    t, h, w = LATENT_T, LATENT_H // 2, LATENT_W // 2
    # Encode each patch's (t, h, w) index in its own value.
    ids = mx.arange(t * h * w, dtype=mx.float32).reshape(1, 1, t, h, w)
    x = mx.broadcast_to(ids, (1, 1, t, h, w))
    x = mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)
    rows = model.patchify_video(x, PATCH)
    assert [int(v) for v in rows[:, 0].tolist()] == list(range(t * h * w))

    video = next(s for s in packed.segments if s.kind == "video")
    grid = packed.positions[video.start : video.stop]
    assert len(grid) == rows.shape[0]
    # Frame index advances every h*w rows in both.
    for frame in range(t):
        block = grid[frame * h * w : (frame + 1) * h * w]
        assert len({row[0] for row in block}) == 1


def test_pack_audio_round_trip():
    x = mx.random.normal((1, 8, 2, AUDIO_T))
    rows = model.pack_audio(x)
    assert rows.shape == (2 * AUDIO_T, 8)
    assert mx.array_equal(model.unpack_audio(rows), x)


def test_pack_audio_is_channel_major(packed):
    """Channel 0's whole timeline, then channel 1's -- matching `layout.audio_grid`."""
    x = mx.zeros((1, 8, 2, AUDIO_T))
    x[0, :, 1, :] = 1.0
    rows = model.pack_audio(x)
    assert mx.array_equal(rows[:AUDIO_T], mx.zeros((AUDIO_T, 8)))
    assert mx.array_equal(rows[AUDIO_T:], mx.ones((AUDIO_T, 8)))

    audio = next(s for s in packed.segments if s.kind == "audio")
    grid = packed.positions[audio.start : audio.stop]
    assert len({row[2] for row in grid[:AUDIO_T]}) == 1
    assert grid[0][2] != grid[AUDIO_T][2], "stereo channels must sit at opposite w"


# --- modulation plan ------------------------------------------------------


def test_plan_t2va_has_two_timesteps(packed):
    step = model.plan(packed, sigma_video=0.5)
    assert len(step.t_vals) == 2, "video and audio schedules only"
    assert step.t_vals == tuple(sorted(step.t_vals))


def test_plan_fl2va_adds_the_conditioning_timestep():
    packed = layout.pack(
        text_len=TEXT_LEN,
        latent_t=LATENT_T,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=AUDIO_T,
        frame_count=layout.align_frame_count(LATENT_T),
        keyframes=(layout.Keyframe(0),),
    )
    step = model.plan(packed, sigma_video=0.5)
    assert len(step.t_vals) == 3
    assert layout.VISUAL_COND_TIMESTEP in step.t_vals


def test_plan_runs_partition_the_sequence(packed):
    step = model.plan(packed, sigma_video=0.5)
    assert step.runs[0][0] == 0
    assert step.runs[-1][1] == packed.seq_len
    for prev, nxt in zip(step.runs, step.runs[1:], strict=False):
        assert prev[1] == nxt[0]
    n_rows = len(step.t_vals) * dit.ADALN_MODALITIES
    assert all(0 <= r < n_rows for _, _, r in step.runs)


def test_plan_tags_each_stream_distinctly(packed):
    step = model.plan(packed, sigma_video=0.5)
    by_kind = {seg.kind: seg for seg in packed.segments}
    rows = {kind: next(r for a, _, r in step.runs if a == seg.start)
            for kind, seg in by_kind.items()}
    assert rows["text"] % dit.ADALN_MODALITIES == dit.SEG_TAG["text"]
    assert rows["video"] % dit.ADALN_MODALITIES == dit.SEG_TAG["video"]
    assert rows["audio"] % dit.ADALN_MODALITIES == dit.SEG_TAG["audio"]
    assert rows["video"] != rows["audio"], "streams must not share a modulation row"


def test_plan_text_tags_split_the_span(packed):
    """A vision block inside the presentation splits text into tag runs."""
    tags = [1] * TEXT_LEN
    tags[3:6] = [0, 0, 0]
    step = model.plan(packed, sigma_video=0.5, text_tags=tags)
    text_runs = [r for r in step.runs if r[0] < TEXT_LEN]
    assert [(a, b) for a, b, _ in text_runs] == [(0, 3), (3, 6), (6, 8)]
    assert [r % dit.ADALN_MODALITIES for _, _, r in text_runs] == [1, 0, 1]


def test_plan_rejects_mismatched_text_tags(packed):
    with pytest.raises(ValueError, match="text tags"):
        model.plan(packed, sigma_video=0.5, text_tags=[1] * (TEXT_LEN + 1))


def test_plan_audio_timestep_follows_the_shifted_schedule(packed):
    sigma = 0.5
    step = model.plan(packed, sigma_video=sigma)
    assert 1.0 - sigma in step.t_vals
    assert 1.0 - layout.audio_sigma(sigma) in step.t_vals

    explicit = model.plan(packed, sigma_video=sigma, sigma_audio=0.25)
    assert 0.75 in explicit.t_vals


def test_final_layer_targets_the_last_two_segments(packed):
    step = model.plan(packed, sigma_video=0.5)
    video = next(s for s in packed.segments if s.kind == "video")
    audio = next(s for s in packed.segments if s.kind == "audio")
    assert step.video_seg[:2] == (video.start, video.stop)
    assert step.audio_seg[:2] == (audio.start, audio.stop)


def test_sigma_schedule_agrees_with_the_layout_fixture(local_file):
    """The one part of this file backed by a real reference run."""
    path = local_file("MLX_H3_LAYOUT_FIXTURE")
    for case in json.loads(path.read_text())["sigma_schedule"]:
        assert layout.audio_sigma(case["sigma_v"]) == pytest.approx(
            case["sigma_a"], rel=1e-12
        )


# --- assembly -------------------------------------------------------------


def test_embed_fills_every_segment(packed, tiny):
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))
    video = mx.random.normal((packed.img_target_rows, TINY.video_patch_dim))
    audio = mx.random.normal((packed.audio_target_rows, TINY.audio_latents_dim))
    h = tiny.embed(packed, text, video, audio)
    assert h.shape == (packed.seq_len, TINY.hidden_size)


def test_embed_rejects_wrong_row_counts(packed, tiny):
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))
    video = mx.random.normal((packed.img_target_rows + 1, TINY.video_patch_dim))
    audio = mx.random.normal((packed.audio_target_rows, TINY.audio_latents_dim))
    with pytest.raises(ValueError, match="do not fill the layout"):
        tiny.embed(packed, text, video, audio)


def test_step_shapes_round_trip_to_the_latents(packed, tiny):
    """A full step must hand back exactly the latent shapes it was given."""
    video = mx.random.normal((1, TINY.latents_dim, LATENT_T, LATENT_H, LATENT_W))
    audio = mx.random.normal((1, TINY.audio_latents_dim, 2, AUDIO_T))
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))
    dv, da = tiny(video, audio, text, packed, sigma_video=0.5)
    assert dv.shape == video.shape
    assert da.shape == audio.shape
    assert mx.isfinite(dv).all().item() and mx.isfinite(da).all().item()


def test_step_sees_every_input(packed, tiny):
    """Perturbing the text, the video or the audio must move the video output.

    Joint attention is the mechanism; if audio could not reach video, the two
    streams would drift apart and nothing else in the suite would notice.
    """
    video = mx.random.normal((1, TINY.latents_dim, LATENT_T, LATENT_H, LATENT_W))
    audio = mx.random.normal((1, TINY.audio_latents_dim, 2, AUDIO_T))
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))
    base, _ = tiny(video, audio, text, packed, sigma_video=0.5)

    for name, kwargs in (
        ("text", {"text_embed": text + 1.0}),
        ("video", {"video_latent": video + 1.0}),
        ("audio", {"audio_latent": audio + 1.0}),
    ):
        args = {
            "video_latent": video,
            "audio_latent": audio,
            "text_embed": text,
            **kwargs,
        }
        got, _ = tiny(
            args["video_latent"], args["audio_latent"], args["text_embed"],
            packed, sigma_video=0.5,
        )
        assert mx.abs(got - base).max().item() > 1e-4, f"{name} does not reach video"


def test_step_returns_raw_data_ward_velocities(packed, tiny):
    """The model contract excludes solver signs and schedule derivatives."""
    video = mx.random.normal((1, TINY.latents_dim, LATENT_T, LATENT_H, LATENT_W))
    audio = mx.random.normal((1, TINY.audio_latents_dim, 2, AUDIO_T))
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))

    step = model.plan(packed, sigma_video=0.5)
    t_emb = tiny.time_embedder(mx.array(step.t_vals, dtype=mx.float32))
    from mlx_h3.rope import angles, tables

    cos, sin = tables(angles(layout.to_mlx(packed.positions), tiny.rope.inv_freq))
    h = tiny.embed(
        packed, text,
        model.patchify_video(video, TINY.patch_size),
        model.pack_audio(audio),
    )
    for block in tiny.blocks:
        h = block(h, t_emb, step.runs, cos, sin)
    v, a = tiny.final_layer(h, t_emb, step.video_seg, step.audio_seg)

    dv, da = tiny(video, audio, text, packed, sigma_video=0.5)
    want_v = model.unpatchify_video(
        v, LATENT_T, LATENT_H // 2, LATENT_W // 2, TINY.latents_dim, TINY.patch_size
    )
    want_a = model.unpack_audio(a)
    assert mx.abs(dv - want_v).max().item() < 1e-5
    assert mx.abs(da - want_a).max().item() < 1e-5


def test_precomputed_adaln_matches_weight_path_and_releases_projections(packed):
    mx.random.seed(7)
    tiny = model.MiniMaxH3(TINY)
    tiny.rope.inv_freq = mx.array([1.0, 0.1, 0.01, 0.001])
    video = mx.random.normal((1, TINY.latents_dim, LATENT_T, LATENT_H, LATENT_W))
    audio = mx.random.normal((1, TINY.audio_latents_dim, 2, AUDIO_T))
    text = mx.random.normal((TEXT_LEN, TINY.hidden_size))
    sigma_video = 0.5
    sigma_audio = layout.audio_sigma(sigma_video)
    step = model.plan(
        packed,
        sigma_video=sigma_video,
        sigma_audio=sigma_audio,
    )

    expected = tiny(
        video,
        audio,
        text,
        packed,
        sigma_video=sigma_video,
        sigma_audio=sigma_audio,
    )
    mx.eval(expected)

    tiny.precompute_adaln((step,), dtype=text.dtype)
    actual = tiny(
        video,
        audio,
        text,
        packed,
        sigma_video=sigma_video,
        sigma_audio=sigma_audio,
        step_index=0,
    )
    mx.eval(actual)

    max_error = max(
        mx.abs(got - want).max().item()
        for got, want in zip(actual, expected, strict=True)
    )
    assert max_error < 1e-5
    assert tiny.has_precomputed_adaln
    assert tiny.time_embedder is None
    assert all(block.adaln_proj is None for block in tiny.blocks)
    assert tiny.final_layer.adaln_proj is None


def test_config_matches_the_checkpoint():
    """Geometry is read off the checkpoint, not trusted from the dataclass."""
    ckpt = Path(__file__).resolve().parents[1] / "weights/mlx-8bit/dit_fl2va_a8g32.safetensors"
    if not ckpt.exists():
        pytest.skip("checkpoint absent")
    import struct

    with ckpt.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    cfg = model.H3Config()

    assert header["blocks.0.norm1.weight"]["shape"] == [cfg.hidden_size]
    assert header["final_layer.video_out.weight"]["shape"] == [
        cfg.video_patch_dim,
        cfg.hidden_size,
    ]
    assert header["final_layer.audio_out.weight"]["shape"] == [
        cfg.audio_latents_dim,
        cfg.hidden_size,
    ]
    assert header["rope.inv_freq"]["shape"] == [cfg.rope_inv_freq_len]
    assert header["time_embedder.proj_in.weight"]["shape"] == [
        cfg.time_embed_hidden_size,
        cfg.timestep_input_dim,
    ]
    assert header["time_embedder.proj_out.weight"]["shape"] == [
        cfg.time_embed_dim,
        cfg.time_embed_hidden_size,
    ]
    # adaLN emits 6 vectors x 3 modalities of hidden width.
    assert header["blocks.0.adaln_proj.linear.bias"]["shape"] == [
        dit.ADALN_EXPAND * dit.ADALN_MODALITIES * cfg.hidden_size
    ]
    assert header["final_layer.adaln_proj.linear.bias"]["shape"] == [2 * cfg.hidden_size]
    assert (
        max(int(k.split(".")[1]) for k in header if k.startswith("blocks."))
        == cfg.num_layers - 1
    )
