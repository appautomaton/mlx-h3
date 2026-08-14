"""Weightless invariants for layout behavior independent of golden fixtures."""

from __future__ import annotations

import pytest

from mlx_h3 import layout


def test_audio_grid_is_channel_major():
    rows = layout.audio_grid(cursor=512.0, t=4, w_low=-5.0, w_high=7.0)

    assert len(rows) == 8
    assert [row[0] for row in rows] == [512.0, 513.0, 514.0, 515.0] * 2
    assert all(row[1] == 0.0 for row in rows)
    assert [row[2] for row in rows[:4]] == [-5.0] * 4
    assert [row[2] for row in rows[4:]] == [7.0] * 4


def test_video_grid_composes_temporal_and_spatial_positions():
    frame, _ = layout.frame_grid(30, 54)
    rows = layout.video_grid(17, frame, cursor=512.0)
    temporal = layout.video_t_grid(17, 512.0)

    assert len(rows) == 17 * len(frame)
    assert rows[0] == (temporal[0], frame[0][0], frame[0][1])
    assert rows[-1] == (temporal[-1], frame[-1][0], frame[-1][1])


def test_keyframe_anchor_is_rejected_in_the_middle():
    with pytest.raises(ValueError, match="neither first nor last"):
        layout.pack(
            text_len=8,
            latent_t=17,
            latent_h=30,
            latent_w=54,
            audio_t=93,
            frame_count=56,
            keyframes=(layout.Keyframe(20),),
        )
