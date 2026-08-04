"""Tests for memory pressure decisions independent of macOS counters."""

from __future__ import annotations

import pytest

from mlx_h3 import memory


def sample(*, active=0, swap=0, swapouts=0, compressor=0, free=100):
    return memory.Sample(active, active, 0, swap, swapouts, compressor, free)


def test_compressor_growth_requires_low_free_memory(monkeypatch):
    readings = iter(
        [
            sample(compressor=100, free=90),
            sample(compressor=100 + memory.COMPRESSOR_SLACK_PAGES + 1, free=63),
        ]
    )
    monkeypatch.setattr(memory, "sample", lambda: next(readings))
    memory.Guard("test").check()


def test_compressor_growth_at_low_free_memory_fails(monkeypatch):
    readings = iter(
        [
            sample(compressor=100, free=90),
            sample(compressor=100 + memory.COMPRESSOR_SLACK_PAGES + 1, free=10),
        ]
    )
    monkeypatch.setattr(memory, "sample", lambda: next(readings))
    with pytest.raises(memory.BudgetExceeded, match="compressor"):
        memory.Guard("test").check()


def test_swap_and_active_budget_remain_unconditional(monkeypatch):
    swap_readings = iter(
        [sample(free=90), sample(swapouts=1, free=90)]
    )
    monkeypatch.setattr(memory, "sample", lambda: next(swap_readings))
    with pytest.raises(memory.BudgetExceeded, match="SWAPPING"):
        memory.Guard("test").check()

    active_readings = iter(
        [sample(free=90), sample(active=71 * memory.GIB, free=90)]
    )
    monkeypatch.setattr(memory, "sample", lambda: next(active_readings))
    with pytest.raises(memory.BudgetExceeded, match="exceeds budget"):
        memory.Guard("test", budget_gib=70).check()
