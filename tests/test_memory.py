"""Tests for memory pressure decisions independent of macOS counters."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_vm_stat_and_swap_usage_parse_macos_counters(monkeypatch):
    outputs = iter(
        [
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 123.\n"
            "Swapouts: 7.\n"
            "Pages occupied by compressor: 456.\n",
            "total = 4096.00M  used = 1.25G  free = 2816.00M\n",
        ]
    )
    monkeypatch.setattr(
        memory.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout=next(outputs)),
    )

    stats, page_size = memory._vm_stat()

    assert page_size == 16_384
    assert stats["Pages free"] == 123
    assert stats["Swapouts"] == 7
    assert stats["Pages occupied by compressor"] == 456
    assert memory._swap_used() == int(1.25 * memory.GIB)


def test_sample_combines_mlx_and_kernel_counters(monkeypatch):
    monkeypatch.setattr(
        memory,
        "_vm_stat",
        lambda: ({"Swapouts": 3, "Pages occupied by compressor": 9}, 16_384),
    )
    monkeypatch.setattr(memory, "_swap_used", lambda: 17)
    monkeypatch.setattr(
        memory.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="64\n"),
    )
    monkeypatch.setattr(memory.mx, "get_active_memory", lambda: 11)
    monkeypatch.setattr(memory.mx, "get_peak_memory", lambda: 12)
    monkeypatch.setattr(memory.mx, "get_cache_memory", lambda: 13)

    assert memory.sample() == memory.Sample(11, 12, 13, 17, 3, 9, 64)


def test_configure_sets_both_mlx_limits_and_rejects_oversubscription(monkeypatch):
    calls = []
    monkeypatch.setattr(
        memory.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 80 * memory.GIB},
    )
    monkeypatch.setattr(
        memory.mx, "set_wired_limit", lambda value: calls.append(("wired", value))
    )
    monkeypatch.setattr(
        memory.mx, "set_memory_limit", lambda value: calls.append(("memory", value))
    )

    memory.configure(70)

    assert calls == [("wired", 70 * memory.GIB), ("memory", 70 * memory.GIB)]
    with pytest.raises(memory.BudgetExceeded, match="system wired limit"):
        memory.configure(80)


def test_guard_context_checks_exit_and_report_is_scalar_only(monkeypatch):
    readings = iter([sample(active=1), sample(active=2), sample(active=3)])
    monkeypatch.setattr(memory, "sample", lambda: next(readings))

    with memory.Guard("context"):
        pass

    report = memory.report("status ")
    assert report == "status active   0.0  peak   0.0  cache  0.0 GiB  |  swap 0 MiB  free 100%"
