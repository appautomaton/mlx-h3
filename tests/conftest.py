"""Shared controls for optional local validation resources."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Make explicit validation tiers fail closed when requested."""
    group = parser.getgroup("local validation")
    group.addoption(
        "--require-fixtures",
        action="store_true",
        help="fail instead of skip when a local reference fixture is absent",
    )
    group.addoption(
        "--require-checkpoints",
        action="store_true",
        help="fail instead of skip when a local model checkpoint is absent",
    )
    group.addoption(
        "--require-nax",
        action="store_true",
        help="fail instead of skip when the mlx_nax_int extension is absent",
    )


@pytest.fixture(scope="session")
def local_file(request):
    """Resolve an optional local test file from an environment variable."""

    def resolve(variable: str) -> Path:
        value = os.environ.get(variable)
        if not value:
            message = f"{variable} is not set"
            if request.config.getoption("--require-fixtures"):
                pytest.fail(message, pytrace=False)
            pytest.skip(message)
        path = Path(value).expanduser()
        if not path.is_file():
            message = f"{variable} does not point to a file: {path}"
            if request.config.getoption("--require-fixtures"):
                pytest.fail(message, pytrace=False)
            pytest.skip(message)
        return path

    return resolve


@pytest.fixture(scope="session")
def nax_extension(request):
    """Resolve the optional NAX extension, with a fail-closed validation mode.

    The extension is built by `uv sync --extra nax` and any plain `uv sync`
    removes it again. Skipping silently in that state would let the M5 W8A8 path
    stop being exercised while the suite still reported green, so CI and any
    deliberate M5 run should pass --require-nax.
    """
    required = ("grouped_quantize", "grouped_matmul")
    try:
        extension = import_module("mlx_nax_int")
    except ImportError as error:
        message = f"mlx_nax_int is not installed: {error}"
    else:
        missing = [name for name in required if not hasattr(extension, name)]
        if not missing:
            return extension
        message = f"mlx_nax_int lacks fused grouped W8A8 operations: {missing}"

    if request.config.getoption("--require-nax"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture(scope="session")
def local_checkpoint(request):
    """Resolve an optional checkpoint, with a fail-closed validation mode."""

    def resolve(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate
        message = f"local checkpoint is absent: {candidate}"
        if request.config.getoption("--require-checkpoints"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    return resolve
