"""Shared controls for optional local validation resources."""

from __future__ import annotations

import os
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
