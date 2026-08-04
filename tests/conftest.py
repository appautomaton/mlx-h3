"""Shared test fixtures for optional local validation data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def local_file():
    """Resolve an optional local test file from an environment variable."""

    def resolve(variable: str) -> Path:
        value = os.environ.get(variable)
        if not value:
            pytest.skip(f"{variable} is not set")
        path = Path(value).expanduser()
        if not path.is_file():
            pytest.skip(f"{variable} does not point to a file")
        return path

    return resolve
