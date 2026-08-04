"""Validate that GitHub release metadata matches the package version."""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path


PRE_RELEASE = re.compile(r"(?:a|b|rc)\d+|\.dev\d+")


def main() -> int:
    with Path("pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]

    tag = os.environ.get("GITHUB_REF_NAME")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not tag or not event_path:
        raise RuntimeError("GitHub release environment is missing")

    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    release_is_prerelease = bool(event["release"]["prerelease"])
    version_is_prerelease = PRE_RELEASE.search(version) is not None

    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise RuntimeError(f"release tag {tag!r} must be {expected_tag!r}")
    if release_is_prerelease != version_is_prerelease:
        raise RuntimeError(
            "GitHub pre-release state must match the Python package version"
        )

    print(f"Release metadata is valid: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
