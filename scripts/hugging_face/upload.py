#!/usr/bin/env python3
"""Publish the MLX 8-bit MiniMax-H3 artifacts to Hugging Face.

Usage:
    python scripts/hugging_face/upload.py --dry-run     # plan only, no transfer
    python scripts/hugging_face/upload.py --card-only   # refresh README.md
    python scripts/hugging_face/upload.py               # weights, then the card

The local tree is flat and the repository is foldered, so the artifacts are
hardlinked into a staging directory first. Hardlinks share inodes, so this costs
no disk even at 97 GiB, and `upload-large-folder` sees an ordinary tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "appautomaton/minimax-h3-base-8bit-mlx"

# local file under weights/mlx-8bit/ -> directory it occupies in the repository
ARTIFACTS = {
    "dit_fl2va_a8g32.safetensors": "dit-fl2va",
    "dit_ref2va_a8g32.safetensors": "dit-ref2va",
    "te_qwen3vl_a8g32.safetensors": "text-encoder",
}

SOURCE = Path("weights/mlx-8bit")
STAGING = Path("weights/.hf-staging")
CARD = Path("scripts/hugging_face/model_cards/appautomaton/minimax-h3-base-8bit-mlx.md")


def resolve_hf() -> str:
    hf = shutil.which("hf")
    if hf is None:
        raise FileNotFoundError(
            "no hf CLI on PATH; install huggingface_hub or activate the environment"
        )
    return hf


def stage(root: Path) -> Path:
    """Hardlink the artifacts into the layout the repository uses."""
    staging = root / STAGING
    if staging.exists():
        shutil.rmtree(staging)

    for name, folder in ARTIFACTS.items():
        source = root / SOURCE / name
        if not source.is_file():
            raise FileNotFoundError(f"missing artifact: {source}")
        target = staging / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Fall back to a copy across filesystems; hardlinking is the normal path.
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return staging


def run(command: list[str], *, env: dict[str, str]) -> None:
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, check=False, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload README.md without touching the weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the commands without transferring anything.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    card = root / CARD
    if not card.is_file():
        raise FileNotFoundError(f"missing model card: {card}")

    env = os.environ.copy()
    # Xet is disabled for every appautomaton upload; it has not been reliable on
    # artifacts this size.
    env["HF_HUB_DISABLE_XET"] = "1"

    if args.dry_run:
        print(f"repo: {REPO_ID}")
        total = 0
        for name, folder in ARTIFACTS.items():
            source = root / SOURCE / name
            size = source.stat().st_size if source.is_file() else 0
            total += size
            state = "ok" if size else "MISSING"
            print(f"  {folder}/{name}  {size / 1024**3:.1f} GiB  {state}")
        print(f"  README.md <- {CARD}")
        print(f"  total: {total / 1024**3:.1f} GiB")
        return 0

    hf = resolve_hf()

    if not args.card_only:
        staging = stage(root)
        # One worker: these are 35 GiB single files, and parallel streams have
        # been the thing that breaks rather than the thing that helps.
        run(
            [
                hf,
                "upload-large-folder",
                "--repo-type",
                "model",
                "--num-workers",
                "1",
                REPO_ID,
                str(staging),
            ],
            env=env,
        )

    run(
        [hf, "upload", "--repo-type", "model", REPO_ID, str(card), "README.md"],
        env=env,
    )

    print(f"Done. https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
