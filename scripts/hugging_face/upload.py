#!/usr/bin/env python3
"""Publish the MLX 8-bit MiniMax-H3 artifacts to Hugging Face.

python scripts/hugging_face/upload.py --dry-run     plan only
python scripts/hugging_face/upload.py --card-only   refresh README.md
python scripts/hugging_face/upload.py --only te     one artifact
python scripts/hugging_face/upload.py               weights, then the card

Uploads resume, so an interrupted run is restarted with the same command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "appautomaton/minimax-h3-base-8bit-mlx"
SOURCE = Path("weights/mlx-8bit")
CARD = Path("scripts/hugging_face/model_cards/appautomaton/minimax-h3-base-8bit-mlx.md")
# Short name -> filename. Ordered smallest first, which is also the order to
# upload them in: the cheapest artifact proves the path before the big ones.
ARTIFACTS = {
    "te": "te_qwen3vl_a8g32.safetensors",
    "fl2va": "dit_fl2va_a8g32.safetensors",
    "ref2va": "dit_ref2va_a8g32.safetensors",
}


def run(command: list[str], env: dict[str, str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    if subprocess.run(command, check=False, env=env).returncode != 0:
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        choices=tuple(ARTIFACTS),
        help="Upload a single artifact and skip the card.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    card = root / CARD
    source = root / SOURCE
    if not card.is_file():
        raise FileNotFoundError(f"missing model card: {card}")

    selected = [ARTIFACTS[args.only]] if args.only else list(ARTIFACTS.values())
    missing = [name for name in selected if not (source / name).is_file()]
    if missing and not args.card_only:
        raise FileNotFoundError(f"missing under {source}: {', '.join(missing)}")

    if args.dry_run:
        total = sum((source / name).stat().st_size for name in selected)
        print(f"repo: {REPO_ID}")
        for name in selected:
            print(f"  {name}  {(source / name).stat().st_size / 1024**3:.1f} GiB")
        if not args.only:
            print(f"  README.md <- {CARD}")
        print(f"  total: {total / 1024**3:.1f} GiB")
        return 0

    hf = shutil.which("hf")
    if hf is None:
        raise FileNotFoundError("no hf CLI on PATH; install huggingface_hub")

    env = os.environ.copy()
    # Xet stalls on artifacts this size. Every appautomaton upload disables it.
    env["HF_HUB_DISABLE_XET"] = "1"

    if not args.card_only:
        for name in selected:
            # One worker: a single stream already saturates the uplink, so more
            # of them buy nothing and only widen the window for a stall.
            run(
                [
                    hf,
                    "upload-large-folder",
                    "--repo-type",
                    "model",
                    "--num-workers",
                    "1",
                    "--include",
                    name,
                    REPO_ID,
                    str(source),
                ],
                env,
            )

    if args.only:
        print(f"Done. https://huggingface.co/{REPO_ID}")
        return 0

    run([hf, "upload", "--repo-type", "model", REPO_ID, str(card), "README.md"], env)
    print(f"Done. https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
