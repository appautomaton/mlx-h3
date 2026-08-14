#!/usr/bin/env python3
"""Publish the complete MLX MiniMax-H3 runtime bundle to Hugging Face.

python scripts/hugging_face/upload.py --dry-run     plan only
python scripts/hugging_face/upload.py --card-only   refresh README.md
python scripts/hugging_face/upload.py --only te     one artifact
python scripts/hugging_face/upload.py               runtime assets, then metadata

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
SOURCE = Path("weights")
CARD = Path("scripts/hugging_face/model_cards/appautomaton/minimax-h3-base-8bit-mlx.md")
LICENSE = Path(
    "scripts/hugging_face/model_cards/appautomaton/minimax-h3-base-8bit-mlx.LICENSE"
)
QWEN_LICENSE = Path(
    "scripts/hugging_face/model_cards/appautomaton/"
    "minimax-h3-base-8bit-mlx.LICENSE-QWEN"
)
NOTICE = Path(
    "scripts/hugging_face/model_cards/appautomaton/minimax-h3-base-8bit-mlx.NOTICE"
)
# Short name -> filename. Ordered smallest first, which is also the order to
# upload them in: the cheapest artifact proves the path before the big ones.
ARTIFACTS = {
    "tokenizer": "tokenizer/tokenizer.json",
    "audio-vae": "bf16/vae/minimax_h3_audio_vae_fp32.safetensors",
    "video-vae": "bf16/vae/minimax_h3_video_vae_fp16.safetensors",
    "te": "mlx-8bit/te_qwen3vl_a8g32.safetensors",
    "fl2va": "mlx-8bit/dit_fl2va_a8g32.safetensors",
    "ref2va": "mlx-8bit/dit_ref2va_a8g32.safetensors",
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
    license_file = root / LICENSE
    qwen_license = root / QWEN_LICENSE
    notice = root / NOTICE
    source = root / SOURCE
    metadata_files = {
        "README.md": card,
        "LICENSE": license_file,
        "LICENSE-QWEN": qwen_license,
        "NOTICE": notice,
    }
    missing_metadata = [
        str(path) for path in metadata_files.values() if not path.is_file()
    ]
    if missing_metadata:
        raise FileNotFoundError(f"missing metadata: {', '.join(missing_metadata)}")

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
            for destination, path in metadata_files.items():
                print(f"  {destination} <- {path.relative_to(root)}")
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

    for destination, path in metadata_files.items():
        run(
            [
                hf,
                "upload",
                "--repo-type",
                "model",
                REPO_ID,
                str(path),
                destination,
            ],
            env,
        )
    print(f"Done. https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
