"""Reject private or generated assets from the public Git tree."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath


MAX_FILE_BYTES = 10 * 1024 * 1024

FORBIDDEN_PARTS = {
    "artifacts",
    "cases",
    "checkpoints",
    "examples",
    "inputs",
    "logs",
    "model_cache",
    "models",
    "outputs",
    "prompts",
    "runs",
    "samples",
    "tokenizer_cache",
    "weights",
}

ALLOWED_HIDDEN_DIRECTORIES = {".github", ".githooks"}
ALLOWED_HIDDEN_FILES = {".gitignore", ".python-version"}

FORBIDDEN_SUFFIXES = {
    ".aac",
    ".avi",
    ".bin",
    ".case",
    ".ckpt",
    ".flac",
    ".gguf",
    ".gif",
    ".jpeg",
    ".jpg",
    ".log",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".png",
    ".prompt",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".webm",
    ".webp",
}

# Field names from H3's structured briefs, in their assigned form. The text-only
# brief carries the first three; a reference brief adds the rest. Each is split
# across adjacent literals so the joined marker never appears in this file, which
# is itself scanned. Do not join them.
#
# A brief written as free prose, naming no field, still passes. No content check
# recognises that; the directory, suffix, and filename rules above are what keep
# working input out.
PRIVATE_PROMPT_MARKERS = {
    "integrated_multimodal_" "description:",
    "non_diegetic_" "music:",
    "overall_" "soundscape:",
    "subject_" "definitions:",
    "retention_" "analysis:",
    "detailed_" "description:",
}

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def git_paths(*, cached: bool) -> list[Path]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if cached
        else ["git", "ls-files", "-z"]
    )
    result = subprocess.run(command, check=True, capture_output=True)
    return [Path(item) for item in result.stdout.decode().split("\0") if item]


def check_path(path: Path) -> list[str]:
    failures: list[str] = []
    posix = PurePosixPath(path.as_posix())
    lowered_parts = {part.lower() for part in posix.parts}
    hidden_directories = {
        part.lower()
        for part in posix.parts[:-1]
        if part.startswith(".") and part.lower() not in ALLOWED_HIDDEN_DIRECTORIES
    }
    if hidden_directories:
        failures.append("hidden local directory")
    forbidden_parts = lowered_parts & FORBIDDEN_PARTS
    if forbidden_parts:
        failures.append(f"forbidden directory: {', '.join(sorted(forbidden_parts))}")

    lower_name = path.name.lower()
    if lower_name.startswith(".") and lower_name not in ALLOWED_HIDDEN_FILES:
        failures.append("hidden local file")
    if lower_name == ".ds_store":
        failures.append("forbidden system metadata file")
    if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        failures.append("forbidden model, prompt, case, or media file")
    if ".prompt." in lower_name or ".case." in lower_name:
        failures.append("forbidden prompt or case filename")

    if path.is_symlink():
        failures.append("symbolic links are not allowed in the public tree")
        return failures
    if not path.is_file():
        return failures
    if path.stat().st_size > MAX_FILE_BYTES:
        failures.append(f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB")

    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append("non-text content uses a text-file extension")
        else:
            marker = next(
                (item for item in PRIVATE_PROMPT_MARKERS if item in content), None
            )
            if marker is not None:
                failures.append(f"private structured-prompt marker: {marker}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Check only paths staged for commit instead of every tracked path.",
    )
    args = parser.parse_args()

    failures = {
        path: reasons
        for path in git_paths(cached=args.cached)
        if (reasons := check_path(path))
    }
    if not failures:
        print("Public tree check passed.")
        return 0

    print("Public tree check failed:")
    for path, reasons in failures.items():
        print(f"  {path}: {'; '.join(reasons)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
