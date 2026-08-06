<div align="center">

# mlx-h3

**Pure MLX MiniMax-H3 text-to-video-and-audio inference for Apple Silicon.**

[![Pre-release](https://img.shields.io/badge/release-v0.0.1a3-F59E0B?style=flat-square)](https://github.com/appautomaton/mlx-h3/releases)
[![PyPI](https://img.shields.io/pypi/v/mlx-h3?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-h3/)
[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-native-000000?style=flat-square&logo=apple&logoColor=white)](https://support.apple.com/mac/)
[![MLX](https://img.shields.io/badge/backend-MLX-7C3AED?style=flat-square)](https://github.com/ml-explore/mlx)

[**appautomaton.renocrypt.com/mlx-h3**](https://appautomaton.renocrypt.com/mlx-h3/)

</div>

`mlx-h3` is an independent, pure-MLX inference runtime for
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). It generates video and
stereo audio jointly, keeps model residency phase-scoped, and targets large-memory
Apple silicon systems without using PyTorch at runtime.

> [!IMPORTANT]
> This project is pre-alpha. This package version is released as the `v0.0.1a3`
> GitHub pre-release. Model files are not included in the repository or PyPI package.

## Why mlx-h3

- **Joint audio and video** — one DiT denoises both modalities in a shared sequence.
- **Pure MLX runtime** — no PyTorch execution and no CUDA dependency.
- **Bounded model residency** — text encoder, DiT, Video VAE, and Audio VAE load and
  release in separate phases.
- **Current sampling baseline** — 20 `simple` schedule steps with the second-order
  `res_multistep` solver.
- **Dependency-light tokenizer** — byte-level BPE implemented locally from
  `tokenizer.json`.
- **Fail-fast memory guard** — configurable active-memory budget and swap detection.

## Current scope

| Capability | Status |
|---|---|
| Text-to-video-and-audio (T2VA) | Working |
| Synchronized H.264/AAC MP4 output | Working |
| 8-bit DiT and text encoder loading | Working |
| First/last-frame conditioning (FL2VA) | Working |
| Ordered image/video/audio references (Ref2VA) | Working |
| Reference-video soundtrack conditioning | Working |
| Context-IR and 2K regeneration | Not available locally |

## Requirements

- Apple silicon Mac
- macOS with a recent MLX-compatible toolchain
- Python 3.13 or newer
- `ffmpeg` available on `PATH`
- Local MiniMax-H3 tokenizer and checkpoints
- Enough unified memory for the selected canvas and frame count

The default runtime memory budget is 70 GiB. It is a guardrail, not a promise that
every system workload will remain swap-free.

## Install

From a local checkout:

```sh
git clone https://github.com/appautomaton/mlx-h3.git
cd mlx-h3
uv sync
```

Install the current PyPI pre-release:

```sh
uv tool install --prerelease allow mlx-h3==0.0.1a3
```

## Local model layout

Model files stay outside version control. The default paths are:

```text
weights/
├── tokenizer/tokenizer.json
├── mlx-8bit/te_qwen3vl_a8g32.safetensors
├── mlx-8bit/dit_fl2va_a8g32.safetensors
├── mlx-8bit/dit_ref2va_a8g32.safetensors
└── bf16/vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
```

Dense DiT and text-encoder weights may be retained locally for requantization, but
inference never loads them. The dense Video VAE and Audio VAE checkpoints are runtime
inputs.

## Generate

Keep private input text in your shell environment rather than a tracked file:

```sh
uv run mlx-h3 "$MLX_H3_INPUT_TEXT" \
  --width 512 \
  --height 288 \
  --frames 124 \
  --steps 20 \
  --seed 42 \
  --output outputs/result.mp4
```

Long structured prompts can instead stay in an untracked UTF-8 file:

```sh
uv run mlx-h3 --prompt-file "$MLX_H3_PROMPT_FILE" \
  --width 768 \
  --height 448 \
  --frames 124 \
  --steps 10 \
  --output outputs/preview.mp4
```

Conditioning inputs are explicit. `--first-frame` and `--last-frame` select the
FL2VA path. Repeat `--ref-image`, `--ref-video`, and `--ref-audio` in the order
Ref2VA should read them; use `--ref-video-silent` to ignore embedded audio or
`--ref-video-with-audio VIDEO AUDIO` to override a video's soundtrack.

Canvas dimensions must be multiples of 32 and may not exceed `768 * 1344` pixels.
Frame requests are aligned to the Video VAE's `17n + 5` rule and capped at the
released 15-second limit. Use `--steps 10` for a faster preview; `--steps 20` is the
quality baseline.

Run `uv run mlx-h3 --help` for checkpoint path overrides and all generation options.

## Memory model

The pipeline intentionally keeps only one large model phase resident at a time:

```text
reference encoders -> release -> text/vision encode -> release
                   -> joint denoise -> release -> video decode -> release
                   -> audio decode -> release -> mux
```

Safety checks remain enabled in release runs. Scalar telemetry is emitted only when a
callback is attached, so normal inference does not retain diagnostic tensors or model
objects.

## Development

```sh
uv run ruff check .
uv run pytest -q
python dev/check_public_tree.py
uv build --no-sources
```

The public-tree check rejects model files, media, private inputs, generated artifacts,
large files, hidden local state, symlinks, and structured private prompt payloads. A local
pre-commit hook runs the same check against staged files.

Reference notes live in [docs/](docs/): [architecture](docs/architecture.md) (what H3 is),
[weights](docs/weights.md) (what is on disk), [porting](docs/porting.md) (validation and pitfalls),
[prompting](docs/prompting.md) (what to send the encoder).

## Project identity

- Distribution and CLI: `mlx-h3`
- Python import package: `mlx_h3`
- Project page: [appautomaton.renocrypt.com/mlx-h3](https://appautomaton.renocrypt.com/mlx-h3/)
- Repository: [appautomaton/mlx-h3](https://github.com/appautomaton/mlx-h3)
- Runtime: pure MLX on Apple silicon

This project is not affiliated with or endorsed by MiniMax.
