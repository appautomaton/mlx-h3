<div align="center">

# mlx-h3

**Pure MLX MiniMax-H3 text-to-video-and-audio inference for Apple Silicon.**

[![Release](https://img.shields.io/github/v/release/appautomaton/mlx-h3?include_prereleases&style=flat-square&color=F59E0B&label=release)](https://github.com/appautomaton/mlx-h3/releases)
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
> This project is pre-alpha. Every package version ships as a GitHub pre-release under
> the matching `v{version}` tag. Model files are not included in the repository or in
> the PyPI package.

## Why mlx-h3

- **Joint audio and video.** One DiT denoises both modalities in a shared sequence.
- **Pure MLX runtime.** No PyTorch execution and no CUDA dependency.
- **Bounded model residency.** Text encoder, DiT, Video VAE, and Audio VAE load and
  release in separate phases.
- **Current sampling baseline.** 20 `simple` schedule steps with the second-order
  `res_multistep` solver.
- **Dependency-light tokenizer.** Byte-level BPE implemented locally from
  `tokenizer.json`.
- **Fail-fast memory guard.** Configurable active-memory budget and swap detection.

## Current scope

| Capability | Status |
|---|---|
| Text-to-video-and-audio (T2VA) | Working |
| Synchronized H.264/AAC MP4 output | Working |
| 8-bit DiT and text encoder loading | Working |
| First/last-frame conditioning (FL2VA) | Working |
| Ordered image/video/audio references (Ref2VA) | Working |
| Reference-video soundtrack conditioning | Working |
| Community Turbo LoRA with paired-schedule Euler | Working (opt-in) |
| M5 NAX W8A8 DiT trunk | Experimental (opt-in) |
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

From PyPI. The `--prerelease allow` flag selects the current pre-release, so no
version needs pinning:

```sh
uv tool install --prerelease allow mlx-h3
```

## Local model layout

Model files stay outside version control. The default paths are:

```text
weights/
├── tokenizer/tokenizer.json
├── mlx-8bit/te_qwen3vl_a8g32.safetensors
├── mlx-8bit/dit_fl2va_a8g32.safetensors
├── mlx-8bit/dit_ref2va_a8g32.safetensors
├── adapters/minimax-h3-turbo/
│   ├── minimax_h3_turbo_v4_step600_ema.safetensors
│   └── minimax_h3_turbo_4step_ema_ckpt850.safetensors
└── bf16/vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
```

Exactly one DiT loads per request, selected by conditioning mode. Both VAE checkpoints
stay dense and are runtime inputs. Dense DiT and text-encoder checkpoints are
requantization sources only, and inference never reads them.

### Getting the model files

The complete mixed-precision runtime bundle is published as
[appautomaton/minimax-h3-base-8bit-mlx](https://huggingface.co/appautomaton/minimax-h3-base-8bit-mlx).
It contains the 8-bit DiTs and text encoder, both dense VAEs at their released precision,
and the tokenizer in the directory layout expected by the runtime:

```sh
hf download appautomaton/minimax-h3-base-8bit-mlx --local-dir weights
```

To build the 8-bit files locally from the BF16 sources instead of downloading them,
see [docs/weights.md](docs/weights.md).

### Optional Turbo adapters

The optional community Turbo adapters stay BF16 and separate from both 8-bit DiT
checkpoints. Download the two EMA checkpoints validated by this runtime from
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
without moving or merging any existing weight:

```sh
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors \
  minimax_h3_turbo_4step_ema_ckpt850.safetensors \
  --local-dir weights/adapters/minimax-h3-turbo
```

Use `v4-step600 EMA` for most requests, preferably at six to eight sampling steps.
The older `v1-850 EMA` is an optional fallback for heavy or fast motion when a request
must stay at exactly four steps. Here `v4` names the training recipe and `step600` the
training checkpoint. Neither is the inference step count.

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
Ref2VA should read them. Use `--ref-video-silent` to ignore embedded audio, or
`--ref-video-with-audio VIDEO AUDIO` to override a video's soundtrack.

Canvas dimensions must be multiples of 32 and may not exceed `768 * 1344` pixels.
Frame requests are aligned to the Video VAE's `17n + 5` rule and capped at the
released 15-second limit. Use `--steps 10` for a faster preview. `--steps 20` is the
quality baseline.

To use the community Turbo LoRA, provide its local path explicitly. This switches the
DiT phase to first-order Euler with video and audio advancing on their own sigma grids:

```sh
uv run mlx-h3 "$MLX_H3_INPUT_TEXT" \
  --turbo-lora weights/adapters/minimax-h3-turbo/minimax_h3_turbo_v4_step600_ema.safetensors \
  --steps 6 \
  --output outputs/turbo-result.mp4
```

Turbo mode defaults to six steps and accepts explicit values from four through eight.
Values outside that range fail before model loading. Both checkpoints use the same module
geometry and inference path, so selecting one does not change per-step cost. The adapters
are a community early preview, not an official MiniMax release. Their module geometry
matches both local DiTs, and T2VA/FL2VA has the author-supported base path. Ref2VA is
structurally compatible and has passed local smoke validation, but remains experimental
because its author has not yet declared Ref2VA support.

Run `uv run mlx-h3 --help` for checkpoint path overrides and all generation options.

On M5 hardware, the local experimental NAX extension can replace the 200 main
DiT trunk linears with group-scaled W8A8 execution. This remains opt-in and does
not load dense DiT weights:

```sh
uv run mlx-h3 "$MLX_H3_INPUT_TEXT" \
  --nax-group-size 896 \
  --output outputs/nax-result.mp4
```

Accepted group sizes are 64, 256, 448, and 896. The extension must first be installed
from the matching local MLX experiment. The default remains MLX W8A16. One fixed-seed
full generation has passed local numerical and visual A/B checks, but that is not a
general perceptual-quality baseline.

## Memory model

The pipeline intentionally keeps only one large model phase resident at a time:

```text
reference encoders -> release -> text/vision encode -> release
                   -> joint denoise (+ optional LoRA) -> release -> video decode -> release
                   -> audio decode -> release -> mux
```

Safety checks remain enabled in release runs. Scalar telemetry is emitted only when a
callback is attached, so normal inference does not retain diagnostic tensors or model
objects.

## Development

```sh
uv run ruff check .
uv run pytest -q
uv run pytest -q -m "not checkpoint and not fixture and not runtime"
uv run pytest -q -m "not checkpoint and not fixture and not runtime" \
  --cov=mlx_h3 --cov-branch --cov-report=term-missing
python dev/check_public_tree.py
uv build --no-sources
```

The unmarked suite is deterministic, weightless, and runs on every change. Optional
validation tiers fail closed when explicitly required:

```sh
uv run pytest -q -m fixture --require-fixtures
uv run pytest -q -m checkpoint --require-checkpoints
uv run pytest -q -m runtime
```

Reference fixtures are supplied through the local environment variables documented by
the relevant tests. Checkpoint tests read local safetensors structure without loading
the full inference payload. Runtime tests may require FFmpeg, Metal extensions, or other
machine-specific capabilities.

Branch coverage is gated at the measured fast-suite baseline. Raise the floor as
new tests close identified gaps; do not lower it to accommodate a change.

The public-tree check rejects model files, media, private inputs, generated artifacts,
large files, hidden local state, symlinks, and structured private prompt payloads. A local
pre-commit hook runs the same check against staged files.

Reference notes live in [docs/](docs/): [architecture](docs/architecture.md) (what H3 is),
[weights](docs/weights.md) (what is on disk), [porting](docs/porting.md) (validation and pitfalls),
[prompting](docs/prompting.md) (what to send the encoder).

## Project identity

- Distribution and CLI: `mlx-h3`
- Python import package: `mlx_h3`
- Published weights: [appautomaton/minimax-h3-base-8bit-mlx](https://huggingface.co/appautomaton/minimax-h3-base-8bit-mlx)
- Project page: [appautomaton.renocrypt.com/mlx-h3](https://appautomaton.renocrypt.com/mlx-h3/)
- Repository: [appautomaton/mlx-h3](https://github.com/appautomaton/mlx-h3)
- Runtime: pure MLX on Apple silicon

This project is not affiliated with or endorsed by MiniMax.
