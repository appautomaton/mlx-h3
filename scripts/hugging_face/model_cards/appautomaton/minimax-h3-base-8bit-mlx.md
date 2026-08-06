---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
library_name: mlx
pipeline_tag: text-to-video
base_model: MiniMaxAI/MiniMax-H3
tags:
- mlx
- apple-silicon
- minimax
- text-to-video
- text-to-audio-video
- audio-video-generation
- quantized
---

# MiniMax-H3-Base — MLX (8-bit)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--h3-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-h3)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.renocrypt.com/mlx-h3/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

MLX affine 8-bit quantization of **H3-Base**, the open stage of
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — text to synchronized video and
stereo audio, denoised together in one packed sequence. Runs through the
[`mlx-h3`](https://github.com/appautomaton/mlx-h3) runtime on Apple silicon, with no
PyTorch at inference time.

## Contents

| Path | Size | What it is |
| --- | --- | --- |
| `dit-fl2va/dit_fl2va_a8g32.safetensors` | 34.8 GiB | DiT for text-only and 0–2 keyframe conditioning |
| `dit-ref2va/dit_ref2va_a8g32.safetensors` | 34.8 GiB | DiT for reference conditioning — up to 9 images, 3 videos, 3 audio clips |
| `text-encoder/te_qwen3vl_a8g32.safetensors` | 27.7 GiB | Qwen3-VL-32B text encoder and vision tower |

The two DiTs are structurally identical and differ only in input packing. Load one,
selected by conditioning mode — never both.

## Quantization

MLX affine, **8 bits, group size 32**. Per-group scale and bias in BF16, so the effective
cost is closer to 9 bits per weight than 8.

Not to be confused with ComfyUI's `int8_convrot`, which is tensor-wise int8 plus a
256-group rotation. That format is unusable from MLX — the rotation has to be folded into
the compute path, and taking the weights without it destroys accuracy.

Layers under 2 dimensions, and any whose last axis is not divisible by 32, stay dense.

## This is H3-Base only

H3 ships as three stages and only the middle one is open:

| Stage | Role | Here |
| --- | --- | --- |
| H3-Context-IR | prompt → structured brief | No, API only |
| **H3-Base** | 768p joint audio-video generation | **Yes** |
| H3-Regenerate-2K | 768p → 2K upscale | No, API only |

Two consequences worth knowing before you download 97 GiB:

**Nothing rewrites your prompt.** The hosted product expands a one-line request into a
structured brief before generation. That stage is not released, so the text you pass is the
text the encoder sees. See [the prompting notes](https://github.com/appautomaton/mlx-h3/blob/main/docs/prompting.md).

**2K figures from hosted guides do not apply.** Upscaling is the third stage. This is 768p,
capped at 768×1344.

## Not included

The two VAEs are required at runtime and are **not** quantized — they stay dense, so there
is no MLX-specific build to publish. Get them from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), along with the
tokenizer.

## Getting started

```sh
uv tool install --prerelease allow mlx-h3==0.0.1a3
hf download appautomaton/minimax-h3-base-8bit-mlx --local-dir weights/mlx-8bit
```

Point the runtime at the files with `--dit`, `--ref-dit`, and `--text-encoder`, or lay them
out as `mlx-h3` expects and rely on the defaults. Pull one component with `--include` if you
do not want all three.

## Requirements

Apple silicon with enough unified memory to hold one model at a time. The DiT and text
encoder are **never co-resident** — `mlx-h3` loads one, materializes its output, releases
it, and asserts the memory came back. Unstaged they would be 62.5 GiB of weights before a
single activation. The runtime targets a 70 GiB active budget and treats swap as a failure.

## Links

- Source code: [`appautomaton/mlx-h3`](https://github.com/appautomaton/mlx-h3)
- Project page: [appautomaton.renocrypt.com/mlx-h3](https://appautomaton.renocrypt.com/mlx-h3/)
- Upstream model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

## License

These are quantized derivatives of MiniMax-H3 — the weights have been modified from the
original release.

MiniMax-H3 is licensed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which carries use restrictions and a territorial scope. Read it before using these files.

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement,
> Copyright © 2026 MiniMax. All Rights Reserved.

`text-encoder/` derives from Qwen3-VL-32B, licensed under
[Apache 2.0](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE).

The `mlx-h3` runtime code is separately licensed under MIT.
