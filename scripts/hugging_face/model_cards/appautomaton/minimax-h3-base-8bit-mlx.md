---
language:
- en
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
- minimax-h3
- text-to-video
- text-to-audio-video
- audio-video-generation
- synchronized-audio-video
- quantized
---

# MiniMax-H3-Base — MLX (8-bit)

[![PyPI](https://img.shields.io/pypi/v/mlx-h3?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-h3/)
[![GitHub](https://img.shields.io/badge/GitHub-mlx--h3-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/appautomaton/mlx-h3)
[![Project page](https://img.shields.io/badge/project-appautomaton.renocrypt.com-F59E0B?style=flat-square)](https://appautomaton.renocrypt.com/mlx-h3/)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb?style=flat-square)](https://appautomaton.renocrypt.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow?style=flat-square)](https://huggingface.co/appautomaton)

MLX affine 8-bit conversion of **H3-Base**, the open stage of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — text to synchronized video and stereo audio, denoised together in one packed sequence. Runs on Apple silicon through the [`mlx-h3`](https://github.com/appautomaton/mlx-h3) runtime with no PyTorch, CUDA, or cloud API at inference time.

## Contents

| File | Size | Role |
| --- | ---: | --- |
| `dit_fl2va_a8g32.safetensors` | 34.8 GiB | DiT for text and 0–2 keyframes |
| `dit_ref2va_a8g32.safetensors` | 34.8 GiB | DiT for reference conditioning |
| `te_qwen3vl_a8g32.safetensors` | 27.7 GiB | Qwen3-VL-32B text encoder |

Pull one with `--include`; you do not need all three.

The two DiTs are structurally identical and differ only in input packing. Load one, selected by conditioning mode — not both.

## Quantization

MLX **affine, 8 bits, group size 32** — the `a8g32` in each filename. One scale and one bias per 32 weights, stored as bf16 alongside the packed integers.

This is not ComfyUI's `int8_convrot`, which is tensor-wise int8 plus a 256-group rotation and cannot be loaded from MLX. The distinction is format, not precision.

Layers stay dense where quantizing them costs accuracy for no useful saving: anything under two dimensions, and anything whose last axis is not divisible by the group size. 8-bit is the serving configuration here rather than a compromise — the workload is compute-bound, so lower precision buys memory without buying speed.

## This is H3-Base only

MiniMax-H3 ships as three stages. Only the middle one is released.

| Stage | Role | Here |
| --- | --- | --- |
| H3-Context-IR | prompt → structured brief | No |
| **H3-Base** | 768p joint audio-video generation | **Yes** |
| H3-Regenerate-2K | 768p → 2K upscale | No |

Two consequences worth knowing before you download 97 GiB:

- **Nothing rewrites your prompt.** The hosted product expands a one-line request into a structured brief first. Here the encoder sees exactly what you send. See [the prompting guide](https://github.com/appautomaton/mlx-h3/blob/main/docs/prompting.md).
- **Output is 768p.** The 2K figures in hosted guides describe the third stage, which is not open.

## Not included

The VAEs and tokenizer are required at runtime and are **not** in this repository. They are unmodified and available upstream from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```
video VAE   fp16   4.8 GiB
audio VAE   fp32   577 MiB
tokenizer          6.7 MiB
```

## How to get started

```sh
hf download appautomaton/minimax-h3-base-8bit-mlx --local-dir weights/mlx-8bit
```

Runtime install and usage: [`mlx-h3` on PyPI](https://pypi.org/project/mlx-h3/) ·
[project page](https://appautomaton.renocrypt.com/mlx-h3/) ·
[GitHub](https://github.com/appautomaton/mlx-h3)

## Requirements

Apple silicon with enough unified memory to hold one model at a time. The DiT and text encoder are **never co-resident** — the runtime loads one, materializes its output, releases it, and asserts the memory came back. Unstaged they would be 62.5 GiB of weights before a single activation. Default active-memory budget is 70 GiB; swap activity is treated as a failure, not a slowdown.

## Links

- Source code: [`appautomaton/mlx-h3`](https://github.com/appautomaton/mlx-h3)
- Package: [`mlx-h3` on PyPI](https://pypi.org/project/mlx-h3/)
- Project page: [appautomaton.renocrypt.com/mlx-h3](https://appautomaton.renocrypt.com/mlx-h3/)
- More from App Automaton: [Project](https://appautomaton.renocrypt.com) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

These files are **modified** — quantized derivatives of the original release, not the original weights.

The two DiTs are governed by the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), which applies to you as a recipient. It carries territorial limits and an acceptable-use policy; read it before use.

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

`text-encoder/` derives from [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct), licensed under [Apache 2.0](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE).

The `mlx-h3` runtime code is separately licensed under MIT.
