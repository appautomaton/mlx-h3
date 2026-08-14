---
language:
- en
license: other
license_name: minimax-h3-community-license
license_link: LICENSE
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

Complete mixed-precision MLX runtime bundle for **H3-Base**, the open stage of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). The DiTs and text encoder use MLX affine 8-bit serving weights, while the quality-sensitive Video and Audio VAEs remain at their released FP16 and FP32 precision. It turns text into synchronized video and stereo audio, denoised together in one packed sequence, with no PyTorch, CUDA, or cloud API at inference time.

## Contents

| File | Size | Role |
| --- | ---: | --- |
| `mlx-8bit/dit_fl2va_a8g32.safetensors` | 34.8 GiB | DiT for text and 0–2 keyframes; affine 8-bit |
| `mlx-8bit/dit_ref2va_a8g32.safetensors` | 34.8 GiB | DiT for reference conditioning; affine 8-bit |
| `mlx-8bit/te_qwen3vl_a8g32.safetensors` | 27.7 GiB | Qwen3-VL-32B text encoder; affine 8-bit |
| `bf16/vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB | Unmodified Video VAE; FP16 |
| `bf16/vae/minimax_h3_audio_vae_fp32.safetensors` | 577 MiB | Unmodified Audio VAE; FP32 |
| `tokenizer/tokenizer.json` | 6.7 MiB | Unmodified Qwen tokenizer |

The complete bundle is 102.7 GiB. Downloading it into `weights/` produces exactly the directory layout expected by `mlx-h3`.

The two DiTs are structurally identical and differ only in input packing. The runtime loads one, selected by conditioning mode, never both. A selective download can therefore omit the DiT mode that will not be used.

## Quantization

MLX **affine, 8 bits, group size 32**, which is the `a8g32` in each filename. One scale and one bias per 32 weights, stored as bf16 alongside the packed integers. Each file also records `quantization.mode`, `quantization.bits`, `quantization.group_size`, and its `source` filename in safetensors metadata.

A tensor is packed only when all of these hold: it is bf16, it is named `.weight`, it is rank 2, and its last axis divides by 32. Three categories are held back on purpose.

- **F32 tensors.** The release marks its own precision-sensitive tensors as F32, covering patch projections, the time embedder, the final output heads, and `rope.inv_freq`. Filtering by dtype is exact where a name whitelist borrowed from another framework is not.
- **Gathered lookup tables.** `embed_tokens` and `pos_embed` are read with `take_axis`, and packing them returns uint32 garbage after the gather.
- **Everything a scale cannot hang from**, meaning biases, norms, and any tensor whose last axis is not a multiple of the group size.

The result is 260 of 535 DiT tensors packed, and 439 of 902 in the text encoder.

This is not ComfyUI's `int8_convrot`, which is tensor-wise int8 plus a 256-group rotation and cannot be loaded from MLX. The distinction is format, not precision.

8-bit is the serving configuration here rather than a compromise. In MLX's affine path the activations stay bf16, so lower weight precision buys residency rather than throughput.

## This is H3-Base only

MiniMax-H3 ships as three stages. Only the middle one is released.

| Stage | Role | Here |
| --- | --- | --- |
| H3-Context-IR | prompt → structured brief | No |
| **H3-Base** | 768p joint audio-video generation | **Yes** |
| H3-Regenerate-2K | 768p → 2K upscale | No |

Two consequences worth knowing before you download the bundle:

- **Nothing rewrites your prompt.** The hosted product expands a one-line request into a structured brief first. Here the encoder sees exactly what you send. See [the prompting guide](https://github.com/appautomaton/mlx-h3/blob/main/docs/prompting.md).
- **Output is 768p.** The 2K figures in hosted guides describe the third stage, which is not open.

## Upstream assets included unchanged

Three small or dense runtime assets are included unchanged so one repository download is sufficient.

Both VAEs come from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), and the tokenizer comes from the [official MiniMax-H3 release](https://huggingface.co/MiniMaxAI/MiniMax-H3). Their SHA-256 digests are recorded here for provenance.

| File | SHA-256 |
| --- | --- |
| `minimax_h3_video_vae_fp16.safetensors` | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `minimax_h3_audio_vae_fp32.safetensors` | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `tokenizer.json` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |

## Works with these files

- **Community Turbo LoRA.** The BF16 adapters from [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) target 259 DiT linears, all of which are present in both checkpoints here. The runtime loads an adapter inside the DiT phase and releases it with that phase, so nothing is merged into these files.
- **Experimental M5 W8A8.** The runtime's opt-in NAX path requantizes the already-loaded 8-bit trunk linears into symmetric W8A8 groups and dispatches native integer TensorOps. It reads these files, never the dense source.

## How to get started

```sh
hf download appautomaton/minimax-h3-base-8bit-mlx --local-dir weights
```

Runtime install and usage: [`mlx-h3` on PyPI](https://pypi.org/project/mlx-h3/) ·
[project page](https://appautomaton.renocrypt.com/mlx-h3/) ·
[GitHub](https://github.com/appautomaton/mlx-h3)

## Requirements

Apple silicon with enough unified memory to hold one model at a time. The DiT and text encoder are **never co-resident**. The runtime loads one, materializes its output, releases it, and asserts the memory came back. Unstaged they would be 62.5 GiB of weights before a single activation. The default active-memory budget is 70 GiB, and swap activity is treated as a failure rather than a slowdown.

## Links

- Source code: [`appautomaton/mlx-h3`](https://github.com/appautomaton/mlx-h3)
- Package: [`mlx-h3` on PyPI](https://pypi.org/project/mlx-h3/)
- Project page: [appautomaton.renocrypt.com/mlx-h3](https://appautomaton.renocrypt.com/mlx-h3/)
- More from App Automaton: [Project](https://appautomaton.renocrypt.com) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

The DiTs and text encoder are **modified** quantized derivatives. The tokenizer and both VAEs are unmodified upstream assets included to make the runtime bundle complete.

The DiTs and VAEs are governed by the [MiniMax H3 Community License Agreement](LICENSE), which applies to you as a recipient. It carries territorial limits and an acceptable-use policy, so read it before use. The required redistribution notice is provided in [`NOTICE`](NOTICE).

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

`mlx-8bit/te_qwen3vl_a8g32.safetensors` and `tokenizer/tokenizer.json` derive from [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct), licensed under [Apache 2.0](LICENSE-QWEN).

The `mlx-h3` runtime code is separately licensed under MIT.
