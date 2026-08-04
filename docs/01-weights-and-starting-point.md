# Weights: what to download

Target machine: MacBook Pro / M5 Max / 128 GB.

## Decision

**`Comfy-Org/MiniMax-H3`, bf16, 123.6 GB. Quantize to MLX affine group-64 in-house.**

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir weights/bf16
```

Downloaded and integrity-verified 2026-08-04:

| File | Size | Tensors | Structure |
|---|---|---|---|
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 61.7 GiB | 535 | layers 0–49, 13xF32 + 522xBF16 |
| `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 48.0 GiB | 902 | layers 0–49 (truncated), all BF16 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.9 GiB | 562 | all F16 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.6 GiB | 917 | all F32 |

Integrity check used: header-declared data end + header length == actual file size. All four match.

Keep the released configuration and tokenizer metadata with the local model files.

## Why bf16 and not a pre-quantized build

**ComfyUI's `int8_convrot` is not usable from MLX.** Not a precision problem, a format problem.
From `comfy/quant_ops.py`:

```python
QUANT_ALGOS["int8_tensorwise"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},          # one scale for the whole tensor
    "comfy_tensor_layout": "TensorWiseINT8Layout",
}
```

`int8_convrot` = `int8_tensorwise` plus a 256-group rotation. The rotation suppresses outliers
(QuaRot / SpinQuant family) and the forward pass must apply the matching rotation to activations
to cancel it — `comfy/ops.py:1588` notes "rotated embedding table: record it so the forward
un-rotates after lookup". The rotation itself lives in comfy_kitchen CUDA code
(`comfy.quant_ops.ck`), not in the Python.

|  | Granularity | Precision strategy |
|---|---|---|
| ComfyUI int8 | one scale per tensor | rotation-based outlier suppression |
| MLX affine | one scale per 64 elements | fine granularity |

These are different strategies and you cannot adopt half of one. Taking ComfyUI's int8 weights
without implementing the rotation destroys accuracy; implementing it means reverse-engineering
the CUDA side first.

**Third-party MLX builds are cross-checks, not foundations.** `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit`
is a one-day-old personal project. Its structure is verified sound (see below) and its converter's
reasoning is worth reading, but the provenance-clean path is to quantize the official-partner bf16
release ourselves.

**Selecting FL2VA over Ref2VA.** The two DiTs are structurally identical and differ only in input
packing. Ref2VA's `packing_ref2va` is the most complex piece in the diffusers integration (841 lines)
and must advance the shared rotary clock across ordered references. Build FL2VA first; Ref2VA later
adds a packing layer without touching the DiT.

## Text encoder truncation (verified)

H3 reads only the layer-50 unnormalized hidden state and never uses the lm_head, so layers 51–64
plus the lm_head are dead weight:

```
14 layers x ~488M params x 2 bytes  ~= 13.7 GB
lm_head 151936 x 5120 x 2 bytes     ~=  1.6 GB
                                       --------
                                       15.3 GB

66.7 - 15.3 = 51.4 ~= 51.5 GB   (matches the published file size)
```

Confirmed by reading the safetensors header of the downloaded file:

```
layer indices:  0 ~ 49   (50 layers, 51-64 absent)
lm_head:        absent
tensors:        902, all BF16
```

Comfy-Org already performed this truncation. It is lossless. Do not download the official
66.7 GB text encoder.

## The 13 F32 tensors

The DiT checkpoint marks its own precision-sensitive tensors as F32. **Filter by dtype rather than
maintaining a name whitelist** — the diffusers docs' `modules_to_not_convert` list uses diffusers
naming, which does not match this Comfy-Org repack.

```
video_patch_proj.{weight,bias}      [5376, 96]
audio_patch_proj.{weight,bias}      [5376, 32]
time_embedder.proj_in.{weight,bias} [5376, 256]
time_embedder.proj_out.{weight,bias}[2688, 5376]
final_layer.video_out.{weight,bias} [96, 5376]
final_layer.audio_out.{weight,bias} [32, 5376]
rope.inv_freq                       [16]
```

Also keep dense regardless of dtype: gathered embedding tables. A shape-based "is this a linear?"
test will happily pack `embed_tokens.weight` and `visual.pos_embed.weight`, which are read with
`take_axis`; the gather then returns packed uint32 garbage. Lookup tables are the one case shape
cannot decide.

## Tensor naming (Comfy-Org repack)

```
blocks.N.adaln_proj.linear.{weight,bias}    AdaLN, precomputable into a table
blocks.N.attn.qkv_proj.weight               fused qkv; split order is a gotcha
blocks.N.attn.{q_norm,k_norm}.weight        per-head RMSNorm, applied BEFORE rope
blocks.N.attn.out_proj.weight
blocks.N.mlp.{fc1,fc2}.weight               fc1 is fused gate+up; half order is a gotcha
blocks.N.{norm1,norm2}.weight
```

Top level: `blocks` (500 tensors), `token_refiner` (17), `final_layer` (7), `time_embedder` (4),
`audio_patch_proj` (2), `condition_proj` (2), `video_patch_proj` (2), `rope` (1).

50 blocks x 10 tensors = 500, matching `num_layers: 50`; `token_refiner` matches
`num_refiner_layers: 2`.

`adaln_t_table` is **absent** from the bf16 checkpoint — it holds live AdaLN weights and the table
must be computed at load time.

## Memory budget

Post-quantization residency is ~69 GB, leaving ~58 GB for activations.

Measured by mlx-serve on M4 Max / 128 GB: bf16 at 62 GB resident is already "JITTERY, sits near the
working-set edge", 41–48 s/step and unstable; 8-bit stock qmm 42 s/step; 8-bit with wide-M dq-gemm
**36.6 s/step**.

**8-bit is the correct serving configuration, not a compromise.** Corollary: a dequant-once bf16
weight cache is a dead end — it recreates the bf16 residency regime that measured slower.

## AdaLN: two routes, do not stack them

Roughly 13B of the 33B sits in AdaLN branches, and it is precomputable into a table at inference
time. Two equivalent approaches:

- **Weight pruning** (ComfyUI): `int8_convrot` 34.0 GB → `pruned_int8_convrot` 21.0 GB
- **Engine-side precompute** (mlx-serve): weights retain AdaLN, the table is built at load

We take the second route. Measured AdaLN cost is **~0%** of step time — see `02-mlx-port-notes.md`.

## Reference artifacts for cross-checking

`ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` — MLX affine 8-bit (group 64), self-contained.
Header inspection confirms: 50 layers, no lm_head, `.scales`/`.biases` markers, U32 x 439 packed
plus BF16 x 1341 left dense. Quantized sizes 35.2 GB (DiT) and 28.2 GB (TE) are useful sanity
targets for our own quantization output.
