# Weights

What is on disk, what inference loads, and the rules for producing the quantized files again.

## Layout

    weights/bf16/       source, from Comfy-Org/MiniMax-H3
      diffusion_models/minimax_h3_fl2va_bf16.safetensors     61.7 GiB  535 tensors
      diffusion_models/minimax_h3_ref2va_bf16.safetensors    61.7 GiB  535 tensors
      text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors  48.0 GiB  902 tensors
      vae/minimax_h3_video_vae_fp16.safetensors               4.9 GiB  562 tensors
      vae/minimax_h3_audio_vae_fp32.safetensors               0.6 GiB  917 tensors
    weights/mlx-8bit/   produced by dev/quantize.py
      dit_fl2va_a8g32.safetensors                            34.8 GiB
      dit_ref2va_a8g32.safetensors                           34.8 GiB
      te_qwen3vl_a8g32.safetensors                           27.7 GiB

Inference loads one 8-bit DiT selected by the conditioning mode, the 8-bit text encoder, and the
two **dense** VAE files. The bf16 DiT and text encoder exist only as requantization inputs and are
never loaded at runtime.

To fetch the sources:

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir weights/bf16
```

The Comfy-Org text encoder is already truncated to layers 0–49 with no `lm_head`, which is exactly
what H3 reads. That truncation is lossless — do not download the official 66.7 GB text encoder.

## A pre-quantized build cannot be substituted

**ComfyUI's `int8_convrot` is unusable from MLX.** This is a format problem, not a precision one.
`int8_convrot` is tensor-wise int8 plus a 256-group rotation (QuaRot / SpinQuant family) that
suppresses outliers, and the forward pass must apply the matching rotation to activations to cancel
it. That rotation lives in comfy_kitchen CUDA code (`comfy.quant_ops.ck`), not in Python.

|  | Granularity | Precision strategy |
|---|---|---|
| ComfyUI int8 | one scale per tensor | rotation-based outlier suppression |
| MLX affine | one scale per 32 elements | fine granularity |

You cannot adopt half of one strategy: taking the int8 weights without the rotation destroys
accuracy, and implementing the rotation means reverse-engineering the CUDA side first.

**8-bit is the correct serving configuration, not a compromise.** The workload is compute-bound, so
quantization buys footprint and not speed. Corollary: a dequant-once bf16 weight cache is a dead
end — it recreates the bf16 residency regime without buying anything.

## Requantization rules

`dev/quantize.py` is MLX affine, 8-bit, group size 32. Four rules decide what stays dense, and each
one exists because the obvious alternative fails silently:

**Filter by dtype, not by name.** The checkpoint marks its own precision-sensitive tensors as F32 —
patch projections, time embedder, final output heads, `rope.inv_freq`. A name whitelist copied from
the diffusers docs uses diffusers naming and will not match this repack.

**Keep gathered lookup tables dense explicitly.** A shape-based "is this a linear?" test happily
packs `embed_tokens.weight` and `visual.pos_embed.weight`, which are read with `take_axis`; the
gather then returns packed uint32 garbage. This is the one case shape cannot decide.

**Write scales and biases as siblings**, `<module>.scales`, not as children of the weight. That is
where `nn.QuantizedLinear` looks for them.

**Read lazily.** Seek tensor by tensor into the safetensors payload so the 61.7 GiB DiT never lands
in RAM whole; peak is one tensor plus the accumulating output.

`loading.py` then reads *from the checkpoint* which modules are quantized — a module is quantized
if and only if the file carries `.scales` beside its weight. Do not duplicate the converter's
predicate in the loader; a predicate written twice eventually disagrees with itself.

## Residency

The selected quantized DiT and TE are **34.8 GiB and 27.7 GiB, never co-resident**. That is the
invariant `pipeline.run_phase` enforces: load one model, materialize its output, release it, assert
the memory came back. Unstaged they would be 62.5 GiB of weights before a single activation.

## DiT tensor naming

```
blocks.N.adaln_proj.linear.{weight,bias}    AdaLN
blocks.N.attn.qkv_proj.weight               fused qkv; split order is a gotcha
blocks.N.attn.{q_norm,k_norm}.weight        per-head RMSNorm, applied BEFORE rope
blocks.N.attn.out_proj.weight
blocks.N.mlp.{fc1,fc2}.weight               fc1 is fused gate+up; half order is a gotcha
blocks.N.{norm1,norm2}.weight
```

Top level: `blocks` (500), `token_refiner` (17), `final_layer` (7), `time_embedder` (4),
`audio_patch_proj` (2), `condition_proj` (2), `video_patch_proj` (2), `rope` (1).

`adaln_t_table` is **absent** from this checkpoint, which carries live AdaLN weights instead.
Loading therefore takes the reference's `time_embedder` branch, not its curve branch.

## AdaLN schedule precompute

About 39% of the parameters (13B of 33B) sit in AdaLN branches, whose output depends only on
`(timestep, modality)`. Before denoising, the runtime builds the exact schedule for the request,
materializes every block and final-layer modulation table, then releases the timestep embedder and
AdaLN projections block by block. Real-checkpoint parity against the weight-based path is exact.

The checkpoint remains 34.8 GiB on disk, but active DiT residency falls from 34.756 GiB to about
21.2 GiB before the first sampling step. Ten steps use about 0.172 GiB of tables. This optimization
targets residency, not step time: AdaLN cost does not scale with sequence length and is negligible
beside attention. A curve-form pruned checkpoint could later reduce disk and load footprint, but it
is not required for the current memory budget or output path.
