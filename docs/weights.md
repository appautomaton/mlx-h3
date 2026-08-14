# Weights

What is on disk, what inference loads, and the rules for producing the quantized files again.

## Layout

    weights/tokenizer/  bundled unchanged from MiniMaxAI/MiniMax-H3
      tokenizer.json                                          6.7 MiB
    weights/mlx-8bit/   produced by dev/quantize.py, published as appautomaton/minimax-h3-base-8bit-mlx
      dit_fl2va_a8g32.safetensors                            34.8 GiB  260 of 535 modules quantized
      dit_ref2va_a8g32.safetensors                           34.8 GiB  260 of 535 modules quantized
      te_qwen3vl_a8g32.safetensors                           27.7 GiB  439 of 902 modules quantized
    weights/bf16/       dense runtime VAEs plus optional requantization sources
      diffusion_models/minimax_h3_fl2va_bf16.safetensors     61.7 GiB  535 tensors
      diffusion_models/minimax_h3_ref2va_bf16.safetensors    61.7 GiB  535 tensors
      text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors  48.0 GiB  902 tensors
      vae/minimax_h3_video_vae_fp16.safetensors               4.85 GiB 562 tensors
      vae/minimax_h3_audio_vae_fp32.safetensors               0.56 GiB 917 tensors
    weights/adapters/   optional runtime adapters
      minimax-h3-turbo/minimax_h3_turbo_v4_step600_ema.safetensors     744 MiB
      minimax-h3-turbo/minimax_h3_turbo_4step_ema_ckpt850.safetensors  744 MiB

Inference loads one 8-bit DiT selected by the conditioning mode, the 8-bit text encoder, the
tokenizer, and the two **dense** VAE files. The BF16 DiT and text encoder exist only as
requantization inputs and are never loaded at runtime. Once a quantized artifact is written and
verified, its BF16 source can be deleted and re-downloaded on demand. Each quantized file records
its own `source` filename in safetensors metadata, alongside `quantization.mode`,
`quantization.bits`, and `quantization.group_size`.

## Getting the files

The complete mixed-precision runtime bundle is published, so one download installs the
8-bit DiTs and text encoder, both dense VAEs, and the tokenizer at their default paths:

```bash
hf download appautomaton/minimax-h3-base-8bit-mlx --local-dir weights
```

The bundled VAEs are unmodified files from Comfy-Org/MiniMax-H3, and the bundled tokenizer is
the unmodified `tokenizer/tokenizer.json` from MiniMaxAI/MiniMax-H3. Their source hashes are
recorded in the published model card.

To reproduce the 8-bit artifacts, fetch the BF16 sources separately:

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --local-dir weights/bf16
```

The Comfy-Org text encoder is already truncated to layers 0–49 with no `lm_head`, which is exactly
what H3 reads. That truncation is lossless, so do not download the official 66.7 GB text encoder.

Fetch the optional community adapters separately:

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors \
  minimax_h3_turbo_4step_ema_ckpt850.safetensors \
  --local-dir weights/adapters/minimax-h3-turbo
```

When requested, the BF16 Turbo LoRA is loaded only inside the selected DiT phase and is
released with that DiT. It is never merged into either base checkpoint.

Use `v4-step600 EMA` for most requests at six to eight sampling steps. Keep `v1-850 EMA`
as an optional fallback for heavy or fast motion at exactly four steps. Both reviewed
checkpoints contain 518 BF16 tensors, meaning A/B pairs for 259 DiT linears. Rank 16 adapters
cover 50 block AdaLN projections and final AdaLN. Rank 64 adapters cover each block's attention
and MLP projections plus the two token-refiner blocks. Patch projections, condition projection,
time embedder, norms, and output heads are untouched.

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

**8-bit is the correct serving configuration, not a compromise.** In the default MLX affine path,
activations remain BF16, so weight quantization buys footprint rather than compute throughput.
A dequant-once BF16 weight cache is therefore a dead end: it recreates the BF16 residency regime
without buying anything.

The opt-in M5 NAX experiment is a different execution path. It requantizes only the already-loaded
8-bit trunk linears into symmetric W8A8 groups and uses native integer TensorOps. It never reads the
dense source checkpoint. This path remains a development experiment because its speed and error
depend on group size, and full multi-step generation quality is not yet an accepted serving
baseline.

## Requantization rules

`dev/quantize.py` is MLX affine, 8-bit, group size 32. Four rules decide what stays dense, and each
one exists because the obvious alternative fails silently:

**Filter by dtype, not by name.** The checkpoint marks its own precision-sensitive tensors as F32,
covering patch projections, time embedder, final output heads, and `rope.inv_freq`. A name whitelist
copied from the diffusers docs uses diffusers naming and will not match this repack.

**Keep gathered lookup tables dense explicitly.** A shape-based "is this a linear?" test happily
packs `embed_tokens.weight` and `visual.pos_embed.weight`, which are read with `take_axis`. The
gather then returns packed uint32 garbage. This is the one case shape cannot decide.

**Write scales and biases as siblings**, `<module>.scales`, not as children of the weight. That is
where `nn.QuantizedLinear` looks for them.

**Read lazily.** Seek tensor by tensor into the safetensors payload so the 61.7 GiB DiT never lands
in RAM whole. Peak is one tensor plus the accumulating output.

Everything else follows from shape: a tensor is quantized only if it is BF16, named `.weight`, rank
2, and has a last axis divisible by the group size. The DiT therefore quantizes 260 of its 535
tensors and the text encoder 439 of its 902, with biases, norms, and F32 heads left dense.

`loading.py` then reads *from the checkpoint* which modules are quantized. A module is quantized
if and only if the file carries `.scales` beside its weight. Do not duplicate the converter's
predicate in the loader, because a predicate written twice eventually disagrees with itself.

## Residency

The selected quantized DiT and TE are **34.8 GiB and 27.7 GiB, never co-resident**. That is the
invariant `pipeline.run_phase` enforces: load one model, materialize its output, release it, assert
the memory came back. Unstaged they would be 62.5 GiB of weights before a single activation.

Each Turbo adapter adds 779.8 MB on disk, but only the explicitly selected file is loaded.
About 159.6 MB of the selected adapter belongs to AdaLN targets and is released after
adapter-aware modulation precompute. About 620.2 MB remains with the DiT trunk.

## DiT tensor naming

```
blocks.N.adaln_proj.linear.{weight,bias}    AdaLN
blocks.N.attn.qkv_proj.weight               fused qkv (split order is a gotcha)
blocks.N.attn.{q_norm,k_norm}.weight        per-head RMSNorm, applied BEFORE rope
blocks.N.attn.out_proj.weight
blocks.N.mlp.{fc1,fc2}.weight               fc1 is fused gate+up (half order is a gotcha)
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

If the Turbo adapter is present, its low-rank AdaLN branch is evaluated by the same projection
call. The completed base-plus-adapter modulation is stored in the table, after which both the
quantized base projection and its BF16 LoRA tensors are released.

The checkpoint remains 34.8 GiB on disk, but active DiT residency falls from 34.756 GiB to about
21.2 GiB before the first sampling step. Ten steps use about 0.172 GiB of tables. This optimization
targets residency, not step time: AdaLN cost does not scale with sequence length and is negligible
beside attention. A curve-form pruned checkpoint could later reduce disk and load footprint, but it
is not required for the current memory budget or output path.
