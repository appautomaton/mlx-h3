# MLX porting notes

## There is no official DiT implementation

**MiniMax ships VAE source only. No DiT source.**

```
FL2VA/video_vae/*.py     24 .py files -- klvae / vae_vit / vae_cnn / attention / flash / parallel
FL2VA/audio_vae/dac_*.py              -- DAC + BigVGAN lineage
FL2VA/transformer/                    -- config.json and weights, no code
```

Every DiT implementation is therefore an independent rewrite from `config.json`. Three exist:

| Implementation | File | Character |
|---|---|---|
| ComfyUI | `comfy/ldm/minimax/model.py` (33 KB) | **de-facto spec, most compact and readable** |
| diffusers | `models/transformers/transformer_minimax_h3.py` | best documented |
| SGLang | `runtime/models/dits/minimax_h3.py` | production serving, only one with multi-GPU |

Use ComfyUI's as the porting baseline.

## Hard constraint: no end-to-end torch reference on a Mac

ComfyUI's `model.py` calls comfy_kitchen CUDA kernels (fused RMSNorm+rope, fused SwiGLU, its
attention entry point). Those cannot run here.

The upstream fixture generator states the consequence plainly:

> the math below is a TRANSCRIPTION of `comfy/ldm/minimax/model.py` rather than the reference
> executing. A green test therefore proves the port agrees with an **independently written
> implementation of the same spec** — it catches the MLX-side slips this port is actually prone
> to but it **cannot catch a misreading shared by both implementations**.

### Validation tiers, strongest first

| Tier | Method | Strength |
|---|---|---|
| layout | `minimax_h3_layout.json` — **actually executes** the ComfyUI reference, weightless | golden |
| DiT block | `minimax_h3_dit.safetensors` — f32 CPU parity against an independent transcription | catches port slips, not shared misreadings |
| end-to-end | live run, eyeball or compare against a known-good implementation | weakest |

Both fixtures are pre-generated and committed upstream, so **no torch is required** — they are
plain data, readable with `mx.load()` and `json`.

`minimax_h3_dit.safetensors` (26 tensors, toy dimensions: hidden 256, 32 tokens) carries a full
single-block trace: `x.h_in` → `x.attn_in` → `x.attn_out` → `x.mlp_out` → `x.h_out`, plus
`x.rope_cos` / `x.rope_sin` / `x.positions` / `x.t_emb` / `x.t_vals` / `x.runs` and the block's
own weights. Feed `h_in`, assert `h_out`.

`minimax_h3_layout.json` carries `constants`, `frame_grid`, `temporal_shape`, `adapt_canvas`,
`sigma_schedule`, `frame_position_grid`, `video_t_grid`, `rope_freqs`, `packed_layout`.

## Seven ways to be silently wrong

The first four are what the block fixture exists to catch:

1. **AdaLN reshape/chunk order** — modality stride, expand order
2. **qkv split; per-head RMSNorm applied BEFORE rope; partial split-half rope with the top 32
   of 128 dims left unrotated**
3. **SwiGLU gate/up half order inside the fused fc1**
4. **cos before sin in the timestep embedding**

Two more from the diffusers documentation:

5. **One generator, three draws**, in order: conditioning noise → video noise → audio noise.
   Passing `latents` / `audio_latents` replaces the corresponding draw. Wrong order means seeds
   do not reproduce.
6. The older diffusers integration defines **`num_inference_steps` as sigma grid points including
   terminal 0**, so it drives one fewer model evaluation than its value suggests. The current
   runtime instead follows the released Comfy workflow: 20 `simple` steps mean 20 model calls and
   21 sigma points, with `res_multistep` rather than Euler.
7. **Do not copy the Turbo custom sampler's audio slope division into this runtime.** Comfy's
   sampler receives a packed derivative whose audio component is already mapped onto the video
   sigma grid. `MiniMaxH3.__call__` instead returns raw audio velocity. Paired Euler must advance
   that raw value directly from `sigma_audio` to `sigma_audio_next`; dividing by the slope again
   applies the conversion twice.

## Performance: the DiT is already at the compute roofline

Timings anywhere in this repo are preliminary and machine-specific; do not treat them as targets.
What is durable is the shape of the cost, which follows from the architecture rather than from any
measurement:

**Attention dominates and grows as O(S²).** Everything else in a block is linear in S. So sequence
length — canvas × frames — is the only lever with real leverage on wall clock. `height`/`width`
need only be multiples of 32; use small canvases while developing.

**Do not write a custom attention kernel.** MLX's full-attention kernel is already near roofline at
these shapes. The whole line item a hand-written kernel could win is a couple of percent of a step.

**AdaLN is precomputed for residency, not speed.** Its cost does not scale with S at all — one
`[t_dim -> 6*hidden*3]` matmul per block against 2–4 rows. The runtime materializes the request's
exact schedule and releases roughly 13 GiB of AdaLN weights before denoising. See `weights.md`.

**Default MLX affine quantization buys footprint, not speed.** Its activations remain BF16, so the
8-bit checkpoint is chosen for residency rather than lower-precision matrix throughput.

An explicit M5-only development path in `mlx_h3.nax` instead quantizes activations and dispatches
native W8A8 TensorOps through a local MLX extension. `dev/one_step.py --nax-group-size ...` is the
validation entry point. It converts only the 200 attention/MLP trunk linears after AdaLN precompute;
the default runtime and public CLI remain W8A16. Activation max reduction, BF16 scaling, rounding,
and int8 casting are fused into one Metal kernel before the group-scaled integer matrix multiply.

On an M5 Max at the `dev` shape (56 frames, 864x480, text length 512, sequence length 7,583), three
fixed-seed one-step runs measured a 23.01 s median for group 896. The corresponding default W8A16
median was 35.30 s, giving a 1.53x speedup and 34.8% lower step time. Fusing activation
quantization first improved the earlier W8A8 median from 31.62 s to 25.43 s. Staging each output
tile's activation and weight scales in threadgroup memory then reduced the median to 23.01 s. Both
optimizations produce bit-identical one-step W8A8 video and audio tensors. Against W8A16, the
one-step NRMSE was 1.51% for video and 1.20% for audio. These measurements establish one-step
numerical parity and performance.

A fixed-seed, 20-step T2VA A/B at the same canvas and frame count, with 23 prompt tokens and a
7,094-token packed sequence, completed end to end in 7.3 minutes for W8A8 versus 9.3 minutes for
W8A16. DiT execution was 391.7 s (19.59 s/step) versus 520.0 s (26.00 s/step), a 1.33x speedup and
24.7% lower DiT time. W8A8 DiT active memory was 19.2 GiB versus 21.4 GiB. The common text-encoder
phase set the overall 27.0 GiB peak in both runs. Video decode took roughly 29 s and audio decode
roughly 0.6 s, so neither is the next material bottleneck at this shape.

The encoded 20-step outputs had video SSIM 0.9588 and PSNR 33.83 dB. Decoded audio had cosine
similarity 0.9904 and mean absolute error 0.000787. First, middle, and final frame inspection found
small cloud and wave-detail differences without structural failure. This is one complete quality
sample, not a generally accepted perceptual baseline; the sequential run order may also include
thermal-state effects in the timing comparison.

The largest wall-clock levers remain **fewer forwards** (fewer steps, TeaCache-style step cache) and
**less math per forward** (sparse attention — still withheld upstream; MiniMax says it is coming).
Native lower-precision trunk GEMM is a smaller, hardware-specific lever now covered by the NAX
experiment above.

The community Turbo LoRA realizes the first option. Its BF16 low-rank branch stays separate from
the MLX affine-8-bit base, and its paired-schedule Euler path reduces the requested model calls to
four through eight, defaulting to six, without changing phase residency. Pure-MLX adapter loading
and bounded end-to-end execution are validated without using a Torch reference run.

Quantization mechanics (dtype filtering, lookup tables, lazy reads) live in
`weights.md`, not here.
