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
the default runtime and public CLI remain W8A16. One-step numerical parity and performance are
measured, but that does not establish 20-step perceptual quality.

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
