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

## Six ways to be silently wrong

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

**Quantization buys footprint, not speed.** The workload is compute-bound, so 8-bit is chosen for
residency, not throughput.

Wall clock only moves via **fewer forwards** (fewer steps, TeaCache-style step cache) or **less
math per forward** (sparse attention — still withheld upstream; MiniMax says it is coming).

Quantization mechanics (dtype filtering, lookup tables, lazy reads) live in
`weights.md`, not here.
