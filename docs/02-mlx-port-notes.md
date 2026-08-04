# MLX porting notes

## There is no official DiT implementation

After pulling all 158 non-weight files from `MiniMaxAI/MiniMax-H3`:

**MiniMax shipped VAE source only. No DiT source.**

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

Measured upstream by in-run ablation ladder (864x480 / 73 frames / 8-bit, stock qmm baseline
41.3 s/step):

| Component | s/step | Share |
|---|---|---|
| attention total (qkv + SDPA + out) | 22.4 | 54% |
| — of which SDPA | 10.1 | 24% |
| MLP (fc1 + SwiGLU + fc2) | 18.6 | 45% |
| **AdaLN** | **~0** | **0%** |

Linear GEMMs measured ~13.6 TFLOPS effective, SDPA 13.3 TFLOPS.

### Three settled conclusions

**1. Quantization buys footprint, not speed.** The workload is compute-bound at roughly
**192,000 FLOPs per weight byte**. bf16 41–48 s/step and jittery; 8-bit 42; 8-bit with wide-M
dq-gemm 36.6, output byte-identical at u8.

**2. Do not write a custom attention kernel.** One SDPA at `[1, 56, 9266, 128]` is ~2.5 TFLOP and
microbenches at **186 ms = 13.3 TFLOPS**. A step has fifty of them: 50 x 186 ms = 9.3 s, matching
the ladder's measured 10.1 s. MLX's full-attention kernel is near-roofline at this shape; a custom
kernel's ceiling is ~2% of the step.

**3. Never fuse AdaLN — precompute it.** Measured at 0% of step time. Roughly 13B of the 33B sits
in AdaLN branches and is precomputable into a table at load. `adaln_t_table` is absent from the
bf16 checkpoint, so the table must be built by us at load time.

### What is left

Wall clock only moves via **fewer forwards** (fewer steps, TeaCache-style step cache) or **less
math per forward** (sparse attention — still withheld upstream; MiniMax says it is coming in a
future update).

## Baseline numbers (M4 Max / 128 GB)

Our target is M5 Max / 128 GB, same class or faster:

| Config | Weights | Wall clock |
|---|---|---|
| 256x256, 56 frames, 30 steps | bf16 | ~4 min |
| 864x480, 73 frames, 30 steps | 8-bit | ~22 min |

Cold-start breakdown: weight load ~36 s cold, TE ~15 s cold / ~1 s warm, VAE decode ~1–2 s,
audio ~0.05 s. Everything else is the step loop.

For scale: SGLang on 4xH200 does 1344x768 / 124 frames / 50 steps in 75.1 s.

### Canvas is the biggest lever

`960x544` runs about **2.3x faster per step** than the trained `1344x768`. `height`/`width` only
need to be multiples of 32. Use small canvases throughout development.

## Quantization mechanics

**Filter by dtype, not by name.** The checkpoint marks its own precision-sensitive tensors as F32
(13 of them). A name whitelist copied from the diffusers docs will not match the Comfy-Org repack's
naming.

**Lookup tables are the exception shape cannot decide.** A shape-based "is this a linear?" test
packs `embed_tokens.weight` and `visual.pos_embed.weight`, which are read with `take_axis`; the
gather then returns packed uint32 garbage. Keep gathered embedding tables dense explicitly.

**Read lazily.** Convert tensor by tensor so the 62 GB DiT never lands in RAM whole; peak should be
the size of the output being accumulated.

## Sequencing

1. **Layout** (0 GB) — packing, frame grid, position grids, sigma schedule, validated against
   `minimax_h3_layout.json`. Weightless, and the only tier with a true executed reference.
2. **VAEs** (5.5 GiB) — independently verifiable; get encode/decode working.
3. **DiT block parity** — f32 CPU against `minimax_h3_dit.safetensors`, watching the six gotchas.
4. **End-to-end live run** — small canvas first (256x256 / 56 frames / 30 steps).
5. **Quantization** — bf16 → MLX affine group-64; cross-check output sizes against 35.2 / 28.2 GB.
6. **Text encoder** — until then, feed a dumped fixed prompt embedding.
7. **Ref2VA packing** — last, no DiT changes required.
