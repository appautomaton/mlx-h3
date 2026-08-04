# mlx-h3 / docs

Research for porting MiniMax-H3 to Apple Silicon. Target machine: MacBook Pro / M5 Max / 128 GB.

Researched 2026-08-04, one day after the H3 weights dropped.

| Doc | Contents |
|---|---|
| [00-minimax-h3-overview.md](00-minimax-h3-overview.md) | What H3 is, the three-stage system, architecture, license |
| [01-weights-and-starting-point.md](01-weights-and-starting-point.md) | **Which weights, from where, in what order** |
| [02-mlx-port-notes.md](02-mlx-port-notes.md) | Gotchas, performance ceiling, validation strategy, sequencing |

## Three-sentence version

1. **H3 is a three-stage system and only the middle stage is open.** Context-IR (prompt
   structuring) in front and Regenerate-2K (768p→2K) behind are both API-only, so local
   inference tops out at 768p.

2. **One weight source: `Comfy-Org/MiniMax-H3` bf16, 123.6 GB, quantized in-house.**
   ComfyUI's own int8 is `int8_tensorwise` + a 256-group rotation bound to CUDA kernels —
   not usable from MLX without reimplementing the rotation.

3. **There is no official DiT implementation.** MiniMax shipped VAE source only; every DiT
   implementation is an independent rewrite from `config.json`. Use ComfyUI's
   `comfy/ldm/minimax/model.py` as the porting baseline.

## Current state

- Weights downloaded and integrity-verified: `../weights/bf16/`, 115 GiB, 4 files.
- Environment: uv, Python 3.13, single runtime dependency (`mlx`).
- Next: Layer 0 — packing layout, frame grid, position grids, sigma schedule,
  validated against `minimax_h3_layout.json`. Weightless.
