# Agent Guide

This file applies to the entire repository. Keep it short: it defines durable
constraints and routes deeper reading to the relevant document.

## Start Here

`mlx-h3` is a pure-MLX MiniMax-H3 inference runtime for synchronized video and
stereo audio generation on Apple silicon.

Before changing code:

1. Read this file.
2. Read [README.md](README.md) for the public contract and current feature scope.
3. Open only the task-relevant document from [docs/README.md](docs/README.md).

## Non-Negotiable Constraints

- Runtime code is pure MLX. Do not add or execute PyTorch, CUDA, or Triton paths.
- Code, tests, comments, commit messages, and documentation must be written in
  English.
- Never commit model weights, checkpoints, tokenizer assets, generated media,
  private input text, prompts, cases, logs, or local agent state.
- Preserve phase-scoped model residency: text encoder -> release -> DiT ->
  release -> Video VAE -> release -> Audio VAE -> release -> mux.
- Inference loads the 8-bit DiT and text encoder. Dense DiT/text weights are only
  local requantization inputs; the Video and Audio VAEs remain dense runtime inputs.
- The default active-memory budget is 70 GiB. Swap activity is a hard failure.
  Do not weaken, bypass, or suppress the memory guard to make a run pass.
- Do not commit real generation prompts or generated examples, including in tests,
  docs, fixtures, issue templates, or release notes.

Run `python dev/check_public_tree.py` before exposing repository contents. The
pre-commit hook applies the same policy to staged files.

## Architecture Invariants

- H3 jointly denoises audio and video in one packed sequence and one DiT call per
  sampling step.
- The quality baseline is 20 `simple` schedule steps with `res_multistep`.
- Video and audio use different sigma mappings, but advance together on the video
  sigma grid.
- Guidance is distilled into the weights; there is no classifier-free guidance
  branch or negative-prompt pass.
- Canvas axes are multiples of 32, total area is at most `768 * 1344`, and frame
  counts follow the Video VAE's `17n + 5` alignment.

If a change contradicts one of these points, stop and verify the model contract
before implementing it.

## Working Loop

1. Inspect the relevant implementation and documentation before editing.
2. Make the smallest coherent change; avoid speculative abstractions.
3. Validate in proportion to risk:

   ```sh
   uv run ruff check .
   uv run pytest -q
   python dev/check_public_tree.py
   ```

4. For packaging or release changes, also run:

   ```sh
   uv build --no-sources
   ```

Do not start expensive checkpoint or full-generation runs unless the task requires
them. Never use a PyTorch run as local validation.

## Documentation Router

- [README.md](README.md): installation, CLI, supported features, local model layout,
  and user-facing memory behavior.
- [docs/00-minimax-h3-overview.md](docs/00-minimax-h3-overview.md): model stages,
  architecture, conditioning variants, and output constraints.
- [docs/01-weights-and-starting-point.md](docs/01-weights-and-starting-point.md):
  checkpoint selection, quantization, tensor naming, and residency decisions.
- [docs/02-mlx-port-notes.md](docs/02-mlx-port-notes.md): validation strength,
  numerical pitfalls, sampler semantics, and performance boundaries.
- [src/mlx_h3/pipeline.py](src/mlx_h3/pipeline.py): phase orchestration and the
  end-to-end runtime contract.
- [src/mlx_h3/memory.py](src/mlx_h3/memory.py): memory and swap safety policy.
- [src/mlx_h3/sampler.py](src/mlx_h3/sampler.py): scheduler and solver behavior.
- [.github/workflows/workflow.yml](.github/workflows/workflow.yml) and
  [dev/check_release.py](dev/check_release.py): PyPI release automation and metadata
  validation.

Put detailed or evolving knowledge in the appropriate document above and link to it
from here only when an agent must discover it early. Do not duplicate long explanations
in this file.

## Releases

- Distribution and CLI name: `mlx-h3`; Python import package: `mlx_h3`.
- The version in `pyproject.toml`, Git tag `v{version}`, and GitHub Release type must
  agree. The workflow rejects mismatches.
- PEP 440 alpha, beta, release-candidate, and development versions must be GitHub
  pre-releases. Final versions must be normal releases.
- PyPI versions are immutable. Never reuse or overwrite a published version.
- Publishing is an external release action; perform it only when explicitly requested.
