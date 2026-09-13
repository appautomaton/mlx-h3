# Local experimental native integer NAX extension

This experimental MLX extension exposes the integer TensorOps available on
M5-class Apple GPUs:

- `int8 x int8 -> int32` (`bits=8`)
- `int8 x packed int4 -> int32` (`bits=4`)

The raw matrix operations require two-dimensional, row-contiguous inputs whose
logical `M`, `N`, and `K` dimensions are multiples of 64. The 4-bit weights are
packed along the output-column dimension, with the even column in the low nibble
and the odd column in the high nibble.

The extension also contains experimental fused operations for realistic W8A8
inference:

- row/column-scaled and affine-corrected W8A8 matrix multiplication;
- group-scaled W8A8 matrix multiplication with BF16 output; and
- fused BF16-to-int8 activation quantization for K-axis groups of 64, 256, 448,
  or 896 elements.

`grouped_quantize` reproduces the BF16 scale and intermediate rounding used by
the corresponding MLX expression. `grouped_matmul` stages each 64-row activation
scale tile and 64-column weight scale tile in threadgroup memory so every scale
is fetched once per group rather than once per output element. These APIs are
prototypes, not part of MLX's public API. This repository is a local development
dependency of `mlx-h3`, built from `extensions/nax-int` through the optional
`nax` extra. It is not published to PyPI, so the `--nax-group-size` option is
reachable only from a repository checkout.

The group-896 kernel uses two SIMD groups per 64-by-64 output tile. At the H3
shapes this leaves more independent threadgroups available to the GPU than the
four-group configuration while producing bit-identical block outputs.

On an M5 Max at the H3 development shape (7,583 rows), three fixed-seed
end-to-end DiT step measurements were 17.27, 20.76, and 15.93 seconds, for a
17.27-second median. The previous four-SIMD-group kernel measured a 23.01-second
median under the same setup, so the two-group configuration reduced step time
by 24.9% (1.33x). Saved video and audio velocities were bit-identical to the
four-group result.

## Build

The build requires macOS 26.4 or newer, an M5-class GPU, Xcode 26, and the Metal
Toolchain component.

This directory is part of the `mlx-h3` repository and is declared there as the
optional `nax` extra, so it builds against whatever MLX the project lockfile
pins. Build it from the repository root:

```sh
uv sync --extra nax
```

A plain `uv sync` removes it again and leaves a pure-MLX environment that needs
no compiler. Pass `--require-nax` to pytest so the W8A8 tests fail rather than
skip when the extension is absent.

The supported MLX version is pinned exactly, not as a range, because
`nanobind_add_module(... NB_DOMAIN mlx)` shares MLX's nanobind type registry:
0.32.0 is verified, 0.32.2 is verified broken, 0.32.1 is untested. Rebuild and
re-run `tests/test_nax.py` before widening it.

## Test

```sh
python test.py
```

## Benchmark

```sh
python benchmark.py
```

The benchmark compares raw, already-quantized integer matrix multiplication
against `mx.matmul` with float16 inputs. It intentionally excludes activation
quantization and output rescaling; application benchmarks should include both.
