#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "nax_int.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Experimental native integer NAX operations for MLX";
  m.def(
      "matmul",
      &mlx_nax_int::matmul,
      "x"_a,
      "weight"_a,
      "bits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multiply integer matrices with native M5 NAX TensorOps.

        ``bits=8`` computes ``int8 x int8 -> int32``. ``weight`` has shape
        ``[K, N]``.

        ``bits=4`` computes ``int8 x packed-int4 -> int32``. ``weight`` has
        dtype ``uint8`` and shape ``[K, N // 2]``. The even output column is
        stored in the low nibble.

        This experimental operation requires 2D row-contiguous inputs and
        logical M, N, and K dimensions divisible by 64.
      )");
  m.def(
      "scaled_matmul",
      &mlx_nax_int::scaled_matmul,
      "x"_a,
      "weight"_a,
      "x_scale"_a,
      "weight_scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multiply int8 matrices and fuse row/column scaling into a bfloat16
        output. Scales must be bfloat16 and contain M and N elements.
      )");
  m.def(
      "affine_matmul",
      &mlx_nax_int::affine_matmul,
      "x"_a,
      "weight"_a,
      "x_scale"_a,
      "weight_scale"_a,
      "x_sum"_a,
      "weight_center"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multiply int8 matrices and fuse a per-output weight center correction
        into a bfloat16 output.
      )");
  m.def(
      "grouped_matmul",
      &mlx_nax_int::grouped_matmul,
      "x"_a,
      "weight"_a,
      "x_scale"_a,
      "weight_scale"_a,
      "group_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multiply int8 matrices with independent symmetric scales for each K
        group and return bfloat16 output.
      )");
  m.def(
      "grouped_quantize",
      &mlx_nax_int::grouped_quantize,
      "x"_a,
      "group_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Symmetrically quantize each K-axis group of a 2D bfloat16 matrix and
        return the int8 values plus bfloat16 scales.
      )");
}
