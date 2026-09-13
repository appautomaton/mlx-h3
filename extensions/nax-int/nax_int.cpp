#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "mlx/backend/cpu/encoder.h"
#include "nax_int.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_nax_int {

namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to locate the extension binary.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

int checked_dim(int64_t value, const char* name) {
  if (value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::string("[nax_int.matmul] ") + name + " exceeds int32.");
  }
  return static_cast<int>(value);
}

int8_t unpack_int4(uint8_t packed, int column) {
  int value = (column & 1) ? (packed >> 4) : (packed & 0x0f);
  return static_cast<int8_t>(value >= 8 ? value - 16 : value);
}

} // namespace

mx::array matmul(
    const mx::array& x,
    const mx::array& weight,
    int bits,
    mx::StreamOrDevice stream) {
  if (bits != 4 && bits != 8) {
    throw std::invalid_argument("[nax_int.matmul] bits must be 4 or 8.");
  }
  if (x.ndim() != 2 || weight.ndim() != 2) {
    throw std::invalid_argument("[nax_int.matmul] inputs must be 2D.");
  }
  if (x.dtype() != mx::int8) {
    throw std::invalid_argument("[nax_int.matmul] x must have dtype int8.");
  }
  if ((bits == 8 && weight.dtype() != mx::int8) ||
      (bits == 4 && weight.dtype() != mx::uint8)) {
    throw std::invalid_argument(
        bits == 8
            ? "[nax_int.matmul] 8-bit weight must have dtype int8."
            : "[nax_int.matmul] packed 4-bit weight must have dtype uint8.");
  }

  int M = checked_dim(x.shape(0), "M");
  int K = checked_dim(x.shape(1), "K");
  int weight_k = checked_dim(weight.shape(0), "weight K");
  int packed_n = checked_dim(weight.shape(1), "weight N");
  int N = bits == 4 ? checked_dim(int64_t{packed_n} * 2, "N") : packed_n;
  if (K != weight_k) {
    throw std::invalid_argument(
        "[nax_int.matmul] x.shape[1] must equal weight.shape[0].");
  }
  if (M % 64 || N % 64 || K % 64) {
    throw std::invalid_argument(
        "[nax_int.matmul] M, N, and K must be divisible by 64.");
  }

  auto x_contiguous =
      x.flags().row_contiguous ? x : mx::contiguous(x, false, stream);
  auto weight_contiguous = weight.flags().row_contiguous
      ? weight
      : mx::contiguous(weight, false, stream);
  return mx::array(
      {M, N},
      mx::int32,
      std::make_shared<NaxIntMatmul>(mx::to_stream(stream), bits),
      {x_contiguous, weight_contiguous});
}

mx::array scaled_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    mx::StreamOrDevice stream) {
  if (x.ndim() != 2 || weight.ndim() != 2) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] x and weight must be 2D.");
  }
  if (x.dtype() != mx::int8 || weight.dtype() != mx::int8) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] x and weight must have dtype int8.");
  }
  if (x_scale.dtype() != mx::bfloat16 || weight_scale.dtype() != mx::bfloat16) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] scales must have dtype bfloat16.");
  }

  int M = checked_dim(x.shape(0), "M");
  int K = checked_dim(x.shape(1), "K");
  int weight_k = checked_dim(weight.shape(0), "weight K");
  int N = checked_dim(weight.shape(1), "N");
  if (K != weight_k) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] x.shape[1] must equal weight.shape[0].");
  }
  if (x_scale.size() != M || weight_scale.size() != N) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] scales must contain M and N elements.");
  }
  if (M % 64 || N % 64 || K % 64) {
    throw std::invalid_argument(
        "[nax_int.scaled_matmul] M, N, and K must be divisible by 64.");
  }

  auto make_contiguous = [&](const mx::array& value) {
    return value.flags().row_contiguous ? value
                                        : mx::contiguous(value, false, stream);
  };
  return mx::array(
      {M, N},
      mx::bfloat16,
      std::make_shared<NaxIntScaledMatmul>(mx::to_stream(stream)),
      {make_contiguous(x),
       make_contiguous(weight),
       make_contiguous(x_scale),
       make_contiguous(weight_scale)});
}

mx::array affine_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    const mx::array& x_sum,
    const mx::array& weight_center,
    mx::StreamOrDevice stream) {
  if (x.ndim() != 2 || weight.ndim() != 2) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] x and weight must be 2D.");
  }
  if (x.dtype() != mx::int8 || weight.dtype() != mx::int8) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] x and weight must have dtype int8.");
  }
  if (x_scale.dtype() != mx::bfloat16 || weight_scale.dtype() != mx::bfloat16 ||
      weight_center.dtype() != mx::bfloat16 || x_sum.dtype() != mx::int32) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] scales/center must be bfloat16 and x_sum int32.");
  }

  int M = checked_dim(x.shape(0), "M");
  int K = checked_dim(x.shape(1), "K");
  int weight_k = checked_dim(weight.shape(0), "weight K");
  int N = checked_dim(weight.shape(1), "N");
  if (K != weight_k) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] x.shape[1] must equal weight.shape[0].");
  }
  if (x_scale.size() != M || x_sum.size() != M || weight_scale.size() != N ||
      weight_center.size() != N) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] row values must contain M and column values N elements.");
  }
  if (M % 64 || N % 64 || K % 64) {
    throw std::invalid_argument(
        "[nax_int.affine_matmul] M, N, and K must be divisible by 64.");
  }

  auto make_contiguous = [&](const mx::array& value) {
    return value.flags().row_contiguous ? value
                                        : mx::contiguous(value, false, stream);
  };
  return mx::array(
      {M, N},
      mx::bfloat16,
      std::make_shared<NaxIntAffineMatmul>(mx::to_stream(stream)),
      {make_contiguous(x),
       make_contiguous(weight),
       make_contiguous(x_scale),
       make_contiguous(weight_scale),
       make_contiguous(x_sum),
       make_contiguous(weight_center)});
}

mx::array grouped_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    int group_size,
    mx::StreamOrDevice stream) {
  if (group_size != 64 && group_size != 256 && group_size != 448 &&
      group_size != 896) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] group_size must be 64, 256, 448, or 896.");
  }
  if (x.ndim() != 2 || weight.ndim() != 2) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] x and weight must be 2D.");
  }
  if (x.dtype() != mx::int8 || weight.dtype() != mx::int8) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] x and weight must have dtype int8.");
  }
  if (x_scale.dtype() != mx::bfloat16 || weight_scale.dtype() != mx::bfloat16) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] scales must have dtype bfloat16.");
  }

  int M = checked_dim(x.shape(0), "M");
  int K = checked_dim(x.shape(1), "K");
  int weight_k = checked_dim(weight.shape(0), "weight K");
  int N = checked_dim(weight.shape(1), "N");
  int groups = K / group_size;
  if (K != weight_k) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] x.shape[1] must equal weight.shape[0].");
  }
  if (x_scale.size() != M * groups || weight_scale.size() != groups * N) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] scale shapes do not match the selected group size.");
  }
  if (M % 64 || N % 64 || K % group_size) {
    throw std::invalid_argument(
        "[nax_int.grouped_matmul] M/N must be divisible by 64 and K by the group size.");
  }

  auto make_contiguous = [&](const mx::array& value) {
    return value.flags().row_contiguous ? value
                                        : mx::contiguous(value, false, stream);
  };
  return mx::array(
      {M, N},
      mx::bfloat16,
      std::make_shared<NaxIntGroup64Matmul>(mx::to_stream(stream), group_size),
      {make_contiguous(x),
       make_contiguous(weight),
       make_contiguous(x_scale),
       make_contiguous(weight_scale)});
}

std::vector<mx::array> grouped_quantize(
    const mx::array& x,
    int group_size,
    mx::StreamOrDevice stream) {
  if (group_size != 64 && group_size != 256 && group_size != 448 &&
      group_size != 896) {
    throw std::invalid_argument(
        "[nax_int.grouped_quantize] group_size must be 64, 256, 448, or 896.");
  }
  if (x.ndim() != 2 || x.dtype() != mx::bfloat16) {
    throw std::invalid_argument(
        "[nax_int.grouped_quantize] x must be a 2D bfloat16 array.");
  }
  int M = checked_dim(x.shape(0), "M");
  int K = checked_dim(x.shape(1), "K");
  if (K % group_size) {
    throw std::invalid_argument(
        "[nax_int.grouped_quantize] K must be divisible by group_size.");
  }
  auto x_contiguous =
      x.flags().row_contiguous ? x : mx::contiguous(x, false, stream);
  return mx::array::make_arrays(
      {{M, K}, {M, K / group_size}},
      {mx::int8, mx::bfloat16},
      std::make_shared<NaxGroupedQuantize>(mx::to_stream(stream), group_size),
      {x_contiguous});
}

void NaxIntMatmul::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(out.shape(1));
  int bits = bits_;
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(weight);
  encoder.set_output_array(out);
  encoder.dispatch([x_ptr = x.data<int8_t>(),
                    weight_ptr = weight.data<uint8_t>(),
                    out_ptr = out.data<int32_t>(),
                    M,
                    N,
                    K,
                    bits]() {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          int8_t w;
          if (bits == 8) {
            w = reinterpret_cast<const int8_t*>(weight_ptr)[k * N + n];
          } else {
            w = unpack_int4(weight_ptr[k * (N / 2) + n / 2], n);
          }
          sum += static_cast<int32_t>(x_ptr[m * K + k]) * w;
        }
        out_ptr[m * N + n] = sum;
      }
    }
  });
}

#ifdef _METAL_

void NaxIntMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  auto& out = outputs[0];
  auto& device = mx::metal::device(stream().device);

  if (!x.flags().row_contiguous || !weight.flags().row_contiguous) {
    throw std::runtime_error(
        "[nax_int.matmul] internal inputs are not contiguous.");
  }

  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(out.shape(1));

  auto library = device.get_library("mlx_nax_int", current_binary_dir());
  auto kernel =
      device.get_kernel(bits_ == 8 ? "nax_w8a8" : "nax_w4a8", library);
  auto& encoder = mx::metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weight, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(M, 3);
  encoder.set_bytes(N, 4);
  encoder.set_bytes(K, 5);

  MTL::Size group_dims(4 * kernel->threadExecutionWidth(), 1, 1);
  MTL::Size grid_dims(N / 64, M / 64, 1);
  encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void NaxIntMatmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.matmul] Metal is unavailable.");
}

#endif

std::vector<mx::array> NaxIntMatmul::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.matmul] JVP is not implemented.");
}

std::vector<mx::array> NaxIntMatmul::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.matmul] VJP is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> NaxIntMatmul::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.matmul] vmap is not implemented.");
}

bool NaxIntMatmul::is_equivalent(const mx::Primitive& other) const {
  return bits_ == static_cast<const NaxIntMatmul&>(other).bits_;
}

void NaxIntScaledMatmul::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(weight);
  encoder.set_input_array(x_scale);
  encoder.set_input_array(weight_scale);
  encoder.set_output_array(out);
  encoder.dispatch([x_ptr = x.data<int8_t>(),
                    weight_ptr = weight.data<int8_t>(),
                    x_scale_ptr = x_scale.data<mx::bfloat16_t>(),
                    weight_scale_ptr = weight_scale.data<mx::bfloat16_t>(),
                    out_ptr = out.data<mx::bfloat16_t>(),
                    M,
                    N,
                    K]() {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += static_cast<int32_t>(x_ptr[m * K + k]) * weight_ptr[k * N + n];
        }
        out_ptr[m * N + n] = mx::bfloat16_t(
            static_cast<float>(sum) * static_cast<float>(x_scale_ptr[m]) *
            static_cast<float>(weight_scale_ptr[n]));
      }
    }
  });
}

#ifdef _METAL_

void NaxIntScaledMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  auto& out = outputs[0];
  auto& device = mx::metal::device(stream().device);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));

  auto library = device.get_library("mlx_nax_int", current_binary_dir());
  auto kernel = device.get_kernel("nax_w8a8_scaled_bf16", library);
  auto& encoder = mx::metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weight, 1);
  encoder.set_input_array(x_scale, 2);
  encoder.set_input_array(weight_scale, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(M, 5);
  encoder.set_bytes(N, 6);
  encoder.set_bytes(K, 7);

  MTL::Size group_dims(4 * kernel->threadExecutionWidth(), 1, 1);
  MTL::Size grid_dims(N / 64, M / 64, 1);
  encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void NaxIntScaledMatmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.scaled_matmul] Metal is unavailable.");
}

#endif

std::vector<mx::array> NaxIntScaledMatmul::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.scaled_matmul] JVP is not implemented.");
}

std::vector<mx::array> NaxIntScaledMatmul::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.scaled_matmul] VJP is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> NaxIntScaledMatmul::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.scaled_matmul] vmap is not implemented.");
}

bool NaxIntScaledMatmul::is_equivalent(const mx::Primitive&) const {
  return true;
}

void NaxIntAffineMatmul::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  const auto& x_sum = inputs[4];
  const auto& weight_center = inputs[5];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(weight);
  encoder.set_input_array(x_scale);
  encoder.set_input_array(weight_scale);
  encoder.set_input_array(x_sum);
  encoder.set_input_array(weight_center);
  encoder.set_output_array(out);
  encoder.dispatch([x_ptr = x.data<int8_t>(),
                    weight_ptr = weight.data<int8_t>(),
                    x_scale_ptr = x_scale.data<mx::bfloat16_t>(),
                    weight_scale_ptr = weight_scale.data<mx::bfloat16_t>(),
                    x_sum_ptr = x_sum.data<int32_t>(),
                    weight_center_ptr = weight_center.data<mx::bfloat16_t>(),
                    out_ptr = out.data<mx::bfloat16_t>(),
                    M,
                    N,
                    K]() {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += static_cast<int32_t>(x_ptr[m * K + k]) * weight_ptr[k * N + n];
        }
        float value = static_cast<float>(weight_scale_ptr[n]) * sum +
            static_cast<float>(weight_center_ptr[n]) * x_sum_ptr[m];
        out_ptr[m * N + n] =
            mx::bfloat16_t(static_cast<float>(x_scale_ptr[m]) * value);
      }
    }
  });
}

#ifdef _METAL_

void NaxIntAffineMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  const auto& x_sum = inputs[4];
  const auto& weight_center = inputs[5];
  auto& out = outputs[0];
  auto& device = mx::metal::device(stream().device);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));
  auto library = device.get_library("mlx_nax_int", current_binary_dir());
  auto kernel = device.get_kernel("nax_w8a8_affine_bf16", library);
  auto& encoder = mx::metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weight, 1);
  encoder.set_input_array(x_scale, 2);
  encoder.set_input_array(weight_scale, 3);
  encoder.set_input_array(x_sum, 4);
  encoder.set_input_array(weight_center, 5);
  encoder.set_output_array(out, 6);
  encoder.set_bytes(M, 7);
  encoder.set_bytes(N, 8);
  encoder.set_bytes(K, 9);

  MTL::Size group_dims(4 * kernel->threadExecutionWidth(), 1, 1);
  MTL::Size grid_dims(N / 64, M / 64, 1);
  encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void NaxIntAffineMatmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.affine_matmul] Metal is unavailable.");
}

#endif

std::vector<mx::array> NaxIntAffineMatmul::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.affine_matmul] JVP is not implemented.");
}

std::vector<mx::array> NaxIntAffineMatmul::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.affine_matmul] VJP is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> NaxIntAffineMatmul::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.affine_matmul] vmap is not implemented.");
}

bool NaxIntAffineMatmul::is_equivalent(const mx::Primitive&) const {
  return true;
}

void NaxIntGroup64Matmul::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));
  int group_size = group_size_;
  int groups = K / group_size;
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(weight);
  encoder.set_input_array(x_scale);
  encoder.set_input_array(weight_scale);
  encoder.set_output_array(out);
  encoder.dispatch([x_ptr = x.data<int8_t>(),
                    weight_ptr = weight.data<int8_t>(),
                    x_scale_ptr = x_scale.data<mx::bfloat16_t>(),
                    weight_scale_ptr = weight_scale.data<mx::bfloat16_t>(),
                    out_ptr = out.data<mx::bfloat16_t>(),
                    M,
                    N,
                    K,
                    groups,
                    group_size]() {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float total = 0.0f;
        for (int group = 0; group < groups; ++group) {
          int32_t sum = 0;
          for (int inner = 0; inner < group_size; ++inner) {
            int k = group * group_size + inner;
            sum +=
                static_cast<int32_t>(x_ptr[m * K + k]) * weight_ptr[k * N + n];
          }
          total += static_cast<float>(sum) *
              static_cast<float>(x_scale_ptr[m * groups + group]) *
              static_cast<float>(weight_scale_ptr[group * N + n]);
        }
        out_ptr[m * N + n] = mx::bfloat16_t(total);
      }
    }
  });
}

#ifdef _METAL_

void NaxIntGroup64Matmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& weight = inputs[1];
  const auto& x_scale = inputs[2];
  const auto& weight_scale = inputs[3];
  auto& out = outputs[0];
  auto& device = mx::metal::device(stream().device);

  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int N = static_cast<int>(weight.shape(1));
  auto library = device.get_library("mlx_nax_int", current_binary_dir());
  const char* kernel_name = group_size_ == 64 ? "nax_w8a8_group64_bf16"
      : group_size_ == 256                    ? "nax_w8a8_group256_bf16"
      : group_size_ == 448                    ? "nax_w8a8_group448_bf16"
                                              : "nax_w8a8_group896_bf16";
  int simdgroups = group_size_ == 896 ? 2 : 4;
  auto kernel = device.get_kernel(kernel_name, library);
  auto& encoder = mx::metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weight, 1);
  encoder.set_input_array(x_scale, 2);
  encoder.set_input_array(weight_scale, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(M, 5);
  encoder.set_bytes(N, 6);
  encoder.set_bytes(K, 7);

  MTL::Size group_dims(simdgroups * kernel->threadExecutionWidth(), 1, 1);
  MTL::Size grid_dims(N / 64, M / 64, 1);
  encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void NaxIntGroup64Matmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.grouped_matmul] Metal is unavailable.");
}

#endif

std::vector<mx::array> NaxIntGroup64Matmul::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.grouped_matmul] JVP is not implemented.");
}

std::vector<mx::array> NaxIntGroup64Matmul::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.grouped_matmul] VJP is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> NaxIntGroup64Matmul::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[nax_int.grouped_matmul] vmap is not implemented.");
}

bool NaxIntGroup64Matmul::is_equivalent(const mx::Primitive& other) const {
  return group_size_ ==
      static_cast<const NaxIntGroup64Matmul&>(other).group_size_;
}

void NaxGroupedQuantize::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  auto& quantized = outputs[0];
  auto& scales = outputs[1];
  quantized.set_data(mx::allocator::malloc(quantized.nbytes()));
  scales.set_data(mx::allocator::malloc(scales.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int group_size = group_size_;
  int groups = K / group_size;
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_output_array(quantized);
  encoder.set_output_array(scales);
  encoder.dispatch([x_ptr = x.data<mx::bfloat16_t>(),
                    quantized_ptr = quantized.data<int8_t>(),
                    scales_ptr = scales.data<mx::bfloat16_t>(),
                    M,
                    K,
                    groups,
                    group_size]() {
    for (int m = 0; m < M; ++m) {
      for (int group = 0; group < groups; ++group) {
        int offset = m * K + group * group_size;
        float maximum = 0.0f;
        for (int i = 0; i < group_size; ++i) {
          maximum = std::max(
              maximum, std::abs(static_cast<float>(x_ptr[offset + i])));
        }
        auto stored_scale = mx::bfloat16_t(
            std::max(maximum / 127.0f, 1e-8f));
        scales_ptr[m * groups + group] = stored_scale;
        float scale = static_cast<float>(stored_scale);
        for (int i = 0; i < group_size; ++i) {
          auto normalized = mx::bfloat16_t(
              static_cast<float>(x_ptr[offset + i]) / scale);
          float value = std::round(static_cast<float>(normalized));
          quantized_ptr[offset + i] =
              static_cast<int8_t>(std::clamp(value, -127.0f, 127.0f));
        }
      }
    }
  });
}

#ifdef _METAL_

void NaxGroupedQuantize::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  auto& quantized = outputs[0];
  auto& scales = outputs[1];
  auto& device = mx::metal::device(stream().device);
  quantized.set_data(mx::allocator::malloc(quantized.nbytes()));
  scales.set_data(mx::allocator::malloc(scales.nbytes()));

  int M = static_cast<int>(x.shape(0));
  int K = static_cast<int>(x.shape(1));
  int groups = K / group_size_;
  const char* kernel_name = group_size_ == 64 ? "nax_quantize_group64_bf16"
      : group_size_ == 256                    ? "nax_quantize_group256_bf16"
      : group_size_ == 448                    ? "nax_quantize_group448_bf16"
                                              : "nax_quantize_group896_bf16";
  auto library = device.get_library("mlx_nax_int", current_binary_dir());
  auto kernel = device.get_kernel(kernel_name, library);
  auto& encoder = mx::metal::get_command_encoder(stream());
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_output_array(quantized, 1);
  encoder.set_output_array(scales, 2);
  encoder.set_bytes(K, 3);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims(groups, M, 1);
  encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else

void NaxGroupedQuantize::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[nax_int.grouped_quantize] Metal is unavailable.");
}

#endif

std::vector<mx::array> NaxGroupedQuantize::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "[nax_int.grouped_quantize] JVP is not implemented.");
}

std::vector<mx::array> NaxGroupedQuantize::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[nax_int.grouped_quantize] VJP is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> NaxGroupedQuantize::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "[nax_int.grouped_quantize] vmap is not implemented.");
}

bool NaxGroupedQuantize::is_equivalent(const mx::Primitive& other) const {
  return group_size_ ==
      static_cast<const NaxGroupedQuantize&>(other).group_size_;
}

} // namespace mlx_nax_int
