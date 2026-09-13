#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

template <typename BType, typename BPointerType>
inline void nax_int_matmul(
    device int8_t* x,
    device BPointerType* weight,
    device int32_t* out,
    int M,
    int N,
    int K,
    uint2 tgid) {
  dextents<int, 2> x_extents(K, M);
  dextents<int, 2> weight_extents(N, K);
  dextents<int, 2> out_extents(N, M);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> X(x, x_extents);
  tensor<device BType, dextents<int, 2>, tensor_inline> W(
      weight, weight_extents);
  tensor<device int32_t, dextents<int, 2>, tensor_inline> Y(out, out_extents);

  auto x_tile = X.template slice<dynamic_extent, 64>(0, tgid.y * 64);
  auto w_tile = W.template slice<64, dynamic_extent>(tgid.x * 64, 0);
  auto y_tile = Y.template slice<64, 64>(tgid.x * 64, tgid.y * 64);

  constexpr auto descriptor =
      matmul2d_descriptor(64, 64, dynamic_length_v<int>);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  operation.run(x_tile, w_tile, y_tile);
}

kernel void nax_w8a8(
    device int8_t* x [[buffer(0)]],
    device int8_t* weight [[buffer(1)]],
    device int32_t* out [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  nax_int_matmul<int8_t, int8_t>(x, weight, out, M, N, K, tgid);
}

kernel void nax_w4a8(
    device int8_t* x [[buffer(0)]],
    device uint8_t* weight [[buffer(1)]],
    device int32_t* out [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  nax_int_matmul<int4b_format, uint8_t>(x, weight, out, M, N, K, tgid);
}

kernel void nax_w8a8_scaled_bf16(
    device int8_t* x [[buffer(0)]],
    device int8_t* weight [[buffer(1)]],
    device bfloat* x_scale [[buffer(2)]],
    device bfloat* weight_scale [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  dextents<int, 2> x_extents(K, M);
  dextents<int, 2> weight_extents(N, K);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> X(x, x_extents);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> W(
      weight, weight_extents);

  auto x_tile = X.template slice<dynamic_extent, 64>(0, tgid.y * 64);
  auto w_tile = W.template slice<64, dynamic_extent>(tgid.x * 64, 0);

  constexpr auto descriptor =
      matmul2d_descriptor(64, 64, dynamic_length_v<int>);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accum = operation.template get_destination_cooperative_tensor<
      decltype(x_tile),
      decltype(w_tile),
      int32_t>();

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      accum[i] = 0;
    }
  }
  operation.run(x_tile, w_tile, accum);

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto index = accum.get_multidimensional_index(i);
      int column = tgid.x * 64 + index[0];
      int row = tgid.y * 64 + index[1];
      out[row * N + column] = bfloat(
          float(accum[i]) * float(x_scale[row]) * float(weight_scale[column]));
    }
  }
}

kernel void nax_w8a8_affine_bf16(
    device int8_t* x [[buffer(0)]],
    device int8_t* weight [[buffer(1)]],
    device bfloat* x_scale [[buffer(2)]],
    device bfloat* weight_scale [[buffer(3)]],
    device int32_t* x_sum [[buffer(4)]],
    device bfloat* weight_center [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant int& M [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant int& K [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  dextents<int, 2> x_extents(K, M);
  dextents<int, 2> weight_extents(N, K);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> X(x, x_extents);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> W(
      weight, weight_extents);

  auto x_tile = X.template slice<dynamic_extent, 64>(0, tgid.y * 64);
  auto w_tile = W.template slice<64, dynamic_extent>(tgid.x * 64, 0);

  constexpr auto descriptor =
      matmul2d_descriptor(64, 64, dynamic_length_v<int>);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accum = operation.template get_destination_cooperative_tensor<
      decltype(x_tile),
      decltype(w_tile),
      int32_t>();

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      accum[i] = 0;
    }
  }
  operation.run(x_tile, w_tile, accum);

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto index = accum.get_multidimensional_index(i);
      int column = tgid.x * 64 + index[0];
      int row = tgid.y * 64 + index[1];
      float value = float(weight_scale[column]) * float(accum[i]) +
          float(weight_center[column]) * float(x_sum[row]);
      out[row * N + column] = bfloat(float(x_scale[row]) * value);
    }
  }
}

template <int GROUP_SIZE, int SIMD_GROUPS = 4>
inline void nax_w8a8_grouped(
    device int8_t* x,
    device int8_t* weight,
    device bfloat* x_scale,
    device bfloat* weight_scale,
    device bfloat* out,
    threadgroup bfloat* x_scale_tile,
    threadgroup bfloat* weight_scale_tile,
    int M,
    int N,
    int K,
    uint2 tgid,
    uint tid) {
  dextents<int, 2> x_extents(K, M);
  dextents<int, 2> weight_extents(N, K);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> X(x, x_extents);
  tensor<device int8_t, dextents<int, 2>, tensor_inline> W(
      weight, weight_extents);

  constexpr auto descriptor = matmul2d_descriptor(64, 64, GROUP_SIZE);
  matmul2d<descriptor, execution_simdgroups<SIMD_GROUPS>> operation;
  auto first_x = X.template slice<GROUP_SIZE, 64>(0, tgid.y * 64);
  auto first_w = W.template slice<64, GROUP_SIZE>(tgid.x * 64, 0);
  auto accum = operation.template get_destination_cooperative_tensor<
      decltype(first_x),
      decltype(first_w),
      int32_t>();
  metal::array<float, 128 / SIMD_GROUPS> total;

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 128 / SIMD_GROUPS; ++i) {
    total[i] = 0.0f;
  }

  int groups = K / GROUP_SIZE;
  for (int group = 0; group < groups; ++group) {
    if (tid < 64) {
      x_scale_tile[tid] =
          x_scale[(tgid.y * 64 + tid) * groups + group];
      weight_scale_tile[tid] =
          weight_scale[group * N + tgid.x * 64 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto x_tile =
        X.template slice<GROUP_SIZE, 64>(group * GROUP_SIZE, tgid.y * 64);
    auto w_tile =
        W.template slice<64, GROUP_SIZE>(tgid.x * 64, group * GROUP_SIZE);
    operation.run(x_tile, w_tile, accum);

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
      if (accum.is_valid_element(i)) {
        auto index = accum.get_multidimensional_index(i);
        total[i] += float(accum[i]) * float(x_scale_tile[index[1]]) *
            float(weight_scale_tile[index[0]]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto index = accum.get_multidimensional_index(i);
      int column = tgid.x * 64 + index[0];
      int row = tgid.y * 64 + index[1];
      out[row * N + column] = bfloat(total[i]);
    }
  }
}

#define instantiate_grouped_kernel(name, group_size, simdgroups) \
  kernel void name(                                               \
      device int8_t* x [[buffer(0)]],                             \
      device int8_t* weight [[buffer(1)]],                        \
      device bfloat* x_scale [[buffer(2)]],                       \
      device bfloat* weight_scale [[buffer(3)]],                  \
      device bfloat* out [[buffer(4)]],                           \
      constant int& M [[buffer(5)]],                              \
      constant int& N [[buffer(6)]],                              \
      constant int& K [[buffer(7)]],                              \
      uint2 tgid [[threadgroup_position_in_grid]],                \
      uint tid [[thread_index_in_threadgroup]]) {                 \
    threadgroup bfloat x_scale_tile[64];                          \
    threadgroup bfloat weight_scale_tile[64];                     \
    nax_w8a8_grouped<group_size, simdgroups>(                     \
        x,                                                        \
        weight,                                                   \
        x_scale,                                                  \
        weight_scale,                                             \
        out,                                                      \
        x_scale_tile,                                             \
        weight_scale_tile,                                        \
        M,                                                        \
        N,                                                        \
        K,                                                        \
        tgid,                                                     \
        tid);                                                     \
  }

instantiate_grouped_kernel(nax_w8a8_group64_bf16, 64, 4);
instantiate_grouped_kernel(nax_w8a8_group256_bf16, 256, 4);
instantiate_grouped_kernel(nax_w8a8_group448_bf16, 448, 4);
instantiate_grouped_kernel(nax_w8a8_group896_bf16, 896, 2);

template <int GROUP_SIZE>
inline void nax_grouped_quantize_bf16(
    device bfloat* x,
    device int8_t* quantized,
    device bfloat* scales,
    threadgroup float* partial_max,
    threadgroup bfloat* values,
    int K,
    uint2 tgid,
    uint tid) {
  constexpr uint threads = 256;
  int group = tgid.x;
  int row = tgid.y;
  int offset = row * K + group * GROUP_SIZE;
  float local_max = 0.0f;

  for (int i = tid; i < GROUP_SIZE; i += threads) {
    bfloat value = x[offset + i];
    values[i] = value;
    local_max = max(local_max, abs(float(value)));
  }
  partial_max[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  bfloat stored_scale = bfloat(max(partial_max[0] / 127.0f, 1e-8f));
  float scale = float(stored_scale);
  if (tid == 0) {
    scales[row * (K / GROUP_SIZE) + group] = stored_scale;
  }
  for (int i = tid; i < GROUP_SIZE; i += threads) {
    bfloat normalized = bfloat(float(values[i]) / scale);
    float value = clamp(rint(float(normalized)), -127.0f, 127.0f);
    quantized[offset + i] = int8_t(value);
  }
}

#define instantiate_quantize_kernel(name, group_size) \
  kernel void name(                                   \
      device bfloat* x [[buffer(0)]],                 \
      device int8_t* quantized [[buffer(1)]],         \
      device bfloat* scales [[buffer(2)]],            \
      constant int& K [[buffer(3)]],                  \
      uint2 tgid [[threadgroup_position_in_grid]],    \
      uint tid [[thread_index_in_threadgroup]]) {     \
    threadgroup float partial_max[256];               \
    threadgroup bfloat values[group_size];            \
    nax_grouped_quantize_bf16<group_size>(            \
        x, quantized, scales, partial_max, values, K, tgid, tid); \
  }

instantiate_quantize_kernel(nax_quantize_group64_bf16, 64);
instantiate_quantize_kernel(nax_quantize_group256_bf16, 256);
instantiate_quantize_kernel(nax_quantize_group448_bf16, 448);
instantiate_quantize_kernel(nax_quantize_group896_bf16, 896);
