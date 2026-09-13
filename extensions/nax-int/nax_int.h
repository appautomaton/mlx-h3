#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mlx_nax_int {

mx::array matmul(
    const mx::array& x,
    const mx::array& weight,
    int bits,
    mx::StreamOrDevice stream = {});

mx::array scaled_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    mx::StreamOrDevice stream = {});

mx::array affine_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    const mx::array& x_sum,
    const mx::array& weight_center,
    mx::StreamOrDevice stream = {});

mx::array grouped_matmul(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& x_scale,
    const mx::array& weight_scale,
    int group_size,
    mx::StreamOrDevice stream = {});

std::vector<mx::array> grouped_quantize(
    const mx::array& x,
    int group_size,
    mx::StreamOrDevice stream = {});

class NaxIntMatmul : public mx::Primitive {
 public:
  explicit NaxIntMatmul(mx::Stream stream, int bits)
      : mx::Primitive(stream), bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "NaxIntMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int bits_;
};

class NaxIntScaledMatmul : public mx::Primitive {
 public:
  explicit NaxIntScaledMatmul(mx::Stream stream) : mx::Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "NaxIntScaledMatmul";
  }

  bool is_equivalent(const mx::Primitive& other) const override;
};

class NaxIntAffineMatmul : public mx::Primitive {
 public:
  explicit NaxIntAffineMatmul(mx::Stream stream) : mx::Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;
  const char* name() const override {
    return "NaxIntAffineMatmul";
  }
  bool is_equivalent(const mx::Primitive& other) const override;
};

class NaxIntGroup64Matmul : public mx::Primitive {
 public:
  explicit NaxIntGroup64Matmul(mx::Stream stream, int group_size)
      : mx::Primitive(stream), group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;
  const char* name() const override {
    return "NaxIntGroup64Matmul";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
};

class NaxGroupedQuantize : public mx::Primitive {
 public:
  explicit NaxGroupedQuantize(mx::Stream stream, int group_size)
      : mx::Primitive(stream), group_size_(group_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;
  const char* name() const override {
    return "NaxGroupedQuantize";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
};

} // namespace mlx_nax_int
