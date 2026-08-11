"""Experimental M5 NAX W8A8 linears for the H3 DiT trunk.

The public runtime keeps MLX's affine W8A16 path by default. This module is an
explicit experiment: it requantizes one loaded linear at a time into symmetric
W8A8 groups and dispatches a local Metal 4 extension. Dense source checkpoints
are never loaded or required.
"""

from __future__ import annotations

import gc
from importlib import import_module

import mlx.core as mx
import mlx.nn as nn

from . import lora

SUPPORTED_GROUP_SIZES = (64, 256, 448, 896)
_TRUNK_LINEAR_NAMES = ("qkv_proj", "out_proj", "fc1", "fc2")


def _operations():
    try:
        extension = import_module("mlx_nax_int")
        return extension.grouped_quantize, extension.grouped_matmul
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "NAX W8A8 requires the local mlx_nax_int extension with "
            "grouped_quantize and grouped_matmul support"
        ) from error


class GroupedW8A8Linear(nn.Module):
    """A BF16-input linear backed by group-scaled int8 NAX matmul."""

    def __init__(self, base: nn.QuantizedLinear, *, group_size: int) -> None:
        super().__init__()
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"NAX group size must be one of {SUPPORTED_GROUP_SIZES}"
            )
        output_dims, packed_input_dims = base.weight.shape
        input_dims = packed_input_dims * 32 // base.bits
        if input_dims % group_size:
            raise ValueError(
                f"input dimension {input_dims} is not divisible by NAX group "
                f"size {group_size}"
            )
        if output_dims % 64:
            raise ValueError(
                f"output dimension {output_dims} is not divisible by 64"
            )

        dense = mx.dequantize(
            base.weight,
            base.scales,
            base.biases,
            group_size=base.group_size,
            bits=base.bits,
            mode=base.mode,
        )
        grouped = dense.reshape(output_dims, input_dims // group_size, group_size)
        scales = mx.maximum(
            mx.max(mx.abs(grouped), axis=-1, keepdims=True) / 127.0,
            1e-8,
        ).astype(mx.bfloat16)
        quantized = mx.clip(mx.round(grouped / scales), -127, 127).astype(
            mx.int8
        )

        self.weight = mx.contiguous(
            mx.transpose(quantized.reshape(output_dims, input_dims))
        )
        self.scales = mx.contiguous(mx.transpose(scales[..., 0]))
        self.bias = getattr(base, "bias", None)
        self.group_size = group_size
        mx.eval(self.weight, self.scales)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 2 or x.dtype != mx.bfloat16:
            raise ValueError("NAX W8A8 linears require a 2D bfloat16 input")
        rows, input_dims = x.shape
        padded_rows = (rows + 63) // 64 * 64
        if padded_rows != rows:
            x = mx.pad(x, ((0, padded_rows - rows), (0, 0)))

        grouped_quantize, grouped_matmul = _operations()
        quantized, scales = grouped_quantize(x, self.group_size)
        output = grouped_matmul(
            quantized,
            self.weight,
            scales,
            self.scales,
            self.group_size,
        )[:rows]
        return output if self.bias is None else output + self.bias


def _convert(module: nn.Module, group_size: int) -> nn.Module:
    if isinstance(module, lora.LoRALinear):
        if not isinstance(module.base, nn.QuantizedLinear):
            raise TypeError("NAX LoRA base must be an MLX QuantizedLinear")
        module.base = GroupedW8A8Linear(module.base, group_size=group_size)
        return module
    if not isinstance(module, nn.QuantizedLinear):
        raise TypeError("NAX target must be an MLX QuantizedLinear")
    return GroupedW8A8Linear(module, group_size=group_size)


def convert_dit(model: nn.Module, *, group_size: int) -> int:
    """Replace the four compute-dominant linears in every DiT block."""
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"NAX group size must be one of {SUPPORTED_GROUP_SIZES}")
    converted = 0
    for block in model.blocks:
        block.attn.qkv_proj = _convert(block.attn.qkv_proj, group_size)
        block.attn.out_proj = _convert(block.attn.out_proj, group_size)
        block.mlp.fc1 = _convert(block.mlp.fc1, group_size)
        block.mlp.fc2 = _convert(block.mlp.fc2, group_size)
        converted += len(_TRUNK_LINEAR_NAMES)
        gc.collect()
        mx.clear_cache()
    return converted
