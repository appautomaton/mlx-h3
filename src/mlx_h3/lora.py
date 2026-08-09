"""Validated activation-space LoRA adapters for the H3 DiT.

The serving checkpoint stays quantized. Adapter tensors remain dense BF16 and
are evaluated as a separate low-rank branch: ``base(x) + B(A(x))``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


@dataclass(frozen=True)
class Target:
    module: str
    a_key: str
    b_key: str
    input_dims: int
    output_dims: int
    rank: int


class LoRALinear(nn.Module):
    """A dense BF16 low-rank branch over a dense or quantized base linear."""

    def __init__(
        self,
        base: nn.Module,
        lora_a: mx.array,
        lora_b: mx.array,
        *,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(strength):
            raise ValueError("LoRA strength must be finite")
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.strength = float(strength)

    def __call__(self, x: mx.array) -> mx.array:
        base = self.base(x)
        adapter_input = x.astype(self.lora_a.dtype)
        delta = (adapter_input @ self.lora_a.T) @ self.lora_b.T
        return base + (self.strength * delta).astype(base.dtype)


def targets(header: Mapping[str, dict], metadata: Mapping[str, str]) -> tuple[Target, ...]:
    """Validate a standard MiniMax-H3 BF16 LoRA safetensors header."""
    if metadata.get("base_model") != "MiniMax-H3":
        raise ValueError("LoRA metadata base_model must be MiniMax-H3")
    metadata_dtype = metadata.get("dtype")
    if metadata_dtype is not None and metadata_dtype != "bfloat16":
        raise ValueError("LoRA metadata dtype must be bfloat16")
    if metadata.get("application") != "W_eff = W + lora_B @ lora_A":
        raise ValueError("unsupported LoRA application metadata")

    keys = set(header)
    unsupported = sorted(
        key for key in keys if not key.endswith(A_SUFFIX) and not key.endswith(B_SUFFIX)
    )
    if unsupported:
        raise ValueError(f"unsupported LoRA tensor keys: {unsupported[:3]}")

    modules_a = {key.removesuffix(A_SUFFIX) for key in keys if key.endswith(A_SUFFIX)}
    modules_b = {key.removesuffix(B_SUFFIX) for key in keys if key.endswith(B_SUFFIX)}
    if modules_a != modules_b:
        missing_a = sorted(modules_b - modules_a)
        missing_b = sorted(modules_a - modules_b)
        raise ValueError(
            f"unpaired LoRA tensors: missing A {missing_a[:3]}, missing B {missing_b[:3]}"
        )
    if not modules_a:
        raise ValueError("LoRA checkpoint contains no tensor pairs")

    out = []
    for module in sorted(modules_a):
        a_key = module + A_SUFFIX
        b_key = module + B_SUFFIX
        a_meta, b_meta = header[a_key], header[b_key]
        if a_meta.get("dtype") != "BF16" or b_meta.get("dtype") != "BF16":
            raise ValueError(f"LoRA tensors for {module} must be BF16")
        a_shape = tuple(a_meta.get("shape", ()))
        b_shape = tuple(b_meta.get("shape", ()))
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"LoRA tensors for {module} must be rank 2")
        rank, input_dims = a_shape
        output_dims, b_rank = b_shape
        if min(rank, input_dims, output_dims) < 1 or b_rank != rank:
            raise ValueError(
                f"incompatible LoRA shapes for {module}: A {a_shape}, B {b_shape}"
            )
        out.append(
            Target(module, a_key, b_key, input_dims, output_dims, rank)
        )
    return tuple(out)


def _linear_shape(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        output_dims, input_dims = module.weight.shape
        return input_dims, output_dims
    if isinstance(module, nn.QuantizedLinear):
        output_dims, packed_input_dims = module.weight.shape
        input_dims = packed_input_dims * 32 // module.bits
        return input_dims, output_dims
    raise ValueError(f"LoRA target is not a supported linear: {type(module).__name__}")


def attach(
    model: nn.Module,
    header: Mapping[str, dict],
    metadata: Mapping[str, str],
    weights: Mapping[str, mx.array],
    *,
    strength: float = 1.0,
) -> tuple[Target, ...]:
    """Attach every validated adapter pair to its existing model module."""
    selected = targets(header, metadata)
    modules = dict(model.named_modules())
    replacements = []
    for target in selected:
        if target.module not in modules:
            raise ValueError(f"LoRA target is absent from the DiT: {target.module}")
        base = modules[target.module]
        actual_input, actual_output = _linear_shape(base)
        expected = (target.input_dims, target.output_dims)
        if (actual_input, actual_output) != expected:
            raise ValueError(
                f"LoRA target shape mismatch for {target.module}: "
                f"model {(actual_input, actual_output)}, adapter {expected}"
            )
        try:
            a, b = weights[target.a_key], weights[target.b_key]
        except KeyError as error:
            raise ValueError(f"LoRA payload is missing {error.args[0]}") from error
        if a.shape != (target.rank, target.input_dims):
            raise ValueError(f"LoRA payload shape mismatch for {target.a_key}: {a.shape}")
        if b.shape != (target.output_dims, target.rank):
            raise ValueError(f"LoRA payload shape mismatch for {target.b_key}: {b.shape}")
        if a.dtype != mx.bfloat16 or b.dtype != mx.bfloat16:
            raise ValueError(f"LoRA payload tensors for {target.module} must be BF16")
        replacements.append(
            (target.module, LoRALinear(base, a, b, strength=strength))
        )

    model.update_modules(tree_unflatten(replacements))
    return selected
