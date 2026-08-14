"""Synthetic contracts for BF16 LoRA branches over MLX H3 linears."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_h3 import loading, lora, model


METADATA = {
    "base_model": "MiniMax-H3",
    "dtype": "bfloat16",
    "application": "W_eff = W + lora_B @ lora_A",
}


def pair(module: str, a_shape: tuple[int, int], b_shape: tuple[int, int]):
    header = {
        module + lora.A_SUFFIX: {"dtype": "BF16", "shape": list(a_shape)},
        module + lora.B_SUFFIX: {"dtype": "BF16", "shape": list(b_shape)},
    }
    weights = {
        module + lora.A_SUFFIX: mx.ones(a_shape, dtype=mx.bfloat16),
        module + lora.B_SUFFIX: mx.ones(b_shape, dtype=mx.bfloat16),
    }
    return header, weights


def test_quantized_base_and_bf16_lora_stay_separate():
    mx.random.seed(3)
    dense = nn.Linear(32, 3, bias=False)
    base = dense.to_quantized(group_size=32, bits=8)
    a = mx.random.normal((2, 32)).astype(mx.bfloat16)
    b = mx.random.normal((3, 2)).astype(mx.bfloat16)
    wrapped = lora.LoRALinear(base, a, b)
    x = mx.random.normal((4, 32)).astype(mx.bfloat16)

    got = wrapped(x)
    want = base(x) + ((x @ a.T) @ b.T).astype(base(x).dtype)
    assert isinstance(wrapped.base, nn.QuantizedLinear)
    assert mx.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_attach_validates_and_replaces_the_named_linear():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)

    tiny = Tiny()
    header, weights = pair("proj", (2, 4), (3, 2))
    selected = lora.attach(tiny, header, METADATA, weights)

    assert len(selected) == 1
    assert isinstance(tiny.proj, lora.LoRALinear)
    assert tiny.proj.base.weight.shape == (3, 4)


def test_header_rejects_unpaired_and_wrong_dtype_tensors():
    header, _ = pair("proj", (2, 4), (3, 2))
    del header["proj" + lora.B_SUFFIX]
    with pytest.raises(ValueError, match="unpaired"):
        lora.targets(header, METADATA)

    header, _ = pair("proj", (2, 4), (3, 2))
    header["proj" + lora.A_SUFFIX]["dtype"] = "F16"
    with pytest.raises(ValueError, match="BF16"):
        lora.targets(header, METADATA)


def test_header_allows_missing_dtype_metadata_but_rejects_wrong_value():
    header, _ = pair("proj", (2, 4), (3, 2))
    metadata_without_dtype = {
        key: value for key, value in METADATA.items() if key != "dtype"
    }

    assert len(lora.targets(header, metadata_without_dtype)) == 1
    with pytest.raises(ValueError, match="metadata dtype"):
        lora.targets(header, METADATA | {"dtype": "float16"})


def test_adaln_lora_is_baked_into_precompute_then_released():
    cfg = model.H3Config(
        hidden_size=8,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        rope_inv_freq_len=1,
    )
    tiny = model.MiniMaxH3(cfg)
    block_header, block_weights = pair(
        "blocks.0.adaln_proj.linear", (2, 8), (144, 2)
    )
    final_header, final_weights = pair(
        "final_layer.adaln_proj.linear", (2, 8), (16, 2)
    )
    header = block_header | final_header
    weights = block_weights | final_weights
    lora.attach(tiny, header, METADATA, weights)

    step = model.Plan((0.1, 0.6), [], (0, 0, 0), (0, 0, 1))
    embedding = tiny.time_embedder(mx.array(step.t_vals, dtype=mx.float32)).astype(
        mx.bfloat16
    )
    expected_block = tuple(tiny.blocks[0].adaln_proj(embedding))
    expected_final = tuple(tiny.final_layer.adaln_proj(embedding))
    mx.eval(*expected_block, *expected_final)

    tiny.precompute_adaln((step,), dtype=mx.bfloat16)
    actual_block = tiny._adaln_schedule.blocks[0][0]
    actual_final = tiny._adaln_schedule.final[0]

    assert all(
        mx.allclose(got, want, rtol=1e-3, atol=1e-3)
        for got, want in zip(actual_block, expected_block, strict=True)
    )
    assert all(
        mx.allclose(got, want, rtol=1e-3, atol=1e-3)
        for got, want in zip(actual_final, expected_final, strict=True)
    )
    assert tiny.blocks[0].adaln_proj is None
    assert tiny.final_layer.adaln_proj is None


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "adapter_name",
    (
        "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        "minimax_h3_turbo_v4_step600_ema.safetensors",
    ),
)
def test_real_adapter_header_matches_both_quantized_dit_trees(
    adapter_name: str, local_checkpoint
):
    root = Path(__file__).resolve().parents[1]
    adapter_path = (
        root
        / "weights/adapters/minimax-h3-turbo"
        / adapter_name
    )
    base_paths = (
        root / "weights/mlx-8bit/dit_fl2va_a8g32.safetensors",
        root / "weights/mlx-8bit/dit_ref2va_a8g32.safetensors",
    )
    adapter_path = local_checkpoint(adapter_path)
    base_paths = tuple(local_checkpoint(path) for path in base_paths)

    adapter_header, metadata = loading.read_header(adapter_path)
    selected = lora.targets(adapter_header, metadata)
    base_headers = [loading.read_header(path)[0] for path in base_paths]

    assert len(selected) == 259
    assert {target.rank for target in selected} == {16, 64}
    for target in selected:
        keys = tuple(
            target.module + suffix
            for suffix in (".weight", ".scales", ".biases")
        )
        for key in keys:
            assert key in base_headers[0]
            assert key in base_headers[1]
            left = base_headers[0][key]
            right = base_headers[1][key]
            assert (left["dtype"], left["shape"]) == (right["dtype"], right["shape"])
