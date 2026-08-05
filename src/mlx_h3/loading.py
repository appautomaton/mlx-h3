"""Loading quantized checkpoints into the module tree.

Which tensors are quantized is read off the checkpoint, not decided here: a
predicate duplicated between the converter and the loader is a predicate that
will eventually disagree with itself. A module is quantized if and only if the
file carries scales beside its weight.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .audio_vae import AudioVAE, AudioVAEConfig, AudioVAEEncoder
from .model import H3Config, MiniMaxH3, Plan
from .text_encoder import MultimodalTextEncoder, TextEncoder, TextEncoderConfig
from .video_vae import VideoVAE, VideoVAEConfig, VideoVAEEncoder


def read_header(path: str | Path) -> tuple[dict, dict]:
    """Tensor table and metadata, without touching the payload."""
    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, header.pop("__metadata__", {})


def quantization(metadata: dict) -> dict:
    return {
        "group_size": int(metadata.get("quantization.group_size", 32)),
        "bits": int(metadata.get("quantization.bits", 8)),
        "mode": metadata.get("quantization.mode", "affine"),
    }


def quantized_modules(header: dict) -> set[str]:
    """Module paths whose weight is packed, e.g. ``blocks.0.attn.qkv_proj``."""
    return {k.removesuffix(".scales") for k in header if k.endswith(".scales")}


def prepare(model: nn.Module, header: dict, metadata: dict) -> nn.Module:
    """Swap in QuantizedLinear wherever the checkpoint says to, in place."""
    packed = quantized_modules(header)
    nn.quantize(model, **quantization(metadata), class_predicate=lambda p, _: p in packed)
    return model


def check(model: nn.Module, header: dict) -> tuple[set[str], set[str]]:
    """(missing from the checkpoint, unexpected in it) -- both empty means a clean load."""
    wanted = {k for k, _ in tree_flatten(model.parameters())}
    stored = set(header)
    return wanted - stored, stored - wanted


def load_dit(
    path: str | Path,
    config: H3Config | None = None,
    *,
    plans: tuple[Plan, ...] | None = None,
    modulation_dtype: mx.Dtype | None = None,
) -> MiniMaxH3:
    header, metadata = read_header(path)
    model = prepare(MiniMaxH3(config), header, metadata)
    missing, unexpected = check(model, header)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the module tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})"
        )
    model.load_weights(str(path))
    if plans is not None:
        if modulation_dtype is None:
            raise ValueError("modulation_dtype is required with step plans")
        model.precompute_adaln(plans, dtype=modulation_dtype)
    # Materialize now rather than on first use: mx.load is lazy, and letting the
    # weights fault in mid-step is exactly the paging this project must avoid.
    mx.eval(model.parameters())
    return model


def load_text_encoder(
    path: str | Path, config: TextEncoderConfig | None = None
) -> TextEncoder:
    """Load only the 50-layer text decoder; the ViT payload stays untouched."""
    header, metadata = read_header(path)
    selected_header = {key: value for key, value in header.items() if key.startswith("model.")}
    model = prepare(TextEncoder(config), selected_header, metadata)
    missing, unexpected = check(model, selected_header)
    if missing or unexpected:
        raise ValueError(
            "text encoder checkpoint does not match the decoder tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = [(key, checkpoint[key]) for key in sorted(selected_header)]
    model.load_weights(selected)
    del checkpoint, selected
    # Materialize before inference so paging cannot be hidden inside layer 0.
    mx.eval(model.parameters())
    return model


def load_multimodal_text_encoder(
    path: str | Path,
    config: TextEncoderConfig | None = None,
) -> MultimodalTextEncoder:
    """Load the text decoder and Qwen3-VL tower for image conditioning."""
    header, metadata = read_header(path)
    model = prepare(MultimodalTextEncoder(config), header, metadata)
    missing, unexpected = check(model, header)
    if missing or unexpected:
        raise ValueError(
            "multimodal text checkpoint does not match the model tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = []
    for key in sorted(header):
        value = checkpoint[key]
        if key == "visual.patch_embed.proj.weight":
            # Torch Conv3d [O,C,T,H,W] -> the equivalent flattened Linear.
            value = value.reshape(value.shape[0], -1)
        selected.append((key, value))
    model.load_weights(selected)
    del checkpoint, selected
    mx.eval(model.parameters())
    return model


def load_video_vae(
    path: str | Path, config: VideoVAEConfig | None = None
) -> VideoVAE:
    """Load only the visual decoder subset from the dense VAE checkpoint."""
    header, _ = read_header(path)
    model = VideoVAE(config)
    wanted = {key for key, _ in tree_flatten(model.parameters())}
    stored = set(header)
    missing = wanted - stored
    unused = stored - wanted
    invalid_unused = {
        key
        for key in unused
        if not key.startswith(("encoder.", "quant_conv."))
    }
    if missing or invalid_unused:
        raise ValueError(
            "video VAE checkpoint does not match the decoder tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(invalid_unused)} unexpected (e.g. {sorted(invalid_unused)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = []
    for key in sorted(wanted):
        value = checkpoint[key]
        if key in ("latents_mean", "latents_std"):
            value = value.astype(mx.float32)
        elif key == "post_quant_conv.weight":
            value = value.reshape(
                model.config.latent_channels, model.config.latent_channels
            )
        selected.append((key, value))
    model.load_weights(selected)
    del checkpoint, selected
    # The source file also carries an encoder. Only decoder parameters are
    # materialized here, so t2va never pays its memory cost.
    mx.eval(model.parameters())
    return model


def load_video_vae_encoder(
    path: str | Path, config: VideoVAEConfig | None = None
) -> VideoVAEEncoder:
    """Load only the causal CNN encoder subset from the dense visual VAE."""
    header, _ = read_header(path)
    model = VideoVAEEncoder(config)
    wanted = {key for key, _ in tree_flatten(model.parameters())}
    stored = set(header)
    missing = wanted - stored
    invalid = {
        key
        for key in stored - wanted
        if not key.startswith(("decoder.", "post_quant_conv."))
    }
    if missing or invalid:
        raise ValueError(
            "video VAE checkpoint does not match the encoder tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(invalid)} unexpected (e.g. {sorted(invalid)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = []
    for key in sorted(wanted):
        value = checkpoint[key]
        if key in ("latents_mean", "latents_std"):
            value = value.astype(mx.float32)
        elif key.endswith(".weight") and value.ndim == 5:
            # Torch Conv3d [O,I,T,H,W] -> MLX [O,T,H,W,I].
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        selected.append((key, value))
    model.load_weights(selected)
    del checkpoint, selected
    mx.eval(model.parameters())
    return model


def load_audio_vae(
    path: str | Path, config: AudioVAEConfig | None = None
) -> AudioVAE:
    """Load only the BigVGAN decode subset from the dense audio checkpoint."""
    header, _ = read_header(path)
    model = AudioVAE(config)
    wanted = {key for key, _ in tree_flatten(model.parameters())}
    stored = set(header)
    missing = wanted - stored
    unused = stored - wanted
    invalid_unused = {
        key
        for key in unused
        if not key.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))
    }
    if missing or invalid_unused:
        raise ValueError(
            "audio VAE checkpoint does not match the decoder tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(invalid_unused)} unexpected (e.g. {sorted(invalid_unused)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = []
    for key in sorted(wanted):
        value = checkpoint[key]
        if key.endswith(".weight") and value.ndim == 3:
            if key.startswith("decoder.ups."):
                # Torch ConvTranspose1d [I,O,K] -> MLX [O,K,I].
                value = mx.transpose(value, (1, 2, 0))
            else:
                # Torch Conv1d [O,I,K] -> MLX [O,K,I].
                value = mx.transpose(value, (0, 2, 1))
        selected.append((key, value))
    model.load_weights(selected)
    del checkpoint, selected
    mx.eval(model.parameters())
    return model


def load_audio_vae_encoder(
    path: str | Path, config: AudioVAEConfig | None = None
) -> AudioVAEEncoder:
    """Load only the deterministic audio posterior-mean encoder."""
    header, _ = read_header(path)
    model = AudioVAEEncoder(config)
    wanted = {key for key, _ in tree_flatten(model.parameters())}
    stored = set(header)
    missing = wanted - stored
    invalid = {
        key
        for key in stored - wanted
        if not key.startswith(("decoder.", "dec_in_proj.", "logs_proj."))
    }
    if missing or invalid:
        raise ValueError(
            "audio VAE checkpoint does not match the encoder tree: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(invalid)} unexpected (e.g. {sorted(invalid)[:3]})"
        )

    checkpoint = mx.load(str(path))
    selected = []
    for key in sorted(wanted):
        value = checkpoint[key]
        if key in ("latents_mean", "latents_std"):
            value = value.astype(mx.float32)
        elif key.endswith(".weight") and value.ndim == 3:
            # Torch Conv1d [O,I,K] -> MLX [O,K,I].
            value = mx.transpose(value, (0, 2, 1))
        selected.append((key, value))
    model.load_weights(selected)
    del checkpoint, selected
    mx.eval(model.parameters())
    return model
