"""Experimental native integer NAX matrix multiplication."""

# Load MLX's native runtime before this extension.  The extension links against
# ``@rpath/libmlx.dylib`` and must resolve it from the active Python environment,
# not the isolated environment that built the editable wheel.
import mlx.core as _mlx_core  # noqa: F401  -- imported for its load-order effect

from ._ext import (
    affine_matmul,
    grouped_matmul,
    grouped_quantize,
    matmul,
    scaled_matmul,
)

__all__ = [
    "affine_matmul",
    "grouped_matmul",
    "grouped_quantize",
    "matmul",
    "scaled_matmul",
]
