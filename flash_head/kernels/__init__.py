"""YEP-48 fused Triton kernels with graceful fallback.

Enabled by default when Triton + CUDA are present. Set
FLASH_HEAD_FUSED_KERNELS=0 to force the original eager math paths
(useful for A/B parity checks — bench/bench_kernels.py does this
automatically).
"""

import os
import torch

from .fused_rope import TRITON_AVAILABLE as _ROPE_OK, fused_rope, RopeTableCache
from .fused_norm import (
    TRITON_AVAILABLE as _NORM_OK,
    fused_rms_norm,
    fused_modulated_layer_norm,
)


def kernels_enabled() -> bool:
    if os.environ.get("FLASH_HEAD_FUSED_KERNELS", "1") == "0":
        return False
    return _ROPE_OK and _NORM_OK and torch.cuda.is_available()


__all__ = [
    "kernels_enabled",
    "fused_rope",
    "fused_rms_norm",
    "fused_modulated_layer_norm",
    "RopeTableCache",
]
