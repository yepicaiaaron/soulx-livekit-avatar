"""YEP-48: Fused Triton norm kernels for the FlashHead DiT.

Two inference-only kernels:

- fused_rms_norm: single-pass RMSNorm (fp32 accumulation) replacing the
  eager pow/mean/rsqrt/mul chain in RMSNorm.forward (4+ kernel launches
  and two extra fp32 round-trips through HBM per call).

- fused_modulated_layer_norm: LayerNorm fused with the AdaLN modulation
  `LN(x) * (1 + scale) + shift` that every DiT block applies twice per
  layer. Optionally applies an elementwise affine (weight/bias) instead,
  covering norm3. Collapses 3-4 launches into one.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / D
        inv = 1.0 / tl.sqrt(ms + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * D + offs, x * inv * w, mask=mask)

    @triton.jit
    def _mod_ln_kernel(
        x_ptr, out_ptr,
        scale_ptr, shift_ptr, w_ptr, b_ptr,
        D, eps,
        HAS_MOD: tl.constexpr, HAS_AFFINE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / D
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / D
        y = xc / tl.sqrt(var + eps)

        if HAS_AFFINE:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = y * w + b
        if HAS_MOD:
            sc = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = y * (1.0 + sc) + sh

        tl.store(out_ptr + row * D + offs, y, mask=mask)


from typing import Optional


def _rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.contiguous().view(-1, D)
    out = torch.empty_like(x2)
    _rms_norm_kernel[(x2.shape[0],)](
        x2, weight, out, D, eps, BLOCK=triton.next_power_of_2(D)
    )
    return out.view_as(x)


def _mod_ln_impl(
    x: torch.Tensor,
    scale: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    D = x.shape[-1]
    has_mod = scale is not None
    has_affine = weight is not None
    x2 = x.contiguous().view(-1, D)
    out = torch.empty_like(x2)
    _mod_ln_kernel[(x2.shape[0],)](
        x2, out,
        scale.reshape(-1) if has_mod else x2,
        shift.reshape(-1) if has_mod else x2,
        weight if has_affine else x2,
        bias if has_affine else x2,
        D, eps,
        HAS_MOD=has_mod, HAS_AFFINE=has_affine,
        BLOCK=triton.next_power_of_2(D),
    )
    return out.view_as(x)


# Register as opaque custom ops so torch.compile never traces kernel
# internals (same FakeTensor-safety rationale as fused_rope).
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

if TRITON_AVAILABLE and _HAS_CUSTOM_OP:

    @torch.library.custom_op("flash_head::fused_rms_norm", mutates_args=())
    def _rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _rms_norm_impl(x, weight, eps)

    @_rms_norm_op.register_fake
    def _(x, weight, eps):
        return torch.empty_like(x)

    @torch.library.custom_op("flash_head::fused_mod_ln", mutates_args=())
    def _mod_ln_op(x: torch.Tensor, scale: Optional[torch.Tensor],
                   shift: Optional[torch.Tensor], weight: Optional[torch.Tensor],
                   bias: Optional[torch.Tensor], eps: float) -> torch.Tensor:
        return _mod_ln_impl(x, scale, shift, weight, bias, eps)

    @_mod_ln_op.register_fake
    def _(x, scale, shift, weight, bias, eps):
        return torch.empty_like(x)

    def fused_rms_norm(x, weight, eps):
        return _rms_norm_op(x, weight, eps)

    def fused_modulated_layer_norm(x, scale=None, shift=None, weight=None,
                                   bias=None, eps=1e-6):
        return _mod_ln_op(x, scale, shift, weight, bias, eps)

elif TRITON_AVAILABLE:

    def fused_rms_norm(x, weight, eps):
        return _rms_norm_impl(x, weight, eps)

    def fused_modulated_layer_norm(x, scale=None, shift=None, weight=None,
                                   bias=None, eps=1e-6):
        return _mod_ln_impl(x, scale, shift, weight, bias, eps)

else:
    fused_rms_norm = None
    fused_modulated_layer_norm = None
