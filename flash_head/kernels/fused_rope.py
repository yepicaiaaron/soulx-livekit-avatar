"""YEP-48: Fused Triton RoPE for the FlashHead DiT.

Replaces `rope_apply` in flash_head_model.py, which rotates q/k through a
complex-float64 path (`view_as_complex` + fp64 polar multiply). That path is
memory-bound, runs fp64 math at 1/64 throughput on consumer GPUs, and is the
source of the torch.compile FakeTensor shape mismatches at small frame_num.

Here the 3D (T, H, W) rotary tables are materialized once per grid size as
real-valued fp32 cos/sin tensors, and a single Triton kernel applies the
rotation in fp32 with the output cast back to the input dtype. The op is
registered through torch.library so torch.compile treats it as opaque —
no complex dtypes ever enter the traced graph.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rope_fwd_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr,
        n_heads, head_dim,
        SEQ_LEN,          # rotated prefix; tokens beyond pass through
        HALF_DIM: tl.constexpr,
    ):
        pid = tl.program_id(0)          # one program per (token, head)
        token = pid // n_heads
        head = pid % n_heads

        offs = tl.arange(0, HALF_DIM)
        mask = offs < head_dim // 2

        base = (token * n_heads + head) * head_dim
        x0 = tl.load(x_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)

        in_prefix = token < SEQ_LEN
        tbase = token * (head_dim // 2)
        cos = tl.load(cos_ptr + tbase + offs, mask=mask & in_prefix, other=1.0)
        sin = tl.load(sin_ptr + tbase + offs, mask=mask & in_prefix, other=0.0)

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        tl.store(out_ptr + base + 2 * offs, y0, mask=mask)
        tl.store(out_ptr + base + 2 * offs + 1, y1, mask=mask)


def _rope_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 seq_len: int) -> torch.Tensor:
    b, s, n, d = x.shape
    assert b == 1, "streaming pipeline is batch-1"
    out = torch.empty_like(x)
    half = d // 2
    HALF_DIM = triton.next_power_of_2(half)
    grid = (s * n,)
    _rope_fwd_kernel[grid](
        x, cos, sin, out,
        n, d, min(seq_len, s),
        HALF_DIM=HALF_DIM,
    )
    return out


# ---------------------------------------------------------------------------
# torch.compile-safe registration
# ---------------------------------------------------------------------------
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

if TRITON_AVAILABLE and _HAS_CUSTOM_OP:

    @torch.library.custom_op("flash_head::fused_rope", mutates_args=())
    def fused_rope_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      seq_len: int) -> torch.Tensor:
        return _rope_triton(x, cos, sin, seq_len)

    @fused_rope_op.register_fake
    def _(x, cos, sin, seq_len):
        return torch.empty_like(x)

    def fused_rope(x, cos, sin, seq_len):
        return fused_rope_op(x, cos, sin, seq_len)

elif TRITON_AVAILABLE:  # torch < 2.4 fallback: plain callable, no fake tensor

    def fused_rope(x, cos, sin, seq_len):
        return _rope_triton(x, cos, sin, seq_len)

else:
    fused_rope = None


class RopeTableCache:
    """Materializes and caches fp32 cos/sin tables per (f, h, w) grid.

    Tables are built from the model's existing complex `freqs` buffer so the
    rotation matches the eager path bit-for-bit up to fp32 rounding.
    """

    def __init__(self):
        self._cache = {}

    def get(self, freqs: torch.Tensor, grid_sizes, head_dim: int, device):
        f, h, w = int(grid_sizes[0]), int(grid_sizes[1]), int(grid_sizes[2])
        key = (f, h, w, str(device))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        c = head_dim // 2
        split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        table = torch.cat([
            split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, -1)

        cos = table.real.to(device=device, dtype=torch.float32).contiguous()
        sin = table.imag.to(device=device, dtype=torch.float32).contiguous()
        self._cache[key] = (cos, sin, f * h * w)
        return self._cache[key]
