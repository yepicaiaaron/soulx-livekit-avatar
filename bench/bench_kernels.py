"""YEP-48/49 kernel verification + microbenchmarks.

Run on the GPU box BEFORE live testing:

    python bench/bench_kernels.py

Verifies fused Triton kernels match the eager reference paths within
bf16 tolerance on the exact shapes the FlashHead DiT uses, then times
both paths. Exits non-zero on parity failure so deploy scripts can gate
on it.
"""

import sys
import time

import torch

sys.path.insert(0, ".")

DIM = 1536
HEADS = 12
HEAD_DIM = DIM // HEADS
# (frame_num, latent f) pairs: 9->3, 13->4, 33->9 latents; h=w=32 at 512px
GRIDS = {9: (3, 32, 32), 13: (4, 32, 32), 33: (9, 32, 32)}
DTYPE = torch.bfloat16
TOL = 1.6e-2  # bf16 resolution at unit scale


def eager_rmsnorm(x, w, eps):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def eager_mod_ln(x, scale, shift, eps):
    ln = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    return ln * (1 + scale) + shift


def rel_err(out, ref):
    # Relative error with unit floor: fused kernels round once (fp32 end-to-end),
    # eager rounds mid-chain — bf16 ulp disagreement on large values is expected.
    return ((out.float() - ref.float()).abs() / (ref.float().abs() + 1.0)).max().item()


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    if not torch.cuda.is_available():
        print("FATAL: CUDA required for kernel benchmarks")
        return 2

    from flash_head.kernels import (
        kernels_enabled, fused_rope, fused_rms_norm,
        fused_modulated_layer_norm, RopeTableCache,
    )
    from flash_head.src.modules.flash_head_model import (
        precompute_freqs_cis_3d, rope_apply,
    )

    if not kernels_enabled():
        print("FATAL: fused kernels unavailable (triton missing?)")
        return 2

    dev = "cuda"
    failures = 0
    freqs = precompute_freqs_cis_3d(HEAD_DIM)
    tables = RopeTableCache()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'case':<34}{'eager ms':>10}{'fused ms':>10}{'speedup':>9}{'max err':>10}")

    for frame_num, grid in GRIDS.items():
        s = grid[0] * grid[1] * grid[2]

        # --- RoPE ---
        x = torch.randn(1, s, HEADS, HEAD_DIM, device=dev, dtype=DTYPE)
        ref = rope_apply(x, freqs.to(dev), grid)
        cos, sin, seq_len = tables.get(freqs.to(dev), grid, HEAD_DIM, dev)
        out = fused_rope(x.contiguous(), cos, sin, seq_len)
        err = (out.float() - ref.float()).abs().max().item()
        ok = err < TOL
        failures += not ok
        te = timeit(lambda: rope_apply(x, freqs.to(dev), grid))
        tf = timeit(lambda: fused_rope(x.contiguous(), cos, sin, seq_len))
        print(f"rope f={frame_num:<3} seq={s:<6} {'OK' if ok else 'FAIL':<8}"
              f"{te:>10.3f}{tf:>10.3f}{te/tf:>8.1f}x{err:>10.1e}")

        # --- RMSNorm ---
        xf = torch.randn(1, s, DIM, device=dev, dtype=DTYPE)
        w = torch.randn(DIM, device=dev)
        ref = eager_rmsnorm(xf, w, 1e-6)
        out = fused_rms_norm(xf, w, 1e-6)
        err = rel_err(out, ref)
        ok = err < TOL
        failures += not ok
        te = timeit(lambda: eager_rmsnorm(xf, w, 1e-6))
        tf = timeit(lambda: fused_rms_norm(xf, w, 1e-6))
        print(f"rmsnorm seq={s:<6} {'OK' if ok else 'FAIL':<13}"
              f"{te:>10.3f}{tf:>10.3f}{te/tf:>8.1f}x{err:>10.1e}")

        # --- modulated LN ---
        scale = torch.randn(1, 1, DIM, device=dev, dtype=DTYPE) * 0.1
        shift = torch.randn(1, 1, DIM, device=dev, dtype=DTYPE) * 0.1
        ref = eager_mod_ln(xf, scale, shift, 1e-6)
        out = fused_modulated_layer_norm(xf, scale=scale, shift=shift, eps=1e-6)
        err = rel_err(out, ref)
        ok = err < TOL
        failures += not ok
        te = timeit(lambda: eager_mod_ln(xf, scale, shift, 1e-6))
        tf = timeit(lambda: fused_modulated_layer_norm(xf, scale=scale, shift=shift, eps=1e-6))
        print(f"mod-ln seq={s:<6} {'OK' if ok else 'FAIL':<14}"
              f"{te:>10.3f}{tf:>10.3f}{te/tf:>8.1f}x{err:>10.1e}")

    from flash_head.src.modules import flash_head_model as fhm
    print(f"\nattention backend (YEP-49): {fhm._ATTN_BACKEND}"
          f"  [fa3={fhm.FLASH_ATTN_3_AVAILABLE} sage={fhm.SAGE_ATTN_AVAILABLE}"
          f" fa2={fhm.FLASH_ATTN_2_AVAILABLE}]")

    if failures:
        print(f"\n{failures} parity FAILURES — do not deploy fused kernels")
        return 1
    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
