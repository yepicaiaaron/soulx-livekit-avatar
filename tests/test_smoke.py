"""Unit/smoke tests for the YEP-48/49 low-latency work. Run on the GPU box:

    pytest tests/ -v

CUDA-dependent tests skip cleanly on CPU-only machines; the pure-math
tests (audio windowing, config profiles, chunk arithmetic) always run.
"""

import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Pure logic — always runs
# ---------------------------------------------------------------------------

def _load_params(profile):
    import yaml
    path = {
        "default": "flash_head/configs/infer_params.yaml",
        "balanced": "flash_head/configs/infer_params_balanced.yaml",
        "lowlat": "flash_head/configs/infer_params_lowlat.yaml",
    }[profile]
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("profile,frame_num,slice_ms", [
    ("default", 33, 1120.0), ("balanced", 13, 320.0), ("lowlat", 9, 160.0),
])
def test_profile_chunk_math(profile, frame_num, slice_ms):
    p = _load_params(profile)
    assert p["frame_num"] == frame_num
    # motion frames: (latent_num-1)*vae_stride_t + 1 with WanVAE stride 4
    motion = (p["motion_frames_latent_num"] - 1) * 4 + 1
    slice_len = p["frame_num"] - motion
    assert slice_len > 0, "chunk must deliver net new frames"
    assert slice_len / p["tgt_fps"] * 1000 == pytest.approx(slice_ms)
    # frame_num must map to whole VAE latents: (n-1) % stride == 0
    assert (p["frame_num"] - 1) % 4 == 0


def test_audio_window_indices():
    """The embedding window must stay in-bounds for every profile."""
    for profile in ("default", "balanced", "lowlat"):
        p = _load_params(profile)
        end_idx = p["cached_audio_duration"] * p["tgt_fps"]
        start_idx = end_idx - p["frame_num"]
        assert start_idx > 2, f"{profile}: need >=2 frames of left context"
        indices = (torch.arange(2 * 2 + 1) - 2)
        centers = torch.arange(start_idx, end_idx).unsqueeze(1) + indices
        centers = torch.clamp(centers, min=0, max=end_idx - 1)
        assert centers.min() >= 0 and centers.max() < end_idx


def test_config_profile_env_selects_file():
    env = dict(os.environ, FLASH_HEAD_PROFILE="lowlat")
    out = subprocess.run(
        [sys.executable, "-c",
         "import yaml,os;"
         "from flash_head import inference as i;"
         "print(i.infer_params['frame_num'])"],
        capture_output=True, text=True, env=env, cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "9"


def test_attention_dispatch_override():
    env = dict(os.environ, FLASH_HEAD_ATTN="sdpa")
    out = subprocess.run(
        [sys.executable, "-c",
         "from flash_head.src.modules import flash_head_model as m;"
         "print(m._ATTN_BACKEND)"],
        capture_output=True, text=True, env=env, cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "sdpa"


def test_no_hardcoded_livekit_secrets():
    with open("webrtc_sync.py") as f:
        src = f.read()
    assert "livekit.cloud" not in src, "hardcoded LiveKit URL back in source"
    assert "API6pGtbWcmZpMs" not in src, "leaked API key back in source"


# ---------------------------------------------------------------------------
# CUDA kernels — GPU box only
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


@needs_cuda
@pytest.mark.parametrize("grid", [(3, 32, 32), (4, 32, 32), (9, 32, 32)])
def test_fused_rope_matches_eager(grid):
    from flash_head.kernels import fused_rope, RopeTableCache
    from flash_head.src.modules.flash_head_model import (
        precompute_freqs_cis_3d, rope_apply)
    head_dim, heads = 128, 12
    s = grid[0] * grid[1] * grid[2]
    freqs = precompute_freqs_cis_3d(head_dim).cuda()
    x = torch.randn(1, s, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    ref = rope_apply(x, freqs, grid)
    cos, sin, seq_len = RopeTableCache().get(freqs, grid, head_dim, "cuda")
    out = fused_rope(x.contiguous(), cos, sin, seq_len)
    assert (out.float() - ref.float()).abs().max().item() < 1.6e-2


@needs_cuda
def test_fused_rmsnorm_matches_eager():
    from flash_head.kernels import fused_rms_norm
    x = torch.randn(1, 3072, 1536, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1536, device="cuda")
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
           ).to(x.dtype) * w
    out = fused_rms_norm(x, w, 1e-6)
    assert (out.float() - ref.float()).abs().max().item() < 1.6e-2


@needs_cuda
def test_fused_modln_matches_eager():
    from flash_head.kernels import fused_modulated_layer_norm
    x = torch.randn(1, 3072, 1536, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(1, 1, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = torch.nn.functional.layer_norm(
        x.float(), (1536,), eps=1e-6).to(x.dtype) * (1 + scale) + shift
    out = fused_modulated_layer_norm(x, scale=scale, shift=shift, eps=1e-6)
    assert (out.float() - ref.float()).abs().max().item() < 1.6e-2


@needs_cuda
def test_rope_small_frame_num_compiles():
    """The historical failure: frame_num<33 broke torch.compile via the
    complex-fp64 path. The fused op must survive compilation at f=3."""
    from flash_head.kernels import fused_rope, RopeTableCache
    from flash_head.src.modules.flash_head_model import precompute_freqs_cis_3d
    grid = (3, 32, 32)
    freqs = precompute_freqs_cis_3d(128).cuda()
    cos, sin, seq_len = RopeTableCache().get(freqs, grid, 128, "cuda")

    def f(x):
        return fused_rope(x, cos, sin, seq_len) * 2

    x = torch.randn(1, 3072, 12, 128, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(f)
    out = compiled(x)
    assert out.shape == x.shape
