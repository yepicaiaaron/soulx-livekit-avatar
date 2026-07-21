"""Quality-regression gate: fused kernels must not change the pictures.

Generates a deterministic chunk (fixed seed, fixed synthetic audio) and
saves the frames; then compares two such runs. The deploy script runs it
twice — FLASH_HEAD_FUSED_KERNELS=0 (eager reference) and =1 (YEP-48) —
holding FLASH_HEAD_ATTN constant so the attention backend doesn't
confound the comparison, and gates on PSNR.

    FLASH_HEAD_ATTN=fa2 FLASH_HEAD_FUSED_KERNELS=0 python bench/bench_quality.py --out /tmp/ref.npy
    FLASH_HEAD_ATTN=fa2 FLASH_HEAD_FUSED_KERNELS=1 python bench/bench_quality.py --out /tmp/fused.npy
    python bench/bench_quality.py --compare /tmp/ref.npy /tmp/fused.npy

PSNR >= 35 dB passes (bf16 run-to-run noise sits well above 40 dB; a
real kernel bug lands far below 30).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")

PSNR_PASS = 35.0


def generate(out_path):
    import torch
    from flash_head.inference import (
        get_pipeline, get_base_data, get_infer_params,
        get_audio_embedding, run_pipeline,
    )

    params = get_infer_params()
    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=os.environ.get("SOULX_CKPT_DIR", "models/SoulX-FlashHead-1_3B"),
        wav2vec_dir=os.environ.get("SOULX_WAV2VEC_DIR", "models/wav2vec2-base-960h"),
        model_type=os.environ.get("SOULX_MODEL_TYPE", "pro"),
    )
    get_base_data(pipeline,
                  cond_image_path_or_dir=os.environ.get(
                      "SOULX_COND_IMAGE", "examples/omani_character.png"),
                  base_seed=42, use_face_crop=False)

    params = get_infer_params()
    sr, fps = params["sample_rate"], params["tgt_fps"]
    cached = sr * params["cached_audio_duration"]
    end_idx = params["cached_audio_duration"] * fps
    start_idx = end_idx - params["frame_num"]

    # deterministic pseudo-speech: amplitude-modulated tone
    t = np.arange(cached) / sr
    audio = (0.1 * np.sin(2 * np.pi * 220 * t) *
             (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

    emb = get_audio_embedding(pipeline, audio, start_idx, end_idx)
    frames = run_pipeline(pipeline, emb)  # (T, H, W, C) float
    torch.cuda.synchronize()
    np.save(out_path, frames.cpu().numpy().astype(np.uint8))
    print(f"saved {frames.shape} -> {out_path}")


def compare(a_path, b_path):
    a = np.load(a_path).astype(np.float64)
    b = np.load(b_path).astype(np.float64)
    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {a.shape} vs {b.shape}")
        return 1
    mse = np.mean((a - b) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    per_frame = [
        99.0 if (m := np.mean((a[i] - b[i]) ** 2)) == 0 else 10 * np.log10(255.0 ** 2 / m)
        for i in range(a.shape[0])
    ]
    print(f"PSNR overall: {psnr:.1f} dB  (worst frame: {min(per_frame):.1f} dB, "
          f"threshold {PSNR_PASS} dB)")
    if min(per_frame) >= PSNR_PASS:
        print("QUALITY GATE: PASS — fused kernels are visually lossless")
        return 0
    print("QUALITY GATE: FAIL — investigate before deploying fused kernels")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    elif args.out:
        generate(args.out)
    else:
        ap.error("need --out or --compare")
