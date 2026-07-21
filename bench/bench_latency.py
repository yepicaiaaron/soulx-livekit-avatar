"""End-to-end chunk latency benchmark for the FlashHead pipeline.

Measures, for the selected profile (FLASH_HEAD_PROFILE=lowlat|balanced|default)
and model type (SOULX_MODEL_TYPE, default pro):

  - time-to-first-chunk (includes torch.compile warm-up separately)
  - steady-state per-chunk generation time vs the real-time budget
  - a sustainability verdict: can this GPU hold 25fps at this profile?

Run:  FLASH_HEAD_PROFILE=lowlat python bench/bench_latency.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from flash_head.inference import (  # noqa: E402
    get_pipeline, get_base_data, get_infer_params,
    get_audio_embedding, run_pipeline,
)

N_WARMUP = 3
N_CHUNKS = 20


def main():
    params = get_infer_params()
    model_type = os.environ.get("SOULX_MODEL_TYPE", "pro")
    ckpt_dir = os.environ.get("SOULX_CKPT_DIR", "models/SoulX-FlashHead-1_3B")
    wav2vec_dir = os.environ.get("SOULX_WAV2VEC_DIR", "models/wav2vec2-base-960h")
    cond_image = os.environ.get("SOULX_COND_IMAGE", "examples/omani_character.png")

    print(f"profile: frame_num={params['frame_num']}  model_type={model_type}")

    pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir,
                            wav2vec_dir=wav2vec_dir, model_type=model_type)
    get_base_data(pipeline, cond_image_path_or_dir=cond_image,
                  base_seed=42, use_face_crop=False)

    params = get_infer_params()  # motion_frames_num now populated
    sr = params["sample_rate"]
    fps = params["tgt_fps"]
    frame_num = params["frame_num"]
    slice_len = frame_num - params["motion_frames_num"]
    budget = slice_len / fps
    cached = sr * params["cached_audio_duration"]
    end_idx = params["cached_audio_duration"] * fps
    start_idx = end_idx - frame_num

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(cached) * 0.05).astype(np.float32)

    # torch.compile warm-up (excluded from steady-state stats)
    t0 = time.perf_counter()
    emb = get_audio_embedding(pipeline, audio, start_idx, end_idx)
    run_pipeline(pipeline, emb)
    torch.cuda.synchronize()
    print(f"warm-up (compile) first chunk: {time.perf_counter()-t0:.1f}s")

    for _ in range(N_WARMUP):
        run_pipeline(pipeline, emb)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_CHUNKS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = get_audio_embedding(pipeline, audio, start_idx, end_idx)
        run_pipeline(pipeline, emb)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    p50, p95 = np.percentile(times, [50, 95])
    print(f"\nchunk: {frame_num} frames gen, {slice_len} net new "
          f"({slice_len/fps*1000:.0f}ms of video)")
    print(f"gen time  p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  "
          f"budget={budget*1000:.0f}ms")
    print(f"RTF p95: {p95/budget:.2f}  (<1.0 sustains 25fps)")

    audio_lookahead = frame_num / fps
    ttfr = audio_lookahead + p50  # audio fill + first gen; excludes ASR/LLM/TTS/net
    print(f"avatar-side TTFR estimate: {ttfr*1000:.0f}ms "
          f"(={frame_num/fps*1000:.0f}ms audio lookahead + {p50*1000:.0f}ms gen)")

    if p95 < budget:
        print("VERDICT: SUSTAINS real-time at this profile.")
        return 0
    print("VERDICT: does NOT sustain — try FLASH_HEAD_PROFILE=balanced "
          "(or default), or a faster GPU.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
