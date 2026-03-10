from fastapi import FastAPI, UploadFile, File
import subprocess
import shutil
import os
import time
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

@app.post("/generate")
async def generate_video(image: UploadFile = File(...), audio: UploadFile = File(...)):
    img_path = f"examples/api_in_{int(time.time())}.png"
    aud_path = f"examples/api_in_{int(time.time())}.wav"
    
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    with open(aud_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    cmd = [
        "python", "generate_video.py",
        "--ckpt_dir", "models/SoulX-FlashHead-1_3B",
        "--wav2vec_dir", "models/wav2vec2-base-960h",
        "--model_type", "lite",
        "--cond_image", img_path,
        "--audio_path", aud_path,
        "--audio_encode_mode", "stream"
    ]
    
    subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    
    results_dir = "sample_results"
    if not os.path.exists(results_dir):
        return {"error": "Generation failed"}
        
    files = sorted([f for f in os.listdir(results_dir) if f.endswith('.mp4')], 
                   key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
                   
    if files:
        return FileResponse(os.path.join(results_dir, files[-1]))
    return {"error": "No video generated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
