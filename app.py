import gradio as gr
import subprocess
import os
import shutil

def generate(image, audio):
    if not image or not audio:
        return None
    
    img_path = "examples/gradio_input.png"
    aud_path = "examples/gradio_input.wav"
    shutil.copy(image, img_path)
    shutil.copy(audio, aud_path)
    
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
        return None
    files = sorted([f for f in os.listdir(results_dir) if f.endswith('.mp4')], 
                   key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    if files:
        return os.path.join(results_dir, files[-1])
    return None

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Image(type="filepath", label="Reference Image"), gr.Audio(type="filepath", label="Voice Audio")],
    outputs=gr.Video(label="Generated Video"),
    title="SoulX-FlashHead Live Demo",
    description="Oracle-guided Generation of Real-time Streaming Talking Heads (Lite Model - 96 FPS on RTX 4090)"
)

demo.launch(server_name="0.0.0.0", server_port=7860)