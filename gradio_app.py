import os
import gradio as gr
import torch
from PIL import Image
from typing import Optional
import imageio
import numpy as np

from wan.pipelines.i2v_pipeline import WanI2VPipeline
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video

# Depth estimation optional
try:
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    _HAS_DEPTH = True
except Exception:
    _HAS_DEPTH = False

_depth_models = {}

def get_depth_model(model_name: str = "Intel/dpt-hybrid-midas", device: str = "cuda"):
    if model_name not in _depth_models:
        feat = DPTFeatureExtractor.from_pretrained(model_name)
        mod = DPTForDepthEstimation.from_pretrained(model_name).to(device)
        _depth_models[model_name] = (feat, mod)
    return _depth_models[model_name]


def estimate_depth(image: Image.Image, model_name: str = "Intel/dpt-hybrid-midas"):
    if not _HAS_DEPTH:
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat, mod = get_depth_model(model_name, device)
    inputs = feat(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = mod(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=image.size[::-1], mode='bicubic', align_corners=False)
    depth = pred.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def extract_depth_from_video(video_path: str, model_name: str = "Intel/dpt-hybrid-midas", max_frames: int = 120):
    """Return path to saved depth video (grayscale) for given input video."""
    if not _HAS_DEPTH:
        return None, "Depth model not available (install transformers)."
    if not os.path.exists(video_path):
        return None, f"Video not found: {video_path}"
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get('fps', 16)
    frames = []
    try:
        for idx, frame in enumerate(reader):
            if idx >= max_frames:
                break
            pil = Image.fromarray(frame).convert("RGB")
            d = estimate_depth(pil, model_name=model_name)
            if d is None:
                return None, "Depth estimation failed."
            d_img = (d * 255).astype(np.uint8)
            d_img = np.stack([d_img]*3, axis=-1)  # RGB grayscale
            frames.append(d_img)
    finally:
        reader.close()
    if not frames:
        return None, "No frames processed."
    depth_path = os.path.splitext(video_path)[0] + "_depth.mp4"
    writer = imageio.get_writer(depth_path, fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    return depth_path, f"Depth video saved: {depth_path}"

_pipeline: Optional[WanI2VPipeline] = None

def get_pipeline(ckpt_dir: str):
    global _pipeline
    if _pipeline is None or _pipeline.ckpt_dir != ckpt_dir:
        _pipeline = WanI2VPipeline(ckpt_dir=ckpt_dir, task='i2v-A14B')
        _pipeline.load_model()
    return _pipeline


def generate(ckpt_dir, prompt, image, size, frame_num, guide_scale, sample_steps, sample_shift, seed, offload_model, negative_prompt, return_depth):
    if image is None:
        return None, None, "Please upload an image"
    pipe = get_pipeline(ckpt_dir)
    res = pipe(
        prompt=prompt,
        image=image,
        size_key=size,
        frame_num=frame_num if frame_num else None,
        guide_scale=guide_scale,
        sample_steps=sample_steps if sample_steps else None,
        sample_shift=sample_shift if sample_shift else None,
        seed=seed if seed >= 0 else -1,
        offload_model=offload_model,
        negative_prompt=negative_prompt or '',
    )
    save_path = f"i2v_{res.seed}.mp4"
    save_video(res.frames[None], save_path, fps=pipe.cfg.sample_fps, nrow=1)
    depth_map = estimate_depth(image) if (return_depth and _HAS_DEPTH) else None
    return save_path, depth_map, f"Done. Saved to {save_path}"


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Wan 2.2 Image ↦ Video (A14B) + Depth Tools")
        with gr.Tabs():
            with gr.Tab("Image → Video"):
                with gr.Row():
                    ckpt_dir = gr.Textbox(label="Checkpoint Dir", value=os.environ.get('WAN_CKPT_DIR','./ckpts'))
                    size = gr.Dropdown(choices=SUPPORTED_SIZES['i2v-A14B'], value='720*1280', label='Target Max Size (area)')
                    frame_num = gr.Slider(5, 161, value=81, step=4, label='Frames (4n+1)')
                    guide_scale = gr.Slider(0.1, 15.0, value=3.5, step=0.1, label='Guide Scale')
                with gr.Row():
                    sample_steps = gr.Slider(1, 60, value=40, step=1, label='Sampling Steps')
                    sample_shift = gr.Slider(0.0, 8.0, value=5.0, step=0.1, label='Shift')
                    seed = gr.Number(value=-1, label='Seed (-1 random)', precision=0)
                    offload_model = gr.Checkbox(label='Offload Model', value=True)
                prompt = gr.Textbox(label='Prompt', lines=4, value="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.")
                negative_prompt = gr.Textbox(label='Negative Prompt', lines=2)
                image = gr.Image(label='Input Image', type='pil')
                return_depth = gr.Checkbox(label='Return Depth (if depth model available)', value=False)
                run_btn = gr.Button("Generate")
                video = gr.Video(label='Generated Video')
                depth_output = gr.Image(label='Depth Map (optional)')
                status = gr.Textbox(label='Status')

                run_btn.click(
                    fn=generate,
                    inputs=[ckpt_dir, prompt, image, size, frame_num, guide_scale, sample_steps, sample_shift, seed, offload_model, negative_prompt, return_depth],
                    outputs=[video, depth_output, status]
                )
            with gr.Tab("Video Depth Extract"):
                gr.Markdown("Upload a video to produce a depth visualization (grayscale). Not used as guidance yet.")
                in_vid = gr.Video(label='Input Video')
                depth_btn = gr.Button("Extract Depth")
                out_depth_vid = gr.Video(label='Depth Video')
                depth_status = gr.Textbox(label='Status')

                def depth_video_interface(video_file):
                    if video_file is None:
                        return None, "Please provide a video." 
                    depth_path, msg = extract_depth_from_video(video_file)
                    return depth_path, msg

                depth_btn.click(depth_video_interface, inputs=[in_vid], outputs=[out_depth_vid, depth_status])
    return demo

if __name__ == '__main__':
    demo = build_demo()
    demo.launch(server_name='0.0.0.0', server_port=7860)
