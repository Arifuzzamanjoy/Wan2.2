# Wan 2.2 Gradio Interface

A web interface for Wan 2.2 Image-to-Video generation with optional depth estimation features.

## Features

- **Image → Video Generation**: Convert static images into dynamic videos using text prompts
- **Depth Estimation**: Optional depth map visualization for input images
- **Video Depth Extraction**: Extract depth information from uploaded videos
- **Full Parameter Control**: Adjust sampling steps, guidance scale, shift, seed, and more

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model Checkpoints**:
   Place your Wan 2.2 A14B model checkpoints in a directory structure like:
   ```
   ckpts/
   ├── low_noise_model/
   ├── high_noise_model/
   ├── Wan2.1_VAE.pth
   └── models_t5_umt5-xxl-enc-bf16.pth
   ```

3. **Launch Interface**:
   ```bash
   python gradio_app.py
   ```
   
   Or set checkpoint directory via environment:
   ```bash
   export WAN_CKPT_DIR=/path/to/your/checkpoints
   python gradio_app.py
   ```

4. **Access Interface**: Open http://localhost:7860 in your browser

## Interface Overview

### Image → Video Tab
- **Checkpoint Dir**: Path to model checkpoints
- **Target Max Size**: Output resolution (maintains input aspect ratio)
- **Frames**: Number of output frames (must be 4n+1, default 81)
- **Guide Scale**: Classifier-free guidance strength (default 3.5)
- **Sampling Steps**: Diffusion steps (default 40)
- **Shift**: Noise schedule parameter (default 5.0)
- **Seed**: Random seed for reproducibility (-1 for random)
- **Prompt**: Text description for video generation
- **Negative Prompt**: What to avoid in generation
- **Input Image**: Upload your source image
- **Return Depth**: Optionally generate depth map of input image

### Video Depth Extract Tab
- Upload a video to generate a grayscale depth visualization
- Useful for understanding scene geometry
- Currently visualization-only (not used for guidance)

## Alignment with Codebase

The interface is built on:
- `wan.pipelines.i2v_pipeline.WanI2VPipeline`: Diffusers-style wrapper around `WanI2V`
- `wan.configs`: Uses official config parameters and size constraints
- `wan.utils.utils.save_video`: Standard video output format
- Optional depth estimation via `transformers` DPT models

## Configuration Alignment

| Parameter | Config Default | UI Default | Notes |
|-----------|---------------|------------|-------|
| frame_num | 81 | 81 | Must be 4n+1 |
| sample_steps | 40 | 40 | Diffusion sampling steps |
| sample_shift | 5.0 | 5.0 | Noise schedule shift |
| sample_guide_scale | (3.5, 3.5) | 3.5 | Unified guidance scale |
| sample_fps | 16 | 16 | Output video framerate |

## Advanced Usage

### Custom Checkpoints
```python
from wan.pipelines.i2v_pipeline import WanI2VPipeline

pipeline = WanI2VPipeline(
    ckpt_dir="/path/to/checkpoints",
    task='i2v-A14B',
    device=0
)

result = pipeline(
    prompt="A cat playing in the garden",
    image=your_pil_image,
    size_key='720*1280',
    seed=42
)
```

### Programmatic Depth Estimation
```python
from gradio_app import estimate_depth

depth_map = estimate_depth(your_image)  # Returns numpy array
```

## Requirements

- CUDA-capable GPU (recommended)
- Python 3.8+
- PyTorch 2.4+
- 16GB+ VRAM for full resolution
- Optional: `transformers` for depth estimation

## Notes

- First run will download depth estimation models if enabled
- Large videos are processed in chunks (max 120 frames for depth)
- Model offloading can reduce VRAM usage at the cost of speed
- Generated videos are saved as MP4 files in the working directory

## Troubleshooting

- **CUDA OOM**: Enable model offloading or reduce frame count
- **Slow generation**: Disable model offloading if you have sufficient VRAM
- **Import errors**: Ensure all dependencies are installed in the correct environment
- **Depth estimation fails**: Install `transformers` or disable depth features
