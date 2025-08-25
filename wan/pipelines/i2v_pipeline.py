# Lightweight Diffusers-style wrapper for Wan I2V
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from PIL import Image
import torch

from ..image2video import WanI2V
from ..configs import WAN_CONFIGS, MAX_AREA_CONFIGS

@dataclass
class I2VResult:
    frames: torch.Tensor  # (C, F, H, W) after decode range [-1,1]
    seed: int

class WanI2VPipeline:
    """A small convenience wrapper mimicking a diffusers pipeline API.
    It handles model lazy loading and a callable interface.
    """
    def __init__(self, ckpt_dir: str, task: str = 'i2v-A14B', device: Union[str,int] = 0, **kwargs):
        assert task in WAN_CONFIGS, f"Unknown task {task}"
        self.cfg = WAN_CONFIGS[task]
        self.ckpt_dir = ckpt_dir
        self.device = device
        self._model: Optional[WanI2V] = None
        self._init_kwargs = kwargs

    def load_model(self):
        if self._model is None:
            self._model = WanI2V(
                config=self.cfg,
                checkpoint_dir=self.ckpt_dir,
                device_id=self.device if isinstance(self.device,int) else 0,
                **self._init_kwargs
            )
        return self._model

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        size_key: str = '720*1280',
        frame_num: Optional[int] = None,
        guide_scale: Union[float,Tuple[float,float]] = None,
        sample_solver: str = 'unipc',
        sample_steps: Optional[int] = None,
        sample_shift: Optional[float] = None,
        seed: int = -1,
        offload_model: bool = True,
        negative_prompt: str = '',
    ) -> I2VResult:
        self.load_model()
        if frame_num is None:
            frame_num = self.cfg.frame_num
        if sample_steps is None:
            sample_steps = self.cfg.sample_steps
        if sample_shift is None:
            sample_shift = self.cfg.sample_shift
        if guide_scale is None:
            guide_scale = self.cfg.sample_guide_scale
        # Ensure a concrete positive seed like CLI behavior
        if seed < 0:
            seed = random.randint(0, 2**63 - 1)
        video = self._model.generate(
            input_prompt=prompt,
            img=image,
            max_area=MAX_AREA_CONFIGS[size_key],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=offload_model,
        )
        return I2VResult(frames=video, seed=seed)

__all__ = ["WanI2VPipeline", "I2VResult"]
