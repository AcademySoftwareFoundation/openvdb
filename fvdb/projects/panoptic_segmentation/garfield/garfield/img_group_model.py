"""
Quick wrapper for Segment Anything Model
"""

from dataclasses import dataclass, field
from typing import Literal, Type, Union

import numpy as np
import torch
from nerfstudio.configs import base_config as cfg
from PIL import Image
from transformers import pipeline


@dataclass
class ImgGroupModelConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: ImgGroupModel)
    """target class to instantiate"""
    model_type: Literal["sam_fb", "sam_hf", "maskformer"] = "sam_fb"
    """
    Currently supports:
     - "sam_fb": Original SAM model (from facebook github)
     - "sam_hf": SAM model from huggingface
     - "maskformer": MaskFormer model from huggingface (experimental)
    """

    sam_model_type: str = ""
    sam_model_ckpt: str = ""
    sam_kwargs: dict = field(default_factory=lambda: {})
    "Arguments for SAM model (fb)."

    # # Settings used for the paper:
    # model_type="sam_fb",
    # sam_model_type="vit_h",
    # sam_model_ckpt="models/sam_vit_h_4b8939.pth",
    # sam_kwargs={
    #     "points_per_side": 32,  # 32 in original
    #     "pred_iou_thresh": 0.90,
    #     "stability_score_thresh": 0.90,
    # },

    device: Union[torch.device, str] = ("cpu",)


class ImgGroupModel:
    """
    Wrapper for 2D image segmentation models (e.g. MaskFormer, SAM)
    Original paper uses SAM, but we can use any model that outputs masks.
    The code currently assumes that every image has at least one group/mask.
    """

    def __init__(self, config: ImgGroupModelConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = self.config.device = self.kwargs["device"]
        self.model = None

        # also, assert that model_type doesn't have a "/" in it! Will mess with h5df.
        assert "/" not in self.config.model_type, "model_type cannot have a '/' in it!"

    def __call__(self, img: np.ndarray):
        # takes in range 0-255... HxWx3
        # For using huggingface transformer's SAM model
        if self.config.model_type == "sam_hf":
            if self.model is None:
                self.model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
            masks = masks["masks"]
            masks = sorted(masks, key=lambda x: x.sum())
            return masks

        elif self.config.model_type == "sam_fb":
            # For using the original SAM model
            if self.model is None:
                from segment_anything import (
                    SamAutomaticMaskGenerator,
                    sam_model_registry,
                )

                registry = sam_model_registry[self.config.sam_model_type]
                model = registry(checkpoint=self.config.sam_model_ckpt)
                model = model.to(device=self.config.device)
                self.model = SamAutomaticMaskGenerator(model=model, **self.config.sam_kwargs)
            masks = self.model.generate(img)
            masks = [m["segmentation"] for m in masks]  # already as bool
            masks = sorted(masks, key=lambda x: x.sum())
            return masks

        elif self.config.model_type == "maskformer":
            # For using another model (e.g., MaskFormer)
            if self.model is None:
                self.model = pipeline(model="facebook/maskformer-swin-large-coco", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img)
            masks = [(np.array(m["mask"]) != 0) for m in masks]
            masks = sorted(masks, key=lambda x: x.sum())
            return masks

        raise NotImplementedError(f"Model type {self.config.model_type} not implemented")
