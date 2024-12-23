"""
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
from jaxtyping import Int
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from rich.progress import Console
from torch import Tensor

CONSOLE = Console(width=120)


@dataclass
class GarfieldPixelSamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: GarfieldPixelSampler)
    num_rays_per_image: int = 256  # different from num_rays_per_batch


class GarfieldPixelSampler(PixelSampler):
    def __init__(self, config: GarfieldPixelSamplerConfig, **kwargs):
        self.num_rays_per_image = (
            1  # Start with 1 (i.e., no indices grouped by image. Will be updated later in pipeline)
        )
        super().__init__(config, **kwargs)

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Equivalent to PixelSampler, but with the following differences when `grouping_enabled` is True:
        - `batch_size` is expected to be a multiple of `num_rays_per_image`.
        - Indices are grouped by image, with `num_rays_per_image` rays per image.
        [
            [image_0, x_0, y_0], [image_0, x_1, y_1], ..., [image_0, x_n, y_n],
            [image_1, x_0, y_0], [image_1, x_1, y_1], ..., [image_1, x_n, y_n],
            ...
        ]
        """
        if isinstance(mask, Tensor):
            raise NotImplementedError("GarfieldPixelSampler does not support masks yet.")

        indices = super().sample_method(
            batch_size,
            num_images,
            image_height,
            image_width,
            mask,
            device,
        )

        if self.num_rays_per_image == 1:
            return indices

        sub_bs = batch_size // (self.num_rays_per_image)
        if (sub_bs * self.num_rays_per_image) != batch_size:
            raise ValueError(
                f"Batch size {batch_size} is not a multiple of num_rays_per_image {self.num_rays_per_image}."
            )

        image_indices = torch.randint(low=0, high=num_images, size=(sub_bs,))
        indices[:, 0] = image_indices.repeat_interleave(self.num_rays_per_image)

        return indices
