"""
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from rich.progress import Console
from typing_extensions import TypeVar

CONSOLE = Console(width=120)

import os
import os.path as osp

import h5py
import numpy as np
from garfield.garfield_pixel_sampler import GarfieldPixelSampler
from garfield.img_group_model import ImgGroupModel, ImgGroupModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class GarfieldDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: GarfieldDataManager)
    """The datamanager class to use."""
    img_group_model: ImgGroupModelConfig = field(default_factory=lambda: ImgGroupModelConfig())
    """The SAM model to use. This can be any other model that outputs masks..."""


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class GarfieldDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """
    Tacking on grouping info to the normal VanillaDataManager.
    """

    config: GarfieldDataManagerConfig
    train_pixel_sampler: Optional[GarfieldPixelSampler] = None

    def __init__(
        self,
        config: GarfieldDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.img_group_model: ImgGroupModel = self.config.img_group_model.setup(device=self.device)

        # This is where all the group data + statistics is stored.
        # Note that this can get quite big (~10GB if 300 images, ...)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        self.sam_data_path = Path(cache_dir) / "sam_data.hdf5"

        self.pixel_level_keys = None
        self.scale_3d = None
        self.group_cdf = None
        self.scale_3d_statistics = None

    def load_sam_data(self) -> bool:
        """
        Loads the SAM data (masks, 3D scales, etc.) through hdf5.
        If the file doesn't exist, returns False.
        """
        prefix = self.img_group_model.config.model_type
        if osp.exists(self.sam_data_path):
            sam_data = h5py.File(self.sam_data_path, "r")
            if prefix not in sam_data.keys():
                return False

            sam_data = sam_data[prefix]

            pixel_level_keys_list, scales_3d_list, group_cdf_list = [], [], []

            num_entries = len(sam_data["pixel_level_keys"].keys())
            for i in range(num_entries):
                pixel_level_keys_list.append(torch.from_numpy(sam_data["pixel_level_keys"][str(i)][...]))
            self.pixel_level_keys = torch.nested.nested_tensor(pixel_level_keys_list)
            del pixel_level_keys_list

            for i in range(num_entries):
                scales_3d_list.append(torch.from_numpy(sam_data["scale_3d"][str(i)][...]))
            self.scale_3d = torch.nested.nested_tensor(scales_3d_list)
            self.scale_3d_statistics = torch.cat(scales_3d_list)
            del scales_3d_list

            for i in range(num_entries):
                group_cdf_list.append(torch.from_numpy(sam_data["group_cdf"][str(i)][...]))
            self.group_cdf = torch.nested.nested_tensor(group_cdf_list)
            del group_cdf_list

            return True

        return False

    def save_sam_data(self, pixel_level_keys, scale_3d, group_cdf):
        """Save the SAM grouping data to hdf5."""
        prefix = self.img_group_model.config.model_type
        # make the directory if it doesn't exist
        if not osp.exists(self.sam_data_path.parent):
            os.makedirs(self.sam_data_path.parent)

        # Append, not overwrite -- in case of multiple runs with different settings.
        with h5py.File(self.sam_data_path, "a") as f:
            for i in range(len(pixel_level_keys)):
                f.create_dataset(f"{prefix}/pixel_level_keys/{i}", data=pixel_level_keys[i])
                f.create_dataset(f"{prefix}/scale_3d/{i}", data=scale_3d[i])
                f.create_dataset(f"{prefix}/group_cdf/{i}", data=group_cdf[i])

    @staticmethod
    def create_pixel_mask_array(masks: torch.Tensor):
        """
        Create per-pixel data structure for grouping supervision.
        pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
        where Area(m1) < Area(m2) < ... (sorted by area).
        """
        max_masks = masks.sum(dim=0).max().item()
        image_shape = masks.shape[1:]
        pixel_mask_array = torch.full((max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int).to(masks.device)

        for m, mask in enumerate(masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_mask_array[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_mask_array[i][right_index]) != 0:
                    pixel_mask_array[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

        return pixel_mask_array

    def _calculate_3d_groups(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        point: torch.Tensor,
        max_scale: float = 2.0,
    ):
        """
        Calculate the set of groups and their 3D scale for each pixel, and the cdf.
        Returns:
            - pixel_level_keys: [H, W, max_masks]
            - scale: [num_masks, 1]
            - mask_cdf: [H, W, max_masks]
        max_masks is the maximum number of masks that was assigned to a pixel in the image,
         padded with -1s. mask_cdf does *not* include the -1s.
        Refer to the main paper for more details.
        """
        image_shape = rgb.shape[:2]
        depth = depth.view(-1, 1)  # (H*W, 1)
        point = point.view(-1, 3)  # (H*W, 3)

        def helper_return_no_masks():
            # Fail gracefully when no masks are found.
            # Create dummy data (all -1s), which will be ignored later.
            # See: `get_loss_dict_group` in `garfield_model.py`
            pixel_level_keys = torch.full((image_shape[0], image_shape[1], 1), -1, dtype=torch.int)
            scale = torch.Tensor([0.0]).view(-1, 1)
            mask_cdf = torch.full((image_shape[0], image_shape[1], 1), 1, dtype=torch.float)
            return (pixel_level_keys, scale, mask_cdf)

        # Calculate SAM masks
        masks = self.img_group_model((rgb.numpy() * 255).astype(np.uint8))

        # If no masks are found, return dummy data.
        if len(masks) == 0:
            return helper_return_no_masks()

        sam_mask = []
        scale = []

        # For all 2D groups,
        # 1) Denoise the masks (through eroding)
        all_masks = torch.stack(
            # [torch.from_numpy(_["segmentation"]).to(self.device) for _ in masks]
            [torch.from_numpy(_).to(self.device) for _ in masks]
        )
        # erode all masks using 3x3 kernel
        eroded_masks = torch.conv2d(
            all_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        # 2) Calculate 3D scale
        # Don't include groups with scale > max_scale (likely to be too noisy to be useful)
        for i in range(len(masks)):
            curr_mask = eroded_masks[i]
            curr_mask = curr_mask.flatten()
            curr_points = point[curr_mask]
            extent = (curr_points.std(dim=0) * 2).norm()
            if extent.item() < max_scale:
                sam_mask.append(curr_mask.reshape(image_shape))
                scale.append(extent.item())

        # If no masks are found, after postprocessing, return dummy data.
        if len(sam_mask) == 0:
            return helper_return_no_masks()

        sam_mask = torch.stack(sam_mask)  # (num_masks, H, W)
        scale = torch.Tensor(scale).view(-1, 1).to(self.device)  # (num_masks, 1)

        # Calculate "pixel level keys", which is a 2D array of shape (H, W, max_masks)
        # Each pixel has a list of group indices that it belongs to, in order of increasing scale.
        pixel_level_keys = self.create_pixel_mask_array(sam_mask).long()  # (H, W, max_masks)

        # Calculate group sampling CDF, to bias sampling towards smaller groups
        # Be careful to not include -1s in the CDF (padding, or unlabeled pixels)
        # Inversely proportional to log of mask size.
        mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
        mask_sorted = torch.argsort(counts)
        mask_inds, counts = mask_inds[mask_sorted], counts[mask_sorted]
        counts[0] = 0  # don't include -1
        probs = counts / counts.sum()  # [-1, 0, ...]
        mask_probs = torch.gather(probs, 0, pixel_level_keys.reshape(-1) + 1).view(pixel_level_keys.shape)
        mask_log_probs = torch.log(mask_probs)
        never_masked = mask_log_probs.isinf()
        mask_log_probs[never_masked] = 0.0
        mask_log_probs = mask_log_probs / (mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6)
        mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
        mask_cdf[never_masked] = 1.0

        return (pixel_level_keys.cpu(), scale.cpu(), mask_cdf.cpu())

    def next_group(self, ray_bundle: RayBundle, batch: Dict[str, Any]):
        """Returns the rays' mask and 3D scales for grouping.
        We add to `batch` the following:
            - "mask_id": [batch_size,]
            - "scale": [batch_size,]
            - "nPxImg": int == `num_rays_per_image`
        This function also adds `scale` to `ray_bundle.metadata`.

        We're using torch nested tensors -- this means that it's difficult to index into them.
        At least now, it seems possible to index normally into a leaf tensor.
        """
        indices = batch["indices"].long().detach().cpu()
        npximg = self.train_pixel_sampler.num_rays_per_image
        img_ind = indices[:, 0]
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        # sampled_imgs = img_ind[::npximg]
        mask_id = torch.zeros((indices.shape[0],), device=self.device)
        scale = torch.zeros((indices.shape[0],), device=self.device)

        random_vec_sampling = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)
        random_vec_densify = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)

        for i in range(0, indices.shape[0], npximg):
            img_idx = img_ind[i]

            # Use `random_vec` to choose a group for each pixel.
            per_pixel_index = self.pixel_level_keys[img_idx][x_ind[i : i + npximg], y_ind[i : i + npximg]]
            random_index = torch.sum(
                random_vec_sampling.view(-1, 1) > self.group_cdf[img_idx][x_ind[i : i + npximg], y_ind[i : i + npximg]],
                dim=-1,
            )

            # `per_pixel_index` encodes the list of groups that each pixel belongs to.
            # If there's only one group, then `per_pixel_index` is a 1D tensor
            # -- this will mess up the future `gather` operations.
            if per_pixel_index.shape[-1] == 1:
                per_pixel_mask = per_pixel_index.squeeze()
            else:
                per_pixel_mask = torch.gather(per_pixel_index, 1, random_index.unsqueeze(-1)).squeeze()
                per_pixel_mask_ = torch.gather(
                    per_pixel_index,
                    1,
                    torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
                ).squeeze()

            mask_id[i : i + npximg] = per_pixel_mask.to(self.device)

            # interval scale supervision
            curr_scale = self.scale_3d[img_idx][per_pixel_mask]
            curr_scale[random_index == 0] = (
                self.scale_3d[img_idx][per_pixel_mask][random_index == 0] * random_vec_densify[random_index == 0]
            )
            for j in range(1, self.group_cdf[img_idx].shape[-1]):
                if (random_index == j).sum() == 0:
                    continue
                curr_scale[random_index == j] = (
                    self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    + (
                        self.scale_3d[img_idx][per_pixel_mask][random_index == j]
                        - self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    )
                    * random_vec_densify[random_index == j]
                )
            scale[i : i + npximg] = curr_scale.squeeze().to(self.device)

        batch["mask_id"] = mask_id
        batch["scale"] = scale
        batch["nPxImg"] = npximg
        ray_bundle.metadata["scale"] = batch["scale"]
