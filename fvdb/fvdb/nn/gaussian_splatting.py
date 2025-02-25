# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from .. import (
    gaussian_render,
    gaussian_render_depth,
    precompute_gaussian_render_state,
    render_pixels_from_precomputed_gaussian_render_state,
    save_gaussian_ply,
)


class GaussianSplat3D(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        rgbs: torch.Tensor,
        initial_covariance_scale: float = 1.0,
        initial_opacity: float = 1.0,
        scene_size: float = 1.0,
        sh_degree: int = 3,
    ):
        super().__init__()

        self.scene_size = scene_size

        dist2_avg = (self._knn(means, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * initial_covariance_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        num_means = means.shape[0]
        quats = torch.rand((num_means, 4))  # [N, 4]
        opacities = torch.logit(torch.full((num_means,), initial_opacity))  # [N,]

        # TODO (Francis): Don't hardcode number of channels to 3 here
        _sh_and_colors = torch.zeros(((sh_degree + 1) ** 2, num_means, 3))  # [N, K, 3]
        _sh_and_colors[0, :, :] = self._rgb_to_sh(rgbs)

        # sh = nn.Parameter(_sh_and_colors)  # [N, K, 3]
        means = torch.nn.Parameter(means)  # [N, 3]
        scales = torch.nn.Parameter(scales)  # [N, 3]
        quats = torch.nn.Parameter(quats)  # [N, 4]
        opacities = torch.nn.Parameter(opacities)  # [N,]

        # FIXME (Francis): I don't like splitting these but during training we need
        #                  a seperate learning rate for each of them. I wonder if we can
        #                  just create a view for the optimizer but keep them in the same tensor
        sh_0 = torch.nn.Parameter(_sh_and_colors[:1, :, :])  # [1, N, 3]
        sh_n = torch.nn.Parameter(_sh_and_colors[1:, :, :])  # [K, N, 3]

        self._params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "sh0": sh_0,
                "shN": sh_n,
            }
        )
        self._default_lrs = {
            "means": 1.6e-4 * scene_size,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }

        self.sh_degree = sh_degree
        self._info_cache = {}

        # You can load Gaussians from a statedict with a different number of Gaussians than the current model.
        # This hook will resize the parameter tensors so that you can do this
        def _load_state_dict_hook(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            if len(missing_keys) > 0:
                raise ValueError(f"Missing keys {missing_keys}")

            for k, v in state_dict.items():
                if k.startswith("_params"):
                    ktrim = k[len("_params.") :]
                    if ktrim in self._params:
                        self._params[ktrim].data = torch.empty_like(v)
            self.clear_cache()

        self._register_load_state_dict_pre_hook(_load_state_dict_hook)

    @staticmethod
    def from_colmap(
        colmap_points: np.ndarray,
        colmap_points_rgb: np.ndarray,
        initial_covariance_scale: float = 1.0,
        initial_opacity: float = 1.0,
        scene_size: float = 1.0,
        sh_degree: int = 3,
    ) -> "GaussianSplat3D":
        points = torch.from_numpy(colmap_points).float()
        rgbs = torch.from_numpy(colmap_points_rgb / 255.0).float()
        return GaussianSplat3D(
            points,
            rgbs,
            initial_covariance_scale=initial_covariance_scale,
            initial_opacity=initial_opacity,
            scene_size=scene_size,
            sh_degree=sh_degree,
        )

    @staticmethod
    def from_random(
        num_means: int,
        means_range: float = 1.0,
        initial_covariance_scale: float = 1.0,
        initial_opacity: float = 1.0,
        scene_size: float = 1.0,
        sh_degree: int = 3,
    ) -> "GaussianSplat3D":
        # [N, 3] in [-scene_size * means_range, scene_size * means_range]
        points = scene_size * means_range * (torch.rand([num_means, 3]) * 2.0 - 1.0)
        rgbs = torch.rand([num_means, 3])  # [N, 3]
        return GaussianSplat3D(
            points,
            rgbs,
            initial_covariance_scale=initial_covariance_scale,
            initial_opacity=initial_opacity,
            scene_size=scene_size,
            sh_degree=sh_degree,
        )

    def set_spherical_harmonic_coeffs(self, new_sh_coeffs: torch.Tensor):
        self._params["sh0"].data = new_sh_coeffs[:1, :, :]
        self._params["shN"].data = new_sh_coeffs[1:, :, :]
        k = new_sh_coeffs.shape[1]
        self.sh_degree = int(math.sqrt(k) - 1)
        self.clear_cache()

    @property
    def num_gaussians(self) -> int:
        return self._params["means"].shape[0]

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._params["opacities"])

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._params["scales"])

    @property
    def means(self) -> torch.Tensor:
        return self._params["means"]

    @staticmethod
    def _knn(x: torch.Tensor, k: int = 4) -> torch.Tensor:
        x_np = x.cpu().numpy()
        model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
        distances, _ = model.kneighbors(x_np)
        return torch.from_numpy(distances).to(x)

    @staticmethod
    def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    def clear_cache(self):
        self._info_cache = {}

    def save_ply(self, path: str):

        means = self._params["means"].detach()
        scales = self._params["scales"].detach()
        opacities = self._params["opacities"].detach()
        quats = self._params["quats"].detach()
        sh0 = self._params["sh0"].detach()
        shN = self._params["shN"].detach()
        sh = torch.cat([sh0, shN], 0).permute(1, 0, 2).detach()

        save_gaussian_ply(path, means, quats, scales, opacities, sh)

    def render_rgb_and_depth(
        self,
        image_w: int,
        image_h: int,
        extrinsics_mats: torch.Tensor,
        intrinsics_mats: torch.Tensor,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        sh_degree: int = -1,
        eps_2d: float = 0.3,
        radius_clip: float = 0.0,
        tile_size: int = 16,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        ortho: bool = False,
    ):
        rgbd, alphas, _ = self(
            image_w=image_w,
            image_h=image_h,
            extrinsics_mats=extrinsics_mats,
            intrinsics_mats=intrinsics_mats,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            eps_2d=eps_2d,
            radius_clip=radius_clip,
            tile_size=tile_size,
            image_crop=False,
            render_depth=True,
            rasterize_mode=rasterize_mode,
            cache_info=False,
            depth_only=False,
            ortho=ortho,
        )
        rgb = rgbd[..., :3]  # [B, H, W, 1]
        depth = rgbd[..., 3:4] / alphas.clamp(min=1e-10)  # [B, H, W, 1]

        return rgb, depth

    def render_depth_points(
        self,
        image_w: int,
        image_h: int,
        extrinsics_mats: torch.Tensor,
        intrinsics_mats: torch.Tensor,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        sh_degree: int = -1,
        eps_2d: float = 0.3,
        radius_clip: float = 0.0,
        tile_size: int = 16,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        ortho: bool = False,
    ):
        depth, alphas, _ = self(
            image_w=image_w,
            image_h=image_h,
            extrinsics_mats=extrinsics_mats,
            intrinsics_mats=intrinsics_mats,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            eps_2d=eps_2d,
            radius_clip=radius_clip,
            tile_size=tile_size,
            image_crop=False,
            render_depth=True,
            rasterize_mode=rasterize_mode,
            cache_info=False,
            depth_only=True,
            ortho=ortho,
        )
        depth = depth / alphas.clamp(min=1e-10)  # [B, H, W, 1]

        device, dtype = depth.device, depth.dtype
        row, col = torch.meshgrid(
            torch.arange(0, image_h, device=device, dtype=dtype),
            torch.arange(0, image_w, device=device, dtype=dtype),
            indexing="ij",
        )
        cam_pts = torch.stack([col, row, torch.ones_like(row)])  # [3, H, W]
        cam_pts = torch.linalg.inv(intrinsics_mats) @ cam_pts.view(3, -1)  # [B, 3, H * W]
        cam_pts = cam_pts.permute(0, 2, 1).reshape(depth.shape[0], image_h, image_w, 3) * depth  # [B, H, W, 3]

        return depth, cam_pts

    def forward(
        self,
        image_w: int,
        image_h: int,
        extrinsics_mats: torch.Tensor,
        intrinsics_mats: torch.Tensor,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        sh_degree: int = -1,
        eps_2d: float = 0.3,
        radius_clip: float = 0.0,
        tile_size: int = 16,
        image_crop=None,
        render_depth: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        cache_info: bool = False,
        depth_only: bool = False,
        ortho: bool = False,
    ):
        if rasterize_mode not in ["classic", "antialiased"]:
            raise ValueError(f"Invalid rasterize_mode {rasterize_mode}")

        # if camera_model not in ["pinhole", "ortho", "fisheye"]:
        #     raise ValueError(f"Invalid camera_model {camera_model}")
        sh_degree = self.sh_degree if sh_degree < 0 else sh_degree
        if sh_degree > self.sh_degree:
            raise ValueError(f"sh_degree {sh_degree} is larger than the maximum {self.sh_degree}")

        means = self._params["means"]  # [N, 3]
        quats = self._params["quats"]  # [N, 4]
        scales = torch.exp(self._params["scales"])  # [N, 3]
        opacities = torch.sigmoid(self._params["opacities"])  # [N,]
        # FIXME (Francis): It sucks that we need to concatenate here
        #                  but we do this so we can optimize
        #                  these parameters seperately. I wonder if we can
        #                  fix this
        sh0 = self._params["sh0"]  # [1, N, 3]
        shN = self._params["shN"]  # [K, N, 3]
        sh = sh0 if shN.numel() == 0 or sh_degree == 0 else torch.cat([sh0, shN], 0)  # [N, K, 3]

        image_crop = (0, 0, image_w, image_h) if image_crop is None else image_crop
        if cache_info:
            if len(self._info_cache) == 0:
                self._info_cache = precompute_gaussian_render_state(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    sh_coeffs=sh,
                    viewmats=extrinsics_mats,
                    Ks=intrinsics_mats,
                    image_width=image_w,
                    image_height=image_h,
                    eps2d=eps_2d,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=radius_clip,
                    sh_degree_to_use=sh_degree,
                    tile_size=tile_size,
                    antialias=(rasterize_mode == "antialiased"),
                    render_depth_channel=render_depth,
                    ortho=ortho,
                )
            colors, alphas = render_pixels_from_precomputed_gaussian_render_state(
                self._info_cache["means2d"],
                self._info_cache["conics"],
                self._info_cache["colors"],
                self._info_cache["opacities"],
                image_crop[2],
                image_crop[3],
                image_crop[0],
                image_crop[1],
                tile_size,
                self._info_cache["tile_offsets"],
                self._info_cache["tile_gaussian_ids"],
            )
            info = self._info_cache

            if depth_only:
                # FIXME (Francis): Implement this properly
                return colors[..., -1:], alphas, info
        else:
            if depth_only:
                colors, alphas, info = gaussian_render_depth(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    viewmats=extrinsics_mats,
                    Ks=intrinsics_mats,
                    image_width=image_w,
                    image_height=image_h,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    tile_size=tile_size,
                    radius_clip=radius_clip,
                    eps2d=eps_2d,
                    antialias=(rasterize_mode == "antialiased"),
                    return_debug_info=True,
                    ortho=ortho,
                )
            else:
                colors, alphas, info = gaussian_render(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    sh_coeffs=sh,
                    viewmats=extrinsics_mats,
                    Ks=intrinsics_mats,
                    image_width=image_w,
                    image_height=image_h,
                    eps2d=eps_2d,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=radius_clip,
                    sh_degree_to_use=sh_degree,
                    tile_size=tile_size,
                    antialias=(rasterize_mode == "antialiased"),
                    render_depth_channel=render_depth,
                    return_debug_info=True,
                    ortho=ortho,
                )

        # if render_mode in ["ED", "RGB+ED"]:
        #     assert False, "ass"
        #     colors = torch.cat(
        #         [
        #             colors[..., :-1],
        #             colors[..., -1:] / alphas.clamp(min=1e-10),
        #         ],
        #         dim=-1,
        #     )

        info["width"] = image_w
        info["height"] = image_h
        info["tile_size"] = tile_size
        info["n_cameras"] = extrinsics_mats.shape[0]
        info["tile_width"] = math.ceil(image_w / float(tile_size))
        info["tile_height"] = math.ceil(image_h / float(tile_size))
        info["gaussian_ids"] = None

        return colors, alphas, info
