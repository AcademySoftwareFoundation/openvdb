# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import os

from fvdb import GaussianSplat3d, JaggedTensor, gaussian_render_jagged

device = "cuda:0"

# path for adding data to the test data REPOSITORY
test_data_path = Path(os.path.dirname(__file__)) / "../../../../../fvdb-test-data/unit_tests/gsplat/"
# path for adding to the local FVDB_TEST_DATA_DIR
# local_test_data_path = Path(os.getenv("FVDB_TEST_DATA_DIR")) / "unit_tests/gsplat/"

data_path = test_data_path / "test_garden_cropped.npz"


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


data = np.load(data_path)
means = torch.from_numpy(data["means3d"]).float().to(device)
quats = torch.from_numpy(data["quats"]).float().to(device)
scales = torch.from_numpy(data["scales"]).float().to(device)
opacities = torch.from_numpy(data["opacities"]).float().to(device)
colors = torch.from_numpy(data["colors"]).float().to(device)
cam_to_world_mats = torch.from_numpy(data["viewmats"]).float().to(device)
projection_mats = torch.from_numpy(data["Ks"]).float().to(device)
width = data["width"].item()
height = data["height"].item()

sh_degree = 3
sh_coeffs = torch.zeros((means.shape[0], (sh_degree + 1) ** 2, 3), device=device)
sh_coeffs[:, 0, :] = rgb_to_sh(colors)
sh_0 = sh_coeffs[:, 0:1, :].clone()
sh_n = sh_coeffs[:, 1:, :].clone()

gs3d = GaussianSplat3d(
    means=means,
    quats=quats,
    log_scales=torch.log(scales),
    logit_opacities=torch.logit(opacities),
    sh0=sh_0,
    shN=sh_n,
    requires_grad=True,
)

num_cameras = cam_to_world_mats.shape[0]
near_plane = 0.01
far_plane = 1e10

projected_gaussians = gs3d.project_gaussians_for_images(
    cam_to_world_mats,
    projection_mats,
    width,
    height,
    near_plane,
    far_plane,
    "perspective",
    sh_degree,
    min_radius_2d=0.0,
    eps_2d=1e-4,
    antialias=True,
)

print("Width: ", width)
print("Height: ", height)
print("Width in tiles: ", width / 16)
print("Height in tiles: ", height / 16)
print("Num cameras: ", num_cameras)
print("Tile offsets shape: ", projected_gaussians.tile_offsets.shape)
print("Tile gaussian ids shape: ", projected_gaussians.tile_gaussian_ids.shape)

image_dims = torch.tensor([width, height], device=device, dtype=torch.int32)

projected_data_to_save = {
    "means2d": projected_gaussians.means2d,
    "conics": projected_gaussians.conics,
    "colors": projected_gaussians.render_quantities,
    "opacities": projected_gaussians.opacities,
    "tile_offsets": projected_gaussians.tile_offsets,
    "tile_gaussian_ids": projected_gaussians.tile_gaussian_ids,
    "image_dims": image_dims,
}


def save_test_data(data):
    class Container(torch.nn.Module):
        def __init__(self, my_values):
            super().__init__()
            for key in my_values:
                setattr(self, key, my_values[key])

    container = torch.jit.script(Container(data))
    container.save("rasterize_forward_inputs_3cams.pt")


save_test_data(projected_data_to_save)
