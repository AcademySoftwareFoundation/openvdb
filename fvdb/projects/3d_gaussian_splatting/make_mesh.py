# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os

import numpy as np
import open3d as o3d
import torch
import torch.utils.data
import tqdm
import tyro
from datasets import ColmapDataset, ColmapParser

from fvdb.nn.gaussian_splatting import GaussianSplat3D


def to_cam_open3d(image_width, image_height, K, world_view_transform):

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_width,
        height=image_height,
        cx=K[0, 2].item(),
        cy=K[1, 2].item(),
        fx=K[0, 0].item(),
        fy=K[1, 1].item(),
    )

    extrinsic = np.asarray((world_view_transform).cpu().numpy())
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic

    return camera


@torch.inference_mode()
def mesh_gs(
    model: GaussianSplat3D,
    dataset: ColmapDataset,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    device: str = "cuda",
):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for data in tqdm.tqdm(dataloader):
        img = data["image"].to(device).squeeze()
        K = data["K"].to(device).squeeze()
        cam_to_world = data["camtoworld"].to(device).squeeze()
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        rgbd, alphas, info = model(
            image_w=img.shape[1],
            image_h=img.shape[0],
            extrinsics_mats=world_to_cam.unsqueeze(0),
            intrinsics_mats=K.unsqueeze(0),
            rasterize_mode="classic",
            render_depth=True,
        )

        rgb = rgbd[0, ..., 0:3]

        # img = rgbd[0,...,0:3].cpu().detach().numpy()*255.0
        # imageio.imwrite('rgb.jpg', img.astype(np.uint8))

        depth = rgbd[0, ..., 3:4]

        depth = depth / alphas[0].clamp(min=1e-10)

        # img = depth.cpu().detach().numpy().squeeze()
        # imageio.imwrite('depth.tif', img.astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(rgb.cpu().numpy() * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth.cpu().numpy().squeeze(), order="C")),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        cam_o3d = to_cam_open3d(img.shape[1], img.shape[0], K, world_to_cam)
        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    return mesh


def main(
    checkpoint_path: str,
    data_path: str,
    results_path: str,
    voxel_size: float = 0.5,  # TODO add logic to auto-tune these parameters
    sdf_trunc: float = 2.0,
    depth_trunc: float = 300.0,
    data_scale_factor: int = 1,
    device: str = "cuda",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianSplat3D(torch.rand([8, 3]), torch.rand([8, 3])).to(device)
    model.load_state_dict(checkpoint["splats"])

    parser = ColmapParser(data_path, test_every=1, factor=data_scale_factor)
    dataset = ColmapDataset(parser, split="test")

    print("extracting mesh")
    mesh = mesh_gs(model, dataset, device=device, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    print("saving mesh")
    o3d.io.write_triangle_mesh(os.path.join(results_path, "mesh.ply"), mesh)
    print("mesh saved")


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
