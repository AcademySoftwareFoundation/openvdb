# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import point_cloud_utils as pcu
import torch
import torch.utils.data
import tqdm
import tyro
from datasets import ColmapDataset, ColmapParser
from skimage import feature, morphology

from fvdb import GaussianSplat3d


def render_depth_pointcloud(
    model: GaussianSplat3d,
    world_to_cam_mats: torch.Tensor,
    projection_mats: torch.Tensor,
    image_w: int,
    image_h: int,
    near: float,
    far: float,
):
    """
    Generate a point cloud (in camera space) by rendering a batch of depth image from a GaussianSplat3d model.

    Args:
        model: GaussianSplat3d model.
        world_to_cam_mats: [B, 4, 4] tensor of world-to-camera matrices.
        projection_mats: [B, 4, 4] tensor of projection matrices.
        image_w: Image width.
        image_h: Image height.
        near: Near clipping plane.
        far: Far clipping plane.

    Returns:
        depths: [B, H, W] tensor of depths.
        cam_pts: [B, H, W, 3] tensor of camera-space points
    """
    depths, alphas = model.render_depths(world_to_cam_mats, projection_mats, image_w, image_h, near, far)

    depths = depths / alphas.clamp(min=1e-10)

    device, dtype = depths.device, depths.dtype
    row, col = torch.meshgrid(
        torch.arange(0, image_h, device=device, dtype=dtype),
        torch.arange(0, image_w, device=device, dtype=dtype),
        indexing="ij",
    )
    cam_pts = torch.stack([col, row, torch.ones_like(row)])  # [3, H, W]
    cam_pts = torch.linalg.inv(projection_mats) @ cam_pts.view(3, -1)  # [B, 3, H * W]
    cam_pts = cam_pts.permute(0, 2, 1).reshape(depths.shape[0], image_h, image_w, 3) * depths  # [B, H, W, 3]

    return depths, cam_pts


@torch.inference_mode()
def accumulated_point_cloud(
    model: GaussianSplat3d,
    dataset: ColmapDataset,
    canny_edge_std: float = 1.0,
    dilation_amt: int = 5,
    downsample_factor: int = 1,
    device: str = "cuda",
):
    """
    Generate a point cloud from a GaussianSplat3d scene by rendering depth points for every image in a dataset and
    concatenating those points into a single point cloud.
    We use a canny edge detector on the depth images to identify the silhouette of the scene and remove points at
    sharp boundaries which are often noisy.

    Args:
        model: GaussianSplat3d model.
        dataset: A ColmapDataset.
        canny_edge_std: Standard deviation of the Canny edge detector used to remove points at sharp boundaries.
        dilation_amt: Amount of dilation to apply to the Canny edge detector.
        downsample_factor: Downsample factor for the depth images.
        device: Device to use.

    Returns:
        accumulated_pts: A [num_points, 3] shaped tensor of accumulated points.
        accumulated_pts_colors: A [num_points, 3] shaped tensor of accumulated RGB point colors.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    accumulated_pts = []
    accumulated_pts_colors = []

    for data in tqdm.tqdm(dataloader):
        img = data["image"].to(device).squeeze()
        projection = data["K"].to(device).squeeze()
        cam_to_world = data["camtoworld"].to(device).squeeze()
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        depth, cam_pts = render_depth_pointcloud(
            model,
            world_to_cam.unsqueeze(0),
            projection.unsqueeze(0),
            img.shape[1],
            img.shape[0],
            near=0.1,
            far=1e10,
        )

        depth, cam_pts = depth.squeeze(), cam_pts.squeeze()

        mask = torch.tensor(
            morphology.dilation(
                feature.canny(depth.cpu().numpy(), sigma=canny_edge_std),
                footprint=np.ones((dilation_amt, dilation_amt)),
            )
            == 0,
            device=device,
        )

        mask = mask[::downsample_factor, ::downsample_factor]
        gt_img_clrs = img[::downsample_factor, ::downsample_factor].to(torch.float32) / 255.0
        depth = depth[::downsample_factor, ::downsample_factor].to(torch.float32)
        cam_pts = cam_pts[::downsample_factor, ::downsample_factor]

        world_pts = torch.cat(
            [cam_pts.permute(2, 0, 1), torch.ones(1, cam_pts.shape[0], cam_pts.shape[1]).to(cam_pts)], dim=0
        )
        world_pts = (cam_to_world @ world_pts.view(4, -1)).view(*world_pts.shape)  # [4, H, W]
        world_pts = (world_pts / world_pts[3].unsqueeze(0))[:3]  # [3, H, W]

        accumulated_pts.append(world_pts.permute(1, 2, 0)[mask].view(-1, 3))
        accumulated_pts_colors.append(gt_img_clrs[mask].view(-1, 3))

    accumulated_pts = torch.cat(accumulated_pts, dim=0).cpu().numpy()
    accumulated_pts_colors = torch.cat(accumulated_pts_colors, dim=0).cpu().numpy()

    return accumulated_pts, accumulated_pts_colors


def main(
    checkpoint_path: str,
    data_path: str,
    data_scale_factor: int = 4,
    visualize_results: bool = False,
    device: str = "cuda",
):
    """
    Main function of the script. This script generates a (colored) point cloud from a GaussianSplat3d model by
    rendering a depth map from each image in a dataset, converting the depths to points, and concatenating all the
    points into a single point cloud. The resulting point cloud is saved as a PLY file and
    (optionally) visualized in Polyscope.

    Args:
        checkpoint_path: Path to the GaussianSplat3d checkpoint.
        data_path: Path to the dataset.
        data_scale_factor: Downsample factor for the depth images.
        visualize_results: Whether to visualize the results in Polyscope.
        device: Device to use

    Returns:
        None
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianSplat3d.from_state_dict(checkpoint["splats"])

    parser = ColmapParser(data_path, test_every=1, factor=data_scale_factor)
    dataset = ColmapDataset(parser, split="test")

    points, colors = accumulated_point_cloud(model, dataset, device=device)
    pcu.save_mesh_vc("accumulated_points.ply", points, colors)

    if visualize_results:
        import polyscope as ps

        ps.init()
        ps.set_background_color([0.0, 0.0, 0.0])
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        pc = ps.register_point_cloud("points", points, point_render_mode="quad", radius=1e-4)
        pc.add_color_quantity("color", colors, enabled=True)
        ps.show()


if __name__ == "__main__":
    with torch.no_grad():
        tyro.cli(main)
