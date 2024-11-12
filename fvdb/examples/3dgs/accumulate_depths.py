import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import torch
import torch.utils.data
import tqdm
import tyro
from skimage import feature, morphology

from fvdb.nn.gaussian_splatting import GaussianSplat3D
from fvdb.utils.data import ColmapDataset, ColmapParser


@torch.inference_mode()
def accumulated_point_cloud(
    model: GaussianSplat3D,
    dataset: ColmapDataset,
    canny_edge_std: float = 1.0,
    dilation_amt: int = 5,
    downsample_factor: int = 1,
    device: str = "cuda",
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    accumulated_pts = []
    accumulated_pts_colors = []

    for data in tqdm.tqdm(dataloader):
        img = data["image"].to(device).squeeze()
        K = data["K"].to(device).squeeze()
        cam_to_world = data["camtoworld"].to(device).squeeze()
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        depth, cam_pts = model.render_depth_points(
            image_w=img.shape[1],
            image_h=img.shape[0],
            extrinsics_mat=world_to_cam.unsqueeze(0),
            intrinsics_mat=K.unsqueeze(0),
            rasterize_mode="classic",
            sh_degree=3,
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


def main(checkpoint_path: str, data_path: str, data_scale_factor: int = 4, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianSplat3D(torch.rand([8, 3]), torch.rand([8, 3])).to(device)
    model.load_state_dict(checkpoint["splats"])

    parser = ColmapParser(data_path, test_every=1, normalize=True, factor=data_scale_factor)
    dataset = ColmapDataset(parser, split="test")

    points, colors = accumulated_point_cloud(model, dataset, device=device)
    pcu.save_mesh_vc("accumulated_points.ply", points, colors)

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
