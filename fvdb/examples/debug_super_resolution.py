import os
import torch
import fvdb
from pycg import vis        # pip install python-pycg
import point_cloud_utils as pcu
import torch.nn.functional as F


if __name__ == '__main__':
    # Prepare training data
    device = torch.device("cuda:0")

    # Load data and normalize it to 0-1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    pts = pcu.load_mesh_v(data_path)
    pts = torch.from_numpy(pts[::10]).float().to(device) * 4 + 0.5
    pts[:, 1] -= 0.4

    # Parameters to be optimized
    base_feature = torch.randn((16, 16, 16, 10), device=device, requires_grad=True)
    conv1_weight = torch.randn((2, 10, 3, 3, 3), device=device, requires_grad=True)
    conv2_weight = torch.randn((2, 10, 3, 3, 3), device=device, requires_grad=True)

    optimizer = torch.optim.Adam(params=[
        base_feature, conv1_weight, conv2_weight
    ], lr=0.01)

    for iter in range(500):
        # Simple network inference.
        feature_0 = base_feature.view(-1, 10)
        grid_0 = fvdb.sparse_grid_from_dense_cube(16, device=device)
        feature_0m = grid_0.sparse_conv_inplace(feature_0, conv1_weight)
        mask_0 = feature_0m[:, 1] > feature_0m[:, 0]

        feature_1, grid_1 = grid_0.subdivide(feature_0, 2, mask=mask_0)
        feature_1m = grid_1.sparse_conv_inplace(feature_1, conv2_weight)
        mask_1 = feature_1m[:, 1] > feature_1m[:, 0]

        # Grid 0 to 1 loss
        ref_grid_0 = fvdb.sparse_grid_from_pointcloud(
            pts, grid_0.voxel_size, grid_0.origin, device=device)
        gt_mask_0 = ref_grid_0.points_in_active_voxel(
            grid_0.grid_to_world(grid_0.active_grid_coords().float()))
        mask_0_loss = F.cross_entropy(feature_0m, gt_mask_0.long())

        # Grid 1 to 2 loss
        ref_grid_1 = fvdb.sparse_grid_from_pointcloud(
            pts, grid_1.voxel_size, grid_1.origin, device=device)
        gt_mask_1 = ref_grid_1.points_in_active_voxel(
            grid_1.grid_to_world(grid_1.active_grid_coords().float()))
        mask_1_loss = F.cross_entropy(feature_1m, gt_mask_1.long())

        loss = mask_0_loss + mask_1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print(iter, loss.item())

    # Visualize
    final_grid = grid_1.subdivided_grid(2, mask_1)
    vis.show_3d([vis.wireframe_bbox(
        final_grid.grid_to_world(final_grid.active_grid_coords() - 0.5),
        final_grid.grid_to_world(final_grid.active_grid_coords() + 0.5),
        solid=True
    )])
