import polyscope as ps
import torch
import numpy as np
from torch.nn.functional import grid_sample
from fvdb import SparseIndexGrid


def make_data(nvox, device, dtype):
    grid_origin = np.zeros(3)
    voxel_size = 1.0 / (np.floor(0.5 * nvox) + 0.5)

    grid = SparseIndexGrid(voxel_size, grid_origin, device=device)

    target_vox = int(2 * np.floor(0.5 * nvox) + 1) ** 3
    target_corners = int(2 * np.floor(0.5 * nvox) + 2) ** 3

    p = 2.0 * torch.rand(1, 3) - 1.0
    print(target_vox, target_corners)
    while grid.num_voxels != target_vox:
        p = (2.0 * torch.rand(10 * p.shape[0], 3) - 1.0).to(dtype)
        grid.build_from_pointcloud(p, [0, 0, 0], [0, 0, 0])
    # print(fvdb.num_voxels, int(2 * np.floor(0.5 * nvox) + 1) ** 3)
    # print(fvdb.num_corners(), int(2 * np.floor(0.5 * nvox) + 2) ** 3)
    dual_grid = grid.dual_grid()
    assert dual_grid.num_voxels == target_corners

    dual_ijk = dual_grid.active_grid_coords()
    dual_corners_xyz = dual_grid.grid_to_world(dual_ijk.float())
    assert torch.allclose(dual_corners_xyz.min(0)[0], -torch.ones(3))
    assert torch.allclose(dual_corners_xyz.max(0)[0], torch.ones(3))

    return grid, dual_grid, p


def make_data_simple(device, dtype):
    p = torch.tensor([[0.175, 0.25, 0.45],
                      [0.45, 0.25, 0.45],
                      [-0.35, 0.75, 0.25],
                      [0.15, -0.25, -0.45]]).to(dtype).to(device)

    grid_origin = [0.0] * 3
    voxel_size = 2.0

    fvdb = SparseIndexGrid(voxel_size, grid_origin, device=device)

    fvdb.build_from_pointcloud(p, [0, 0, 0], [0, 0, 0])

    fvdb_dual = fvdb.dual_grid()

    dual_ijk = fvdb_dual.active_grid_coords()
    dual_corners_xyz = fvdb_dual.grid_to_world(dual_ijk.float())
    assert torch.allclose(dual_corners_xyz.min(0)[0], -torch.ones(3)), \
           f"{dual_corners_xyz.min(0)[0]}"
    assert torch.allclose(dual_corners_xyz.max(0)[0], torch.ones(3)), \
           f"{dual_corners_xyz.max(0)[0]}"

    return fvdb, fvdb_dual, p


def sample_fvdb(fvdb_dual, p, dual_features):
    fs = fvdb_dual.sample_trilinear(p, dual_features).squeeze()
    loss = fs.sum()
    loss.backward()
    return fs.clone().detach(), dual_features.grad.clone().detach()


def sample_pytorch(fvdb_dual, p, dual_features):
    dual_ijk = fvdb_dual.active_grid_coords()
    dual_features_grid = permute_features_into_grid(dual_ijk, dual_features).unsqueeze(0)
    p_in = p.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N, 3]
    res = grid_sample(dual_features_grid, p_in, mode='bilinear', align_corners=True).squeeze()
    loss = res.sum()
    loss.backward()

    return res.clone().detach(), dual_features.grad.detach().clone()


def permute_features_into_grid(dual_coordinates, features):
    dual_coordinates = dual_coordinates - dual_coordinates.min(0)[0]
    dual_gsize = int(np.cbrt(dual_coordinates.shape[0]))
    out_grid = torch.empty(features.shape[-1], dual_gsize, dual_gsize, dual_gsize)
    for i in range(features.shape[0]):
        fi = features[i]
        gi = dual_coordinates[i]
        out_grid[:, gi[2], gi[1], gi[0]] = fi
    return out_grid


def main():
    device = 'cpu'
    dtype = torch.float32
    gsize = 7

    # grid, dual_grid, p = make_data_simple(device, dtype)
    grid, dual_grid, p = make_data(gsize, device, dtype)

    dual_features = torch.rand(dual_grid.num_voxels, 4)
    dual_features.requires_grad = True

    fv, gv = sample_fvdb(dual_grid, p, dual_features)

    assert dual_features.grad is not None
    dual_features.grad.zero_()

    fp, gp = sample_pytorch(dual_grid, p, dual_features)
    print(f"max error: {(fv - fp.permute(1, 0)).abs().max()}")
    print(f"max error grad: {(gv - gp).abs().max()}")

    primal_ijk = grid.active_grid_coords()
    dual_v, dual_e = grid.grid_edge_network()

    # print(f"Dual Vertices: {dual_v}")
    gc = grid.grid_to_world(torch.floor(grid.world_to_grid(p.to(device)) + 0.5))

    ps.init()
    ps.register_point_cloud("round vox", gc.cpu(), enabled=True, radius=0.02, transparency=0.75)
    ps.register_point_cloud("grid centers",
                            grid.grid_to_world(primal_ijk.to(dtype)).cpu(), enabled=True)
    ps.register_point_cloud("grid corners", dual_v.cpu(), enabled=True, radius=0.0025)
    ps.register_curve_network("grid edges", dual_v.cpu(), dual_e.cpu(), enabled=True, radius=0.001)
    ps.register_point_cloud("point", p.cpu(), radius=0.02)
    ps.show()


if __name__ == "__main__":
    main()
