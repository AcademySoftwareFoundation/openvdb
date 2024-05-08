import os
import time

import point_cloud_utils as pcu
import polyscope as ps
import torch

from fvdb import SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32

    vox_size = 0.02
    vox_origin = [0.0]*3
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    index0 = SparseIndexGrid(vox_size, vox_origin, device=device)
    index0.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    grids = [index0]
    colors = [torch.rand(index0.num_voxels, 3).to(p)]

    print("Building coarse grids")
    start = time.time()
    for i in range(2):
        print(f"coarsen {i}")
        pool_factor = i + 2

        fine_colors = colors[-1]
        fine_grid = grids[-1]

        coarse_colors, coarse_grid = fine_grid.max_pool(fine_colors, pool_factor)

        print(f"grid {i + 1}")
        print(fine_colors.shape)
        print(fine_grid.num_voxels, coarse_grid.num_voxels)
        print(coarse_colors.shape)
        print("-------------")

        grids.append(coarse_grid)
        colors.append(coarse_colors)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")

    p, n = p.cpu(), n.cpu()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.register_point_cloud("points", p, radius=0.00075)

    for i, index in enumerate(reversed(grids)):
        index_dual = index.dual_grid()
        gp = index.active_grid_coords()
        gd = index_dual.active_grid_coords()

        gp = index.grid_to_world(gp.to(dtype)).cpu()
        gd = index_dual.grid_to_world(gd.to(dtype)).cpu()
        gp, gd = gp.cpu(), gd.cpu()

        print(gp.shape, colors[i].shape)
        grid_pts = ps.register_point_cloud(f"vox centers {i}", gp, radius=0.05 / (i + 1))
        grid_pts.add_color_quantity("vox colors", colors[-i-1].detach().cpu(),
                                     enabled=True)
    ps.show()


if __name__ == "__main__":
    main()
