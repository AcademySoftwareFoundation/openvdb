import os
import time

import point_cloud_utils as pcu
import polyscope as ps
import torch

from fvdb import SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32

    vox_size = 0.1
    vox_origin = [0.0, 0.0, 0.0]
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    index0 = SparseIndexGrid(vox_size, vox_origin, device=device)
    index0.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    grids = [index0]
    # colors = [index0.primal_grid_coords().to(p)]
    # colors[0] += colors[0].min(0)[0]
    # colors[0] /= colors[0].max()
    colors = [torch.rand(index0.num_voxels, 3).to(p)]

    print("Splatting into grid...")
    start = time.time()
    nsplat, counts = index0.splat_trilinear(p, n, return_counts=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")
    countsn = torch.maximum(counts, torch.ones_like(counts))
    nsplat_avg = nsplat / countsn.unsqueeze(-1)

    print("Building subdivided grids")
    start = time.time()
    for i in range(2):
        subdiv_factor = i + 2
        mask = (torch.rand(grids[i].num_voxels, device=device) > 0.5)
        # coarse_colors = grids[-1].primal_grid_coords().to(p)
        # coarse_colors += coarse_colors.min(0)[0]
        # coarse_colors /= coarse_colors.max()
        coarse_colors = colors[-1]
        coarse_grid = grids[-1]
        fine_colors, fine_grid = coarse_grid.subdivide(coarse_colors, subdiv_factor, mask=mask)
        grids.append(fine_grid)
        colors.append(fine_colors)
        assert mask.sum().item() * subdiv_factor ** 3 == grids[-1].num_voxels
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")

    p, n = p.cpu(), n.cpu()
    nsplat, counts, countsn = nsplat.cpu(), counts.cpu(), countsn.cpu()
    nsplat_avg = nsplat_avg.cpu()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.register_point_cloud("points", p, radius=0.00075)

    for i, index in enumerate(grids):
        index_dual = index.dual_grid()
        gp = index.active_grid_coords()
        gd = index_dual.active_grid_coords()

        gp = index.grid_to_world(gp.to(dtype)).cpu()
        gd = index_dual.grid_to_world(gd.to(dtype)).cpu()
        gp, gd = gp.cpu(), gd.cpu()

        grid_pts = ps.register_point_cloud(f"vox centers {i}", gp, radius=0.01 / (i + 1))
        grid_pts.add_color_quantity("vox colors", colors[i].detach().cpu(), enabled=True)
    ps.show()


if __name__ == "__main__":
    main()
