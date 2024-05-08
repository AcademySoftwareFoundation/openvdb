import os
import time

import point_cloud_utils as pcu
import polyscope as ps
import torch
from fvdb import SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32

    vox_size = 0.0025
    vox_origin = (0.0, 0.0, 0.0)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    # p, n = p[:100], n[:100]
    # p, n = torch.zeros(2, 3).to(p), torch.ones(2, 3).to(n)
    index0 = SparseIndexGrid(vox_size, vox_origin, device=device)
    index0.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    # index0.build_from_padded_pointcloud(p, [0, 0, 0], [0, 0, 0])
    grids = [index0]

    print("Splatting into grid...")
    start = time.time()
    nsplat, counts = index0.splat_trilinear(p, n, return_counts=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")
    countsn = torch.maximum(counts, torch.ones_like(counts))
    nsplat_avg = nsplat / countsn.unsqueeze(-1)

    print("Building coarse grids")
    start = time.time()
    for i in range(3):
        grids.append(grids[-1].coarsened_grid(i + 2))
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")

    p, n = p.cpu(), n.cpu()
    nsplat, counts, countsn = nsplat.cpu(), counts.cpu(), countsn.cpu()
    nsplat_avg = nsplat_avg.cpu()

    ps.init()
    ps.register_point_cloud("points", p, radius=0.00075)

    for i, index in enumerate(grids):
        index_dual = index.dual_grid()
        gp = index.active_grid_coords()
        gd = index_dual.active_grid_coords()
        dual_v, dual_e = index.grid_edge_network()

        dual_v = dual_v.cpu()
        gp = index.grid_to_world(gp.to(dtype)).cpu()
        gd = index_dual.grid_to_world(gd.to(dtype)).cpu()
        gp, gd = gp.cpu(), gd.cpu()

        ps.register_curve_network(f"grid edges {i}", dual_v.cpu(), dual_e.cpu(),
                                  enabled=True, radius=0.0005)
        ps.register_point_cloud(f"vox corners {i}", gd, radius=0.0005 * (i + 1))
        if i == 0:
            grid_pts = ps.register_point_cloud("vox centers", gp, radius=0.0005)
            grid_pts.add_scalar_quantity("counts", counts, enabled=True)
            grid_pts.add_vector_quantity("splatted normals (average)", nsplat_avg,
                                        enabled=True, length=0.05, radius=0.001)
            grid_pts.add_vector_quantity("splatted normals", nsplat,
                                        enabled=True, length=0.05, radius=0.001)
    ps.show()


if __name__ == "__main__":
    main()
