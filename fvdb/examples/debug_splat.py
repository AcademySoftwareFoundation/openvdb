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
    vox_origin = (0, 0, 0)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    index = SparseIndexGrid(vox_size, vox_origin, device=device)
    index.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    index_dual = index.dual_grid()

    print("Splatting into grid...")
    start = time.time()
    nsplat = index.splat_trilinear(p, n)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")
    # countsn = torch.maximum(counts, torch.ones_like(counts))
    # nsplat_avg = nsplat / countsn.unsqueeze(-1)

    gp = index.active_grid_coords()
    gd = index_dual.active_grid_coords()
    gp = index.grid_to_world(gp.to(dtype))
    gd = index_dual.grid_to_world(gd.to(dtype))

    print(p.dtype, n.dtype)
    # print(counts.dtype, nsplat.dtype, countsn.dtype)
    print(gp.dtype, gd.dtype)

    p, n = p.cpu(), n.cpu()
    nsplat = nsplat.cpu()
    # nsplat_avg = nsplat_avg.cpu()
    gp, gd = gp.cpu(), gd.cpu()

    ps.init()
    ps.register_point_cloud("points", p, radius=0.00075)
    grid_pts = ps.register_point_cloud("vox coords", gp, radius=0.0005)
    # grid_pts.add_scalar_quantity("counts", counts, enabled=True)
    # grid_pts.add_vector_quantity("splatted normals (average)", nsplat_avg,
    #                              enabled=True, length=0.05, radius=0.001)
    grid_pts.add_vector_quantity("splatted normals", nsplat,
                                 enabled=True, length=0.05, radius=0.001)
    ps.show()


if __name__ == "__main__":
    main()
