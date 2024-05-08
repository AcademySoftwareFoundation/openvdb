import os

import point_cloud_utils as pcu
import polyscope as ps
import torch

from fvdb import SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32

    vox_size = 0.01
    vox_origin = [0.0, 0.0, 0.0]
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    index = SparseIndexGrid(vox_size, vox_origin, device=device)
    index.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    index_dual = index.dual_grid()

    gp = index.active_grid_coords()
    gd = index_dual.active_grid_coords()
    gdw = index_dual.grid_to_world(gd.to(p))

    bmin, bmax = gdw.min(0)[0].unsqueeze(0), gdw.max(0)[0].unsqueeze(0)
    pts = torch.rand(1_000_000, 3).to(p) * (bmax - bmin) + bmin
    mask = index.points_in_active_voxel(pts)

    p, n = p.cpu(), n.cpu()
    mask = mask.cpu()
    gp, gd = gp.cpu(), gd.cpu()
    pts = pts.cpu()
    print(pts.shape, mask.shape)
    ps.init()
    ps.register_point_cloud("points", p, radius=0.00075)
    ps.register_point_cloud("in samples", pts[mask])
    ps.register_point_cloud("out samples", pts[~mask])
    dual_v, dual_e = index.grid_edge_network()
    ps.register_curve_network("grid edges", dual_v.cpu(), dual_e.cpu(),
                              enabled=True, radius=0.0005)
    ps.show()


if __name__ == "__main__":
    main()
