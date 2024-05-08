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
    vox_origin = torch.zeros(3)
    N = 1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    print(data_path)
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts[::N]).to(device).to(dtype), \
           torch.from_numpy(nms[::N]).to(device).to(dtype)

    index = SparseIndexGrid(vox_size, vox_origin, device=device)
    index.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    index_dual = index.dual_grid()

    p, n = p.to(device), n.to(device)

    nsplat, counts = index.splat_trilinear(p, n, return_counts=True)
    nsplat /= torch.minimum(counts, torch.ones_like(counts)).unsqueeze(-1)
    gp = index.active_grid_coords()
    gd = index_dual.active_grid_coords()
    gp = index.grid_to_world(gp.to(dtype))
    gd = index_dual.grid_to_world(gd.to(dtype))

    features = torch.ones(index_dual.num_voxels, 32).to(device).to(dtype) * \
               torch.norm(gd.to(dtype), dim=-1, keepdim=True)
    features.requires_grad = True

    print("Sampling features....")
    start = time.time()
    features_trilerp = index_dual.sample_trilinear(p, features)
    assert not isinstance(features_trilerp, tuple)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")
    loss = features_trilerp.sum()
    loss.backward()

    print(p.dtype, n.dtype)
    print(nsplat.dtype)
    print(gp.dtype, gd.dtype)
    print(features.dtype)
    print(features_trilerp.dtype)
    p, n = p.cpu(), n.cpu()
    nsplat = nsplat.cpu()
    gp, gd = gp.cpu(), gd.cpu()
    features = features.detach().cpu()
    features_trilerp = features_trilerp.detach().cpu()

    ps.init()
    dual_grid_pts = ps.register_point_cloud("dual grid corners", gd, radius=0.001)
    dual_grid_pts.add_scalar_quantity("feature norms", torch.norm(features, dim=-1), enabled=True)

    primal_grid_pts = ps.register_point_cloud("primal grid corners", gp, radius=0.0005)
    primal_grid_pts.add_vector_quantity("splatted normals", nsplat,
                                        enabled=True, length=0.05, radius=0.001)


    surf_pts = ps.register_point_cloud("points", p, radius=0.0035)
    surf_pts.add_scalar_quantity("sampled feature norms",
                                 torch.norm(features_trilerp, dim=-1),
                                 enabled=True)
    ps.show()



if __name__ == "__main__":
    main()
