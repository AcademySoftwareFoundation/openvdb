import timeit
import logging

import polyscope as ps
import torch
from fvdb import GridBatch

from common import load_dragon_mesh

def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    device = torch.device('cuda')
    dtype = torch.float32
    vox_size = 0.0025
    vox_origin = torch.zeros(3)

    p, n = load_dragon_mesh(skip_every=1, device=device, dtype=dtype)

    index = GridBatch(device=device)
    index.set_from_points(p, voxel_sizes=vox_size,  origins=vox_origin)
    index_dual = index.dual_grid()

    nsplat = index.splat_trilinear(p, n)
    gp = index.ijk
    gd = index_dual.ijk
    gp = index.grid_to_world(gp.type(dtype))
    gd = index_dual.grid_to_world(gd.type(dtype))

    features = torch.ones(index_dual.total_voxels, 32).to(device).to(dtype) * \
               torch.norm(gd.jdata.type(dtype), dim=-1, keepdim=True)
    features.requires_grad = True

    logging.info("Sampling features....")
    start = timeit.default_timer()
    features_trilerp = index_dual.sample_trilinear(p, features)
    if features.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s!")
    loss = features_trilerp.jdata.sum()
    loss.backward()

    p, n = p.cpu(), n.cpu()
    nsplat = nsplat.cpu()
    gp, gd = gp.cpu(), gd.cpu()
    features = features.detach().cpu()
    features_trilerp = features_trilerp.detach().cpu()

    ps.init()
    dual_grid_pts = ps.register_point_cloud("dual grid corners", gd.jdata, radius=0.001)
    dual_grid_pts.add_scalar_quantity("feature norms", torch.norm(features, dim=-1), enabled=True)

    primal_grid_pts = ps.register_point_cloud("primal grid corners", gp.jdata, radius=0.0005)
    primal_grid_pts.add_vector_quantity("splatted normals", nsplat.jdata,
                                        enabled=True, length=0.05, radius=0.001)


    surf_pts = ps.register_point_cloud("points", p, radius=0.0035)
    surf_pts.add_scalar_quantity("sampled feature norms",
                                 torch.norm(features_trilerp.jdata, dim=-1),
                                 enabled=True)
    ps.show()



if __name__ == "__main__":
    main()
