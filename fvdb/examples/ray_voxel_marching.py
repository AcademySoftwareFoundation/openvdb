import timeit
import logging

import polyscope as ps
import torch

import fvdb
from fvdb import JaggedTensor, GridBatch

from common import load_dragon_mesh, make_ray_grid, plot_ray_segments


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    device = torch.device('cuda')
    dtype = torch.float32
    vox_size = 0.01
    vox_origin = [0.0, 0.0, 0.0]

    nrays = 1000
    plot_every = 5000  # only plot every n rays
    N = 20  # maximum number of voxels to intersect along ray

    p, n = load_dragon_mesh(device=device, dtype=dtype)
    p -= p.mean(0)
    p = torch.concatenate([p,
                           p * 2 + torch.tensor([0, 0, .48], device=p.device),
                           p * 3 + torch.tensor([0, 0, .96], device=p.device),
                           p * 4 + torch.tensor([0, 0, 1.44], device=p.device)])
    n = torch.concatenate([n, n, n, n])


    grid = GridBatch(device=device)
    grid.set_from_points(p, [-1]*3, [1]*3, voxel_sizes=vox_size, origins=vox_origin)

    gc, ge = grid.viz_edge_network

    ray_o, ray_d = make_ray_grid(nrays, [0.0, 0.0, -0.1], device=device, dtype=dtype)
    pmt = torch.randperm(ray_o.shape[0]).to(device)
    ray_o, ray_d = ray_o[pmt], ray_d[pmt]

    logging.info(f"Tracing {nrays ** 2} Ray Voxels...")
    start = timeit.default_timer()
    pack_info, vox, times = grid.voxels_along_rays(ray_o, ray_d, N, 1e-4)
    if p.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s")
    pack_info = pack_info.jdata
    vox = vox.jdata
    times = times.jdata

    # Downsample and move the output to the CPU for plotting
    ray_o, ray_d = ray_o[::plot_every], ray_d[::plot_every]
    pack_info = pack_info[::plot_every]

    p, n = p.cpu(), n.cpu()
    ray_o, ray_d = ray_o.cpu(), ray_d.cpu()
    vox, times, pack_info = vox.cpu(), times.cpu(), pack_info.cpu()
    gc, ge = gc.cpu(), ge.cpu()

    ps.init()
    ps.set_ground_plane_mode('shadow_only')

    ps.register_point_cloud("points", p, radius=0.0025)

    plot_ray_segments(ray_o, ray_d, times, pack_info)

    vox_i = torch.cat([vox[pack_info[i][0]: pack_info[i][0] + pack_info[i][1]] for i in range(pack_info.shape[0])])
    vox_i = JaggedTensor([vox_i])
    iv, ie = fvdb.sparse_grid_from_ijk(
        vox_i, voxel_sizes=vox_size, origins=vox_origin).viz_edge_network
    ps.register_curve_network("intersected voxels", iv.jdata, ie.jdata, enabled=True, radius=0.0009)
    ps.register_point_cloud("grid corners", gc.jdata, enabled=True, radius=0.001)
    ps.register_curve_network("grid edges", gc.jdata, ge.jdata, enabled=True, radius=0.00075, transparency=0.7)

    ray_dir_points = torch.cat([ray_o, ray_o + times.max() * ray_d])
    ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]),
                                 torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
    ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
    ps.register_point_cloud("ray origins", ray_o, radius=0.01)
    ps.show()



if __name__ == "__main__":
    main()
