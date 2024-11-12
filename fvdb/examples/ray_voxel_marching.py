# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import timeit

import polyscope as ps
import torch

import fvdb
from fvdb import GridBatch, JaggedTensor
from fvdb.utils.examples import load_dragon_mesh, make_ray_grid, plot_ray_segments


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    device = torch.device("cuda")
    dtype = torch.float32
    vox_size = 0.04
    vox_origin = [0.0, 0.0, 0.0]

    nrays = 1024  # 100 x 100 rays
    plot_every = 512  # only plot every n rays
    max_voxels = 20  # maximum number of voxels to intersect along ray

    p, n = load_dragon_mesh(device=device, dtype=dtype)
    p -= p.mean(0)
    p /= 10.0
    p = torch.concatenate(
        [
            p,
            p + 2 * torch.tensor([0, 0, 0.48], device=p.device),
            p + 2 * torch.tensor([0, 0, 0.96], device=p.device),
            # p + 1 * torch.tensor([0, 0, 1.44], device=p.device),
        ]
    )
    n = torch.concatenate([n, n, n])

    batch_size = 2

    p = fvdb.JaggedTensor([p] * batch_size)
    n = fvdb.JaggedTensor([n] * batch_size)

    grid = GridBatch(device=device)
    grid.set_from_points(p, [-1] * 3, [1] * 3, voxel_sizes=vox_size, origins=vox_origin)

    logging.info(f"Created {len(grid)} grids with {grid.total_voxels} total voxels")
    gc, ge = grid.viz_edge_network

    ray_o, ray_d = make_ray_grid(nrays, [0.0, 0.0, -0.1], device=device, dtype=dtype)
    pmt = torch.randperm(ray_o.shape[0]).to(device)
    ray_o, ray_d = ray_o[pmt], ray_d[pmt]

    ray_o, ray_d = fvdb.JaggedTensor([ray_o] * batch_size), fvdb.JaggedTensor([ray_d] * batch_size)

    logging.info(f"Tracing {nrays ** 2} Rays Per Grid...")
    start = timeit.default_timer()
    vox, times = grid.voxels_along_rays(ray_o, ray_d, max_voxels, 1e-4)
    if p.jdata.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s")

    logging.info(f"There are {len(vox)} sets of intersections in the batch")
    for i, visect in enumerate(vox):
        logging.info(f"There are {len(visect)} rays in the {i}th set of intersections")

    logging.info("Plotting")
    ps.init()
    for i in range(batch_size):
        p_i = p[i].jdata.cpu()
        ray_o_i, ray_d_i = ray_o[i].jdata.cpu(), ray_d[i].jdata.cpu()
        times_i = times[i].cpu()
        gc_i, ge_i = gc[i].cpu(), ge[i].cpu()

        ps.set_ground_plane_mode("shadow_only")

        ps.register_point_cloud("points", p_i, radius=0.00025)
        logging.info("About to plot ray segments")
        plot_ray_segments(ray_o_i, ray_d_i, times_i, plot_every)
        logging.info("Plotted Ray Segments")

        logging.info(f"Creating a new grid of only the voxels intersected by this ray")
        isected_grid = fvdb.gridbatch_from_ijk(vox[i].jflatten(), voxel_sizes=vox_size, origins=vox_origin)
        logging.info(f"Created {len(isected_grid)} grids with {isected_grid.total_voxels} total voxels")
        iv, ie = isected_grid.viz_edge_network
        ps.register_curve_network("intersected voxels", iv.jdata.cpu(), ie.jdata.cpu(), enabled=True, radius=0.0009)
        ps.register_point_cloud("grid corners", gc_i.jdata, enabled=True, radius=0.001)
        ps.register_curve_network("grid edges", gc_i.jdata, ge_i.jdata, enabled=True, radius=0.00015, transparency=0.7)

        # ray_dir_points = torch.cat([ray_o_i, ray_o_i + times_i.jdata.max() * ray_d_i])
        # ray_dir_edges = torch.stack([torch.arange(ray_o_i.shape[0]), torch.arange(ray_o_i.shape[0]) + ray_o_i.shape[0]], dim=-1)
        # ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
        # ps.register_point_cloud("ray origins", ray_o, radius=0.01)
        ps.show()


if __name__ == "__main__":
    main()
