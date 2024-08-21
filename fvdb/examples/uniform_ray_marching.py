# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
import os
import time
import logging
import timeit

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
from fvdb import GridBatch, JaggedTensor
import fvdb

from common import load_dragon_mesh, make_ray_grid, plot_ray_segments


def main():
    torch.random.manual_seed(5)
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))

    device = torch.device("cuda")
    dtype = torch.float32
    vox_size = 0.04
    vox_origin = torch.zeros(3).to(device)
    step_size = 0.5 * vox_size

    N = 10  # Maximum number of segments to intersect along ray
    nrays = 100
    plot_every = 20
    batch_size = 2

    p, n = load_dragon_mesh(device=device, dtype=dtype)

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

    ray_o, ray_d = make_ray_grid(nrays, [0.0, 0.1, -0.1], device=device, dtype=dtype)
    pmt = torch.randperm(ray_o.shape[0]).to(device)
    ray_o, ray_d = ray_o[pmt], ray_d[pmt]

    p, n = JaggedTensor([p] * batch_size), JaggedTensor([n] * batch_size)
    ray_o, ray_d = JaggedTensor([ray_o] * batch_size), JaggedTensor([ray_d] * batch_size)

    grid = GridBatch(device=device)
    grid.set_from_points(p, [-1] * 3, [1] * 3, voxel_sizes=vox_size, origins=vox_origin)

    gc, ge = grid.viz_edge_network

    logging.info(f"Tracing {nrays ** 2} Ray Segments...")
    start = timeit.default_timer()
    segments = grid.segments_along_rays(ray_o, ray_d, N, eps=1e-5)
    if p.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s!")

    tmin = fvdb.jzeros(ray_o.lshape, device=device, dtype=dtype)
    tmax = fvdb.jones(ray_o.lshape, device=device, dtype=dtype) * 1e10

    logging.info(f"Generating samples for {ray_o.rshape[0]} Ray Segments...")
    start = timeit.default_timer()
    ray_ts = grid.uniform_ray_samples(ray_o, ray_d, tmin, tmax, step_size, eps=1e-4)
    if p.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s!")

    print(ray_ts.eshape)
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    for b_i in range(batch_size):
        ps.register_point_cloud("points", p[b_i].jdata.cpu(), radius=0.00025, point_render_mode="quad")
        for i in range(0, len(ray_o[b_i].jdata), plot_every):
            roi = ray_o[b_i].jdata[i].unsqueeze(0)  # [1, 3]
            rdi = ray_d[b_i].jdata[i].unsqueeze(0)  # [1, 3]
            segsi = segments[b_i][i].jdata  # [N, 2]

            if segsi.numel() == 0:
                continue

            rp = torch.cat(
                [
                    roi + segsi[:, 0].unsqueeze(-1) * rdi,
                    roi + segsi[:, 1].unsqueeze(-1) * rdi,
                ]
            )
            re = torch.stack([torch.arange(segsi.shape[0]), torch.arange(segsi.shape[0]) + segsi.shape[0]], dim=-1)

            # ray_segs = ps.register_curve_network(f"ray segments {i}", rp.cpu(), re.cpu(), radius=0.00075)

            ray_ts_i = ray_ts[b_i][i].jdata
            ray_ts_i = 0.5 * (ray_ts_i[:, 0] + ray_ts_i[:, 1])
            ray_samples = roi + ray_ts_i.unsqueeze(-1) * rdi
            ps.register_point_cloud(f"ray samples {i}", ray_samples.cpu(), radius=0.0015)
            # rv = torch.zeros(re.shape[0])
            # rv[::2] = 1.0
            # ray_segs.add_scalar_quantity(f"segment colors {i}", rv.cpu(), defined_on="edges", enabled=True, cmap="jet")

        ps.register_point_cloud("grid corners", gc.jdata.cpu(), enabled=True, radius=0.00025, point_render_mode="quad")
        ps.register_curve_network(
            "grid edges", gc.jdata.cpu(), ge.jdata.cpu(), enabled=True, radius=0.00025, transparency=0.7
        )

        # ray_dir_points = torch.cat([ray_o, ray_o + 0.5 * ray_d])
        # ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]), torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
        # ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
        # ps.register_point_cloud("ray origins", ray_o, radius=0.01)
        ps.show()


if __name__ == "__main__":
    main()
