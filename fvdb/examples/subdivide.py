# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
import time
import logging

import polyscope as ps
import torch
from fvdb import GridBatch

from common import load_dragon_mesh


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    device = torch.device("cuda")
    dtype = torch.float32

    vox_size = 0.01
    vox_origin = (0.0, 0.0, 0.0)
    p, n = load_dragon_mesh(device=device, dtype=dtype)

    index0 = GridBatch(device)
    index0.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, vox_origin)
    grids = [index0]

    logging.info("Splatting into grid...")
    start = time.time()
    nsplat = index0.splat_trilinear(p, n)
    if device == "cuda":
        torch.cuda.synchronize()
    logging.info(f"Done in {time.time() - start}s!")

    logging.info("Building subdivided grids")
    start = time.time()
    for i in range(2):
        subdiv_factor = i + 1
        mask = torch.rand(grids[i].total_voxels, device=device) > 0.5
        grids.append(grids[-1].subdivided_grid(subdiv_factor, mask))
        assert mask.sum().item() * subdiv_factor**3 == grids[-1].total_voxels
    if device == "cuda":
        torch.cuda.synchronize()
    logging.info(f"Done in {time.time() - start}s!")

    p, n = p.cpu(), n.cpu()

    ps.init()
    ps.register_point_cloud("points", p, radius=0.00075)

    for i, index in enumerate(grids):
        dual_index = index.dual_grid()
        gp = index.ijk.jdata
        gd = dual_index.ijk.jdata
        dual_v, dual_e = index.viz_edge_network

        dual_v = dual_v.jdata.cpu()
        dual_e = dual_e.jdata.cpu()
        gp = index.grid_to_world(gp.to(dtype)).cpu()
        gd = dual_index.grid_to_world(gd.to(dtype)).cpu()
        gp, gd = gp.cpu().jdata, gd.cpu().jdata

        ps.register_curve_network(f"grid edges {i}", dual_v.cpu(), dual_e.cpu(), enabled=True, radius=0.0005)
        ps.register_point_cloud(f"vox corners {i}", gd, radius=0.0005 * (i + 1))
        if i == 0:
            grid_pts = ps.register_point_cloud("vox centers", gp, radius=0.0005)
            grid_pts.add_vector_quantity(
                "splatted normals", nsplat.jdata.cpu(), enabled=True, length=0.05, radius=0.001
            )
    ps.show()


if __name__ == "__main__":
    main()
