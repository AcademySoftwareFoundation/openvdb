# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import polyscope as ps
import torch

from fvdb import GridBatch, gridbatch_from_ijk
from fvdb.utils.examples import load_dragon_mesh


def main():
    device = "cuda"

    vox_size = 0.0075
    vox_origin = (0, 0, 0)
    N = 1

    [p] = load_dragon_mesh(mode="v", skip_every=N, device=torch.device(device))

    index = GridBatch(device=device)
    index.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, vox_origin)

    primal_voxels = index.ijk.jdata

    nhood = index.neighbor_indexes(primal_voxels, 1, 0).jdata

    ps.init()
    for _ in range(10):
        randvox = np.random.randint(nhood.shape[0])

        voxijk = primal_voxels[randvox]
        nbrs = primal_voxels[nhood[randvox][nhood[randvox] >= 0]]
        print(nhood[randvox])
        nhood_ijk = torch.cat([voxijk.unsqueeze(0), nbrs], dim=0)

        vp, ve = index.viz_edge_network
        vp, ve = vp.jdata, ve.jdata

        vi, vei = gridbatch_from_ijk(nhood_ijk, voxel_sizes=vox_size, origins=vox_origin).viz_edge_network
        vi, vei = vi.jdata, vei.jdata

        ps.register_curve_network("vox", vp.cpu().numpy(), ve.cpu().numpy(), radius=0.0025)
        ps.register_curve_network("nhd", vi.cpu().numpy(), vei.cpu().numpy(), radius=0.005)
        ps.show()


if __name__ == "__main__":
    main()
