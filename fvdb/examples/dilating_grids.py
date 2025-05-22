# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import time
from typing import List

import numpy as np
import torch
import viser

import fvdb
from fvdb.utils.examples import load_dragon_mesh


class Viewer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.scene: viser.SceneApi = self.server.scene

    def plot_grid_edges(self, name: str, grid: fvdb.GridBatch, color: List[float]):
        gv, ge = grid.viz_edge_network

        for i in range(grid.grid_count):
            gvi, gei = gv[i].jdata, ge[i].jdata

            segments = torch.stack(
                [
                    gvi[gei[:, 0], :],  # [N, 3]
                    gvi[gei[:, 1], :],  # [N, 3]
                ],
                dim=1,
            )

            self.scene.add_line_segments(
                name=f"/{name}/grid_{i}",
                points=segments.cpu().numpy(),
                colors=np.array([color])[None, :],
            )

    def show(self):
        while True:
            time.sleep(10.0)


def main():
    # Load mesh
    vertices_t, faces_t = load_dragon_mesh(mode="vf")
    vertices = vertices_t.cpu().numpy()
    faces = faces_t.cpu().numpy()
    print(vertices.min(0), vertices.max(0))
    print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

    vox_size = 0.25

    grid = fvdb.gridbatch_from_mesh(vertices_t.cuda(), faces=faces_t.cuda(), voxel_sizes=vox_size, origins=[0.0] * 3)
    print(f"Grid has {grid.total_voxels} voxels")

    grid_dilated = grid.dilated_grid(3)

    print(f"Grid has {grid_dilated.total_voxels} voxels")

    viewer = Viewer()
    viewer.plot_grid_edges("base_grid", grid, [1.0, 0.0, 0.0])
    viewer.plot_grid_edges("dilated_grid", grid_dilated, [0.0, 0.0, 1.0])
    viewer.show()


if __name__ == "__main__":
    main()
