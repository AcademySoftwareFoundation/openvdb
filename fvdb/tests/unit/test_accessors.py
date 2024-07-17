# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
import unittest

import torch
import numpy as np
from parameterized import parameterized

from fvdb import GridBatch

all_device_combos = [
    ["cpu"],
    ["cuda"],
]

RESOLUTION = 1292  # over int32_t max limit
# RESOLUTION = 64 # under int32_t max limit


class TestAccessors(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_read_into_dense(self, device):
        dense_origin = torch.tensor([0, 0, 0]).to(torch.long).to(device)
        dense_grid = torch.zeros(
            (1, RESOLUTION, RESOLUTION, RESOLUTION, 1),
            dtype=torch.float16,
            device=device,
        )

        sparse_points = torch.tensor(
            [[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device
        )
        grid = GridBatch(mutable=True, device=device)
        grid.set_from_points(sparse_points, voxel_sizes=0.1, origins=[0.0] * 3)

        grid.read_from_dense(dense_grid, dense_origin)

    @parameterized.expand(all_device_combos)
    def test_read_from_dense(self, device):
        dense_origin = torch.tensor([0, 0, 0]).to(torch.long).to(device)

        zero_points = torch.tensor(
            [[0, 0, 0], [1, 1, 1]], dtype=torch.float16, device=device
        )
        grid = GridBatch(mutable=True, device=device)
        grid.set_from_points(zero_points, voxel_sizes=0.1, origins=[0.0] * 3)

        sparse_data = torch.tensor([[0], [0]], dtype=torch.float16, device=device)
        grid.read_into_dense(
            sparse_data, dense_origin, (RESOLUTION, RESOLUTION, RESOLUTION)
        )

if __name__ == '__main__':
    unittest.main()
