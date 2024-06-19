import unittest

import numpy as np
import torch
from parameterized import parameterized

from fvdb import GridBatch

all_device_combos = [
    ['cpu', False],
    ['cuda', False],
]

class TestBasicOps(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_combos)
    def test_world_to_dual(self, device, mutable):
        torch.manual_seed(42)
        np.random.seed(42)

        # Raw grid:
        # o o x x x
        # o o o x x
        # o x o x x
        # o x x x x
        # x x x x x

        # dual_grid():
        # o o o x x
        # o o o o x
        # o o o o x
        # o o o o x
        # o o x x x

        # dual_grid(exclude_border=True):
        # o x x x x
        # x x x x x
        # x x x x x
        # x x x x x
        # x x x x x

        # dual_grid().dual_grid(exclude_border=True):
        # o o x x x
        # o o o x x
        # o o o x x
        # o x x x x
        # x x x x x

        ij = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 2], [3, 0]], device=device
        )
        ijk = torch.cat([
            torch.nn.functional.pad(ij, (0, 1), mode='constant', value=0),
            torch.nn.functional.pad(ij, (0, 1), mode='constant', value=1),
        ], dim=0)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_ijk(ijk)

        _grid = grid.dual_grid(exclude_border=True)
        _target_ijk = torch.tensor([[0, 0, 0]], device=device)
        assert (_grid.ijk.jdata == _target_ijk).all(), _grid.ijk.jdata

        _grid = grid.dual_grid().dual_grid(exclude_border=True)
        _target_ijk = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
            [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
            [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [3, 0, 0], [3, 0, 1]], device=device)
        assert (_grid.ijk.jdata == _target_ijk).all(), _grid.ijk.jdata


if __name__ == '__main__':
    unittest.main()
