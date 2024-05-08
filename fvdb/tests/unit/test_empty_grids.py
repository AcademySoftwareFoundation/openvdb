import itertools
import os
import pickle
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb

from fvdb import GridBatch, sparse_grid_from_ijk, JaggedTensor
from fvdb.utils import volume_render

from .common import (random_drop_points_if_mutable)

all_device_dtype_combos = [
    ['cuda', torch.float16, False],
    ['cpu', torch.float32, False],
    ['cuda', torch.float32, False],
    ['cpu', torch.float64, False],
    ['cuda', torch.float64, False],
    ['cuda', torch.float16, True],
    ['cpu', torch.float32, True],
    ['cuda', torch.float32, True],
    ['cpu', torch.float64, True],
    ['cuda', torch.float64, True]
]

def dtype_to_atol(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 1e-1
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float64:
        return 1e-5
    raise ValueError("dtype must be a valid torch floating type")


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        # self.test_path = os.path.join(os.path.dirname(
        #     os.path.realpath(__file__)), "..", "data")
        pass

    @parameterized.expand(all_device_dtype_combos)
    def test_building_empty_grids_from_ijk(self, device, dtype, mutable):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (0, 3)) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)
        random_drop_points_if_mutable(grid, 0.5)
        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0, 0].item(), 0)
        self.assertEqual(grid.joffsets[0, 1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (0, 2)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_ijk(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (0,)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_ijk(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (5, 0)) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_ijk(grid_ijk_bad, mutable=mutable)

    @parameterized.expand(all_device_dtype_combos)
    def test_building_grid_with_one_empty_element_in_jagged_tensor(self, device, dtype, mutable):
        shapes = [512, 0, 128]
        batch_size = len(shapes)
        grid_ijk = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (shapes[i], 3)) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)
        random_drop_points_if_mutable(grid, 0.5)
        self.assertEqual(len(grid), batch_size)

        off = 0
        for i in range(batch_size):
            self.assertEqual(grid.num_voxels[i], shapes[i])
            self.assertEqual(grid.joffsets[i, 0].item(), off)
            self.assertEqual(grid.joffsets[i, 1].item(), off + shapes[i])
            off += shapes[i]

    @parameterized.expand(all_device_dtype_combos)
    def test_building_empty_grids_from_points(self, device, dtype, mutable):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor(
            [torch.rand(0, 3) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_points(grid_ijk, mutable=mutable)
        random_drop_points_if_mutable(grid, 0.5)
        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0, 0].item(), 0)
        self.assertEqual(grid.joffsets[0, 1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(0, 2) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_points(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_points(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(5, 0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_points(grid_ijk_bad, mutable=mutable)

    @parameterized.expand(all_device_dtype_combos)
    def test_building_empty_grids_from_nearest_points(self, device, dtype, mutable):
        batch_size = 1
        grid_ijk = fvdb.JaggedTensor(
            [torch.rand(0, 3) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_nearest_voxels_to_points(grid_ijk, mutable=mutable)
        random_drop_points_if_mutable(grid, 0.5)
        self.assertEqual(len(grid), batch_size)
        self.assertEqual(grid.joffsets[0, 0].item(), 0)
        self.assertEqual(grid.joffsets[0, 1].item(), 0)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(0, 2) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_nearest_voxels_to_points(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_nearest_voxels_to_points(grid_ijk_bad, mutable=mutable)

        grid_ijk_bad = fvdb.JaggedTensor(
            [torch.rand(5, 0) for i in range(batch_size)]).to(device)
        with self.assertRaises(ValueError):
            fvdb.sparse_grid_from_nearest_voxels_to_points(grid_ijk_bad, mutable=mutable)

    @parameterized.expand(all_device_dtype_combos)
    def test_fvdb_cat(self, device, dtype, mutable):
        def _make_random_grid(batch_size):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            return fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)

        # Test concat batches with 1 grid
        grid1, grid2 = _make_random_grid(1), _make_random_grid(1)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 2)

        # Test concat batches with 1 grid and multiple grids
        grid1, grid2 = _make_random_grid(2), _make_random_grid(1)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 3)

        # Test concat batches with multiple grids
        grid1, grid2 = _make_random_grid(2), _make_random_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 7)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_random_grid(1), _make_random_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        random_drop_points_if_mutable(grid3, 0.5)
        grid_cat = fvdb.cat([grid1, grid2, grid3])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) +  len(grid3))
        self.assertEqual(len(grid_cat), 8)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_random_grid(4), _make_random_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        random_drop_points_if_mutable(grid3, 0.5)
        grid_cat = fvdb.cat([grid1, grid2, grid3])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) +  len(grid3))
        self.assertEqual(len(grid_cat), 11)

        # Cat to the cat /ᐠ - ˕ -マ < Meow
        grid4, grid5 = _make_random_grid(2), _make_random_grid(5)
        grid_cat2 = fvdb.cat([grid_cat, grid1, grid4, grid5, grid3])
        self.assertTrue(torch.allclose(grid_cat2.ijk.jdata,
                                       torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata,
                                                  grid1.ijk.jdata, grid4.ijk.jdata, grid5.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat2), len(grid1) + len(grid2) +  len(grid3) + len(grid1) + \
                         len(grid4) + len(grid5) + len(grid3))
        self.assertEqual(len(grid_cat2), 25)

    @parameterized.expand(all_device_dtype_combos)
    def test_fvdb_cat_empty_grid(self, device, dtype, mutable):

        def _make_random_grid(batch_size):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            return fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)

        def _make_empty_grid(batch_size):
            sizes = [0 for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            return fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)

        # Test concat batches with 1 grid
        grid1, grid2 = _make_random_grid(1), _make_empty_grid(1)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 2)

        # Test concat batches with 1 grid and multiple grids
        grid1, grid2 = _make_empty_grid(2), _make_random_grid(1)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 3)

        # Test concat batches with multiple grids
        grid1, grid2 = _make_random_grid(2), _make_empty_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        grid_cat = fvdb.cat([grid1, grid2])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2))
        self.assertEqual(len(grid_cat), 7)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_empty_grid(1), _make_random_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        random_drop_points_if_mutable(grid3, 0.5)
        grid_cat = fvdb.cat([grid1, grid2, grid3])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) +  len(grid3))
        self.assertEqual(len(grid_cat), 8)

        # Test concat 3 grids
        grid1, grid2, grid3 = _make_random_grid(2), _make_empty_grid(4), _make_random_grid(5)
        random_drop_points_if_mutable(grid1, 0.5)
        random_drop_points_if_mutable(grid2, 0.5)
        random_drop_points_if_mutable(grid3, 0.5)
        grid_cat = fvdb.cat([grid1, grid2, grid3])
        random_drop_points_if_mutable(grid_cat, 0.5)
        self.assertTrue(torch.allclose(grid_cat.ijk.jdata, torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat), len(grid1) + len(grid2) +  len(grid3))
        self.assertEqual(len(grid_cat), 11)

        # Cat to the cat /ᐠ - ˕ -マ < Meow
        grid4, grid5 = _make_empty_grid(2), _make_random_grid(5)
        grid_cat2 = fvdb.cat([grid_cat, grid1, grid4, grid5, grid3])
        self.assertTrue(torch.allclose(grid_cat2.ijk.jdata,
                                       torch.cat([grid1.ijk.jdata, grid2.ijk.jdata, grid3.ijk.jdata,
                                                  grid1.ijk.jdata, grid4.ijk.jdata, grid5.ijk.jdata, grid3.ijk.jdata])))
        self.assertEqual(len(grid_cat2), len(grid1) + len(grid2) +  len(grid3) + len(grid1) + \
                         len(grid4) + len(grid5) + len(grid3))
        self.assertEqual(len(grid_cat2), 25)


if __name__ == '__main__':
    unittest.main()
