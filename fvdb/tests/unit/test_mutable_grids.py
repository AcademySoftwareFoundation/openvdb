import unittest

import torch
from parameterized import parameterized

from fvdb import GridBatch

all_device_dtype_combos = [
    ['cpu', torch.float32],
    ['cuda', torch.float32],
    ['cpu', torch.float64],
    ['cuda', torch.float64]
]

all_device_combos = [
    ['cpu'],
    ['cuda'],
]


class TestUtils(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_active_grid_coords_is_the_same_as_immutable_if_all_enabled(self, device, dtype):
        p = torch.rand(7777, 3).to(dtype).to(device)

        grid1 = GridBatch(mutable=False, device=device)
        grid1.set_from_points(p, [-1]*3, [1]*3, 0.025, torch.zeros(3))

        grid2 = GridBatch(mutable=True, device=device)
        grid2.set_from_points(p, [-1]*3, [1]*3, 0.025, torch.zeros(3))

        self.assertTrue(torch.all(grid1.ijk.jdata == grid2.ijk.jdata))
        self.assertEqual(grid1.total_voxels, grid2.total_voxels)
        self.assertEqual(grid1.total_enabled_voxels, grid2.total_enabled_voxels)
        self.assertEqual(grid1.total_voxels, grid2.total_enabled_voxels)

        drop_idx = torch.randperm(grid1.ijk.jdata.shape[0])[:444]
        drop_ijk = grid1.ijk.jdata[drop_idx]

        grid2.disable_ijk(drop_ijk)

        self.assertNotEqual(grid1.ijk_enabled.jdata.shape[0],
                            grid2.ijk_enabled.jdata.shape[0])
        self.assertEqual(grid1.ijk.jdata.shape[0],
                         grid2.ijk.jdata.shape[0])
        self.assertEqual(grid1.total_voxels, grid2.total_voxels)
        self.assertNotEqual(grid1.total_enabled_voxels, grid2.total_enabled_voxels)
        self.assertNotEqual(grid1.total_voxels, grid2.total_enabled_voxels)

        grid2.enable_ijk(drop_ijk)

        self.assertTrue(torch.all(grid1.ijk_enabled.jdata == grid2.ijk_enabled.jdata))
        self.assertTrue(torch.all(grid1.ijk.jdata == grid2.ijk.jdata))
        self.assertEqual(grid1.total_voxels, grid2.total_voxels)
        self.assertEqual(grid1.total_enabled_voxels, grid2.total_enabled_voxels)
        self.assertEqual(grid1.total_voxels, grid2.total_enabled_voxels)

    @parameterized.expand(all_device_dtype_combos)
    def test_active_grid_coords_before_and_after_masking(self, device, dtype):
        voxel_size = 0.025
        p = torch.rand(7777, 3).to(dtype).to(device)

        grid_static = GridBatch(mutable=False, device=device)
        grid_static.set_from_points(p, [-1]*3, [1]*3, 0.025, torch.zeros(3))

        p_ijk = grid_static.ijk.jdata

        grid_dynamic = GridBatch(mutable=True, device=device)
        grid_dynamic.set_from_ijk(p_ijk, voxel_sizes=voxel_size, origins=torch.zeros(3))

        self.assertTrue(torch.all(grid_static.ijk.jdata == grid_dynamic.ijk.jdata))
        self.assertTrue(torch.all(grid_static.ijk_enabled.jdata == grid_dynamic.ijk_enabled.jdata))
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_voxels)
        self.assertEqual(grid_static.total_enabled_voxels, grid_dynamic.total_enabled_voxels)
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_enabled_voxels)

        drop_mask = torch.rand(grid_static.total_voxels) > 0.5
        drop_ijk = p_ijk[drop_mask]
        keep_ijk = p_ijk[~drop_mask]

        grid_dynamic.disable_ijk(drop_ijk)

        self.assertTrue(torch.all(grid_dynamic.ijk_enabled.jdata == keep_ijk))
        self.assertEqual(grid_dynamic.total_voxels, grid_static.total_enabled_voxels)
        self.assertEqual(grid_dynamic.total_enabled_voxels, keep_ijk.shape[0])

        re_enable_mask = torch.rand(grid_static.total_voxels) > 0.5
        re_enable_ijk = p_ijk[re_enable_mask]

        grid_dynamic.enable_ijk(re_enable_ijk)

        expected_ijk = p_ijk[torch.logical_or(~drop_mask, re_enable_mask)]
        self.assertTrue(torch.all(grid_dynamic.ijk_enabled.jdata == expected_ijk))

    @parameterized.expand(all_device_dtype_combos)
    def test_mask_and_unmask(self, device, dtype):
        voxel_size = 0.025
        p = torch.rand(7777, 3).to(dtype).to(device)

        grid_static = GridBatch(mutable=False, device=device)
        grid_static.set_from_points(p, [-1]*3, [1]*3, voxel_sizes=0.025, origins=torch.zeros(3))

        p_ijk = grid_static.ijk.jdata

        grid_dynamic = GridBatch(mutable=True, device=device)
        grid_dynamic.set_from_ijk(p_ijk, voxel_sizes=voxel_size, origins=torch.zeros(3))

        self.assertTrue(torch.all(grid_static.ijk.jdata == grid_dynamic.ijk.jdata))
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_voxels)
        self.assertEqual(grid_static.total_enabled_voxels, grid_dynamic.total_enabled_voxels)
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_enabled_voxels)

        grid_dynamic.disable_ijk(grid_static.ijk)

        # Test num_enabled_voxels functionality
        for i in range(grid_dynamic.grid_count):
            self.assertEqual(0, grid_dynamic.num_enabled_voxels_at(i))
            self.assertEqual(0, grid_dynamic.num_enabled_voxels[i])
        self.assertEqual(0, grid_dynamic.total_enabled_voxels)
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_voxels)

        grid_dynamic.enable_ijk(grid_static.ijk.jdata)

        for i in range(grid_dynamic.grid_count):
            self.assertEqual(grid_static.num_enabled_voxels_at(i), grid_dynamic.num_enabled_voxels_at(i))
            self.assertEqual(grid_static.num_enabled_voxels[i], grid_dynamic.num_enabled_voxels[i])
        self.assertTrue(torch.all(grid_static.ijk.jdata == grid_dynamic.ijk.jdata))
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_voxels)
        self.assertEqual(grid_static.total_enabled_voxels, grid_dynamic.total_enabled_voxels)
        self.assertEqual(grid_static.total_voxels, grid_dynamic.total_enabled_voxels)


if __name__ == '__main__':
    unittest.main()
