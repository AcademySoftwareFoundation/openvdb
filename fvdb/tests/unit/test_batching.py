import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb


all_device_dtype_combos = [
    ['cpu', torch.float32],
    ['cuda', torch.float32],
]

NVOX = 1_000_000

class TestBatching(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        idx = np.random.randint(num_grids)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)
        self.assertTrue(gridbatch.is_contiguous())

        voxels_idx_target = gridbatch.ijk.jdata[gridbatch.joffsets[idx, 0]:gridbatch.joffsets[idx, 1]]
        voxels_idx_pred = gridbatch[idx].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(gridbatch.is_contiguous())

        # Negative indices
        idx = -idx
        voxels_idx_target = gridbatch.ijk.jdata[gridbatch.joffsets[idx, 0]:gridbatch.joffsets[idx, 1]]
        voxels_idx_pred = gridbatch[idx].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(gridbatch.is_contiguous())

        # Negative indices
        idx = -1
        voxels_idx_target = gridbatch.ijk.jdata[gridbatch.joffsets[idx, 0]:gridbatch.joffsets[idx, 1]]
        voxels_idx_pred = gridbatch[idx].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertFalse(gridbatch[idx].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids_slice(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        idx = np.random.randint(num_grids)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 40_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)
        self.assertTrue(gridbatch.is_contiguous())

        all_ijk = gridbatch.ijk

        # Slice random segment
        sliced_offsets = gridbatch.joffsets[idx:idx+7]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[sliced_offsets[i, 0]:sliced_offsets[i, 1]] for i in range(sliced_offsets.shape[0])
        ])
        self.assertTrue(torch.equal(gridbatch[idx:idx+7].num_voxels, gridbatch.num_voxels[idx:idx+7]))
        voxels_idx_pred = gridbatch[idx:idx+7].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertFalse(gridbatch[idx:idx+7].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

        # Slice past the end
        sliced_offsets = gridbatch.joffsets[gridbatch.grid_count-3:gridbatch.grid_count+4]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[sliced_offsets[i, 0]:sliced_offsets[i, 1]] for i in range(sliced_offsets.shape[0])
        ])
        voxels_idx_pred = gridbatch[gridbatch.grid_count-3:gridbatch.grid_count+4].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[gridbatch.grid_count-3:gridbatch.grid_count+4].num_voxels,
                                    gridbatch.num_voxels[gridbatch.grid_count-3:gridbatch.grid_count+4]))
        self.assertFalse(gridbatch[gridbatch.grid_count-3:gridbatch.grid_count+4].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

        # Slice with step
        idx = np.random.randint(num_grids)
        step = np.random.randint(2, 4)
        sliced_offsets = gridbatch.joffsets[idx:idx+20:step]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[sliced_offsets[i, 0]:sliced_offsets[i, 1]] for i in range(sliced_offsets.shape[0])
        ])
        voxels_idx_pred = gridbatch[idx:idx+20:step].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[idx:idx+20:step].num_voxels, gridbatch.num_voxels[idx:idx+20:step]))
        self.assertFalse(gridbatch[idx:idx+20:step].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids_integer_array(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)
        self.assertTrue(gridbatch.is_contiguous())

        all_ijk = gridbatch.ijk

        # permutation
        pmt = torch.randperm(gridbatch.grid_count)
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertFalse(gridbatch[pmt].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

        # duplication
        pmt = torch.ones(2 * gridbatch.grid_count, dtype=torch.int32)
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertFalse(gridbatch[pmt].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

        # negative indices
        pmt = -torch.arange(gridbatch.grid_count)
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertFalse(gridbatch[pmt].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

        # mixed negative indices
        pmt = -torch.arange(gridbatch.grid_count)
        pmt = torch.cat([pmt, -pmt])
        pmt = pmt[torch.randperm(pmt.shape[0])]
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertFalse(gridbatch[pmt].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids_integer_array_list(self, device, dtype):

        def listify(t_):
            assert t_.dim() == 1
            return [int(t_[i].item()) for i in range(t_.shape[0])]

        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)
        self.assertTrue(gridbatch.is_contiguous())

        all_ijk = gridbatch.ijk

        # permutation
        pmt = listify(torch.randperm(gridbatch.grid_count))
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gridbatch[pmt].is_contiguous())

        # duplication
        pmt = listify(torch.ones(2 * gridbatch.grid_count, dtype=torch.int32))
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gridbatch[pmt].is_contiguous())

        # negative indices
        pmt = listify(-torch.arange(gridbatch.grid_count))
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gridbatch[pmt].is_contiguous())

        # mixed negative indices
        pmt = -torch.arange(gridbatch.grid_count)
        pmt = torch.cat([pmt, -pmt])
        pmt = listify(pmt[torch.randperm(pmt.shape[0])])
        offsets_pmt = gridbatch.joffsets[pmt]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[pmt].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[pmt].num_voxels, gridbatch.num_voxels[pmt]))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gridbatch[pmt].is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids_boolean_array(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        all_ijk = gridbatch.ijk

        mask = torch.rand(gridbatch.grid_count) > 0.5
        offsets_pmt = gridbatch.joffsets[mask]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        voxels_idx_pred = gridbatch[mask].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(torch.equal(gridbatch[mask].num_voxels, gridbatch.num_voxels[mask]))
        self.assertFalse(gridbatch[mask].is_contiguous())
        self.assertTrue(gridbatch.is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_getting_subgrids_boolean_list(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        all_ijk = gridbatch.ijk

        mask = torch.rand(gridbatch.grid_count) > 0.5
        mask = [bool(mask[i].item()) for i in range(mask.shape[0])]
        offsets_pmt = gridbatch.joffsets[mask]
        voxels_idx_target = torch.cat([
            all_ijk.jdata[offsets_pmt[i, 0]:offsets_pmt[i, 1]] for i in range(offsets_pmt.shape[0])
        ])
        self.assertTrue(torch.equal(gridbatch[mask].num_voxels, gridbatch.num_voxels[mask]))
        voxels_idx_pred = gridbatch[mask].ijk.jdata
        self.assertTrue(torch.equal(voxels_idx_target, voxels_idx_pred))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gridbatch[mask].is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_empty_grid(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device)
        gd = gridbatch.dual_grid()
        self.assertEqual(gd.total_voxels, 0)
        gc = gridbatch.coarsened_grid(2)
        self.assertEqual(gc.total_voxels, 0)
        gs = gridbatch.subdivided_grid(2)
        self.assertEqual(gs.total_voxels, 0)
        self.assertEqual(gridbatch.grid_count, 0)
        self.assertEqual(gridbatch.num_voxels.numel(), 0)
        self.assertEqual(gridbatch.total_voxels, 0)
        self.assertEqual(gridbatch.ijk.jdata.numel(), 0)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)
        self.assertTrue(gridbatch.is_contiguous())

        empty_mask = torch.zeros_like(gridbatch.num_voxels, dtype=torch.bool)
        self.assertEqual(gridbatch[empty_mask].num_voxels.numel(), 0)
        self.assertEqual(gridbatch[empty_mask].total_voxels, 0)
        self.assertEqual(gridbatch[empty_mask].ijk.jdata.numel(), 0)
        self.assertTrue(gridbatch[empty_mask].is_contiguous())

        empty_mask = torch.tensor([], dtype=torch.int32)
        self.assertEqual(gridbatch[empty_mask].num_voxels.numel(), 0)
        self.assertEqual(gridbatch[empty_mask].total_voxels, 0)
        self.assertEqual(gridbatch[empty_mask].ijk.jdata.numel(), 0)
        self.assertTrue(gridbatch[empty_mask].is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_grid_cat(self, device, dtype):
        num_grids = np.random.randint(64, 128)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        randpts_1 = fvdb.JaggedTensor(pts_list[:10])
        gridbatch_1 = fvdb.GridBatch(device)
        gridbatch_1.set_from_points(randpts_1, voxel_sizes=0.01)
        self.assertTrue(gridbatch_1.is_contiguous())

        randpts_2 = fvdb.JaggedTensor(pts_list[10:])
        gridbatch_2 = fvdb.GridBatch(device)
        gridbatch_2.set_from_points(randpts_2, voxel_sizes=0.01)
        self.assertTrue(gridbatch_2.is_contiguous())

        gridbatch_cat = fvdb.cat([gridbatch_1, gridbatch_2])
        self.assertTrue(gridbatch_cat.is_contiguous())

        self.assertEqual(gridbatch_cat.grid_count, gridbatch.grid_count)
        self.assertEqual(len(gridbatch), len(gridbatch_cat))
        self.assertTrue(torch.equal(gridbatch_cat.num_voxels, gridbatch.num_voxels))
        self.assertTrue(torch.equal(gridbatch_cat.ijk.jdata, gridbatch.ijk.jdata))

        gridbatch_cat_2 = fvdb.cat([gridbatch_1, fvdb.GridBatch(device), gridbatch_2])

        self.assertEqual(gridbatch_cat_2.grid_count, gridbatch.grid_count)
        self.assertEqual(len(gridbatch), len(gridbatch_cat_2))
        self.assertTrue(torch.equal(gridbatch_cat_2.num_voxels, gridbatch.num_voxels))
        self.assertTrue(torch.equal(gridbatch_cat_2.ijk.jdata, gridbatch.ijk.jdata))
        self.assertTrue(gridbatch_cat_2.is_contiguous())

        gridbatch_3 = gridbatch[[7, 11, 13, 15, 17, 19]]
        gridbatch_4 = gridbatch[[19, 21, 33, 44, 55]]
        self.assertFalse(gridbatch_3.is_contiguous())
        self.assertFalse(gridbatch_4.is_contiguous())

        pts_list_2 = [pts_list[i] for i in [7, 11, 13, 15, 17, 19, 19, 21, 33, 44, 55]]
        gridbatch_target = fvdb.GridBatch(device)
        gridbatch_target.set_from_points(fvdb.JaggedTensor(pts_list_2), voxel_sizes=0.01)

        gridbatch_cat_3 = fvdb.cat([gridbatch_3, gridbatch_4])
        self.assertEqual(gridbatch_cat_3.grid_count, gridbatch_target.grid_count)
        self.assertEqual(len(gridbatch_cat_3), len(gridbatch_target))
        self.assertTrue(torch.equal(gridbatch_cat_3.num_voxels, gridbatch_target.num_voxels))
        self.assertTrue(torch.equal(gridbatch_cat_3.ijk.jdata, gridbatch_target.ijk.jdata))
        self.assertTrue(gridbatch_cat_3.is_contiguous())

        # Stress test: concatenate a whole bunch of views and empty tensors created in different ways
        grids_to_cat = []
        pts_to_cat = []
        for i in range(7):
            if np.random.rand() > 0.5:
                grids_to_cat.append(fvdb.GridBatch(device))

            indices = torch.randperm(len(gridbatch))[:7]
            num_indices = indices.clone()

            if i % 3 == 0:
                indices = [int(indices[j].item()) for j in range(len(indices))]
            elif i % 2 == 0:
                mask = torch.zeros(len(gridbatch), dtype=torch.bool)
                mask[indices] = True
                indices = mask
                # mask select will produce sorted results
                num_indices, _ = torch.sort(num_indices)

            pts_to_cat.extend(
                [pts_list[int(num_indices[j].item())] for j in range(num_indices.shape[0])]
            )
            grids_to_cat.append(gridbatch[indices])
            self.assertFalse(grids_to_cat[-1].is_contiguous())

        grid_pts = fvdb.JaggedTensor(pts_to_cat)
        target_grid = fvdb.GridBatch(device)
        target_grid.set_from_points(grid_pts, voxel_sizes=0.01)

        gridbatch_cat_4 = fvdb.cat(grids_to_cat)

        self.assertEqual(gridbatch_cat_4.grid_count, target_grid.grid_count)
        self.assertEqual(len(gridbatch_cat_4), len(target_grid))
        self.assertTrue(torch.equal(gridbatch_cat_4.num_voxels, target_grid.num_voxels))
        self.assertTrue(torch.equal(gridbatch_cat_4.ijk.jdata, target_grid.ijk.jdata))
        self.assertTrue(gridbatch_cat_4.is_contiguous())

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_tensor_cat(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts1 = fvdb.JaggedTensor([torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)])
        pts2 = fvdb.JaggedTensor([torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)])
        pts3 = fvdb.JaggedTensor([torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)])

        pts_cat0 = fvdb.cat([pts1, pts2, pts3], dim=0)
        for g in range(num_grids):
            self.assertTrue(torch.equal(pts_cat0[g].jdata, pts1[g].jdata))
            self.assertTrue(torch.equal(pts_cat0[g + num_grids].jdata, pts2[g].jdata))
            self.assertTrue(torch.equal(pts_cat0[g + 2 * num_grids].jdata, pts3[g].jdata))

        pts_cat1 = fvdb.cat([pts1, pts2, pts3], dim=1)
        for g in range(num_grids):
            self.assertTrue(torch.equal(
                pts_cat1[g].jdata,
                torch.cat([pts1[g].jdata, pts2[g].jdata, pts3[g].jdata], dim=0)
            ))

        pts4 = pts1.jagged_like(torch.rand_like(pts1.jdata))
        pts5 = pts1.jagged_like(torch.rand_like(pts1.jdata))
        pts_cat2 = fvdb.cat([pts1, pts4, pts5], dim=2)
        for g in range(num_grids):
            self.assertTrue(torch.equal(
                pts_cat2[g].jdata,
                torch.cat([pts1[g].jdata, pts4[g].jdata, pts5[g].jdata], dim=1)
            ))

    @parameterized.expand(all_device_dtype_combos)
    def test_contiguous(self, device, dtype):
        num_grids = np.random.randint(32, 64)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        # Stress test: concatenate a whole bunch of views and empty tensors created in different ways
        grids_to_cat = []
        pts_to_cat = []
        for i in range(7):
            if np.random.rand() > 0.5:
                grids_to_cat.append(fvdb.GridBatch(device))

            indices = torch.randperm(len(gridbatch))[:7]
            num_indices = indices.clone()

            if i % 3 == 0:
                indices = [int(indices[j].item()) for j in range(len(indices))]
            elif i % 2 == 0:
                mask = torch.zeros(len(gridbatch), dtype=torch.bool)
                mask[indices] = True
                indices = mask
                # mask select will produce sorted results
                num_indices, _ = torch.sort(num_indices)

            pts_to_cat.extend(
                [pts_list[int(num_indices[j].item())] for j in range(num_indices.shape[0])]
            )
            grids_to_cat.append(gridbatch[indices])
            self.assertFalse(grids_to_cat[-1].is_contiguous())
            contig_version = grids_to_cat[-1].contiguous()
            self.assertTrue(contig_version.is_contiguous())
            self.assertEqual(contig_version.grid_count, grids_to_cat[-1].grid_count)
            self.assertTrue(torch.equal(contig_version.ijk.jdata, grids_to_cat[-1].ijk.jdata))
            self.assertTrue(torch.equal(contig_version.num_voxels, grids_to_cat[-1].num_voxels))

    @parameterized.expand(all_device_dtype_combos)
    def test_views_of_views(self, device, dtype):
        num_grids = 64
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        grids_to_cat = []
        last_grid = gridbatch
        global_indices = torch.arange(len(gridbatch))
        for i in range(7):
            if np.random.rand() > 0.5:
                grids_to_cat.append(fvdb.GridBatch(device))

            indices = torch.randperm(len(last_grid))[:len(last_grid)-np.random.randint(1, 3)]
            num_indices = indices.clone()

            if i % 3 == 0:
                indices = [int(indices[j].item()) for j in range(len(indices))]
            elif i % 2 == 0:
                mask = torch.zeros(len(last_grid), dtype=torch.bool)
                mask[indices] = True
                indices = mask
                # mask select will produce sorted results
                num_indices, _ = torch.sort(num_indices)

            global_indices = global_indices[indices]
            last_grid = last_grid[indices]
            self.assertFalse(last_grid.is_contiguous())
            self.assertTrue(gridbatch.is_contiguous())


            self.assertTrue(torch.equal(last_grid.ijk.jdata, gridbatch[global_indices].ijk.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_reverse_steps(self, device, dtype):
        num_grids = 64
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.01)

        gb2 = gridbatch[7:-100]
        self.assertTrue(gb2.is_contiguous())
        self.assertEqual(len(gb2), 0)

        gb2 = gridbatch[9:4:-1]
        self.assertEqual(len(gb2), len(np.arange(len(gridbatch))[9:4:-1]))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[9:4:-1][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())

        gb2 = gridbatch[9::-1]
        self.assertEqual(len(gb2), len(np.arange(len(gridbatch))[9::-1]))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[9::-1][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())

        gb2 = gridbatch[::-1]
        self.assertEqual(len(gb2), len(gridbatch))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[::-1][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())

        gb2 = gridbatch[::-2]
        self.assertEqual(len(gb2), len(np.arange(len(gridbatch))[::-2]))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[::-2][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())

        gb2 = gridbatch[::-3]
        self.assertEqual(len(gb2), len(np.arange(len(gridbatch))[::-3]))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[::-3][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())

        gb2 = gridbatch[:5:-7]
        self.assertEqual(len(gb2), len(np.arange(len(gridbatch))[:5:-7]))
        for i, gb2i in enumerate(gb2):
            idx = np.arange(len(gridbatch))[:5:-7][i]
            self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gb2i.ijk.jdata))
        self.assertTrue(gridbatch.is_contiguous())
        self.assertFalse(gb2.is_contiguous())


if __name__ == '__main__':
    unittest.main()
