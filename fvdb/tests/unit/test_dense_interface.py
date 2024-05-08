import unittest

import torch
import numpy as np
from parameterized import parameterized

from fvdb import GridBatch

from .common import random_drop_points_if_mutable, sparse_grid_from_dense_cube

all_device_dtype_combos = [
    ['cpu', torch.float16, False],
    ['cuda', torch.float16, False],
    ['cpu', torch.float32, False],
    ['cuda', torch.float32, False],
    ['cpu', torch.float64, False],
    ['cuda', torch.float64, False],
    ['cpu', torch.float16, True],
    ['cuda', torch.float16, True],
    ['cpu', torch.float32, True],
    ['cuda', torch.float32, True],
    ['cpu', torch.float64, True],
    ['cuda', torch.float64, True]
]

all_device_combos = [
    ['cpu'],
    ['cuda'],
]


class TestUtils(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_dense(self, device, dtype, mutable):
        dense_vdb = sparse_grid_from_dense_cube([10, 11, 12], (-2.0, -2.0, -2.0), (1.0, 1.0, 1.0),
                                                voxel_center=False, mutable=mutable, device=device)
        self.assertTrue(dense_vdb.total_voxels == 10 * 11 * 12)

        vdb_coords = dense_vdb.grid_to_world(dense_vdb.ijk.float()).jdata
        self.assertAlmostEqual(torch.min(vdb_coords).item(), -2.0 + 3 / 12 * 0.5, places=6)
        self.assertAlmostEqual(torch.max(vdb_coords).item(), 1.0 - 3 / 12 * 0.5, places=6)

        vdb_feature = torch.randn((dense_vdb.total_voxels, 4), device=device, dtype=dtype)
        dense_feature = dense_vdb.read_into_dense(vdb_feature).squeeze(0)
        for i in range(10):
            for j in range(11):
                for k in range(12):
                    vdb_f = vdb_feature[dense_vdb.ijk_to_index(torch.tensor([[i, j, k]], device=device)).jdata]
                    dense_f = dense_feature[i, j, k, :]  # dense_feature[:, k, j, i]
                    self.assertTrue(torch.allclose(vdb_f, dense_f))
        vdb_feature2 = dense_vdb.read_from_dense(dense_feature.unsqueeze(0)).jdata
        self.assertTrue(torch.allclose(vdb_feature, vdb_feature2))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)
        random_drop_points_if_mutable(grid)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, device=device, dtype=dtype)
        ijk = grid.ijk_enabled.jdata

        for _ in range(10):
            dense_origin = torch.tensor([
                np.random.randint(low=ijk.min(0).values[i].item(), high=ijk.max(0).values[i].item()) for i in range(3)
            ]).to(torch.long).to(device)

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < max_bound.unsqueeze(0), dim=-1))

            grid_index = grid.ijk_to_index(ijk).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * dense_size[2] * dense_size[1] + j * dense_size[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.read_from_dense(random_grid.unsqueeze(0), dense_origin).jdata

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_multidim(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)
        random_drop_points_if_mutable(grid)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        ijk = grid.ijk_enabled.jdata

        for _ in range(10):
            dense_origin = torch.tensor([
                np.random.randint(low=ijk.min(0).values[i].item(), high=ijk.max(0).values[i].item()) for i in range(3)
            ]).to(torch.long).to(device)

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1))

            grid_index = grid.ijk_to_index(ijk).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid.shape[2] * random_grid.shape[1] + j * random_grid.shape[2] + k

            feat_shape = [c for c in random_grid.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid.view(-1, *feat_shape)[offset]

            pred_sparse = grid.read_from_dense(random_grid.unsqueeze(0), dense_origin).jdata

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_from_dense_multidim_grad(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)
        random_drop_points_if_mutable(grid)

        dense_size = [np.random.randint(low=10, high=128) for _ in range(3)]
        random_grid = torch.randn(*dense_size, 4, 3, 2, device=device, dtype=dtype)
        random_grid_copy = random_grid.clone()
        random_grid.requires_grad = True
        random_grid_copy.requires_grad = True

        ijk = grid.ijk_enabled.jdata

        for _ in range(10):
            dense_origin = torch.tensor([
                np.random.randint(low=ijk.min(0).values[i].item(), high=ijk.max(0).values[i].item()) for i in range(3)
            ]).to(torch.long).to(device)

            ijk_offset = ijk - dense_origin.unsqueeze(0)
            max_bound = torch.tensor(random_grid.shape[:3], device=device, dtype=torch.long)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < max_bound.unsqueeze(0), dim=1))

            grid_index = grid.ijk_to_index(ijk).jdata[keep_mask]
            i, j, k = ijk_offset[keep_mask, 0], ijk_offset[keep_mask, 1], ijk_offset[keep_mask, 2]
            offset = i * random_grid_copy.shape[2] * random_grid_copy.shape[1] + j * random_grid_copy.shape[2] + k

            feat_shape = [c for c in random_grid_copy.shape[3:]]
            target_sparse = torch.zeros(grid.total_voxels, *feat_shape, device=device, dtype=dtype)
            target_sparse[grid_index] = random_grid_copy.view(-1, *feat_shape)[offset]
            loss_copy = target_sparse.sum()
            loss_copy.backward()

            pred_sparse = grid.read_from_dense(random_grid.unsqueeze(0), dense_origin).jdata
            loss = pred_sparse.sum()
            loss.backward()

            self.assertEqual(torch.abs(target_sparse - pred_sparse).max().item(), 0.0)
            self.assertTrue(torch.all(target_sparse == pred_sparse))

            assert random_grid.grad is not None
            assert random_grid_copy.grad is not None
            self.assertTrue(torch.equal(random_grid.grad, random_grid_copy.grad))


    @parameterized.expand(all_device_dtype_combos)
    def test_read_into_dense(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)

        random_drop_points_if_mutable(grid)

        ijk = grid.ijk_enabled.jdata
        sparse_data = torch.randn((grid.total_voxels, 4), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor([np.random.randint(low=min_crop_coord[i].item(),
                                                       high=max_crop_coord[i].item()) for i in range(3)]).to(device)
            crop_size = torch.tensor([np.random.randint(low=1, high=max_crop_size[i].item()) for i in range(3)]).to(device)


            target_crop = torch.zeros(*crop_size.cpu().numpy(), sparse_data.shape[-1], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1))
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, sparse_data.shape[-1])[idx] = sparse_data[grid.enabled_mask.jdata][keep_mask]

            pred_crop = grid.read_into_dense(sparse_data, crop_min, crop_size).squeeze(0)

            self.assertTrue(torch.all(pred_crop == target_crop))

    @parameterized.expand(all_device_dtype_combos)
    def test_read_into_dense_multidim(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)

        random_drop_points_if_mutable(grid)

        ijk = grid.ijk_enabled.jdata
        sparse_data = torch.randn((grid.total_voxels, 4, 3, 2), device=device, dtype=dtype)

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor([np.random.randint(low=min_crop_coord[i].item(),
                                                       high=max_crop_coord[i].item()) for i in range(3)]).to(device)
            crop_size = torch.tensor([np.random.randint(low=1, high=max_crop_size[i].item()) for i in range(3)]).to(device)

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1))
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data[grid.enabled_mask.jdata][keep_mask]

            pred_crop = grid.read_into_dense(sparse_data, crop_min, crop_size).squeeze(0)

            self.assertTrue(torch.all(pred_crop == target_crop))


    @parameterized.expand(all_device_dtype_combos)
    def test_read_into_dense_multidim_grad(self, device, dtype, mutable):

        random_points = torch.randn(100000, 3).to(device).to(dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(random_points, voxel_sizes=0.1, origins=[0.0] * 3)

        random_drop_points_if_mutable(grid)

        ijk = grid.ijk_enabled.jdata
        sparse_data = torch.randn((grid.total_voxels, 4, 3, 2), device=device, dtype=dtype)
        sparse_data_copy = sparse_data.clone()
        sparse_data.requires_grad = True
        sparse_data_copy.requires_grad = True

        bbmin = ijk.min(0).values
        bbmax = ijk.max(0).values
        bbsize = bbmax - bbmin

        # Generate random crops
        min_crop_coord = bbmin - bbsize // 10
        max_crop_size = bbsize + bbsize // 10
        max_crop_coord = min_crop_coord + max_crop_size
        for _ in range(10):
            crop_min = torch.tensor([np.random.randint(low=min_crop_coord[i].item(),
                                                       high=max_crop_coord[i].item()) for i in range(3)]).to(device)
            crop_size = torch.tensor([np.random.randint(low=1, high=max_crop_size[i].item()) for i in range(3)]).to(device)

            target_crop = torch.zeros(*crop_size.cpu().numpy(), *sparse_data.shape[1:], dtype=dtype, device=device)
            ijk_offset = ijk - crop_min.unsqueeze(0)
            keep_mask = torch.logical_and(
                            torch.all(ijk_offset >= 0, dim=-1),
                            torch.all(ijk_offset < crop_size.unsqueeze(0), dim=1))
            write_ijk = ijk_offset[keep_mask].contiguous()
            idx = write_ijk[:, 0] * crop_size[1] * crop_size[2] + write_ijk[:, 1] * crop_size[2] + write_ijk[:, 2]
            target_crop.view(-1, *sparse_data.shape[1:])[idx] = sparse_data_copy[grid.enabled_mask.jdata][keep_mask]

            loss_copy = target_crop.sum()
            loss_copy.backward()


            pred_crop = grid.read_into_dense(sparse_data, crop_min, crop_size).squeeze(0)
            loss = pred_crop.sum()
            loss.backward()

            assert sparse_data.grad is not None
            assert sparse_data_copy.grad is not None
            self.assertEqual(torch.abs(sparse_data.grad - sparse_data_copy.grad).max().item(), 0.0)
            self.assertTrue(torch.all(pred_crop == target_crop))


    @parameterized.expand(all_device_combos)
    def test_build_from_dense(self, device):
        gorigin = tuple([int(a.item()) for a in torch.randint(-32, 32, (3,))])
        gsize = torch.randint(4, 8, (3,))
        gsize = (int(gsize[0].item()), int(gsize[1].item()), int(gsize[2].item()))
        num_dense_vox = gsize[0] * gsize[1] * gsize[2]

        grid = GridBatch(device=device)
        grid.set_from_dense_grid(1, gsize, gorigin, 0.1, [0.0] * 3)
        grid_ijk = grid.ijk.jdata
        target_min_ijk = torch.zeros(3).to(grid_ijk) + torch.tensor(gorigin).to(grid_ijk)
        target_max_ijk = torch.tensor(gsize).to(grid_ijk) - 1 + torch.tensor(gorigin).to(grid_ijk)

        self.assertEqual(grid.total_voxels, num_dense_vox)
        self.assertTrue(torch.all(grid_ijk.min(0)[0] == target_min_ijk))
        self.assertTrue(torch.all(grid_ijk.max(0)[0] == target_max_ijk))

        ijk_mask = torch.stack([
            torch.randint(0, gsize[0], (100,), device=device),
            torch.randint(0, gsize[1], (100,), device=device),
            torch.randint(0, gsize[2], (100,), device=device),
        ], dim=-1)
        dense_mask = torch.zeros(*gsize, dtype=torch.bool).to(device)
        mask_coord_set = set()
        for idx in range(ijk_mask.shape[0]):
            i, j, k = [a.item() for a in ijk_mask[idx]]
            dense_mask[i, j, k] = True
            mask_coord_set.add((i, j, k))

        grid.set_from_dense_grid(1, gsize, gorigin, 0.1, [0.0] * 3, mask=dense_mask)
        grid_ijk = grid.ijk.jdata

        self.assertEqual(len(mask_coord_set), grid_ijk.shape[0])

        grid_ijk -= torch.tensor(gorigin).unsqueeze(0).to(grid_ijk)
        pred_set = set()
        for idx in range(grid_ijk.shape[0]):
            i, j, k = [a.item() for a in grid_ijk[idx]]
            pred_set.add((i, j, k))

        self.assertEqual(pred_set, mask_coord_set)


if __name__ == '__main__':
    unittest.main()
