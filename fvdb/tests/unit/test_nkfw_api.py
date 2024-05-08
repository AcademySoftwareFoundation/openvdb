import functools
import os
import unittest
import weakref

import numpy as np
import torch
from torch import testing

from .nkfw_api.backend.fvdb import SparseFeatureHierarchy as FVDBBackend
from .nkfw_api.backend.hash_table import SparseFeatureHierarchy as HashTableBackend

assert_tensor_equal = functools.partial(testing.assert_close, rtol=0, atol=0)


def assert_tensor_set_equal(t_a: torch.Tensor, t_b: torch.Tensor, dim: int = 0):
    t_a = torch.sort(t_a, dim=dim).values
    t_b = torch.sort(t_b, dim=dim).values
    assert_tensor_equal(t_a, t_b)


def permute_neighbour(nn_info, perm_src, perm_tgt):

    def transpose_permutation(perm: torch.Tensor):
        p = torch.empty(perm.size(0), dtype=torch.long, device=perm.device)
        p[perm] = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        return p

    src_inds = transpose_permutation(perm_src)[nn_info[:, 0]]
    tgt_inds = transpose_permutation(perm_tgt)[nn_info[:, 1]]
    return torch.concat([torch.stack([src_inds, tgt_inds], dim=1), nn_info[:, 2:]], dim=1)


class TestNKFWAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        self.range_kernel = weakref.WeakMethod(self._get_range_kernel)
        self.backend_cls = FVDBBackend
        # self.backend_cls = HashTableBackend

    cached_kernels = None

    def _get_range_kernel(self, n_range):
        assert n_range % 2 == 1, "target_range must be odd."
        if self.cached_kernels is None:
            data = np.load(os.path.join(os.path.dirname(__file__), "nkfw_api", "kernel.npz"))
            self.cached_kernels = data['kernel']
        kernel = torch.tensor(self.cached_kernels.copy()[:n_range ** 3],
                              dtype=torch.int, device=self.device)
        return kernel

    def test_meta_ops(self):
        index = self.backend_cls(4, 0.1, self.device, self.range_kernel)
        self.assertEqual(index.depth, 4)
        self.assertEqual(index.get_stride(3), 8)

        level_0_ijk = torch.tensor([
            [0, 0, 0], [0, 0, 1],
            [1, 0, 0], [1, 0, 1]
        ], dtype=torch.int, device=self.device)
        level_1_ijk = torch.tensor([
            [0, 2, 0], [0, 0, 0]
        ], dtype=torch.int, device=self.device)

        level_0_ijk, _ = index.update_coords(0, level_0_ijk)
        level_1_ijk, _ = index.update_coords(1, level_1_ijk)

        self.assertEqual(index.get_num_voxels(0), 4)
        self.assertEqual(index.get_num_voxels(1), 2)
        self.assertEqual(index.get_num_voxels(2), 0)
        self.assertEqual(index.get_num_voxels(3), 0)

        self.assertEqual(index.voxel_size[0], 0.1)
        self.assertEqual(index.voxel_size[1], 0.1)
        self.assertEqual(index.voxel_size[2], 0.1)

        assert_tensor_set_equal(index.get_coords(0), level_0_ijk)
        assert_tensor_set_equal(index.get_coords(0, expand=3), torch.tensor([
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, -1, 2], [-1, 0, -1], [-1, 0, 0],
            [-1, 0, 1], [-1, 0, 2], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, 2],
            [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, -1, 2], [0, 0, -1], [0, 0, 0],
            [0, 0, 1], [0, 0, 2], [0, 1, -1], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, -1, -1],
            [1, -1, 0], [1, -1, 1], [1, -1, 2], [1, 0, -1], [1, 0, 0], [1, 0, 1],
            [1, 0, 2], [1, 1, -1], [1, 1, 0], [1, 1, 1], [1, 1, 2], [2, -1, -1],
            [2, -1, 0], [2, -1, 1], [2, -1, 2], [2, 0, -1], [2, 0, 0], [2, 0, 1],
            [2, 0, 2], [2, 1, -1], [2, 1, 0], [2, 1, 2], [2, 1, 1],
        ], dtype=torch.int, device=self.device))
        assert_tensor_set_equal(index.get_voxel_centers(1), torch.tensor([
            [0.1, 0.3, 0.1], [0.1, 0.1, 0.1]
        ], device=self.device))

        # Empty indices
        index.update_coords(2, None)
        self.assertEqual(index.get_coords(2).size(0), 0)

    def test_neighbours(self):
        index = self.backend_cls(2, 1.0, self.device, self.range_kernel)

        level_0_ijk = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]
        ], dtype=torch.int, device=self.device)

        level_1_ijk = torch.tensor([
            [0, 0, 2], [0, 0, 0]
        ], dtype=torch.int, device=self.device)

        level_0_ijk, perm_0 = index.update_coords(0, level_0_ijk)
        level_1_ijk, perm_1 = index.update_coords(1, level_1_ijk)

        # Ring-1 NN -- same layer
        src_ids, tgt_ids, n_types, n_counts = index.get_self_neighbours(0, 0, target_range=1)
        assert_tensor_equal(src_ids, tgt_ids)
        assert_tensor_set_equal(src_ids, torch.arange(4, dtype=torch.long, device=self.device))
        assert_tensor_equal(n_types, torch.zeros((4, 3), device=self.device))

        # Ring-3 NN -- same layer
        src_ids, tgt_ids, n_types, n_counts = index.get_self_neighbours(0, 0, target_range=3)
        nn_info = torch.cat([src_ids[:, None], tgt_ids[:, None], n_types.long()], dim=1)
        assert_tensor_set_equal(nn_info, permute_neighbour(torch.tensor(
            [[0, 0, 0, 0,  0],
             [1, 1, 0, 0,  0],
             [2, 2, 0, 0,  0],
             [3, 3, 0, 0,  0],
             [0, 3, 0, 1,  0],
             [1, 3, 0, 1, -1],
             [0, 1, 0, 0,  1],
             [1, 2, 0, 0,  1],
             [1, 0, 0, 0, -1],
             [2, 1, 0, 0, -1],
             [3, 1, 0, -1, 1],
             [3, 0, 0, -1, 0]],
            dtype=torch.long, device=self.device), perm_0, perm_0))
        assert_tensor_equal(n_counts, torch.tensor([
            4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], dtype=torch.long, device=self.device))

        # Ring-1 NN -- across layer
        src_ids, tgt_ids, n_types, n_counts = index.get_self_neighbours(0, 1, target_range=1)
        nn_info = torch.cat([src_ids[:, None], tgt_ids[:, None], n_types.long()], dim=1)
        assert_tensor_set_equal(nn_info, permute_neighbour(torch.tensor(
            [[0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [2, 0, 0, 0, 0],
             [3, 1, 0, 0, 0]],
            dtype=torch.long, device=self.device), perm_0, perm_1))
        assert_tensor_equal(n_counts, torch.tensor([4], dtype=torch.long, device=self.device))

        # Ring-3 NN -- across layer
        src_ids, tgt_ids, n_types, n_counts = index.get_self_neighbours(0, 1, target_range=3)
        nn_info = torch.cat([src_ids[:, None], tgt_ids[:, None], n_types.long()], dim=1)
        assert_tensor_set_equal(nn_info, permute_neighbour(torch.tensor(
            [[0, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [2, 0, 0, 0, 0],
             [3, 1, 0, 0, 0],
             [0, 0, 0, 0, 2],
             [1, 0, 0, 0, 1],
             [3, 0, 0, 0, 2],
             [2, 1, 0, 0, -1]],
            dtype=torch.long, device=self.device), perm_0, perm_1))
        assert_tensor_equal(n_counts, torch.tensor([
            4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,  # (0,0,2) and (0,0,1) belongs to the same NN.
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], dtype=torch.long, device=self.device))

        # Ring-3 NN -- across layer (the other way around)
        src_ids, tgt_ids, n_types, n_counts = index.get_self_neighbours(1, 0, target_range=3)
        nn_info = torch.cat([src_ids[:, None], tgt_ids[:, None], n_types.long()], dim=1)
        assert_tensor_set_equal(nn_info, permute_neighbour(torch.tensor(
            [[1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 2, 0, 0, 0],
             [1, 3, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 2, 0, 0, 0]], dtype=torch.long, device=self.device), perm_1, perm_0))

        # coordinate-based NN (like hash table)
        src_ids, tgt_ids, n_types, n_counts = index.get_coords_neighbours(torch.tensor([
            [0., 0., 0.], [0., 0., 2.]
        ], device=self.device), 1, 0, nn_kernel=self._get_range_kernel(1))
        nn_info = torch.cat([src_ids[:, None], tgt_ids[:, None], n_types.long()], dim=1)
        assert_tensor_set_equal(nn_info, torch.tensor(
            [[0, 0, 0, 0, 0],
             [1, 2, 0, 0, 0]], dtype=torch.long, device=self.device))

    def test_split_splat(self):
        index = self.backend_cls(2, 1.0, self.device, self.range_kernel)
        index.update_coords(0, torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]
        ], dtype=torch.int, device=self.device))
        index.update_coords(1, torch.tensor([
            [0, 0, 2], [0, 0, 0]
        ], dtype=torch.int, device=self.device))

        res = index.splat_data(
            # World coordinates...
            xyz=torch.tensor([
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 2.0]
            ], device=self.device),
            data_depth=0,
            data=torch.tensor([
                [10.0, 8.0],
                [300., 400.]
            ], device=self.device),
            check_corr=False
        )
        assert_tensor_equal(res, torch.tensor(
            [[10., 8.], [150., 200.],
             [150., 200.], [0., 0.]],
            dtype=torch.float32, device=self.device))


if __name__ == '__main__':
    unittest.main()
