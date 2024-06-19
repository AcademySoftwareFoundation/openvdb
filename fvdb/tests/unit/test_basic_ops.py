import itertools
import os
import pickle
import unittest

import numpy as np
import torch
from parameterized import parameterized

from fvdb import GridBatch, sparse_grid_from_ijk, JaggedTensor
from fvdb.utils import volume_render
import fvdb

from .common import (make_dense_grid_and_point_data,
                     make_sparse_grid_and_point_data,
                     random_drop_points_if_mutable,
                     test_expand)

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

    @parameterized.expand(['cpu', 'cuda'])
    def test_joffsets_mutable(self, device):
        # This is a test for https://github.com/voxel-foundation/feature-vdb/issues/196
        grid = GridBatch(mutable=True, device=device)
        grid.set_from_dense_grid(3, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        joffsets = grid.joffsets
        self.assertTrue(joffsets.shape[0] == 3 and joffsets.shape[1] == 2)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxel_neighborhood(self, device, dtype, mutable):
        randvox = torch.randint(0, 256, size=(10_000, 3), dtype=torch.int32).to(device)
        randvox = torch.cat([randvox, randvox + torch.ones(1, 3).to(randvox)], dim=0)  # Ensure there are always neighbors

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_ijk(randvox)
        random_drop_points_if_mutable(grid)

        gt_nhood = torch.zeros((randvox.shape[0], 3, 3, 3), dtype=torch.int32).to(device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    off = torch.tensor([[i-1, j-1, k-1]]).to(randvox)
                    nh_ijk = randvox + off
                    idx = grid.ijk_to_index(nh_ijk).jdata
                    mask = grid.coords_in_active_voxel(nh_ijk).jdata
                    gt_nhood[:, i, j, k] = torch.where(mask, idx, -torch.ones_like(idx))

        nhood = grid.neighbor_indexes(randvox, 1, 0).jdata

        self.assertTrue(torch.equal(nhood, gt_nhood))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1]*3, [1]*3, vox_size, vox_origin)
        grid = grid.dual_grid()
        random_drop_points_if_mutable(grid)

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid.world_to_grid(pts).jdata

        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)),
                        f"max_diff = {torch.abs(pred_dual_coordinates - target_dual_coordinates).max()}")

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_primal(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1]*3, [1]*3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)

        target_primal_coordinates = (pts - vox_origin) / vox_size
        pred_primal_coordinates = grid.world_to_grid(pts).jdata

        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_dual_grad(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        pts.requires_grad = True

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        grid = grid.dual_grid()
        random_drop_points_if_mutable(grid)

        pred_dual_coordinates = grid.world_to_grid(pts).jdata
        grad_out = torch.rand_like(pred_dual_coordinates)
        pred_dual_coordinates.backward(grad_out)

        assert pts.grad is not None  # Removes type errors with .grad
        pred_grad = pts.grad.clone()

        pts.grad.zero_()
        self.assertFalse(torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(pts.grad, torch.zeros_like(pts.grad)))

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        target_dual_coordinates.backward(grad_out)

        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))


    @parameterized.expand(all_device_dtype_combos)
    def test_world_to_primal_grad(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        pts.requires_grad = True

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)

        pred_primal_coordinates = grid.world_to_grid(pts).jdata
        grad_out = torch.rand_like(pred_primal_coordinates)
        pred_primal_coordinates.backward(grad_out)

        assert pts.grad is not None  # Removes type errors with .grad
        pred_grad = pts.grad.clone()

        pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(pts.grad, torch.zeros_like(pts.grad)))

        target_primal_coordinates = ((pts - vox_origin) / vox_size)
        target_primal_coordinates.backward(grad_out)

        self.assertTrue(torch.allclose(
            target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)
        ))
        # diff_idxs = torch.where(~torch.isclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_primal_to_world(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)

        target_world_pts = (grid_pts * vox_size) + vox_origin
        pred_world_pts = grid.grid_to_world(grid_pts).jdata

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_dual_to_world(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)
        grid = grid.dual_grid()

        target_world_pts = ((grid_pts - 0.5) * vox_size) + vox_origin
        pred_world_pts = grid.grid_to_world(grid_pts).jdata

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_primal_to_world_grad(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)
        grid_pts.requires_grad = True

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)

        pred_world_pts = grid.grid_to_world(grid_pts).jdata
        grad_out = torch.rand_like(pred_world_pts)
        pred_world_pts.backward(grad_out)

        assert grid_pts.grad is not None  # Removes type errors with .grad
        pred_grad = grid_pts.grad.clone()

        grid_pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(grid_pts.grad, torch.zeros_like(grid_pts.grad)))

        target_world_pts = (grid_pts * vox_size) + vox_origin
        target_world_pts.backward(grad_out)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(grid_pts.grad, pred_grad, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_dual_to_world_grad(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid_pts = torch.randint_like(pts, -100, 100).to(dtype) + torch.randn_like(pts)
        grid_pts.requires_grad = True

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)
        grid = grid.dual_grid()

        pred_world_pts = grid.grid_to_world(grid_pts).jdata
        grad_out = torch.rand_like(pred_world_pts)
        pred_world_pts.backward(grad_out)

        assert grid_pts.grad is not None  # Removes type errors with .grad
        pred_grad = grid_pts.grad.clone()

        grid_pts.grad.zero_()
        self.assertTrue(not torch.equal(pred_grad, torch.zeros_like(pred_grad)))
        self.assertTrue(torch.equal(grid_pts.grad, torch.zeros_like(grid_pts.grad)))

        target_world_pts = ((grid_pts - 0.5) * vox_size) + vox_origin
        target_world_pts.backward(grad_out)

        self.assertTrue(torch.allclose(target_world_pts, pred_world_pts, atol=dtype_to_atol(dtype)))
        self.assertTrue(torch.allclose(grid_pts.grad, pred_grad, atol=dtype_to_atol(dtype)))


    @parameterized.expand(all_device_dtype_combos)
    def test_dual_of_dual_is_primal(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(dtype).to(device)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1] * 3, [1] * 3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)
        grid_d = grid.dual_grid()
        grid_dd = grid_d.dual_grid()

        primal_origin = grid.origins[0]
        dual_origin = grid_d.origins[0]

        self.assertFalse(torch.allclose(primal_origin, dual_origin))
        self.assertTrue(torch.all(primal_origin == grid_dd.origins[0]))
        self.assertTrue(torch.all(dual_origin == grid_dd.dual_grid().origins[0]))

        target_primal_coordinates = ((pts - vox_origin) / vox_size)
        pred_primal_coordinates = grid.world_to_grid(pts).jdata

        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates, atol=dtype_to_atol(dtype)),
            f"Max diff = {torch.max(torch.abs(target_primal_coordinates- pred_primal_coordinates)).item()}")

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid_d.world_to_grid(pts).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))

        pred_primal_coordinates_dd = grid_dd.world_to_grid(pts).jdata
        self.assertTrue(
            torch.allclose(target_primal_coordinates, pred_primal_coordinates_dd, atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_index(self, device, dtype, mutable):
        gsize = 7

        grid_p, grid_d, _ = make_dense_grid_and_point_data(gsize, device, dtype, mutable)

        pijk = grid_p.ijk.jdata
        dijk = grid_d.ijk.jdata

        for in_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            pijk, dijk = pijk.to(in_dtype), dijk.to(in_dtype)
            pidx = grid_p.ijk_to_index(pijk).jdata
            didx = grid_d.ijk_to_index(dijk).jdata

            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx))
            self.assertTrue(torch.all(didx == target_didx))

            ppmt = torch.randperm(pidx.shape[0])
            dpmt = torch.randperm(pidx.shape[0])

            pidx = grid_p.ijk_to_index(pijk[ppmt]).jdata
            didx = grid_d.ijk_to_index(dijk[dpmt]).jdata
            target_pidx = torch.arange(pidx.shape[0]).to(pidx)
            target_didx = torch.arange(didx.shape[0]).to(didx)

            self.assertTrue(torch.all(pidx == target_pidx[ppmt]))
            self.assertTrue(torch.all(didx == target_didx[dpmt]))

    @parameterized.expand(all_device_dtype_combos)
    def test_coords_in_grid(self, device, _, mutable):
        num_inside = 1000 if device == 'cpu' else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_ijk(random_coords)
        random_drop_points_if_mutable(grid)

        enabled_coords = grid.ijk.jdata
        num_outside = 1000 if device == 'cpu' else 10_000

        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        pred_mask = grid.coords_in_active_voxel(all_coords, ignore_disabled=True).jdata
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_points_in_grid(self, device, dtype, mutable):
        num_inside = 1000 if device == 'cpu' else 100_000
        random_coords = torch.randint(-1024, 1024, (num_inside, 3), dtype=torch.int32).to(device)
        grid = GridBatch(device, mutable)
        grid.set_from_ijk(random_coords)
        random_drop_points_if_mutable(grid)

        enabled_coords = grid.ijk.jdata
        num_outside = 1000 if device == 'cpu' else 10_000
        outside_random_coords = torch.randint(2048, 4096, (num_outside, 3), dtype=torch.int32).to(device)
        inside_coords = enabled_coords[:num_inside]

        all_coords = torch.cat([outside_random_coords, inside_coords])

        all_world_points = grid.grid_to_world(all_coords.to(dtype)).jdata

        pred_mask = grid.points_in_active_voxel(all_world_points, ignore_disabled=True).jdata
        target_mask = torch.ones(all_coords.shape[0], dtype=torch.bool).to(device)
        target_mask[:num_outside] = False

        self.assertTrue(torch.all(pred_mask == target_mask))

    @parameterized.expand(all_device_dtype_combos)
    def test_cubes_intersect_grid(self, device, dtype, mutable):
        # TODO: (@Caenorst) tests are a bit too light, should test on more variety of range
        #import random
        #torch.random.manual_seed(0)
        #random.seed(0)
        #np.random.seed(0)

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, include_boundary_points=True, mutable=mutable)
        voxel_size = grid.voxel_sizes

        primal_mask = grid.cubes_in_grid(p).jdata
        dual_mask = grid_d.cubes_in_grid(p, -voxel_size / 2, voxel_size / 2).jdata
        # # Here: Note that dual_mask != primal_mask because their connectivities could be different!
        # #   Instead, we can only ensure that dual_mask is always true where primal_mask is true!
        #
        # from pycg import vis
        # vis.show_3d([vis.wireframe_bbox(grid.grid_to_world(grid.ijk - 0.5),
        #                                 grid.grid_to_world(grid.ijk + 0.5), solid=True),
        #              vis.wireframe_bbox(grid_d.grid_to_world(grid_d.ijk - 0.5),
        #                                 grid_d.grid_to_world(grid_d.ijk + 0.5), ucid=1),
        #              vis.pointcloud(p[primal_mask != dual_mask])])
        #
        self.assertTrue(torch.all(dual_mask[primal_mask]))

        primal_mask = grid.cubes_intersect_grid(p, -voxel_size / 2, voxel_size / 2).jdata
        dual_mask = grid_d.cubes_intersect_grid(p).jdata
        # gt_dual_mask = grid_d.points_in_active_voxel(p).jdata
        self.assertTrue(torch.all(primal_mask == dual_mask))

        # TODO: (@Caenorst) not sure what we are testing here
        # We should probably replace that by comparison to pytorch implementation

        # This is to avoid points on voxel faces, that have ambiguous values
        # cubes_intersect_grid uses ceil() while sample_trilinear use floor()
        # TODO: (@Caenorst) this is a bit too strong modification as it doesn't test
        # points on faces inside the volume (although previous tests are testing it)
        dual_grid_p = grid_d.world_to_grid(p).jdata
        p[dual_grid_p == dual_grid_p.int()] += 1e-3
        # dummy_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        _ = grid_d.cubes_intersect_grid(p, -0.5, 0.5).jdata

        # This is to avoid points on voxel faces, that have ambiguous values
        # cubes_intersect_grid uses ceil() while sample_bezier use floor()
        # TODO: (@Caenorst) this is a bit too strong modification as it doesn't test
        # points on faces inside the volume (although previous tests are testing it)
        grid_p = grid.world_to_grid(p).jdata
        p[grid_p == grid_p.int()] += 1e-3
        # dummy_features = torch.rand(grid_d.total_voxels, 4).to(device).to(dtype)
        _ = grid_d.cubes_intersect_grid(p, -1, 1).jdata

    @parameterized.expand(all_device_dtype_combos)
    def test_subdivided_grid(self, device, dtype, mutable):
        p = torch.randn(100, 3).to(device).to(dtype)
        vox_size = 0.1
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, (0.0, 0.0, 0.0))
        random_drop_points_if_mutable(grid)

        grids = [grid]
        for i in range(2):
            subdiv_factor = i + 2
            mask = (torch.rand(grids[i].total_voxels, device=device) > 0.5)

            # This line sets false values in disabled voxels
            # This is not needed for the subdivision, but we use the mask to count how
            # many voxels there should be in the subdivided grid so this needs to be correct
            mask[grids[i].disabled_mask.jdata] = False

            grids.append(grids[-1].subdivided_grid(subdiv_factor, mask))
            self.assertEqual(int(mask.sum().item()) * subdiv_factor ** 3, grids[-1].total_enabled_voxels)

        grids = [grid]
        for i, subdiv_factor in enumerate([(2, 2, 1), (3, 2, 2), (1, 1, 3)]):
            mask = torch.rand(grids[i].total_voxels, device=device) > 0.5

            # This line sets false values in disabled voxels
            # This is not needed for the subdivision, but we use the mask to count how
            # many voxels there should be in the subdivided grid so this needs to be correct
            mask[grids[i].disabled_mask.jdata] = False
            nsubvox = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
            grids.append(grids[-1].subdivided_grid(subdiv_factor, mask))
            self.assertEqual(int(mask.sum().item()) * nsubvox, grids[-1].total_enabled_voxels)
        if device == 'cuda':
            torch.cuda.synchronize()

    @parameterized.expand(all_device_dtype_combos)
    def test_build_from_pointcloud_nearest_voxels(self, device, dtype, mutable):
        p = torch.randn((100, 3), device=device, dtype=dtype)

        vox_size = 0.01
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_nearest_voxels_to_points(p, vox_size)
        random_drop_points_if_mutable(grid)

        if p.dtype == torch.half:
            p = p.float()

        expected_ijk = torch.floor(grid.world_to_grid(p).jdata)
        offsets = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], device=device, dtype=torch.long)
        expected_ijk = expected_ijk.unsqueeze(1) + offsets.unsqueeze(0)
        expected_ijk = expected_ijk.view(-1, 3).to(torch.int32)

        if mutable:
            expected_ijk = expected_ijk[grid.coords_in_active_voxel(expected_ijk).jdata]

        expected_ijk_set = set({(expected_ijk[i, 0].item(),
                                 expected_ijk[i, 1].item(),
                                 expected_ijk[i, 2].item())
                                 for i in range(expected_ijk.shape[0])})

        predicted_ijk = grid.ijk_enabled.jdata

        predicted_ijk_set = set({(predicted_ijk[i, 0].item(),
                                  predicted_ijk[i, 1].item(),
                                  predicted_ijk[i, 2].item())
                                  for i in range(predicted_ijk.shape[0])})

        self.assertEqual(predicted_ijk_set, expected_ijk_set)

    @parameterized.expand(all_device_dtype_combos)
    def test_subdivide(self, device, dtype, mutable):
        p = torch.randn(100, 3).to(device).to(dtype)
        vox_size = 0.01

        for subdiv_factor in (4, (4, 3, 2)):
            if isinstance(subdiv_factor, tuple):
                nvoxsub = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
                fac_sub_one = torch.tensor([subdiv_factor]).to(device) - 1
                subvec = torch.tensor(subdiv_factor).to(device)
            else:
                nvoxsub = subdiv_factor ** 3
                fac_sub_one = subdiv_factor - 1
                subvec = subdiv_factor

            vox_size = 0.01
            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_nearest_voxels_to_points(p, vox_size, (0.0, 0.0, 0.0))
            random_drop_points_if_mutable(grid)

            feats = torch.randn(grid.total_voxels, 32).to(p)
            if mutable:
                # Zero out disabled values so we don't get gradients
                # when we compare
                feats[grid.disabled_mask.jdata] = 0.0
            feats.requires_grad = True

            mask = torch.ones(grid.total_voxels, dtype=torch.bool).to(device)

            feats_fine, grid_fine = grid.subdivide(subdiv_factor, feats, mask=mask)
            self.assertTrue(torch.allclose(grid_fine.voxel_sizes[0], grid.voxel_sizes[0] / subvec))
            self.assertTrue(torch.allclose(grid_fine.origins[0], grid.origins[0] - 0.5 * grid_fine.voxel_sizes[0] * fac_sub_one))

            fine_to_coarse_ijk = (grid_fine.ijk.jdata / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fine_to_coarse_ijk.to(torch.int32)).jdata

            self.assertTrue(torch.all(feats_fine.jdata == feats[fine_to_coarse_idx]))

            loss = feats_fine.jdata.pow(3).sum()
            loss.backward()

            assert feats.grad is not None  # Removes type errors with .grad
            feats_grad_thru_subdiv = feats.grad.clone()

            feats.grad.zero_()
            self.assertTrue(torch.all(feats.grad == torch.zeros_like(feats.grad)))
            self.assertTrue(not torch.all(feats.grad == feats_grad_thru_subdiv))

            loss = (torch.cat([feats] * (nvoxsub)).pow(3)).sum()
            loss.backward()

            self.assertTrue(torch.all(feats_grad_thru_subdiv == feats.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_subdivide_with_mask(self, device, dtype, mutable):
        p = torch.randn(100, 3).to(device).to(dtype)
        vox_size = 0.01
        subdiv_factor = 4

        for subdiv_factor in (4, (4, 3, 2)):
            if isinstance(subdiv_factor, tuple):
                nvoxsub = subdiv_factor[0] * subdiv_factor[1] * subdiv_factor[2]
                fac_sub_one = torch.tensor([subdiv_factor]).to(device) - 1
                subvec = torch.tensor(subdiv_factor).to(device)
            else:
                nvoxsub = subdiv_factor ** 3
                fac_sub_one = subdiv_factor - 1
                subvec = subdiv_factor

            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_nearest_voxels_to_points(p, vox_size, (0.0, 0.0, 0.0))
            random_drop_points_if_mutable(grid)

            feats = torch.randn(grid.total_voxels, 32).to(p)
            if mutable:
                # Zero out disabled values so we don't get gradients
                # when we compare
                feats[grid.disabled_mask.jdata] = 0.0
            feats.requires_grad = True

            mask = torch.rand(grid.total_voxels).to(device) > 0.5

            feats_fine, grid_fine = grid.subdivide(subdiv_factor, feats, mask=mask)
            self.assertTrue(torch.allclose(grid_fine.voxel_sizes[0], grid.voxel_sizes[0] / subvec))
            self.assertTrue(torch.allclose(grid_fine.origins[0], grid.origins[0] - 0.5 * grid_fine.voxel_sizes[0] * fac_sub_one))

            fine_to_coarse_ijk = (grid_fine.ijk.jdata / subvec).floor()
            fine_to_coarse_idx = grid.ijk_to_index(fine_to_coarse_ijk.to(torch.int32)).jdata

            self.assertTrue(torch.all(feats_fine.jdata == feats[fine_to_coarse_idx]))

            loss = feats_fine.jdata.pow(3).sum()
            loss.backward()

            assert feats.grad is not None  # Removes type errors with .grad

            feats_grad_thru_subdiv = feats.grad.clone()
            masked_gradients = feats_grad_thru_subdiv[~mask]
            self.assertTrue(torch.all(masked_gradients == torch.zeros_like(masked_gradients)))

            feats.grad.zero_()
            self.assertTrue(torch.all(feats.grad == torch.zeros_like(feats.grad)))
            self.assertTrue(not torch.all(feats.grad == feats_grad_thru_subdiv))

            loss = (torch.cat([feats[mask]] * nvoxsub).pow(3)).sum()
            loss.backward()

            self.assertTrue(torch.all(feats_grad_thru_subdiv == feats.grad))

            masked_gradients = feats.grad[~mask]
            self.assertTrue(torch.all(masked_gradients == torch.zeros_like(masked_gradients)))

    @parameterized.expand(all_device_dtype_combos)
    def test_max_pool(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

        for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
            grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals)
            grid_vals_coarse = grid_vals_coarse.jdata
            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))
            else:
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device)))

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
            grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device='cpu', dtype=dtype)
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
            grid_vals_t = grid_vals_t.to(device)
            grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device='cpu')
            grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
            for i, coord in enumerate(grid_coarse_ijk_cpu):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
            grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    @parameterized.expand(all_device_dtype_combos)
    def test_strided_max_pool(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

        for pool_factor in ((2, 3, 4), 2, 4, 5, 10):
            # Our behavior differs slightly from PyTorch when pool_factor < stride, so only test this.
            if isinstance(pool_factor, int):
                pools = (pool_factor, pool_factor+1, pool_factor + 2, pool_factor + 5)
            else:
                assert isinstance(pool_factor, tuple)
                def addit(pf, val_):
                    assert isinstance(pf, tuple)
                    return (pf[0] + val_, pf[1] + val_, pf[2] + val_)
                pools = (pool_factor, addit(pool_factor, 1), addit(pool_factor, 2), addit(pool_factor, 5))
            for stride in pools:
                grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals, stride=stride)
                grid_vals_coarse = grid_vals_coarse.jdata
                if isinstance(stride, int):
                    self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * stride))
                    self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (stride - 1)))
                else:
                    self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(stride).to(device)))
                    self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(stride) - 1).to(device)))

                # Pytorch pooling
                torch_pool_op = torch.nn.MaxPool3d(pool_factor, stride=stride, ceil_mode=True)
                # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
                grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device='cpu', dtype=dtype)
                grid_ijk_cpu = grid.ijk.jdata.cpu()
                grid_vals_cpu = grid_vals.cpu()
                for i, coord in enumerate(grid_ijk_cpu):
                    grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
                grid_vals_t = grid_vals_t.to(device)
                grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
                grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

                grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device='cpu')
                grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
                for i, coord in enumerate(grid_coarse_ijk_cpu):
                    grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
                grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
                self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    @parameterized.expand(all_device_dtype_combos)
    def test_max_pool_grad(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        for pool_factor in ((2, 3, 1), 1, 2, 3, 4, 5, 7, 15, 10):
            grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
            grid_vals.requires_grad = True

            grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals)
            grid_vals_coarse = grid_vals_coarse.jdata
            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))
            else:
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device)))

            loss = (grid_vals_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals.grad is not None  # Removes type errors with .grad

            grid_vals_grad = grid_vals.grad.clone()
            self.assertEqual((grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                             grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1])

            mask = grid_vals_grad.abs() > 0
            a = torch.sort(torch.tensor([x.item() for x in grid_vals[mask[:, 0]][:, 0]]))[0]
            b = torch.sort(torch.tensor([x.item() for x in grid_vals_coarse[:, 0]]))[0]
            self.assertEqual(torch.max(a - b).max().item(), 0)

            grid_vals.grad.zero_()

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            dtype_np = np.float32 if dtype == torch.float32 else np.float64
            grid_vals_t_np = np.zeros([gsize, gsize, gsize, 3], dtype=dtype_np)
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t_np[coord[0], coord[1], coord[2]] = grid_vals_cpu[i].detach().cpu().numpy()

            grid_vals_t = torch.from_numpy(np.transpose(grid_vals_t_np, (3, 0, 1, 2))).to(device=device, dtype=dtype)
            grid_vals_t.requires_grad = True

            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
            for i, coord in enumerate(grid_coarse.ijk.jdata):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

            loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals_t.grad is not None  # Removes type errors with .grad

            grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device='cpu')
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
            grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

            expected_nnz = grid_vals_t_coarse.shape[1] * grid_vals_t_coarse.shape[2] * \
                           grid_vals_t_coarse.shape[3] * grid_vals_t_coarse.shape[0]
            self.assertEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(),
                             expected_nnz)

            self.assertEqual(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item(), 0.0)

    @parameterized.expand(all_device_dtype_combos)
    def test_avg_pool_grad(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        for pool_factor in ((2, 4, 5), 1, 2, 4, 5, 10):
            grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
            grid_vals.requires_grad = True

            grid_vals_coarse, grid_coarse = grid.avg_pool(pool_factor, grid_vals)
            grid_vals_coarse = grid_vals_coarse.jdata

            if isinstance(pool_factor, int):
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))
                npool_vox = pool_factor ** 3
            else:
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * torch.tensor(pool_factor).to(device)))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (torch.tensor(pool_factor) - 1).to(device)))
                npool_vox = pool_factor[0] * pool_factor[1] * pool_factor[2]
            # self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
            # self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))

            loss = (grid_vals_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals.grad is not None  # Removes type errors with .grad

            grid_vals_grad = grid_vals.grad.clone()
            self.assertLessEqual((grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                                 grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1] * npool_vox)

            grid_vals.grad.zero_()

            # Pytorch pooling
            torch_pool_op = torch.nn.AvgPool3d(pool_factor, pool_factor, ceil_mode=True)
            dtype_np = np.float32 if dtype == torch.float32 else np.float64
            grid_vals_t_np = np.zeros([gsize, gsize, gsize, 3], dtype=dtype_np)
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t_np[coord[0], coord[1], coord[2]] = grid_vals_cpu[i].detach().cpu().numpy()

            grid_vals_t = torch.from_numpy(np.transpose(grid_vals_t_np, (3, 0, 1, 2))).to(device=device, dtype=dtype)
            grid_vals_t.requires_grad = True

            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
            for i, coord in enumerate(grid_coarse.ijk.jdata):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

            self.assertTrue(torch.max(torch.abs(grid_vals_coarse - grid_vals_coarse_t_flat)) < dtype_to_atol(dtype), str(torch.abs(grid_vals_coarse - grid_vals_coarse_t_flat)))

            loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
            loss.backward()

            assert grid_vals_t.grad is not None  # Removes type errors with .grad

            grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device='cpu')
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
            grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

            expected_nnz_ub = grid_vals_t_coarse.shape[1] * grid_vals_t_coarse.shape[2] * \
                           grid_vals_t_coarse.shape[3] * grid_vals_t_coarse.shape[0] * npool_vox
            self.assertLessEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(),
                                 expected_nnz_ub)

            self.assertTrue(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item() < dtype_to_atol(dtype))

    @parameterized.expand(all_device_dtype_combos)
    def test_strided_max_pool_grad(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        for pool_factor in (2, 4, 5, 10):
            for stride in (pool_factor, pool_factor+1, pool_factor + 2, pool_factor + 5):
                grid_vals = torch.rand(grid.total_voxels, 3).to(device).to(dtype) + 0.5
                grid_vals.requires_grad = True

                grid_vals_coarse, grid_coarse = grid.max_pool(pool_factor, grid_vals, stride=stride)
                grid_vals_coarse = grid_vals_coarse.jdata
                self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * stride))
                self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (stride - 1)))

                loss = (grid_vals_coarse.pow(3) * -1.111).sum()
                loss.backward()

                assert grid_vals.grad is not None  # Removes type errors with .grad

                grid_vals_grad = grid_vals.grad.clone()
                self.assertEqual((grid_vals_grad.abs() > 0).sum().to(torch.int32).item(),
                                grid_vals_coarse.shape[0] * grid_vals_coarse.shape[1])

                mask = grid_vals_grad.abs() > 0
                a = torch.sort(torch.tensor([x.item() for x in grid_vals[mask[:, 0]][:, 0]]))[0]
                b = torch.sort(torch.tensor([x.item() for x in grid_vals_coarse[:, 0]]))[0]
                self.assertEqual(torch.max(a - b).max().item(), 0)

                grid_vals.grad.zero_()

                # Pytorch pooling
                torch_pool_op = torch.nn.MaxPool3d(pool_factor, stride=stride, ceil_mode=True)
                dtype_np = np.float32 if dtype == torch.float32 else np.float64
                grid_vals_t_np = np.zeros([gsize, gsize, gsize, 3], dtype=dtype_np)
                grid_ijk_cpu = grid.ijk.jdata.cpu()
                grid_vals_cpu = grid_vals.cpu()
                for i, coord in enumerate(grid_ijk_cpu):
                    grid_vals_t_np[coord[0], coord[1], coord[2]] = grid_vals_cpu[i].detach().cpu().numpy()

                grid_vals_t = torch.from_numpy(np.transpose(grid_vals_t_np, (3, 0, 1, 2))).to(device=device, dtype=dtype)
                grid_vals_t.requires_grad = True

                grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

                grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse)
                for i, coord in enumerate(grid_coarse.ijk.jdata):
                    grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]

                self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

                loss = (grid_vals_t_coarse.pow(3) * -1.111).sum()
                loss.backward()

                assert grid_vals_t.grad is not None  # Removes type errors with .grad

                grid_vals_grad_t_flat = torch.zeros_like(grid_vals_grad, device='cpu')
                grid_ijk_cpu = grid.ijk.jdata.cpu()
                grid_vals_t_cpu_grad = grid_vals_t.grad.cpu()
                for i, coord in enumerate(grid_ijk_cpu):
                    grid_vals_grad_t_flat[i] = grid_vals_t_cpu_grad[:, coord[0], coord[1], coord[2]]
                grid_vals_grad_t_flat = grid_vals_grad_t_flat.to(device)

                expected_nnz = grid_vals_t_coarse.shape[1] * grid_vals_t_coarse.shape[2] * \
                            grid_vals_t_coarse.shape[3] * grid_vals_t_coarse.shape[0]
                self.assertEqual((grid_vals_grad_t_flat.abs() > 0).to(torch.int32).sum().item(),
                                expected_nnz)

                self.assertEqual(torch.abs(grid_vals_grad_t_flat - grid_vals_grad).max().item(), 0.0)

    @parameterized.expand(all_device_dtype_combos)
    def test_pickle(self, device, dtype, mutable):
        grid, _, _ = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        pkl_str = pickle.dumps(grid)
        grid_2 = pickle.loads(pkl_str)
        self.assertTrue(torch.all(grid.ijk.jdata == grid_2.ijk.jdata))
        self.assertTrue(torch.all(grid.ijk_enabled.jdata == grid_2.ijk_enabled.jdata))
        self.assertEqual(grid.device, grid_2.device)
        self.assertTrue(torch.all(grid.voxel_sizes[0] == grid_2.voxel_sizes[0]))
        self.assertTrue(torch.all(grid.origins[0] == grid_2.origins[0]))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_device(self, device, dtype, mutable):
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1]*3, [1]*3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)
        grid = grid.dual_grid()

        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        pred_dual_coordinates = grid.world_to_grid(pts).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates, target_dual_coordinates, atol=dtype_to_atol(dtype)))
        self.assertEqual(grid.device.type, torch.device(device).type)

        to_device = torch.device('cpu')
        grid2 = grid.to(to_device)
        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        if torch.device(device).type != to_device.type:
            with self.assertRaises(RuntimeError):
                pred_dual_coordinates = grid2.world_to_grid(pts).jdata
        pred_dual_coordinates = grid2.world_to_grid(pts.to(to_device)).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates,
                                       target_dual_coordinates.to(to_device), atol=dtype_to_atol(dtype)))
        self.assertEqual(grid2.device, to_device)

        to_device = torch.device('cuda:0')
        grid2 = grid.to(to_device)
        target_dual_coordinates = ((pts - vox_origin) / vox_size) + 0.5
        if torch.device(device).type != to_device.type:
            with self.assertRaises(RuntimeError):
                pred_dual_coordinates = grid2.world_to_grid(pts).jdata
        pred_dual_coordinates = grid2.world_to_grid(pts.to(to_device)).jdata
        self.assertTrue(torch.allclose(pred_dual_coordinates,
                                       target_dual_coordinates.to(to_device), atol=dtype_to_atol(dtype)))
        self.assertEqual(grid2.device, to_device)

    @parameterized.expand(all_device_dtype_combos)
    def test_volume_render(self, device, dtype, mutable):
        vox_size = np.random.rand(3) * 0.2 + 0.05
        step_size = 0.1 * float(np.linalg.norm(vox_size))
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.rand(10000, 3).to(device=device, dtype=dtype) - 0.5
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1]*3, [1]*3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid, drop_pct=0.3)
        grid_dual = grid.dual_grid()

        def make_ray_grid(origin, nrays, minb=(-0.45, -0.45), maxb=(0.45, 0.45)):
            ray_o = torch.tensor([origin] * nrays**2) #+ p.mean(0, keepdim=True)
            ray_d = torch.from_numpy(
                np.stack([a.ravel() for a in
                        np.mgrid[minb[0]:maxb[0]:nrays*1j,
                                minb[1]:maxb[1]:nrays*1j]] +
                        [np.ones(nrays**2)], axis=-1).astype(np.float32))
            ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

            ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

            return ray_o, ray_d

        def volume_render_pytorch(sigma, color, dt, t, pack_info, t_threshold):
            res_c = []
            res_d = []
            for ray_i in range(pack_info.shape[0]):
                start_idx, num_s = pack_info[ray_i]
                end_idx = start_idx + num_s
                sigma_i = sigma[start_idx:end_idx]
                color_i = color[start_idx:end_idx]
                dt_i = dt[start_idx:end_idx]
                t_i = t[start_idx:end_idx]

                alpha = -sigma_i.squeeze() * dt_i
                transmittance = torch.exp(torch.cumsum(alpha.squeeze(), dim=0))
                tmask = (transmittance > t_threshold).to(transmittance)

                summand = transmittance * (1.0 - torch.exp(alpha)) * tmask
                res_c.append((summand[:, None] * color_i).sum(0))
                res_d.append((summand * t_i).sum(0))
            return torch.stack(res_c), torch.stack(res_d)

        t_threshold = 0.001

        grid_data_rgb = torch.rand(grid_dual.total_voxels, 3).to(device=device, dtype=dtype) * 0.5
        grid_data_sigma = torch.rand(grid_dual.total_voxels, 1).to(device=device, dtype=dtype) * 0.5

        grid_data_rgb.requires_grad = True
        grid_data_sigma.requires_grad = True

        ray_o, ray_d = make_ray_grid((0.0, 0.0, -1.0), 8)
        tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
        tmax = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10
        pack_info, ray_idx, ray_intervals = grid.uniform_ray_samples(
            ray_o, ray_d, tmin, tmax, step_size)
        pack_info, ray_idx, ray_intervals = pack_info.jdata, ray_idx.jdata, ray_intervals.jdata

        ray_t = ray_intervals.mean(1)

        ray_delta_t = ray_intervals[:, 1] - ray_intervals[:, 0]
        ray_pts = ray_o[ray_idx] + ray_t[:, None] * ray_d[ray_idx]

        rgb_samples = grid_dual.sample_trilinear(ray_pts, grid_data_rgb).jdata
        sigma_samples = grid_dual.sample_trilinear(ray_pts, grid_data_sigma).jdata

        assert isinstance(sigma_samples, torch.Tensor)  # Fix type errors

        rgb1, depth1, opacity, ws, tot_samples = volume_render(sigma_samples.squeeze(), rgb_samples,
                                                               ray_delta_t, ray_t, pack_info,
                                                               t_threshold)
        loss = rgb1.sum() + depth1.sum()
        loss.backward()

        assert grid_data_rgb.grad is not None  # Removes type errors with .grad
        assert grid_data_sigma.grad is not None  # Removes type errors with .grad

        rgb_1_grad = grid_data_rgb.grad.detach().clone()
        sigma_1_grad = grid_data_sigma.grad.detach().clone()

        grid_data_rgb.grad.zero_()
        grid_data_sigma.grad.zero_()

        rgb_samples = grid_dual.sample_trilinear(ray_pts, grid_data_rgb).jdata
        sigma_samples = grid_dual.sample_trilinear(ray_pts, grid_data_sigma).jdata
        rgb2, depth2 = volume_render_pytorch(sigma_samples, rgb_samples,
                                             ray_delta_t, ray_t, pack_info,
                                             t_threshold)
        loss = rgb2.sum() + depth2.sum()
        loss.backward()
        rgb_2_grad = grid_data_rgb.grad.detach().clone()
        sigma_2_grad = grid_data_sigma.grad.detach().clone()

        # The tolerances here are high because I think the algo has a large condition number
        # due to sums
        self.assertLess(torch.abs(rgb1 - rgb2).max().item(), 1e-2)
        self.assertLess(torch.abs(depth1 - depth2).max().item(), 2e-1)
        self.assertLess(torch.abs(rgb_1_grad - rgb_2_grad).max().item(), 4e-2)
        self.assertLess(torch.abs(sigma_1_grad - sigma_2_grad).max().item(), 1e-1)

        # import polyscope as ps
        # ps.init()
        # rv = torch.cat([ray_o, ray_o + ray_d])
        # re = torch.tensor([[i, i + ray_o.shape[0]] for i in range(ray_o.shape[0])]).to(ray_o)
        # v, e = grid.grid_edge_network()
        # ps.register_curve_network("grid", v.cpu(), e.cpu())
        # rg = ps.register_curve_network("rays", rv.cpu(), re.cpu())
        # rg.add_color_quantity("colors", rgb1.detach().cpu(), defined_on='edges', enabled=True)
        # pc = ps.register_point_cloud("raypts", ray_pts.cpu())
        # pc.add_scalar_quantity("sigma samples", sigma_samples.detach().squeeze().cpu(), enabled=True)
        # ps.show()

    @parameterized.expand(all_device_dtype_combos)
    def test_uniform_step_size_first_step_is_multiple_of_step_size(self, device, dtype, mutable):
        gsize = 8
        grid, _, _ = make_dense_grid_and_point_data(gsize, device, dtype, mutable)
        random_drop_points_if_mutable(grid)

        grid_centers = grid.grid_to_world(grid.ijk.float()).jdata
        camera_origin_inside = torch.mean(grid_centers, dim=0)
        camera_origin_outside = camera_origin_inside - torch.tensor([0.0, 0.0, 4.0]).to(grid.device)

        ray_d_inside = grid_centers[torch.randperm(grid_centers.shape[0])[:24]] - camera_origin_inside[None, :]
        ray_d_inside /= torch.norm(ray_d_inside, dim=-1, keepdim=True)
        ray_d_outside = grid_centers[torch.randperm(grid_centers.shape[0])[:24]] - camera_origin_outside[None, :]
        ray_d_outside /= torch.norm(ray_d_outside, dim=-1, keepdim=True)

        ray_o_inside = torch.ones_like(ray_d_outside) * camera_origin_inside[None, :]
        ray_o_outside = torch.ones_like(ray_d_outside) * camera_origin_outside[None, :]

        tmin = torch.zeros(ray_d_inside.shape[0]).to(ray_d_inside)
        tmin += torch.rand_like(tmin) * 0.01
        tmax = torch.ones_like(tmin) * 1e10

        step_size = 0.01
        _, ray_idx, ray_times_inside = grid.uniform_ray_samples(ray_o_inside, ray_d_inside, tmin, tmax, step_size, include_end_segments=False)
        ray_idx, ray_times_inside = ray_idx.jdata, ray_times_inside.jdata
        nsteps_inside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_inside, torch.round(nsteps_inside), atol=dtype_to_atol(dtype)))

        _, ray_idx, ray_times_inside = grid.uniform_ray_samples(ray_o_outside, ray_d_outside, tmin, tmax, step_size, include_end_segments=False)
        ray_idx, ray_times_inside = ray_idx.jdata, ray_times_inside.jdata
        nsteps_outside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_outside, torch.round(nsteps_outside), atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_fill_to_grid(self, device, dtype, mutable):
        grid1 = GridBatch(device)
        grid2 = GridBatch(device)

        random_points_b1 = torch.randn(100, 3, device=device, dtype=dtype)
        random_points_b2 = torch.randn(100, 3, device=device, dtype=dtype)

        grid1.set_from_points(JaggedTensor([random_points_b1[:70], random_points_b2[:70]]), voxel_sizes=0.01, origins=[0, 0, 0])
        grid2.set_from_points(JaggedTensor([random_points_b1[30:], random_points_b2[30:]]), voxel_sizes=0.01, origins=[0, 0, 0])

        random_features_b1 = torch.randn(grid1[0].total_voxels, 32, device=device, dtype=dtype)
        random_features_b2 = torch.randn(grid1[1].total_voxels, 32, device=device, dtype=dtype)
        ret = grid2.fill_to_grid(JaggedTensor([random_features_b1, random_features_b2]), grid1)

        # Perform an all pairs comparison between grid1 and grid2 points.
        # All points that match up should have the same features.

        # Test independently for both batches.
        b1_comparison = torch.all(grid1[0].ijk.jdata.unsqueeze(0) == grid2[0].ijk.jdata.unsqueeze(1), dim=-1)
        b2_comparison = torch.all(grid1[1].ijk.jdata.unsqueeze(0) == grid2[1].ijk.jdata.unsqueeze(1), dim=-1)

        toinds, frominds = torch.where(b1_comparison)
        self.assertTrue(torch.all(ret[0].jdata[toinds] == random_features_b1[frominds]))

        toinds, frominds = torch.where(b2_comparison)
        self.assertTrue(torch.all(ret[1].jdata[toinds] == random_features_b2[frominds]))

        # All the rest should be zero.
        self.assertTrue(torch.all(ret[0].jdata[~torch.any(b1_comparison, dim=1)] == 0.0))
        self.assertTrue(torch.all(ret[1].jdata[~torch.any(b2_comparison, dim=1)] == 0.0))

        # Test the gradients
        grid1 = grid1[0]
        grid2 = grid2[0]

        random_features = torch.randn(grid1.total_voxels, 32, device=device, dtype=dtype, requires_grad=True)

        def func(features):
            return grid2.fill_to_grid(features, grid1).jdata.sum()

        out = func(random_features)
        out.backward()

        one_indices = torch.where(torch.all(random_features.grad == 1.0, dim=1))[0]

        toinds, frominds = torch.where(b1_comparison)
        self.assertTrue(torch.all(one_indices == frominds))

    @parameterized.expand(all_device_dtype_combos)
    def test_grid_construction(self, device, dtype, mutable):
        rand_ijk = torch.randint(-100, 100, (1000, 3), device=device)
        rand_pts = torch.randn(1000, 3, device=device, dtype=dtype)

        def build_from_ijk(vsize, vorigin):
            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_ijk(rand_ijk, [0]*3, [0]*3, vsize, vorigin)
            return grid

        def build_from_pts(vsize, vorigin):
            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_points(rand_pts, [0]*3, [0]*3, vsize, vorigin)
            return grid

        def build_from_pts_nn(vsize, vorigin):
            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_nearest_voxels_to_points(rand_pts, vsize, vorigin)
            return grid

        def build_from_dense(vsize, vorigin):
            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_dense_grid(1, [10, 10, 10], [0, 0, 0], vsize, vorigin)
            return grid

        vox_size = np.random.rand(3) * 0.2 + 0.05
        vox_origin = torch.rand(3).to(device).to(dtype)

        pts = torch.randn(10000, 3).to(device=device, dtype=dtype)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_points(pts, [-1]*3, [1]*3, vox_size, vox_origin)
        random_drop_points_if_mutable(grid)

        for builder in [build_from_ijk, build_from_pts, build_from_pts_nn, build_from_dense]:
            with self.assertRaises(TypeError):
                grid = builder(vox_size, 0.01)  # type: ignore

            with self.assertRaises(ValueError):
                grid = builder(-vox_size, [0.01]*3)

            with self.assertRaises(ValueError):
                grid = builder(-1.0, [0.01]*3)

            with self.assertRaises(ValueError):
                grid = builder(vox_size * 0.0, [0.01]*3)

            with self.assertRaises(ValueError):
                grid = builder(0.0, [0.01]*3)

            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01]*4)

            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01]*2)

            with self.assertRaises(ValueError):
                grid = builder(vox_size, [0.01]*1)

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_bug(self, device, dtype, mutable):
        data_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
        data = torch.load(os.path.join(data_path, "repro_bug.pth"))
        grid = sparse_grid_from_ijk(data['ijk'].to(device), voxel_sizes=data['vox_size'], origins=data['vox_origin'])
        ray_o: torch.Tensor = torch.load(
            os.path.join(data_path, "ray_o.pth")).to(device=device, dtype=dtype)
        ray_d: torch.Tensor = torch.load(
            os.path.join(data_path, "ray_d.pth")).to(device=device, dtype=dtype)

        pack_info, seg_idx, segments = grid.segments_along_rays(ray_o.to(dtype), ray_d.to(dtype), 100, 0.0)
        pack_info, seg_idx, segments = pack_info.jdata, seg_idx.jdata, segments.jdata

        # TODO: (@Caenorst) Maybe make a separate test case of fp16?
        if dtype == torch.float16:
            self.assertEqual(pack_info[0, 1].item(), 53)
        else:
            self.assertEqual(pack_info[0, 1].item(), 52)

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_always_sorted(self, device, dtype, mutable):
        for eps in [0., 1e-5]:
            pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            grid = GridBatch(mutable=mutable).to(device)
            grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.0001, torch.zeros(3))
            random_drop_points_if_mutable(grid)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

            pack_info, seg_idx, segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=eps)
            pack_info, seg_idx, segments = pack_info.jdata, seg_idx.jdata, segments.jdata

            for i in range(pack_info.shape[0]):
                segments_i = segments[pack_info[i][0]:pack_info[i][0] + pack_info[i][1]]
                if segments_i.size(0) == 0:
                    continue

                self.assertTrue(torch.all(segments_i[:, 1] - segments_i[:, 0] >= eps))
                self.assertTrue(torch.all(segments_i[1:, 0] - segments_i[:-1, 0] >= eps))
                self.assertTrue(torch.all(segments_i[1:, 1] - segments_i[:-1, 1] >= eps))

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_always_sorted(self, device, dtype, mutable):
        for i in range(3):
            pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            grid = GridBatch(mutable=mutable).to(device)
            grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.0001, torch.zeros(3))
            random_drop_points_if_mutable(grid)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

            pack_info, out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)
            pack_info, out_voxels, out_times = pack_info.jdata, out_voxels.jdata, out_times.jdata

            for i in range(pack_info.shape[0]):
                times_i = out_times[pack_info[i][0]: pack_info[i][0] + pack_info[i][1]]
                voxels_i = out_voxels[pack_info[i][0]: pack_info[i][0] + pack_info[i][1]]
                if times_i.size(0) == 0:
                    continue

                self.assertTrue(torch.all(times_i[:, 0] < times_i[:, 1]), f"Max diff = {(times_i[:, 1] - times_i[:, 0]).max().item()}")
                self.assertTrue(torch.all(times_i[1:, 0] > times_i[:-1, 0]))
                self.assertTrue(torch.all(times_i[1:, 1] > times_i[:-1, 1]))
                # Should always march
                max_diff = torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values.cpu().detach().numpy()
                self.assertTrue(torch.all(torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values >= 1),
                                f"Max diff = {max_diff, voxels_i.cpu().numpy(), times_i.cpu().numpy()}")

    @parameterized.expand(all_device_dtype_combos)
    def test_ijk_to_inv_index(self, device, dtype, mutable):
        vox_size = 0.1

        # Unique IJK since for duplicates the permutation is non-bijective
        ijk = list(
            set(
                [tuple([a for a in (np.random.randn(3) / vox_size).astype(np.int32)]) for _ in range(10000)]
            )
        )
        ijk = torch.from_numpy(np.array([list(a) for a in ijk])).to(torch.int32).to(device)

        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_ijk(ijk, voxel_sizes=vox_size, origins=[0.]*3)

        inv_index = grid.ijk_to_inv_index(ijk).jdata

        target_inv_index = torch.full_like(grid.ijk.jdata[:, 0], -1)
        idx = grid.ijk_to_index(ijk).jdata
        for i in range(ijk.shape[0]):
            target_inv_index[idx[i]] = i

        self.assertTrue(torch.all(inv_index == target_inv_index))

        # Test functionality where size of ijk_to_inv_index's argument != len(grid.ijk)
        # Pick random ijk subset
        rand_ijks = []
        for i in range(grid.grid_count):
            ijks = grid.ijk.jdata[grid.ijk.jidx==i]
            rand_ijks.append(torch.unique(ijks[torch.randint(len(ijks), (50,), device = ijks.device)], dim=0) )

        rand_ijks = fvdb.JaggedTensor(rand_ijks)

        rand_ijk_inv_indices = grid.ijk_to_inv_index(rand_ijks)

        # valid ijk indices
        inv_rand_ijk = grid.ijk.jdata[rand_ijk_inv_indices.jdata!= -1]
        assert(len(inv_rand_ijk) == len(rand_ijks.jdata))
        inv_rand_ijk = rand_ijks.jagged_like(inv_rand_ijk)

        def check_order(t1: torch.Tensor, t2: torch.Tensor):
            t1_list = t1.tolist()
            t2_list = t2.tolist()

            last_index = -1
            for elem in t2_list:
                try:
                    current_index = t1_list.index(elem)
                    # Check if the current index is greater than the last index
                    if current_index > last_index:
                        last_index = current_index
                    else:
                        return False
                except ValueError:
                    return False
            return True

        for i, (inv_ijks, ijks) in enumerate(zip(inv_rand_ijk, rand_ijks)):
            # ensure output of ijk_to_inv_index is a permutation of the input
            inv_ijks_sorted, _ = torch.sort(inv_ijks.jdata, dim=0)
            ijks_sorted, _ = torch.sort(ijks.jdata, dim=0)
            assert torch.equal(inv_ijks_sorted, ijks_sorted)

            # ensure output of ijk_to_inv_index appears in ascending order in ijks
            assert check_order(grid.ijk.jdata[grid.ijk.jidx == i], inv_ijks.jdata)

    @parameterized.expand(all_device_dtype_combos)
    def test_no_use_after_free_on_backward(self, device, dtype, mutable):

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_trilinear(p, primal_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        del grid, grid_d
        fv.backward(grad_out)

    @parameterized.expand(all_device_dtype_combos)
    def test_ray_implicit_intersection(self, device, dtype, mutable):
        if dtype == torch.float16:
            return  # TODO: Implement rayimplicit marching for half

        # Generate the SDF for a sphere on a grid
        N = 32
        sphere_rad = 0.35
        ii, jj, kk, = torch.meshgrid([torch.arange(N)]*3, indexing='ij')
        xx, yy, zz = ii.float() / (float(N) - 1) - 0.5, jj.float() / (float(N) - 1) - 0.5, kk.float() / (float(N) - 1) - 0.5
        sphere_sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - sphere_rad

        # Generate a bunch of points on the sphere which we'll send rays to
        cam_o = torch.tensor([0., 0., -2.]).unsqueeze(0).repeat(100, 1)
        cam_targets = torch.randn(100, 3)
        cam_targets /= torch.norm(cam_targets, dim=-1, keepdim=True)
        cam_targets *= sphere_rad
        cam_targets += (0.5 - 0.5 / N)
        cam_d = cam_targets - cam_o
        cam_d /= torch.norm(cam_d, dim=-1, keepdim=True)

        sphere_sdf, cam_o, cam_d = sphere_sdf.to(device), cam_o.to(device), cam_d.to(device)
        sphere_sdf, cam_o, cam_d = sphere_sdf.to(dtype), cam_o.to(dtype), cam_d.to(dtype)

        # Build a grid with the SDF
        grid = GridBatch(device=device, mutable=mutable)
        grid.set_from_dense_grid(1, [sphere_sdf.shape[i] for i in range(3)], [0]*3, voxel_sizes=1.0 / N, origins=[0]*3)
        sdf_p = grid.read_from_dense(sphere_sdf.unsqueeze(-1).unsqueeze(0)).jdata.squeeze()  # permuted sdf values

        # Intersect rays with the SDF
        isect = grid.ray_implicit_intersection(cam_o, cam_d, sdf_p.squeeze()).jdata
        hit_mask = isect >= 0.0
        self.assertTrue(hit_mask.sum().item() > 25)
        hit_pts = cam_o[hit_mask] + isect[hit_mask, None] * cam_d[hit_mask]

        # Sample intersected values and make sure they're within a half voxel from the true intersection
        sdf_samp = grid.sample_trilinear(hit_pts, sdf_p.unsqueeze(1)).jdata
        self.assertLess(sdf_samp.max().item(), 0.5 * torch.norm(grid.voxel_sizes[0]).item())

        # import polyscope as ps
        # ps.init()
        # xyz = grid.grid_to_world(grid.ijk.to(dtype))
        # pc = ps.register_point_cloud("sdf_p", xyz.cpu().numpy())
        # pc.add_scalar_quantity("sdf", sdf_p.cpu().numpy(), enabled=True)
        # pc.add_scalar_quantity("occ", sdf_p.cpu().numpy() <= 0.0, enabled=True)
        # ps.register_point_cloud("cam_target", cam_targets.numpy())
        # ps.register_point_cloud("cam_o", cam_o.cpu().numpy())
        # ps.register_point_cloud("hits", hit_pts.cpu().numpy())
        # ps.show()

    @test_expand(list(itertools.product(
        ['cpu', 'cuda'],
        [torch.float32, torch.float64]
    )))
    def test_marching_cubes(self, device, dtype):
        # Generate the SDF for a sphere on a grid
        N = 32 if device == 'cpu' else 64
        sphere_rads = [0.5, 0.33, 0.3, 0.28, 0.25]
        for batch_size in [1, 3, 5]:
            # Build a dense tensor of SDF values
            ii, jj, kk, = torch.meshgrid([torch.arange(N, device=device)]*3, indexing='ij')  # index space [0, N-1]
            xx, yy, zz = ii.float() / (float(N) - 1) - 0.5, jj.float() / (float(N) - 1) - 0.5, kk.float() / (float(N) - 1) - 0.5  # normalize to [-1, 1]
            sphere_sdf = torch.stack([-torch.sqrt(xx**2 + yy**2 + zz**2) + sphere_rad for sphere_rad in sphere_rads[:batch_size]]).unsqueeze(-1)  # [B, N, N, N, 1] sdf

            # Build a grid with the SDF
            grid = GridBatch(device=device, mutable=False)
            grid.set_from_dense_grid(batch_size, [sphere_sdf[0].shape[i] for i in range(3)], [0]*3, voxel_sizes=1.0 / N, origins=[0]*3)
            sdf_p = grid.read_from_dense(sphere_sdf)  # permuted sdf values

            for level in [0.0, 0.2, -0.2]:
                v, f, _ = grid.marching_cubes(sdf_p, level)

                for bi in range(batch_size):
                    mesh_radius = torch.linalg.norm(
                        v[bi].jdata - torch.tensor([[0.5]*3], device=device, dtype=dtype),
                        axis=1
                    )
                    vox_size = torch.norm(grid.voxel_sizes[bi])
                    self.assertTrue(torch.all(mesh_radius - sphere_rads[bi] < vox_size / 2.0 - level))
                    self.assertTrue(torch.all(torch.logical_and(
                        f[bi].jdata >= 0, f[bi].jdata < v[bi].jdata.shape[0]
                    )))
                # import polyscope as ps
                # ps.init()
                # ps.register_surface_mesh("marching_cubes", v.cpu()[0].jdata.numpy(), f.cpu()[0].jdata.numpy())
                # ps.show()

    @parameterized.expand(all_device_dtype_combos)
    def test_subdivide_empty_grid(self, device, dtype, mutable):
        grid = GridBatch(device=device, mutable=mutable)
        grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        values = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        values, subgrid = grid.subdivide(1, values, mask=torch.zeros(grid.total_voxels, dtype=torch.bool, device=device))
        self.assertTrue(subgrid.total_voxels == 0)
        self.assertTrue(values.r_shape[0] == 0)
        self.assertTrue(values.r_shape[1] == 17)

    @parameterized.expand(all_device_dtype_combos)
    def test_conv_empty_grid(self, device, dtype, mutable):
        grid = GridBatch(device=device, mutable=mutable)
        grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        values, subgrid = grid.subdivide(1, values_in, mask=torch.zeros(grid.total_voxels, dtype=torch.bool, device=device))
        self.assertTrue(subgrid.total_voxels == 0)
        self.assertTrue(values.r_shape[0] == 0)
        self.assertTrue(values.r_shape[1] == 17)

        kmap, _ = grid.sparse_conv_kernel_map(3, 1, target_grid=subgrid)
        kmap.build_gather_scatter()
        res = kmap.sparse_conv_3d(values_in, torch.randn(17, 17, 3, 3, 3))
        self.assertTrue(res.r_shape[0] == 0)
        self.assertTrue(res.r_shape[1] == 17)

    @parameterized.expand(all_device_dtype_combos)
    def test_setting_transform_on_empty_batch_fails(self, device, dtype, mutable):
        grid = GridBatch(device=device, mutable=mutable)
        with self.assertRaises(RuntimeError):
            grid.set_global_origin(torch.zeros(3).to(device))

        with self.assertRaises(RuntimeError):
            grid.set_global_voxel_size(torch.ones(3).to(device))

    @parameterized.expand(all_device_dtype_combos)
    def test_bbox_attrs(self, device, dtype, mutable):
        grid = GridBatch(device=device, mutable=mutable)
        self.assertTrue(torch.equal(grid.bbox, torch.empty(0,2,3, device=device)))
        grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        self.assertTrue(torch.equal(grid.bbox,torch.tensor([[[ 0,  0,  0],[31, 31, 31]]], device=device)))
        self.assertTrue(torch.equal(grid.dual_bbox,torch.tensor([[[ 0,  0,  0],[32, 32, 32]]], device=device)))
        self.assertTrue(torch.equal(grid.total_bbox,torch.tensor([[ 0,  0,  0],[31, 31, 31]], device=device)))

    @parameterized.expand(all_device_dtype_combos)
    def test_clip_grid(self, device, dtype, mutable):
        # TODO: issue #196
        if device == 'cpu' and mutable is True:
            return

        grid = GridBatch(device=device, mutable=mutable)
        grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(values_in, [[0, 0, 0]], [[5, 5, 5]])
        self.assertTrue(clipped_grid.num_voxels == 6**3)
        self.assertTrue(clipped_data.jdata.shape[0] == 6**3)
        self.assertTrue(torch.equal(clipped_data.joffsets, clipped_grid.joffsets))

        grid.set_from_dense_grid(1, [32, 32, 32], [-2, -2, -2], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        values_in = torch.randn(grid.total_voxels, 17, device=device, dtype=dtype)
        clipped_data, clipped_grid = grid.clip(values_in, [[-2,-2,-2]], [[5, 5, 5]])
        self.assertTrue(clipped_grid.num_voxels == 8**3)
        self.assertTrue(clipped_data.jdata.shape[0] == 8**3)
        self.assertTrue(torch.equal(clipped_data.joffsets, clipped_grid.joffsets))

        # Test gradients through clip
        num_features = 17
        grid.set_from_dense_grid(1, [32, 32, 32], [0, 0, 0], voxel_sizes=1.0 / 32, origins=[0, 0, 0])
        features = torch.randn(grid.total_voxels, num_features, device=device, dtype=dtype, requires_grad=True)

        clipped_features, clipped_grid = grid.clip(features, [[0, 0, 0]], [[5, 5, 5]])

        loss = clipped_features.jdata.pow(3).sum()
        loss.backward()

        assert features.grad is not None  # Removes type errors with .grad
        clipped_features_grad = features.grad.clone()

        features.grad.zero_()
        self.assertTrue(torch.all(features.grad == torch.zeros_like(features.grad)))
        self.assertTrue(not torch.all(features.grad == clipped_features_grad))

        ijk_clip_mask = torch.all(grid.ijk.jdata<=5,1)

        loss = (features[ijk_clip_mask.repeat(num_features,1).swapaxes(0,1)].pow(3)).sum()
        loss.backward()
        self.assertTrue(torch.equal(clipped_features_grad, features.grad))

    @parameterized.expand(all_device_dtype_combos)
    def test_dual_without_border(self, device, dtype, mutable):
        # FIXME: This test fails for mutable grids because dual grid doesn't preserve mask
        vox_size = np.random.rand() * 0.1 + 0.05
        vox_origin = torch.rand(3).to(dtype).to(device)
        for b in [1, 3]:
            pts = JaggedTensor([torch.randn(np.random.randint(100_000, 300_000), 3).to(device=device, dtype=dtype)] * b)
            grid = fvdb.sparse_grid_from_points(pts, [0] * 3, [0] * 3, vox_size, vox_origin)
            dual_grid = grid.dual_grid()

            neighbors = grid.neighbor_indexes(dual_grid.ijk, 1)
            inner_mask = torch.all(neighbors.jdata[:, 1:, 1:, 1:].reshape(-1, 8) != -1, dim=-1)
            inner_ijk = dual_grid.ijk.r_masked_select(inner_mask)
            dual_inner = fvdb.sparse_grid_from_ijk(inner_ijk, voxel_sizes=vox_size, origins=vox_origin)

            dual_outer_with_skip = grid.dual_grid(exclude_border=True)
            for i in range(b):
                ijk1 = dual_inner.ijk[i].jdata
                ijk2 = dual_outer_with_skip.ijk[i].jdata
                ijk1_i = set([tuple(ijk1[j].cpu().numpy().tolist()) for j in range(ijk1.shape[0])])
                ijk2_i = set([tuple(ijk2[j].cpu().numpy().tolist()) for j in range(ijk2.shape[0])])
                self.assertTrue(ijk1_i == ijk2_i)

if __name__ == '__main__':
    unittest.main()
