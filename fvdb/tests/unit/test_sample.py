import itertools
import unittest

import numpy as np
import torch
from parameterized import parameterized

from .common import (make_dense_grid_and_point_data,
                     make_sparse_grid_and_point_data,
                     random_drop_points_if_mutable,
                     sparse_grid_from_dense_cube)

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

def trilinear_sample_pytorch(fvdb, p, features, is_dual: bool):
    dual_features_grid = fvdb.read_into_dense(features).squeeze(0).permute(3, 2, 1, 0).unsqueeze(0)
    p_in = p.reshape(1, 1, 1, -1, 3)  # [1, 1, 1, N, 3]
    res = torch.nn.functional.grid_sample(
        dual_features_grid, p_in, mode='bilinear', align_corners=is_dual
    ).squeeze().transpose(0, 1)
    return res

def upsample_pytorch(small_features, scale: int, mode: str):
    # Two differences with nn.UpsamplingBilinear:
    #   1. align_corners = True
    #   2. Boundary padding instead of zero padding.
    feat = small_features.unsqueeze(0)
    feat = torch.nn.functional.pad(
        feat, (1, 1, 1, 1, 1, 1), mode='constant', value=0.0)
    big_features = torch.nn.functional.interpolate(
        feat, scale_factor=scale, mode=mode,
        align_corners=False if mode == 'trilinear' else None)
    big_features = big_features[0][:, scale:-scale, scale:-scale, scale:-scale]

    return big_features

def sample_trilinear_naive(pts, corner_feats, grid):
    device = corner_feats.device
    dtype = corner_feats.dtype
    feats_dim = corner_feats.shape[-1]

    if pts.dtype == torch.half:
        pts = pts.to(torch.float)

    grid_pts = grid.world_to_grid(pts).jdata
    nearest_ijk = torch.floor(grid_pts)

    offsets = torch.tensor(
        list(itertools.product([0, 1], [0, 1], [0, 1])),
        device=device, dtype=torch.long
    )

    nearest_ijk = nearest_ijk.unsqueeze(1).long() + offsets.unsqueeze(0)
    unique_ijk, ijk_idx = torch.unique(nearest_ijk.reshape(-1, 3), dim=0, return_inverse=True)
    corner_feats_indices = grid.ijk_to_index(nearest_ijk.reshape(-1, 3)).jdata.reshape(-1, 8)
    sel_corner_feats = corner_feats[corner_feats_indices]
    sel_corner_feats[~grid.coords_in_active_voxel(nearest_ijk.reshape(-1, 3), False).jdata.reshape(-1, 8)] = 0.
    uvws = torch.abs(grid_pts.unsqueeze(1) - nearest_ijk.to(pts.dtype))

    trilinear_weights = torch.prod(1. - uvws, dim=-1)
    interpolated_feats = trilinear_weights.unsqueeze(-1) * sel_corner_feats.to(pts.dtype)
    return torch.sum(interpolated_feats, dim=1).to(dtype)


def _bezier(x: torch.Tensor):
    b1 = (x + 1.5) ** 2
    b2 = -2 * (x ** 2) + 1.5
    b3 = (x - 1.5) ** 2
    m1 = (x >= -1.5) & (x < -0.5)
    m2 = (x >= -0.5) & (x < 0.5)
    m3 = (x >= 0.5) & (x < 1.5)
    return m1 * b1 + m2 * b2 + m3 * b3

def sample_bezier_naive(pts, corner_feats, grid):
    device = corner_feats.device
    dtype = corner_feats.dtype
    feats_dim = corner_feats.shape[-1]

    if pts.dtype == torch.half:
        pts = pts.to(torch.float)

    grid_pts = grid.world_to_grid(pts).jdata
    nearest_ijk = torch.round(grid_pts)

    offsets = torch.tensor(
        list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])),
        device=device, dtype=torch.long
    )

    nearest_ijk = nearest_ijk.unsqueeze(1).long() + offsets.unsqueeze(0)
    unique_ijk, ijk_idx = torch.unique(nearest_ijk.reshape(-1, 3), dim=0, return_inverse=True)
    corner_feats_indices = grid.ijk_to_index(nearest_ijk.reshape(-1, 3)).jdata.reshape(-1, 27)
    sel_corner_feats = corner_feats[corner_feats_indices]
    sel_corner_feats[~grid.coords_in_active_voxel(nearest_ijk.reshape(-1, 3), False).jdata.reshape(-1, 27)] = 0.
    bz_dir = _bezier(nearest_ijk.to(pts.dtype) - grid_pts.unsqueeze(1))
    bz_weights = torch.prod(bz_dir, dim=-1)
    interpolated_feats = bz_weights.unsqueeze(-1) * sel_corner_feats.to(pts.dtype)
    return torch.sum(interpolated_feats, dim=1).to(dtype)

def splat_trilinear_naive(pts, feats, grid):
    device = feats.device
    dtype = feats.dtype
    feats_dim = feats.shape[-1]

    if pts.dtype == torch.half:
        pts = pts.to(torch.float)

    grid_pts = grid.world_to_grid(pts).jdata
    nearest_ijk = torch.floor(grid_pts)
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

    nearest_ijk = nearest_ijk.unsqueeze(1).long() + offsets.unsqueeze(0)
    unique_ijk, ijk_idx = torch.unique(nearest_ijk.reshape(-1, 3), dim=0, return_inverse=True)
    unique_ijk = unique_ijk
    uvws = torch.abs(grid_pts.unsqueeze(1) - nearest_ijk.to(pts.dtype))

    trilinear_weights = torch.prod(1. - uvws, dim=-1)
    interpolated_feats = trilinear_weights.unsqueeze(-1) * feats.unsqueeze(-2)
    sum_interpolated_feats = torch.zeros((unique_ijk.shape[0], feats_dim),
                                         device=device, dtype=pts.dtype)
    sum_interpolated_feats.index_add_(
        0, ijk_idx, interpolated_feats.reshape(-1, feats_dim))
    output = torch.zeros((grid.ijk.jdata.shape[0], feats_dim),
                         device=device, dtype=dtype)
    mask = grid.coords_in_active_voxel(unique_ijk, False).jdata
    sum_interpolated_feats = sum_interpolated_feats[mask]
    valid_ijk = grid.ijk_to_index(unique_ijk[mask]).jdata
    output[valid_ijk] = sum_interpolated_feats.to(dtype)
    return output

def splat_bezier_naive(pts, feats, grid):
    device = feats.device
    dtype = feats.dtype
    feats_dim = feats.shape[-1]

    if pts.dtype == torch.half:
        pts = pts.to(torch.float)

    grid_pts = grid.world_to_grid(pts).jdata
    nearest_ijk = torch.round(grid_pts)

    offsets = torch.tensor(
        list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])),
        device=device, dtype=torch.long
    )

    nearest_ijk = nearest_ijk.unsqueeze(1).long() + offsets.unsqueeze(0)
    unique_ijk, ijk_idx = torch.unique(nearest_ijk.reshape(-1, 3), dim=0, return_inverse=True)
    corner_feats_indices = grid.ijk_to_index(nearest_ijk.reshape(-1, 3)).jdata.reshape(-1, 27)
    bz_dir = _bezier(nearest_ijk.to(pts.dtype) - grid_pts.unsqueeze(1))
    bz_weights = torch.prod(bz_dir, dim=-1)
    interpolated_feats = bz_weights.unsqueeze(-1) * feats.unsqueeze(-2).to(pts.dtype)
    sum_interpolated_feats = torch.zeros((unique_ijk.shape[0], feats_dim),
                                         device=device, dtype=pts.dtype)
    sum_interpolated_feats.index_add_(
        0, ijk_idx, interpolated_feats.reshape(-1, feats_dim))
    output = torch.zeros((grid.ijk.jdata.shape[0], feats_dim),
                         device=device, dtype=dtype)
    mask = grid.coords_in_active_voxel(unique_ijk, False).jdata
    sum_interpolated_feats = sum_interpolated_feats[mask]
    valid_ijk = grid.ijk_to_index(unique_ijk[mask]).jdata
    output[valid_ijk] = sum_interpolated_feats.to(dtype)
    return output

class TestSample(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_trilinear_dense_vs_pytorch(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-2
            rtol=1e-2
        elif dtype == torch.float32:
            atol=1e-4
            rtol=1e-4
        else:
            atol=1e-5
            rtol=1e-8
        fvdb, fvdb_d, p = make_dense_grid_and_point_data(7, device, dtype, mutable=mutable)

        # Primal
        primal_features = torch.rand((fvdb.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = fvdb.sample_trilinear(p, primal_features).jdata.squeeze()
        grad_out = torch.rand_like(fv) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()
        fp = trilinear_sample_pytorch(fvdb, p, primal_features, is_dual=False)
        fp.backward(grad_out)
        gp = primal_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((fvdb_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv = fvdb_d.sample_trilinear(p, dual_features).jdata.squeeze()
        grad_out = torch.rand_like(fv) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()
        fp = trilinear_sample_pytorch(fvdb_d, p, dual_features, is_dual=True)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")


    @parameterized.expand(all_device_dtype_combos)
    def test_upsample_dense_vs_pytorch(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-2
            rtol=1e-2
        else:
            atol=1e-5
            rtol=1e-8

        nvox = 7
        scale = 2
        fvdb = sparse_grid_from_dense_cube([nvox] * 3, device=device, mutable=mutable)

        small_features = torch.rand((1, nvox, nvox, nvox), device=device, dtype=dtype)
        small_features.requires_grad = True
        small_features_vdb = fvdb.read_from_dense(small_features.permute(3, 2, 1, 0).contiguous().unsqueeze(0))

        fvdb_big = fvdb.subdivided_grid(scale)
        big_pos = fvdb_big.grid_to_world(fvdb_big.ijk.type(dtype)).jdata
        self.assertEqual(big_pos.dtype, dtype)
        big_features_vdb = fvdb.sample_trilinear(big_pos, small_features_vdb).jdata
        fv = fvdb_big.read_into_dense(big_features_vdb).squeeze(0).permute(3, 2, 1, 0)
        grad_out = torch.rand_like(fv) + 0.1
        fv.backward(grad_out)
        assert small_features.grad is not None
        gv = small_features.grad.clone()
        small_features.grad.zero_()

        fp = upsample_pytorch(small_features, scale, 'trilinear')
        fp.backward(grad_out)
        assert small_features.grad is not None
        gp = small_features.grad.clone()
        small_features.grad.zero_()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")
        small_features_vdb = fvdb.read_from_dense(small_features.permute(3, 2, 1, 0).contiguous().unsqueeze(0))
        fvdb_big = fvdb.subdivided_grid(scale)
        big_pos = fvdb_big.grid_to_world(fvdb_big.ijk.type(dtype)).jdata
        big_features_vdb, _ = fvdb.subdivide(scale, small_features_vdb, fine_grid=fvdb_big)
        fv = fvdb_big.read_into_dense(big_features_vdb).squeeze(0).permute(3, 2, 1, 0)
        fv.backward(grad_out)
        gv = small_features.grad.clone()
        small_features.grad.zero_()

        fp = upsample_pytorch(small_features, scale, 'nearest')
        fp.backward(grad_out)
        gp = small_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_trilinear_sparse_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-3
            rtol=1e-3
        else:
            atol=1e-5
            rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_trilinear(p, primal_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_trilinear_naive(p, primal_features, grid)
        fp.backward(grad_out)
        gp = primal_features.grad.clone()
        primal_features.grad.zero_()
        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv = grid_d.sample_trilinear(p, dual_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()
        fp = sample_trilinear_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()
        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_trilinear_with_grad_sparse_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-3
            rtol=1e-3
        else:
            atol=1e-5
            rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv, dfv = grid.sample_trilinear_with_grad(p, primal_features)
        self.assertEqual(fv.dtype, dtype)
        self.assertEqual(dfv.dtype, dtype)
        grad_out = torch.rand_like(fv.jdata.squeeze()) + 0.1
        fv.jdata.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_trilinear_naive(p, primal_features, grid)
        fp.backward(grad_out)
        gp = primal_features.grad.clone()
        primal_features.grad.zero_()
        self.assertTrue(torch.allclose(fv.jdata, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv.jdata - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv, _ = grid_d.sample_trilinear_with_grad(p, dual_features)
        grad_out = torch.rand_like(fv.jdata.squeeze()) + 0.1
        fv.jdata.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()
        fp = sample_trilinear_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()
        self.assertTrue(torch.allclose(fv.jdata, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv.jdata - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_trilinear_sparse_onbound_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            f_atol=1e-2
            f_rtol=1e-2
            g_atol=1e-2
            g_rtol=1e-2
        else:
            f_atol=1e-5
            f_rtol=1e-8
            g_atol=1e-5
            g_rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, include_boundary_points=True, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        p.requires_grad = True
        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_trilinear(p, primal_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_trilinear_naive(p, primal_features, grid)
        fp.backward(grad_out)
        gp = primal_features.grad.clone()
        primal_features.grad.zero_()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv = grid_d.sample_trilinear(p, dual_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_trilinear_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()
        dual_features.grad.zero_()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_trilinear_with_grad_sparse_onbound_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            f_atol=1e-2
            f_rtol=1e-2
            g_atol=1e-2
            g_rtol=1e-2
        else:
            f_atol=1e-5
            f_rtol=1e-8
            g_atol=1e-5
            g_rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, include_boundary_points=True, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        p.requires_grad = True
        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv, _ = grid.sample_trilinear_with_grad(p, primal_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_trilinear_naive(p, primal_features, grid)
        fp.backward(grad_out)
        gp = primal_features.grad.clone()
        primal_features.grad.zero_()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv, _ = grid_d.sample_trilinear_with_grad(p, dual_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_trilinear_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()
        dual_features.grad.zero_()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_bezier_sparse_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-1
            rtol=1e-2
        else:
            atol=1e-5
            rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_bezier(p, primal_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_bezier_naive(p, primal_features, grid)
        fp.backward(grad_out)
        assert primal_features.grad is not None
        gp = primal_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv = grid_d.sample_bezier(p, dual_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_bezier_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_bezier_with_grad_sparse_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            atol=1e-1
            rtol=1e-2
        else:
            atol=1e-5
            rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(device, dtype, mutable=mutable)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv, _ = grid.sample_bezier_with_grad(p, primal_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_bezier_naive(p, primal_features, grid)
        fp.backward(grad_out)
        assert primal_features.grad is not None
        gp = primal_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv, _ = grid_d.sample_bezier_with_grad(p, dual_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_bezier_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=atol, rtol=rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=atol, rtol=rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_bezier_sparse_onbound_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            f_atol=1e-2
            f_rtol=1e-2
            g_atol=1e-1
            g_rtol=1e-1
        else:
            f_atol=1e-5
            f_rtol=1e-8
            g_atol=1e-5
            g_rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(
            device, dtype, include_boundary_points=True, mutable=mutable, expand=1)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv = grid.sample_bezier(p, primal_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_bezier_naive(p, primal_features, grid)
        fp.backward(grad_out)
        assert primal_features.grad is not None
        gp = primal_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv = grid_d.sample_bezier(p, dual_features).jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_bezier_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_bezier_with_grad_sparse_onbound_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            f_atol=1e-2
            f_rtol=1e-2
            g_atol=1e-1
            g_rtol=1e-1
        else:
            f_atol=1e-5
            f_rtol=1e-8
            g_atol=1e-5
            g_rtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(
            device, dtype, include_boundary_points=True, mutable=mutable, expand=1)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        # Primal
        primal_features = torch.rand((grid.total_voxels, 4), device=device, dtype=dtype)
        primal_features.requires_grad = True
        fv, _ = grid.sample_bezier_with_grad(p, primal_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert primal_features.grad is not None
        gv = primal_features.grad.clone()
        primal_features.grad.zero_()

        fp = sample_bezier_naive(p, primal_features, grid)
        fp.backward(grad_out)
        assert primal_features.grad is not None
        gp = primal_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

        # Dual
        dual_features = torch.rand((grid_d.total_voxels, 4), device=device, dtype=dtype)
        dual_features.requires_grad = True
        fv, _ = grid_d.sample_bezier_with_grad(p, dual_features)
        fv = fv.jdata
        grad_out = torch.rand_like(fv.squeeze()) + 0.1
        fv.backward(grad_out)
        assert dual_features.grad is not None
        gv = dual_features.grad.clone()
        dual_features.grad.zero_()

        fp = sample_bezier_naive(p, dual_features, grid_d)
        fp.backward(grad_out)
        gp = dual_features.grad.clone()

        self.assertTrue(torch.allclose(fv, fp, atol=f_atol, rtol=f_rtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=g_atol, rtol=g_rtol),
                        f"Max grad error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_splat_trilinear_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            fatol=1e-3
            frtol=1e-4
            gatol=1e-3
            grtol=1e-3
        else:
            fatol=1e-5
            frtol=1e-8
            gatol=1e-5
            grtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(
            device, dtype, include_boundary_points=True, mutable=mutable, expand=1)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        points_data = torch.randn(p.shape[0], 7, device=device, dtype=dtype,
                                  requires_grad=True)

        fv = grid.splat_trilinear(p, points_data).jdata
        grad_out = torch.rand_like(fv)
        fv.backward(grad_out)
        assert points_data.grad is not None
        gv = points_data.grad.clone()
        points_data.grad.zero_()

        fp = splat_trilinear_naive(p, points_data, grid)
        fp.backward(grad_out)
        assert points_data.grad is not None
        gp = points_data.grad.clone()
        self.assertTrue(torch.allclose(fv, fp, atol=fatol, rtol=frtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=gatol, rtol=grtol),
                        f"Max error is {torch.max(torch.abs(gv - gp))}")

    @parameterized.expand(all_device_dtype_combos)
    def test_splat_bezier_vs_brute(self, device, dtype, mutable):
        if dtype == torch.half:
            fatol=1e-3
            frtol=1e-3
            gatol=1e-2
            grtol=1e-2
        else:
            fatol=1e-5
            frtol=1e-8
            gatol=1e-5
            grtol=1e-8

        grid, grid_d, p = make_sparse_grid_and_point_data(
            device, dtype, include_boundary_points=True, mutable=mutable, expand=1)
        random_drop_points_if_mutable(grid)
        random_drop_points_if_mutable(grid_d)

        points_data = torch.randn(p.shape[0], 7, device=device, dtype=dtype,
                                  requires_grad=True)

        fv = grid.splat_bezier(p, points_data).jdata
        grad_out = torch.rand_like(fv)
        fv.backward(grad_out)
        assert points_data.grad is not None
        gv = points_data.grad.clone()
        points_data.grad.zero_()

        fp = splat_bezier_naive(p, points_data, grid)
        fp.backward(grad_out)
        assert points_data.grad is not None
        gp = points_data.grad.clone()
        self.assertTrue(torch.allclose(fv, fp, atol=fatol, rtol=frtol),
                        f"Max error is {torch.max(torch.abs(fv - fp))}")
        self.assertTrue(torch.allclose(gv, gp, atol=gatol, rtol=grtol),
                        f"Max error is {torch.max(torch.abs(gv - gp))}")

if __name__ == '__main__':
    unittest.main()
