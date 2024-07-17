# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
import os
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor, sparse_grid_from_ijk
from fvdb.utils import volume_render

from .common import (make_dense_grid_and_point_data,
                     random_drop_points_if_mutable,
                     dtype_to_atol)

all_device_combos = [
    ['cpu', True],
    ['cuda', True],
    ['cpu', False],
    ['cuda', False],
]

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

class TestRayMarching(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_with_misses(self, device, dtype, mutable):
        grid = fvdb.sparse_grid_from_dense(num_grids=1, dense_dims=[32, 32, 32], device=device, voxel_sizes=[0.1, 0.1, 0.1], origins=[0, 0, 0])

        ray_o = torch.tensor([[100,0,0]]).to(device).to(dtype)
        ray_d_hit = torch.tensor([[-1, 0, 0]]).to(device).to(dtype) # towards the grid
        ray_d_nohit = torch.tensor([[1, 0, 0]]).to(device).to(dtype) # away from the grid

        ray_o = ray_o.repeat(10, 1) # shape [10, 3]
        ray_d = torch.cat([ray_d_hit, ray_d_nohit], dim=0).repeat(5, 1) # shape [10, 3]

        segment = grid.segments_along_rays(ray_o, ray_d, 1, eps=1e-3)
        target_lshape = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
        self.assertEqual(len(segment), 1)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

        ray_d = torch.cat([ray_d_hit.repeat(5, 1), ray_d_nohit.repeat(5, 1)], dim=0) # shape [10, 3]
        segment = grid.segments_along_rays(ray_o, ray_d, 1, eps=1e-3)
        target_lshape = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
        self.assertEqual(len(segment), 1)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_with_misses(self, device, dtype, mutable):
        grid = fvdb.sparse_grid_from_dense(num_grids=1, dense_dims=[32, 32, 32], device=device, voxel_sizes=[0.1, 0.1, 0.1], origins=[0, 0, 0])

        ray_o = torch.tensor([[100,0,0]]).to(device).to(dtype)
        ray_d_hit = torch.tensor([[-1, 0, 0]]).to(device).to(dtype) # towards the grid
        ray_d_nohit = torch.tensor([[1, 0, 0]]).to(device).to(dtype) # away from the grid

        ray_o = ray_o.repeat(10, 1) # shape [10, 3]
        ray_d = torch.cat([ray_d_hit, ray_d_nohit], dim=0).repeat(5, 1) # shape [10, 3]

        voxels, times = grid.voxels_along_rays(ray_o, ray_d, 1, eps=1e-3)
        target_lshape = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
        self.assertEqual(len(voxels), 1)
        self.assertEqual(len(times), 1)
        self.assertEqual(len(voxels.lshape), len(target_lshape))
        self.assertEqual(len(times.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = voxels.lshape[i]
            vls = times.lshape[i]
            assert isinstance(sls, list)
            assert isinstance(vls, list)
            self.assertEqual(len(sls), len(tls))
            self.assertEqual(len(sls), len(vls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)
                self.assertEqual(vls[j], tlsj)

        ray_d = torch.cat([ray_d_hit.repeat(5, 1), ray_d_nohit.repeat(5, 1)], dim=0) # shape [10, 3]
        voxels, times = grid.voxels_along_rays(ray_o, ray_d, 1, eps=1e-3)
        target_lshape = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
        self.assertEqual(len(voxels), 1)
        self.assertEqual(len(times), 1)
        self.assertEqual(len(voxels.lshape), len(target_lshape))
        self.assertEqual(len(times.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = voxels.lshape[i]
            vls = times.lshape[i]
            assert isinstance(sls, list)
            assert isinstance(vls, list)
            self.assertEqual(len(sls), len(tls))
            self.assertEqual(len(sls), len(vls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)
                self.assertEqual(vls[j], tlsj)

    @parameterized.expand(all_device_dtype_combos)
    def test_uniform_samples_with_misses(self, device, dtype, mutable):
        grid = fvdb.sparse_grid_from_dense(num_grids=1, dense_dims=[32, 32, 32], device=device, voxel_sizes=[0.1, 0.1, 0.1], origins=[0, 0, 0])

        ray_o = torch.tensor([[100,0,0]]).to(device).to(dtype)
        ray_d_hit = torch.tensor([[-1, 0, 0]]).to(device).to(dtype) # towards the grid
        ray_d_nohit = torch.tensor([[1, 0, 0]]).to(device).to(dtype) # away from the grid

        ray_o = ray_o.repeat(10, 1) # shape [10, 3]
        ray_d = torch.cat([ray_d_hit, ray_d_nohit], dim=0).repeat(5, 1) # shape [10, 3]

        t_min = torch.zeros(ray_o.shape[0]).to(ray_o)
        t_max = torch.ones(ray_o.shape[0]).to(ray_o) * 1e10

        segment = grid.uniform_ray_samples(ray_o, ray_d, t_min, t_max, 0.5, eps=1e-3)
        target_lshape = [[8, 0, 8, 0, 8, 0, 8, 0, 8, 0]] if dtype != torch.float16 else [[7, 0, 7, 0, 7, 0, 7, 0, 7, 0]]
        self.assertEqual(len(segment), 1)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

        ray_d = torch.cat([ray_d_hit.repeat(5, 1), ray_d_nohit.repeat(5, 1)], dim=0) # shape [10, 3]
        segment = grid.uniform_ray_samples(ray_o, ray_d, t_min, t_max, 0.5, eps=1e-3)
        target_lshape = [[8, 8, 8, 8, 8, 0, 0, 0, 0, 0]] if dtype != torch.float16 else [[7, 7, 7, 7, 7, 0, 0, 0, 0, 0]]
        self.assertEqual(len(segment), 1)
        self.assertEqual(len(segment.lshape), len(target_lshape))
        for i, tls in enumerate(target_lshape):
            sls = segment.lshape[i]
            assert isinstance(sls, list)
            self.assertEqual(len(sls), len(tls))
            for j, tlsj in enumerate(tls):
                self.assertEqual(sls[j], tlsj)

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
            for ray_i in range(pack_info.shape[0] - 1):
                start_idx, end_idx = pack_info[ray_i], pack_info[ray_i + 1]
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
        ray_intervals = grid.uniform_ray_samples(
            ray_o, ray_d, tmin, tmax, step_size)

        # FIXME: Francis -- this is a hack before I the fix volume_render API
        ray_idx = ray_intervals.jidx.int()
        pack_info = ray_intervals.joffsets
        ray_intervals = ray_intervals.jdata

        ray_t = ray_intervals.mean(1)
        ray_mids = grid.uniform_ray_samples(
            ray_o, ray_d, tmin, tmax, step_size, return_midpoints=True).jdata
        self.assertTrue(torch.allclose(ray_mids, ray_t, atol=dtype_to_atol(dtype)))

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

    @parameterized.expand(all_device_combos)
    def test_world_to_dual(self, device, include_end_segments: bool):
        grid = GridBatch(device=device)
        grid.set_from_dense_grid(1, [2, 2, 2])

        rays_o = torch.tensor([[-0.6, 0., 0.]], device=device)
        rays_d = torch.tensor([[1., 0., 0.]], device=device)
        nears = torch.tensor([0.], device=device)
        fars = torch.tensor([5.], device=device)
        step_size = 0.4
        cone_angle = 0.0
        if include_end_segments:
            t_targets = torch.tensor([0.1, 0.4, 0.8, 1.2, 1.6, 2.0, 2.1], device=device)
        else:
            t_targets = torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], device=device)

        intervals = grid.uniform_ray_samples(
            rays_o, rays_d, nears, fars, step_size, cone_angle, include_end_segments
        )
        middles = grid.uniform_ray_samples(
            rays_o, rays_d, nears, fars, step_size, cone_angle, include_end_segments, return_midpoints=True
        ).jdata
        t_starts, t_ends = torch.unbind(intervals.jdata, dim=-1)
        self.assertTrue(torch.allclose(middles, (t_starts + t_ends) / 2.0, atol=dtype_to_atol(t_starts.dtype)))

        assert torch.allclose(t_starts, t_targets[:-1])
        assert torch.allclose(t_ends, t_targets[1:])
        assert torch.allclose(intervals.jidx, torch.zeros_like(intervals.jidx))

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
        ray_times_inside = grid.uniform_ray_samples(ray_o_inside, ray_d_inside, tmin, tmax, step_size, include_end_segments=False)
        ray_idx, ray_times_inside = ray_times_inside.jidx.long(), ray_times_inside.jdata
        nsteps_inside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_inside, torch.round(nsteps_inside), atol=dtype_to_atol(dtype)))

        ray_times_inside = grid.uniform_ray_samples(ray_o_outside, ray_d_outside, tmin, tmax, step_size, include_end_segments=False)
        ray_idx, ray_times_inside = ray_times_inside.jidx.long(), ray_times_inside.jdata
        nsteps_outside = (ray_times_inside - tmin[ray_idx, None]) / step_size
        self.assertTrue(torch.allclose(nsteps_outside, torch.round(nsteps_outside), atol=dtype_to_atol(dtype)))

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_bug(self, device, dtype, mutable):
        data_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
        data = torch.load(os.path.join(data_path, "repro_bug.pth"))
        grid = sparse_grid_from_ijk(data['ijk'].to(device), voxel_sizes=data['vox_size'], origins=data['vox_origin'])
        ray_o: torch.Tensor = torch.load(
            os.path.join(data_path, "ray_o.pth")).to(device=device, dtype=dtype)
        ray_d: torch.Tensor = torch.load(
            os.path.join(data_path, "ray_d.pth")).to(device=device, dtype=dtype)

        segments = grid.segments_along_rays(ray_o.to(dtype), ray_d.to(dtype), 100, 0.0)

        if dtype == torch.float16:
            self.assertEqual(segments[0][0].jdata.shape[0], 53)
        else:
            self.assertEqual(segments[0][0].jdata.shape[0], 52)

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

            segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=eps)

            for segments_i in segments[0]:
                if segments_i.rshape[0] == 0:
                    continue
                segments_i = segments_i.jdata
                self.assertTrue(torch.all(segments_i[:, 1] - segments_i[:, 0] >= eps))
                self.assertTrue(torch.all(segments_i[1:, 0] - segments_i[:-1, 0] >= eps),
                                f"mismatch eps = {eps}, diff = {segments_i[1:, 0] >= segments_i[:-1, 0]}, vals = {segments_i}, (1) = {segments_i[1:, 0]}, (2) = {segments_i[:-1, 0]}")
                self.assertTrue(torch.all(segments_i[1:, 1] - segments_i[:-1, 1] >= eps))

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_always_sorted_batched(self, device, dtype, mutable):
        for eps in [0., 1e-5]:
            pts = fvdb.JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)] * 2)
            grid = GridBatch(mutable=mutable).to(device)
            grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.0001, torch.zeros(3))
            random_drop_points_if_mutable(grid)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[0].jdata[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
            rays_o = fvdb.JaggedTensor([rays_o] * 2)
            rays_d = fvdb.JaggedTensor([rays_d] * 2)

            segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=eps)

            for b_i in range(len(pts)):
                for segments_i in segments[b_i]:
                    if segments_i.rshape[0] == 0:
                        continue
                    segments_i = segments_i.jdata
                    self.assertTrue(torch.all(segments_i[:, 1] - segments_i[:, 0] >= eps))
                    self.assertTrue(torch.all(segments_i[1:, 0] - segments_i[:-1, 0] >= eps),
                                    f"mismatch eps = {eps}, diff = {segments_i[1:, 0] >= segments_i[:-1, 0]}, vals = {segments_i}, (1) = {segments_i[1:, 0]}, (2) = {segments_i[:-1, 0]}")
                    self.assertTrue(torch.all(segments_i[1:, 1] - segments_i[:-1, 1] >= eps))

    @parameterized.expand(all_device_dtype_combos)
    def test_segments_along_rays_batch_size_mismatch_throws(self, device, dtype, mutable):
        pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
        # pts = fvdb.JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)]*2)
        grid = GridBatch(mutable=mutable).to(device)
        grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.0001, torch.zeros(3))
        random_drop_points_if_mutable(grid)

        rays_o = -torch.ones(100, 3).to(device).to(dtype)
        rays_d = pts[:100] - rays_o
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = fvdb.JaggedTensor([rays_o]*2)
        rays_d = fvdb.JaggedTensor([rays_d]*2)

        with self.assertRaises(ValueError):
            segments = grid.segments_along_rays(rays_o, rays_d, 100, eps=1e-4)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_always_sorted(self, device, dtype, mutable):
        for i in range(3):
            pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            grid = GridBatch(mutable=mutable).to(device)
            grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.01, torch.zeros(3))
            random_drop_points_if_mutable(grid)

            rays_o = -torch.ones(100, 3).to(device).to(dtype)
            rays_d = pts[:100] - rays_o
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

            out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)

            out_idx, out_times2 = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5, return_ijk=False)
            out_idx2 = grid.ijk_to_index(out_voxels.jflatten(dim=1)).jreshape_as(out_idx)
            self.assertTrue(torch.all(out_idx.jdata == out_idx2.jdata))
            self.assertTrue(torch.allclose(out_times.jdata, out_times2.jdata))

            for times_i, voxels_i in zip(out_times[0], out_voxels[0]):
                if times_i.rshape[0] == 0:
                    continue
                times_i, voxels_i = times_i.jdata, voxels_i.jdata
                self.assertTrue(torch.all(times_i[:, 0] < times_i[:, 1]), f"Max diff = {(times_i[:, 1] - times_i[:, 0]).max().item()}")
                self.assertTrue(torch.all(times_i[1:, 0] > times_i[:-1, 0]))
                self.assertTrue(torch.all(times_i[1:, 1] > times_i[:-1, 1]))
                # Should always march
                max_diff = torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values.cpu().detach().numpy()
                self.assertTrue(torch.all(torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values >= 1),
                                f"Max diff = {max_diff, voxels_i.cpu().numpy(), times_i.cpu().numpy()}")

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_batch_size_mismatch_throws(self, device, dtype, mutable):
        pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
        # pts = fvdb.JaggedTensor([torch.rand(10000, 3).to(device=device, dtype=dtype)]*2)
        grid = GridBatch(mutable=mutable).to(device)
        grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.0001, torch.zeros(3))
        random_drop_points_if_mutable(grid)

        rays_o = -torch.ones(100, 3).to(device).to(dtype)
        rays_d = pts[:100] - rays_o
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = fvdb.JaggedTensor([rays_o]*2)
        rays_d = fvdb.JaggedTensor([rays_d]*2)

        with self.assertRaises(ValueError):
            out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)

    @parameterized.expand(all_device_dtype_combos)
    def test_voxels_along_rays_always_sorted_batched(self, device, dtype, mutable):
        for i in range(3):
            # pts = torch.rand(10000, 3).to(device=device, dtype=dtype)
            pts = fvdb.JaggedTensor([torch.rand(100, 3).to(device=device, dtype=dtype)]*2)
            grid = GridBatch(mutable=mutable).to(device)
            grid.set_from_points(pts, (0, 0, 0), (1, 1, 1), 0.01, torch.zeros(3))
            random_drop_points_if_mutable(grid)

            rays_o = [-torch.ones(100, 3).to(device).to(dtype)]*2
            rays_d = [pts[i].jdata[:100] - rays_o[i] for i in range(2)]
            rays_d = [r / torch.norm(r, dim=-1, keepdim=True) for r in rays_d]
            rays_o = fvdb.JaggedTensor(rays_o)
            rays_d = fvdb.JaggedTensor(rays_d)

            out_voxels, out_times = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5)

            out_idx, out_times2 = grid.voxels_along_rays(rays_o, rays_d, 100, 1.0e-5, return_ijk=False)
            out_idx2 = grid.ijk_to_index(out_voxels.jflatten(dim=1)).jreshape_as(out_idx)
            self.assertTrue(torch.all(out_idx.jdata == out_idx2.jdata))
            self.assertTrue(torch.allclose(out_times.jdata, out_times2.jdata))

            for i, _ in enumerate(zip(out_voxels, out_times)):
                for times_i, voxels_i in zip(out_times[i], out_voxels[i]):
                    if times_i.rshape[0] == 0:
                        continue
                    times_i, voxels_i = times_i.jdata, voxels_i.jdata
                    # print(times_i)
                    self.assertTrue(torch.all(times_i[:, 0] < times_i[:, 1]), f"Max diff = {(times_i[:, 1] - times_i[:, 0]).max().item()}")
                    if times_i[1:, 0].numel() > 0:
                        self.assertTrue(torch.all(times_i[1:, 0] > times_i[:-1, 0]), f"Max diff = {(times_i[1:, 0] - times_i[:-1, 0]).max().item()}")
                    # Should always march
                    max_diff = torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values.cpu().detach().numpy()
                    self.assertTrue(torch.all(torch.max(voxels_i[1:] - voxels_i[:-1], dim=1).values >= 1),
                                    f"Max diff = {max_diff, voxels_i.cpu().numpy(), times_i.cpu().numpy()}")



if __name__ == '__main__':
    unittest.main()
