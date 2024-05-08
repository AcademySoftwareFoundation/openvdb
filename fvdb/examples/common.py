import time

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch

from fvdb import CudaSparseFeatureIndexGrid, SparseFeatureIndexGrid


def make_vdb_from_points(pts, vox_size, vox_origin, padding):
    print("Building vdb...")
    start = time.time()
    if pts.is_cuda:
        print("Building CUDA Grid")
        fvdb = CudaSparseFeatureIndexGrid(vox_size, vox_origin)
    else:
        print("Building CPU Grid")
        fvdb = SparseFeatureIndexGrid(vox_size, vox_origin)
    fvdb.build_from_padded_pointcloud(pts, [-padding]*3, [padding]*3)
    print(f"Done in {time.time() - start}s")
    print(f"VDB has {fvdb.num_voxels()} voxels and {fvdb.num_corners()} corners")

    return fvdb


def make_ray_grid(nrays, theta=0.0, r=0.0, minb=(-0.3, -0.3), maxb=(0.3, 0.3),
                  device='cpu', dtype=torch.float32):
    ray_o = torch.tensor([[0.3500, 0.1075, 1.1210]] * nrays**2)
    offset = torch.tensor([[np.cos(theta), np.sin(theta), 0.0]]) * r
    ray_o += offset
    # ray_o = torch.tensor([[0.3500, 0.075, 1.1210]] * nrays**2) #+ p.mean(0, keepdim=True)
    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays*1j,
                           minb[1]:maxb[1]:nrays*1j]] +
                  [np.ones(nrays**2)], axis=-1).astype(np.float32))
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, -ray_d


def load_pointcloud(data_path, skip_every=1, shuffle=True, device='cuda', dtype=torch.float32):
    print("Loading...")
    start = time.time()
    pts = pcu.load_mesh_v(data_path)
    if shuffle:
        pts = pts[np.random.permutation(pts.shape[0])]
    pts = pts[::skip_every]
    print(f"Done in {time.time() - start}s")
    return torch.from_numpy(pts).to(device).to(dtype)


def plot_ray_segments(ray_o, ray_d, times, counts):
    for i in range(0, times.shape[0]):
        t0s = times[i][:counts[i]][:, 0].unsqueeze(-1)
        t1s = times[i][:counts[i]][:, 1].unsqueeze(-1)
        roi = ray_o[i].unsqueeze(0)
        rdi = ray_d[i].unsqueeze(0)
        rp = torch.cat([roi + t0s * rdi, roi + t1s * rdi])
        re = torch.stack([torch.arange(t0s.shape[0]),
                          torch.arange(t0s.shape[0]) + t0s.shape[0]], dim=-1)

        ray_segs = ps.register_curve_network(f"ray segments {i}", rp, re, radius=0.001)
        rv = torch.zeros(re.shape[0])
        rv[::2] = 1.0
        ray_segs.add_scalar_quantity(f"segment colors {i}", rv, defined_on='edges',
                                     enabled=True, cmap='jet')

