import timeit

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
from typing import Tuple, Union, List
from pathlib import Path
import logging

from fvdb import GridBatch

def make_grid_from_points(pts:torch.Tensor, padding, vox_size, vox_origin) -> GridBatch:
    logging.info("Building GridBatch from points...")
    start = timeit.default_timer()
    grid = GridBatch(device=pts.device)
    grid.set_from_points(pts, [-padding]*3, [padding]*3, voxel_sizes=vox_size, origins=vox_origin)
    torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s")
    logging.info(f"GridBatch has {grid.total_voxels} voxels")

    return grid


def make_ray_grid(nrays: int,
                  origin: Union[torch.Tensor, Tuple, List],
                  minb=(-0.3, -0.3),
                  maxb=(0.3, 0.3),
                  device:Union[str, torch.device]='cpu',
                  dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    ray_o = torch.tensor([origin] * nrays**2)

    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays*1j,
                           minb[1]:maxb[1]:nrays*1j]] +
                  [np.ones(nrays**2)], axis=-1).astype(np.float32))
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


def load_pointcloud(data_path, skip_every=1, shuffle=False, device=torch.device('cuda'), dtype=torch.float32) -> torch.Tensor:
    logging.info(f"Loading pointlcoud {data_path}...")
    start = timeit.default_timer()
    pts = pcu.load_mesh_v(data_path)
    if shuffle:
        pts = pts[np.random.permutation(pts.shape[0])]
    pts = pts[::skip_every]
    logging.info(f"Done in {timeit.default_timer() - start}s")
    return torch.from_numpy(pts).to(device).to(dtype)

def load_mesh(data_path, skip_every=1, device=torch.device('cuda'), dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor] :
    logging.info(f"Loading mesh {data_path}...")
    start = timeit.default_timer()
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts[::skip_every]).to(device).to(dtype), \
           torch.from_numpy(nms[::skip_every]).to(device).to(dtype)
    logging.info(f"Done in {timeit.default_timer() - start}s")
    return p, n

def load_dragon_mesh(skip_every=1, device=torch.device('cuda'), dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    data_path = Path(__file__).resolve().parent.parent / "data/dragon.ply"
    return load_mesh(data_path, skip_every=skip_every, device=device, dtype=dtype)

def plot_ray_segments(ray_o, ray_d, times, pack_info):
    for i in range(0, pack_info.shape[0]):
        t0s = times[pack_info[i][0]:pack_info[i][0]+pack_info[i][1]][:, 0].unsqueeze(-1)
        t1s = times[pack_info[i][0]:pack_info[i][0]+pack_info[i][1]][:, 1].unsqueeze(-1)
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

