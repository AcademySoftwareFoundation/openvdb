import os
import time

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch

import fvdb
from fvdb import GridBatch, JaggedTensor


def make_ray_grid(nrays, minb=(-0.3, -0.3), maxb=(0.3, 0.3), device='cpu', dtype=torch.float32):
    ray_o = torch.tensor([[0.0, 0.0, -0.1]] * nrays**2) #+ p.mean(0, keepdim=True)
    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays*1j,
                           minb[1]:maxb[1]:nrays*1j]] +
                  [np.ones(nrays**2)], axis=-1).astype(np.float32))
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


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


def main():
    device = 'cuda'
    dtype = torch.float32
    vox_size = [0.01, 0.01, 0.01]
    vox_origin = [0.0, 0.0, 0.0]
    nrays = 1000
    plot_every = 5000  # only plot every n rays
    N = 100

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    pts, nms = pcu.load_mesh_vn(data_path)
    pts -= pts.mean(0)
    pts = np.concatenate([pts,
                          pts + np.array([0, 0, .48]),
                          pts + np.array([0, 0, .96]),
                          pts + np.array([0, 0, 1.44])])
    nms = np.concatenate([nms, nms, nms, nms])
    p, n = torch.from_numpy(pts[::N]).to(device).to(dtype), \
           torch.from_numpy(nms[::N]).to(device).to(dtype)

    grid = GridBatch(mutable=False, device=device)
    grid.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, vox_origin)

    # nsplat, counts = grid.splat_trilinear(p.to(device), n.to(device), return_counts=True)
    # nsplat /= torch.minimum(counts, torch.ones_like(counts)).unsqueeze(-1)

    gc, ge = grid.viz_edge_network

    ray_o, ray_d = make_ray_grid(nrays, device=device, dtype=dtype)
    pmt = torch.randperm(ray_o.shape[0]).to(device)
    ray_o, ray_d = ray_o[pmt], ray_d[pmt]

    print(f"Tracing {nrays ** 2} Ray Voxels...")
    start = time.time()
    pack_info, vox, times = grid.voxels_along_rays(ray_o, ray_d, 100, 1e-4)
    pack_info = pack_info.jdata
    vox = vox.jdata
    times = times.jdata
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s")

    # Downsample and move the output to the CPU for plotting
    ray_o, ray_d = ray_o[::plot_every], ray_d[::plot_every]
    pack_info = pack_info[::plot_every]

    print(p.dtype, n.dtype)
    print(ray_o.dtype, ray_d.dtype)
    print(vox.dtype, times.dtype, pack_info.dtype)
    print(gc.dtype, ge.dtype)
    p, n = p.cpu(), n.cpu()
    ray_o, ray_d = ray_o.cpu(), ray_d.cpu()
    vox, times, pack_info = vox.cpu(), times.cpu(), pack_info.cpu()
    gc, ge = gc.cpu(), ge.cpu()

    print(times)
    ps.init()
    ps.set_ground_plane_mode('shadow_only')

    ps.register_point_cloud("points", p, radius=0.0025)

    plot_ray_segments(ray_o, ray_d, times, pack_info)

    vox_i = torch.cat([vox[pack_info[i][0]: pack_info[i][0] + pack_info[i][1]] for i in range(pack_info.shape[0])])
    vox_i = JaggedTensor([vox_i])
    iv, ie = fvdb.sparse_grid_from_ijk(
        vox_i, voxel_sizes=vox_size, origins=vox_origin).viz_edge_network
    ps.register_curve_network("intersected voxels", iv.jdata, ie.jdata, enabled=True, radius=0.0009)
    ps.register_point_cloud("grid corners", gc.jdata, enabled=True, radius=0.001)
    ps.register_curve_network("grid edges", gc.jdata, ge.jdata, enabled=True, radius=0.00075, transparency=0.7)

    ray_dir_points = torch.cat([ray_o, ray_o + times.max() * ray_d])
    ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]),
                                 torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
    ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
    ps.register_point_cloud("ray origins", ray_o, radius=0.01)
    ps.show()



if __name__ == "__main__":
    main()
