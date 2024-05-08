import os
import time

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
from fvdb import GridBatch


def make_ray_grid(origin, nrays, minb=(-0.3, -0.3), maxb=(0.3, 0.3), device='cpu', dtype=torch.float32):
    ray_o = torch.tensor([origin] * nrays**2) #+ p.mean(0, keepdim=True)
    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays*1j,
                           minb[1]:maxb[1]:nrays*1j]] +
                  [np.ones(nrays**2)], axis=-1).astype(np.float32))
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


def main():
    device = 'cuda'
    dtype = torch.float32
    vox_size = 0.01
    vox_origin = (0.0, 0.0, 0.0)
    N = 100
    nrays = 3000
    plot_every = 10000
    step_size = 0.5 * vox_size
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")

    pts, nms = pcu.load_mesh_vn(data_path)
    p = torch.from_numpy(pts[::N]).to(dtype).to(device)
    n = torch.from_numpy(nms[::N]).to(dtype).to(device)
    p = torch.cat([p,
                   p + torch.tensor([[0.025, 0.025, 0.125]]).to(p),
                   p + torch.tensor([[0.05, 0.05, 0.25]]).to(p)])
    n = torch.cat([n, n, n])


    print("Building fVDB...")
    grid = GridBatch(device=device)
    grid.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, vox_origin)
    print("Done!")

    gc, ge = grid.viz_edge_network

    ray_o, ray_d = make_ray_grid([0.0, 0.1, -0.1], nrays, device=device, dtype=dtype)

    # vox, times, counts = fvdb.voxels_along_rays(ray_o, ray_d, 100, 1e-5)
    print(f"Tracing {nrays ** 2} Ray Segments...")
    start = time.time()
    pack_info_segment, seg_idx, segments = grid.segments_along_rays(ray_o, ray_d, 100, 1e-4)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")

    tmin = torch.zeros(ray_o.shape[0]).to(device=device, dtype=dtype)
    tmax = torch.ones(ray_o.shape[0]).to(device=device, dtype=dtype) * 1e10
    print(f"Generating samples for {ray_o.shape[0]} Ray Segments...")
    start = time.time()
    pack_info, ray_idx, ray_ts = grid.uniform_ray_samples(ray_o, ray_d, tmin, tmax, step_size)
    pack_info = pack_info.jdata
    ray_idx = ray_idx.jdata
    ray_ts = ray_ts.jdata
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")
    print(f"Generated {ray_ts.shape[0]} samples")
    print(ray_idx.shape, ray_ts.shape, pack_info.shape)

    pmt = torch.randperm(ray_o.shape[0])[::plot_every]
    ray_o, ray_d = ray_o[pmt], ray_d[pmt]
    pack_info_segment = pack_info_segment.jdata[pmt]

    tmin = torch.zeros(ray_o.shape[0]).to(device=device, dtype=dtype)
    tmax = torch.ones(ray_o.shape[0]).to(device=device, dtype=dtype) * 1e10
    pack_info, ray_idx, ray_ts = grid.uniform_ray_samples(ray_o, ray_d, tmin, tmax, step_size)
    pack_info = pack_info.jdata
    ray_idx = ray_idx.jdata
    ray_ts = ray_ts.jdata
    ray_t = ray_ts.sum(-1) * 0.5
    ray_pts = ray_o[ray_idx] + ray_t.unsqueeze(-1) * ray_d[ray_idx]


    p, n = p.cpu(), n.cpu()
    gc, ge = gc.cpu(), ge.cpu()
    ray_o, ray_d = ray_o.cpu(), ray_d.cpu()
    ray_pts = ray_pts.cpu()

    segments, pack_info_segment = segments.jdata.cpu(), pack_info_segment.cpu()

    ps.init()
    ps.set_ground_plane_mode('shadow_only')

    ps.register_point_cloud("points", p, radius=0.0025, point_render_mode='quad')
    ps.register_point_cloud("ray_points", ray_pts, radius=0.0035)

    for i in range(pack_info_segment.shape[0]):
        roi = ray_o[i].unsqueeze(0)  # [1, 3]
        rdi = ray_d[i].unsqueeze(0)  # [1, 3]
        rp = torch.cat([roi + segments[pack_info_segment[i, 0]: pack_info_segment[i, 0] + pack_info_segment[i, 1], 0].unsqueeze(-1) * rdi,
                        roi + segments[pack_info_segment[i, 0]: pack_info_segment[i, 0] + pack_info_segment[i, 1], 1].unsqueeze(-1) * rdi])
        re = torch.stack([torch.arange(pack_info_segment[i, 1]),
                          torch.arange(pack_info_segment[i, 1]) + pack_info_segment[i, 1]], dim=-1)

        ray_segs = ps.register_curve_network(f"ray segments {i}", rp, re, radius=0.00175)
        rv = torch.zeros(re.shape[0])
        rv[::2] = 1.0
        ray_segs.add_scalar_quantity(f"segment colors {i}", rv, defined_on='edges',
                                     enabled=True, cmap='jet')

    # vox_i = torch.cat([vox[i][:counts[i]] for i in range(vox.shape[0])])
    # iv, ie = curve_network_for_voxel_centers(vox_i)
    # iv = fvdb.dual_to_world(iv)
    # ps.register_curve_network("intersected voxels", iv, ie, enabled=True, radius=0.001)

    ps.register_point_cloud("grid corners", gc.jdata, enabled=True, radius=0.001,
                            point_render_mode='quad')
    ps.register_curve_network("grid edges", gc.jdata, ge.jdata, enabled=True, radius=0.00075,
                              transparency=0.7)


    ray_dir_points = torch.cat([ray_o, ray_o + 0.5 * ray_d])
    ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]),
                                 torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
    ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
    ps.register_point_cloud("ray origins", ray_o, radius=0.01)
    ps.show()



if __name__ == "__main__":
    main()
