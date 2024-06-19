import fvdb
import numpy as np
import polyscope as ps
import torch

def make_ray_grid(nrays, minb=(-0.3, -0.3), maxb=(0.3, 0.3), device='cpu', dtype=torch.float32):
    ray_o = torch.tensor([[0.0, 0.0, -1.2]] * (nrays ** 2))
    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays * 1j,
                           minb[1]:maxb[1]:nrays * 1j]] +
                  [np.ones(nrays ** 2)], axis=-1).astype(np.float32)
    )
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device=device, dtype=dtype), ray_d.to(device=device, dtype=dtype)

    return ray_o, ray_d


def main():
    voxelgrid_size = 1.
    device = 'cuda'
    dim = 32

    grid = fvdb.sparse_grid_from_dense(
        num_grids=1,
        dense_dims=[dim] * 3,
        origins=[-voxelgrid_size / 2] * 3,
        voxel_sizes=voxelgrid_size / dim,
        device=device
    )
    ijk = grid.ijk.jdata.float()

    coords = grid.grid_to_world(grid.ijk.jdata.float()).jdata
    sdf = 0.4 - torch.norm(coords * 2, dim=-1)

    # Truncation regions.
    sdf[sdf < -0.1] = torch.nan

    ray_o, ray_d = make_ray_grid(10, device=device)

    isect_times = grid.ray_implicit_intersection(ray_o, ray_d, sdf).jdata
    mask = isect_times > 0

    ray_hit = ray_o[mask] + isect_times[mask, None] * ray_d[mask]


    ps.init()
    pc = ps.register_point_cloud("grid coords", coords.cpu())
    pc.add_scalar_quantity("sdf", sdf.cpu(), enabled=True)

    ray_dir_points = torch.cat([ray_o, ray_o + 0.5 * ray_d])
    ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]),
                                 torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
    ps.register_curve_network("ray directions", ray_dir_points.cpu(), ray_dir_edges.cpu(), radius=0.0005)

    pc = ps.register_point_cloud("intersections", ray_hit.cpu())
    pc.add_scalar_quantity("times", isect_times[mask].cpu())

    ps.show()

if __name__ == "__main__":
    main()
