import logging
import timeit

from fvdb import GridBatch
import torch
import polyscope as ps

from common import load_dragon_mesh, make_ray_grid


def main():
    torch.random.manual_seed(5)
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))

    device = torch.device('cuda')
    dtype = torch.float32
    vox_size =  0.01
    vox_origin = torch.zeros(3).to(device)

    N = 10 # Maximum number of segments to intersect along ray
    nrays = 100
    plot_every = 500

    p, n = load_dragon_mesh(device=device, dtype=dtype)

    p = torch.cat([p,
                   p + torch.tensor([[0.025, 0.025, 0.125]]).to(p),
                   p + torch.tensor([[0.05, 0.05, 0.25]]).to(p)])
    n = torch.cat([n, n, n])


    grid = GridBatch(device=device)
    grid.set_from_points(p, [-1]*3, [1]*3, voxel_sizes=vox_size, origins=vox_origin)

    gc, ge = grid.viz_edge_network
    ray_o, ray_d = make_ray_grid(nrays, [0.0, 0.1, -0.1], device=device, dtype=dtype)

    logging.info(f"Tracing {nrays ** 2} Ray Segments...")
    start = timeit.default_timer()
    pack_info, seg_idx, segments = grid.segments_along_rays(ray_o, ray_d, N, eps=1e-4)
    if p.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s!")

    pmt = torch.randperm(ray_o.shape[0])
    ray_o, ray_d = ray_o[pmt][::plot_every], ray_d[pmt][::plot_every]
    pack_info = pack_info.jdata[pmt][::plot_every]
    segments = segments.jdata

    p, n = p.cpu(), n.cpu()
    gc, ge = gc.jdata.cpu(), ge.jdata.cpu()
    ray_o, ray_d = ray_o.cpu(), ray_d.cpu()
    segments, pack_info = segments.cpu(), pack_info.cpu()

    ps.init()
    ps.set_ground_plane_mode('shadow_only')

    ps.register_point_cloud("points", p, radius=0.0025, point_render_mode='quad')

    for i in range(pack_info.shape[0]):
        roi = ray_o[i].unsqueeze(0)  # [1, 3]
        rdi = ray_d[i].unsqueeze(0)  # [1, 3]
        rp = torch.cat([roi + segments[pack_info[i][0]: pack_info[i][0] + pack_info[i][1], 0].unsqueeze(-1) * rdi,
                        roi + segments[pack_info[i][0]: pack_info[i][0] + pack_info[i][1], 1].unsqueeze(-1) * rdi])
        re = torch.stack([torch.arange(pack_info[i][1]),
                          torch.arange(pack_info[i][1]) + pack_info[i][1]], dim=-1)

        ray_segs = ps.register_curve_network(f"ray segments {i}", rp, re, radius=0.00175)
        rv = torch.zeros(re.shape[0])
        rv[::2] = 1.0
        ray_segs.add_scalar_quantity(f"segment colors {i}", rv, defined_on='edges',
                                     enabled=True, cmap='jet')


    ps.register_point_cloud("grid corners", gc, enabled=True, radius=0.001,
                            point_render_mode='quad')
    ps.register_curve_network("grid edges", gc, ge, enabled=True, radius=0.00075, transparency=0.7)


    ray_dir_points = torch.cat([ray_o, ray_o + 0.5 * ray_d])
    ray_dir_edges = torch.stack([torch.arange(ray_o.shape[0]),
                                 torch.arange(ray_o.shape[0]) + ray_o.shape[0]], dim=-1)
    ps.register_curve_network("ray directions", ray_dir_points, ray_dir_edges, radius=0.0005)
    ps.register_point_cloud("ray origins", ray_o, radius=0.01)
    ps.show()



if __name__ == "__main__":
    main()
