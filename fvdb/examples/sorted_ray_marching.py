import fvdb
import torch
import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
from pathlib import Path
from fvdb import JaggedTensor, GridBatch


def draw_curve_network(gc: JaggedTensor, ge: JaggedTensor):
    for b in range(gc.joffsets.shape[0]):
        ps.register_curve_network(
            f"grid_{b}", gc[b].jdata.cpu().numpy(), ge[b].jdata.cpu().numpy(), enabled=True, radius=0.001
        )


def draw_ray_segments(ray_o, ray_d, times, pack_info):
    for ray_idx, (ray_start, ray_count) in enumerate(pack_info):
        t0s = times[ray_start:ray_start + ray_count][:, 0].unsqueeze(-1)
        t1s = times[ray_start:ray_start + ray_count][:, 1].unsqueeze(-1)
        roi = ray_o[ray_idx].unsqueeze(0)
        rdi = ray_d[ray_idx].unsqueeze(0)
        rp = torch.cat([roi + t0s * rdi, roi + t1s * rdi])
        re = torch.stack([torch.arange(t0s.shape[0]),
                          torch.arange(t0s.shape[0]) + t0s.shape[0]], dim=-1)

        rp = rp.cpu().numpy()
        re = re.cpu().numpy()

        ray_segs = ps.register_curve_network(f"ray segments {ray_idx}", rp, re, radius=0.005)
        rv = torch.linspace(0., 1., re.shape[0])
        ray_segs.add_scalar_quantity(f"segment", rv, defined_on='edges',
                                     enabled=True, cmap='jet')


def jidx_to_packinfo(jidx):
    # First convert from jidx to joffsets
    joffsets = JaggedTensor.from_data_and_jidx(torch.empty(jidx.shape[0], 0), jidx, jidx[-1].item() + 1).joffsets
    # Then convert from joffsets to packinfo
    return torch.stack([joffsets[:, 0], joffsets[:, 1] - joffsets[:, 0]], dim=-1)


def packinfo_to_jidx(packinfo):
    # First convert from packinfo to joffsets
    joffsets = torch.stack([packinfo[:, 0], packinfo[:, 0] + packinfo[:, 1]], dim=-1)
    # Then convert from joffsets to jidx
    return JaggedTensor.from_data_and_offsets(torch.empty(joffsets[-1, 1], 0), joffsets).jidx.long()


if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    device = "cpu"

    base_path = Path(__file__).parent.parent

    mesh_0_v, mesh_0_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-1.ply"))
    mesh_0_v[:, 2] += 0.48
    mesh_1_v, mesh_1_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-1.ply"))
    mesh_1_v[:, 2] += 0.24
    mesh_2_v, mesh_2_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-2.ply"))

    mesh_v_jagged = JaggedTensor([
        torch.from_numpy(mesh_0_v).float().to(device),
        torch.from_numpy(mesh_1_v).float().to(device),
        torch.from_numpy(mesh_2_v).float().to(device)
    ])
    mesh_f_jagged = JaggedTensor([
        torch.from_numpy(mesh_0_f.astype(np.int64)).long().to(device),
        torch.from_numpy(mesh_1_f.astype(np.int64)).long().to(device),
        torch.from_numpy(mesh_2_f.astype(np.int64)).long().to(device)
    ])

    # Grid creation
    grid = fvdb.sparse_grid_from_mesh(
        mesh_v_jagged, mesh_f_jagged, voxel_sizes=[0.05] * 3, origins=[0.0] * 3, mutable=True)

    # Ray creation
    ray_single = 2
    num_rays = ray_single ** 2
    ray_d = torch.meshgrid(
        torch.linspace(-1, 1, ray_single),
        torch.linspace(-1, 1, ray_single)
    )
    ray_d = torch.stack(list(ray_d) + [torch.ones_like(ray_d[0]) * 20], dim=-1).reshape(-1, 3)
    ray_d /= ray_d.norm(dim=-1, keepdim=True)
    ray_o = torch.zeros_like(ray_d)
    ray_o[:, 2] -= 1.0
    ray_o, ray_d = ray_o.to(device), ray_d.to(device)

    ray_o_double = JaggedTensor([ray_o] * grid.grid_count)
    ray_d_double = JaggedTensor([ray_d] * grid.grid_count)
    ray_inds = JaggedTensor([torch.arange(ray_o.shape[0], device=device)] * grid.grid_count)

    # Ray marching
    pack_info, _, times = grid.voxels_along_rays(ray_o_double, ray_d_double, 100, 0.0)
    ps.remove_all_structures()
    draw_curve_network(*grid.viz_edge_network)
    draw_ray_segments(ray_o_double.jdata, ray_d_double.jdata, times.jdata, pack_info.jdata)
    ps.show()

    # Expand pack info into jagged indices, and obtain ray_inds
    expand_inds = packinfo_to_jidx(pack_info.jdata)
    ray_inds = ray_inds.jdata[expand_inds]
    argsort_ray_inds = torch.argsort(ray_inds)
    ray_inds, times = ray_inds[argsort_ray_inds], times.jdata[argsort_ray_inds]

    # Sort times based on idx
    argsort_shuffle_idx = JaggedTensor.from_data_and_jidx(times.mean(1), ray_inds, ray_o.shape[0]).jagged_argsort()
    ps.remove_all_structures()
    draw_curve_network(*grid.viz_edge_network)
    draw_ray_segments(ray_o, ray_d, times[argsort_shuffle_idx], jidx_to_packinfo(ray_inds))
    ps.show()
