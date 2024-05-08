import fvdb
import torch
import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
from pathlib import Path
from fvdb import JaggedTensor, GridBatch


def visualize_grid_color(grid: GridBatch, rgb: JaggedTensor, ignore_disabled: bool = False):
    for b in range(grid.grid_count):
        grid_mask = grid.enabled_mask[b].jdata.cpu().numpy()
        if ignore_disabled:
            grid_mask.fill(True)

        grid_mesh = pcu.voxel_grid_geometry(
            grid.ijk[b].jdata.cpu().numpy()[grid_mask],
            grid.voxel_sizes[b].cpu().numpy(), gap_fraction=0.1)
        grid_color = rgb[b].jdata.cpu().numpy()[grid_mask].repeat(8, axis=0).reshape(-1, 3)

        ps.register_surface_mesh(
            f"grid_{b}", grid_mesh[0], grid_mesh[1], enabled=True
        ).add_color_quantity("color", grid_color, enabled=True)


if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    base_path = Path(__file__).parent.parent

    mesh_1_v, mesh_1_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-1.ply"))
    mesh_2_v, mesh_2_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-2.ply"))
    mesh_2_v[:, 2] += 0.8

    mesh_v_jagged = JaggedTensor([
        torch.from_numpy(mesh_1_v).float().cuda(),
        torch.from_numpy(mesh_2_v).float().cuda()
    ])
    mesh_f_jagged = JaggedTensor([
        torch.from_numpy(mesh_1_f.astype(np.int64)).long().cuda(),
        torch.from_numpy(mesh_2_f.astype(np.int64)).long().cuda()
    ])

    fi1, bc1 = pcu.sample_mesh_random(mesh_1_v, mesh_1_f, 10000)
    fi2, bc2 = pcu.sample_mesh_random(mesh_2_v, mesh_2_f, 10000)

    pcd_1 = pcu.interpolate_barycentric_coords(mesh_1_f, fi1, bc1, mesh_1_v)
    pcd_2 = pcu.interpolate_barycentric_coords(mesh_2_f, fi2, bc2, mesh_2_v)
    pcd_jagged = JaggedTensor([
        torch.from_numpy(pcd_1).float().cuda(),
        torch.from_numpy(pcd_2).float().cuda()
    ])

    # Grid creation
    grid = fvdb.sparse_grid_from_mesh(
        mesh_v_jagged, mesh_f_jagged, voxel_sizes=[0.01] * 3, origins=[0.0] * 3, mutable=True)
    feature = grid.grid_to_world(grid.ijk.float())
    feature.jdata = (feature.jdata - feature.jdata.min(dim=0).values) / \
        (feature.jdata.max(dim=0).values - feature.jdata.min(dim=0).values)

    # Visualization
    ps.remove_all_structures()
    visualize_grid_color(grid, feature)
    ps.show()

    # Get the IJK coordinates to be disabled
    disable_ijk = grid.ijk.r_masked_select(feature.jdata[:, 0] > 0.5)
    grid.disable_ijk(disable_ijk)

    # Visualize disable mask
    enabled_mask = grid.enabled_mask
    ps.remove_all_structures()
    visualize_grid_color(grid, feature.jagged_like(enabled_mask.jdata.unsqueeze(1).repeat(1, 3).float()),
                         ignore_disabled=True)
    ps.show()

    # Sample features onto points
    pts_feature = grid.sample_trilinear(pcd_jagged, feature)

    # Visualize (disabled grid will no longer function)
    ps.remove_all_structures()
    ps.register_point_cloud("pcd_1", pcd_1, enabled=True).add_color_quantity(
        "feature", pts_feature[0].jdata.cpu().numpy(), enabled=True)
    ps.register_point_cloud("pcd_2", pcd_2, enabled=True).add_color_quantity(
        "feature", pts_feature[1].jdata.cpu().numpy(), enabled=True)
    ps.show()

    # We could enable those IJK back
    grid.enable_ijk(disable_ijk)

    # Sample features onto points
    pts_feature = grid.sample_trilinear(pcd_jagged, feature)

    # Visualize (this time we got the original features back)
    ps.remove_all_structures()
    ps.register_point_cloud("pcd_1", pcd_1, enabled=True).add_color_quantity(
        "feature", pts_feature[0].jdata.cpu().numpy(), enabled=True)
    ps.register_point_cloud("pcd_2", pcd_2, enabled=True).add_color_quantity(
        "feature", pts_feature[1].jdata.cpu().numpy(), enabled=True)
    ps.show()
