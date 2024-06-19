import torch
import fvdb
from fvdb import JaggedTensor
from fvdb.nn import VDBTensor
from pathlib import Path
import numpy as np
import polyscope as ps
import point_cloud_utils as pcu


voxel_size_1 = 0.02
voxel_size_2 = 0.03


def build_from_pointcloud(pcd_1: np.ndarray, pcd_2: np.ndarray):
    # Assemble point clouds into JaggedTensor
    pcd_jagged = JaggedTensor([
        torch.from_numpy(pcd_1).float().cuda(),
        torch.from_numpy(pcd_2).float().cuda()
    ])
    voxel_sizes = [[voxel_size_1, voxel_size_1, voxel_size_1], [voxel_size_2, voxel_size_2, voxel_size_2]]

    # Method 1:
    grid_a1 = fvdb.sparse_grid_from_points(pcd_jagged, voxel_sizes=voxel_sizes, origins=[0.0] * 3)

    # Method 2:
    grid_a2 = fvdb.GridBatch(device=pcd_jagged.device)
    grid_a2.set_from_points(pcd_jagged, voxel_sizes=voxel_sizes, origins=[0.0] * 3)

    # Visualization
    gv_a1, ge_a1 = grid_a1.viz_edge_network
    ps.remove_all_structures()
    ps.register_point_cloud("pcd_1", pcd_1, enabled=True, radius=0.01)
    ps.register_curve_network(
        "grid_a1", gv_a1[0].jdata.cpu().numpy(), ge_a1[0].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.register_point_cloud("pcd_2", pcd_2, enabled=True, radius=0.01)
    ps.register_curve_network(
        "grid_a2", gv_a1[1].jdata.cpu().numpy(), ge_a1[1].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.show()

    # Build grid from containing nearest voxels to the points
    grid_b = fvdb.sparse_grid_from_nearest_voxels_to_points(pcd_jagged, voxel_sizes=voxel_sizes, origins=[0.0] * 3)

    # Visualization
    gv_b, ge_b = grid_b.viz_edge_network
    ps.remove_all_structures()
    ps.register_point_cloud("pcd_1", pcd_1, enabled=True, radius=0.01)
    ps.register_curve_network(
        "grid_b1", gv_b[0].jdata.cpu().numpy(), ge_b[0].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.register_point_cloud("pcd_2", pcd_2, enabled=True, radius=0.01)
    ps.register_curve_network(
        "grid_b2", gv_b[1].jdata.cpu().numpy(), ge_b[1].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.show()


def build_from_coordinates(coords_1: np.ndarray, coords_2: np.ndarray):
    coords_jagged = JaggedTensor([
        torch.from_numpy(coords_1).long().cuda(),
        torch.from_numpy(coords_2).long().cuda()
    ])
    voxel_sizes = [[voxel_size_1, voxel_size_1, voxel_size_1], [voxel_size_2, voxel_size_2, voxel_size_2]]

    grid = fvdb.sparse_grid_from_ijk(coords_jagged, voxel_sizes=voxel_sizes, origins=[0.0] * 3)

    # Visualization
    grid_mesh_1 = pcu.voxel_grid_geometry(
        grid.ijk[0].jdata.cpu().numpy(), grid.voxel_sizes[0].cpu().numpy(), gap_fraction=0.1)
    grid_mesh_2 = pcu.voxel_grid_geometry(
        grid.ijk[1].jdata.cpu().numpy(), grid.voxel_sizes[1].cpu().numpy(), gap_fraction=0.1)
    ps.remove_all_structures()
    ps.register_surface_mesh("grid_1", grid_mesh_1[0], grid_mesh_1[1], enabled=True)
    ps.register_surface_mesh("grid_2", grid_mesh_2[0], grid_mesh_2[1], enabled=True)
    ps.show()


def build_from_mesh(mesh_1_vf, mesh_2_vf):
    mesh_1_v, mesh_1_f = mesh_1_vf
    mesh_2_v, mesh_2_f = mesh_2_vf

    mesh_v_jagged = JaggedTensor([
        torch.from_numpy(mesh_1_v).float().cuda(),
        torch.from_numpy(mesh_2_v).float().cuda()
    ])
    mesh_f_jagged = JaggedTensor([
        torch.from_numpy(mesh_1_f.astype(np.int64)).long().cuda(),
        torch.from_numpy(mesh_2_f.astype(np.int64)).long().cuda()
    ])

    voxel_sizes = [[voxel_size_1, voxel_size_1, voxel_size_1], [voxel_size_2, voxel_size_2, voxel_size_2]]
    grid = fvdb.sparse_grid_from_mesh(mesh_v_jagged, mesh_f_jagged, voxel_sizes=voxel_sizes, origins=[0.0] * 3)

    # Visualization
    gv, ge = grid.viz_edge_network
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh_1", mesh_1_v, mesh_1_f, enabled=True)
    ps.register_curve_network(
        "grid_1", gv[0].jdata.cpu().numpy(), ge[0].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.register_surface_mesh("mesh_2", mesh_2_v, mesh_2_f, enabled=True)
    ps.register_curve_network(
        "grid_2", gv[1].jdata.cpu().numpy(), ge[1].jdata.cpu().numpy(), enabled=True, radius=0.004)
    ps.show()


def build_from_dense():
    grid = fvdb.sparse_grid_from_dense(num_grids=1, dense_dims=[32, 32, 32], device="cuda")

    # Easy way to initialize a VDBTensor from a torch 3D tensor [B, D, H, W, C]
    dense_data = torch.ones(2, 32, 32, 32, 16).cuda()
    sparse_data = VDBTensor.from_dense(dense_data, voxel_sizes=[0.1] * 3)
    dense_data_back = sparse_data.to_dense()
    assert torch.all(dense_data == dense_data_back)

    # Visualization
    grid_mesh = pcu.voxel_grid_geometry(
        grid.ijk[0].jdata.cpu().numpy(), grid.voxel_sizes[0].cpu().numpy(), gap_fraction=0.1)
    ps.remove_all_structures()
    ps.register_surface_mesh("grid_1", grid_mesh[0], grid_mesh[1], enabled=True)
    ps.show()


if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    base_path = Path(__file__).parent.parent

    mesh_1_v, mesh_1_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-1.ply"))
    mesh_2_v, mesh_2_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-2.ply"))
    mesh_2_v[:, 2] += 0.8

    fi1, bc1 = pcu.sample_mesh_random(mesh_1_v, mesh_1_f, 10000)
    fi2, bc2 = pcu.sample_mesh_random(mesh_2_v, mesh_2_f, 10000)

    pcd_1 = pcu.interpolate_barycentric_coords(mesh_1_f, fi1, bc1, mesh_1_v)
    pcd_2 = pcu.interpolate_barycentric_coords(mesh_2_f, fi2, bc2, mesh_2_v)

    ijk_1 = np.unique(np.floor(pcd_1 / voxel_size_1).astype(np.int64), axis=0)
    ijk_2 = np.unique(np.floor(pcd_2 / voxel_size_2).astype(np.int64), axis=0)

    build_from_pointcloud(pcd_1, pcd_2)
    build_from_mesh((mesh_1_v, mesh_1_f), (mesh_2_v, mesh_2_f))
    build_from_coordinates(ijk_1, ijk_2)
    build_from_dense()
