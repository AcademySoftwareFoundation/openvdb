import polyscope as ps
import torch
from fvdb import SparseIndexGrid

def main():
    device = 'cuda'
    dtype = torch.float32

    torch.manual_seed(12345678)

    p = torch.rand(777, 3).to(dtype).to(device)

    grid = SparseIndexGrid(0.025, torch.zeros(3), mutable=True, device=device)
    grid.build_from_pointcloud(p, [-1]*3, [1]*3)

    grid2 = SparseIndexGrid(0.025, torch.zeros(3), mutable=True, device=device)
    grid2.build_from_pointcloud(p, [-1]*3, [1]*3)

    primal_ijk = grid.active_grid_coords()
    disable_mask = torch.rand(primal_ijk.shape[0]).to(p) > 0.3
    disable_ijk = primal_ijk[disable_mask]
    grid2.disable_voxels(disable_ijk)

    gv, ge = grid.grid_edge_network()

    ps.init()
    ps.register_point_cloud("grid centers", grid.grid_to_world(primal_ijk.to(dtype)).cpu(), enabled=True)
    ps.register_point_cloud("grid corners", gv.cpu(), enabled=True, radius=0.0025)
    ps.register_curve_network("grid edges", gv.cpu(), ge.cpu(), enabled=True, radius=0.001)

    gv2, ge2 = grid2.grid_edge_network()
    primal_ijk2 = grid2.active_grid_coords()
    ps.register_point_cloud("grid centers 2", grid2.grid_to_world(primal_ijk2.to(dtype)).cpu(), enabled=True)
    ps.register_point_cloud("grid corners 2", gv2.cpu(), enabled=True, radius=0.0025)
    ps.register_curve_network("grid edges 2", gv2.cpu(), ge2.cpu(), enabled=True, radius=0.001)
    ps.show()



if __name__ == "__main__":
    main()
