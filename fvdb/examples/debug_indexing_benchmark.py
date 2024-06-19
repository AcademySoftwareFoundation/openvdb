import argparse
import math
import time
import torch


from fvdb import SparseIndexGrid, sparse_grid_from_dense_cube, sparse_grid_from_ijk_coords


def parse_args():
    parser = argparse.ArgumentParser(
        prog='fVDB vs MinkowskiEngine',
        description='Compare fVDB convolution to MinkowskiEngine')
    parser.add_argument('--load-points', type=str, default="")
    parser.add_argument('--num-points', type=int, default=10000,
                        help='num of points to sample to generate the grids')
    parser.add_argument('--vox-size', type=float, default=0.0005,
                        help='size of voxels influence the resolution')
    parser.add_argument('--num-indices', type=int, default=100000,
                        help='num of indices')
    parser.add_argument('--in-channels', type=int, default=2,
                        help='input channels')

    return parser.parse_args()

def main(args):
    device = 'cuda'
    dtype = torch.float

    num_points = args.num_points
    in_channels = args.in_channels

    if args.load_points:
        p = torch.load(args.load_points).to(device=device, dtype=dtype)
    else:
        p = torch.clamp(torch.rand((args.num_points, 3), device=device, dtype=dtype), min=1e-6, max=1.-1e-6)
        # torch.save(p, "pts.pth")
    vox_size = args.vox_size
    grid = SparseIndexGrid(vox_size, [0.0] * 3, mutable=False, device=device)
    grid.build_from_pointcloud(p)
    print(f"Grid has {grid.num_voxels} voxels")
    fvdb_features = torch.rand((grid.num_enabled_voxels, in_channels), device=device, dtype=dtype)
    in_coords = grid.active_grid_coords()

    indices = grid.active_grid_coords()[
        torch.randint(0, grid.num_enabled_voxels, (args.num_indices,),
                      device='cuda', dtype=torch.int)
    ]

    for i in range(10):
        fvdb_coords = grid.ijk_to_index(indices)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(1000):
        fvdb_coords = grid.ijk_to_index(indices)
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"fvdb: total = {total_time}, per = {total_time / 1000}")


if __name__ == "__main__":
    main(parse_args())
