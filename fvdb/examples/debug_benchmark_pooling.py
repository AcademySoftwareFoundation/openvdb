from fvdb import SparseIndexGrid
import torch
import time
import tqdm

mutable = False
device = 'cuda'
dtype = torch.float32
vox_size = 0.005
vox_origin = (0.0, 0.0, 0.0)
num_warmup = 3
num_benchmark = 10

print("Building grid...")
torch.manual_seed(0)
pts = torch.randn(1000000, 3).to(device).to(dtype)

grid = SparseIndexGrid(vox_size, vox_origin, mutable=mutable, device=device)
grid.build_from_pointcloud(pts, [-1, -1, -1], [1, 1, 1])
print(f"Build grid with {grid.num_voxels} voxels")

for pool_factor in (1, 2, 3, 4, 5, 7, 15, 10):

    grid_vals = torch.randn(grid.num_voxels, 4).to(device).to(dtype)

    print("Warmup...")
    # warmup
    for _ in tqdm.trange(num_warmup):
        grid_vals_coarse, grid_coarse = grid.max_pool(grid_vals, pool_factor)

    print("Benchmarking...")
    start = time.time()
    for _ in tqdm.trange(num_benchmark):
        grid_vals_coarse, grid_coarse = grid.max_pool(grid_vals, pool_factor)
    torch.cuda.synchronize()
    end = time.time()
    tot_time = end - start
    print(f"Voxels = {grid.num_voxels}, Pool factor: {pool_factor}, time: {tot_time}s, time_per = {tot_time / num_benchmark}s")
