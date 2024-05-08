import os
import time
import numpy as np

import point_cloud_utils as pcu
import torch
import tqdm

from fvdb import SparseIndexGrid


def benchmark_inplace_conv(grid, in_feature, in_kernel):
    start_time = time.perf_counter()
    out_feature = grid.sparse_conv_inplace(in_feature, in_kernel)
    torch.cuda.synchronize()
    return time.perf_counter() - start_time


def benchmark_kmap_conv(grid, in_feature, in_kernel):
    start_time = time.perf_counter()
    kmap, _ = grid.convolution_kernel_map(kernel_size=in_kernel.size(-1), stride=1)
    torch.cuda.synchronize()

    kmap_time = time.perf_counter()
    out_feature, _, _ = grid.sparse_conv(in_feature, in_kernel, kmap=kmap, target_grid=grid)
    torch.cuda.synchronize()

    return kmap_time - start_time, time.perf_counter() - kmap_time


def main():
    device = 'cuda'
    dtype = torch.float32
    kernel_size = 3
    in_channel, out_channel = 128, 64

    vox_size = 0.005
    vox_origin = (0.0, 0.0, 0.0)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")

    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts).to(device).to(dtype), \
        torch.from_numpy(nms).to(device).to(dtype)

    index0 = SparseIndexGrid(vox_size, vox_origin, device=device)
    index0.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])

    grid_feats = torch.rand((index0.num_voxels, in_channel), device=device, dtype=dtype) * 0.5 + 0.5
    kernels = torch.rand(out_channel, in_channel, kernel_size, kernel_size, kernel_size,
                         dtype=dtype, device=device) * 0.5 + 0.5

    torch.cuda.synchronize()

    inplace_time = []
    kmap_time = []
    conv_time = []

    for iter in tqdm.trange(100):
        inplace = benchmark_inplace_conv(index0, grid_feats, kernels)
        kmap, conv = benchmark_kmap_conv(index0, grid_feats, kernels)
        inplace_time.append(inplace)
        kmap_time.append(kmap)
        conv_time.append(conv)

    inplace_time, kmap_time, conv_time = inplace_time[5:], kmap_time[5:], conv_time[5:]

    print(f"Num voxels = {index0.num_voxels}, channel = {in_channel} -> {out_channel}, device = {device}")
    print(f"Convolution Inplace {np.mean(inplace_time):.4f} +/- {np.std(inplace_time):.4f}")
    print(f"Kmap {np.mean(kmap_time):.4f} +/- {np.std(kmap_time):.4f}")
    print(f"Kmap Convolution {np.mean(conv_time):.4f} +/- {np.std(conv_time):.4f}")


if __name__ == "__main__":
    main()
