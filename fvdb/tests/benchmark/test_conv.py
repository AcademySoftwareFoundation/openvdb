import torch
import fvdb
import fvdb.nn as fvdbnn
import random

import pytest
import logging


torch.backends.cudnn.deterministic = True

PTS_CACHE = [torch.empty((10_000, 3), dtype=torch.float32).normal_() for _ in range(100)]

@pytest.mark.parametrize("i_ch", [3, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("o_ch", [3, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("backend", ['default', 'cutlass', 'me', 'halo', 'igemm_mode0', 'igemm_mode1', 'igemm_mode2'])
@pytest.mark.benchmark(group="sparse_conv3d",
                       warmup=True,
                       warmup_iterations=3,
 )
def test_forward_conv3d(benchmark, i_ch, o_ch, backend):
    device = torch.device('cuda')
    pts = random.choice(PTS_CACHE).to(device=device) * 4

    coords = torch.floor(pts / 0.01).to(torch.int32)
    grid = fvdb.sparse_grid_from_ijk(coords).to(device)

    feature = torch.empty(grid.total_voxels, i_ch, dtype=torch.float32, device=device).random_()

    example_inputs = fvdbnn.VDBTensor(grid, fvdb.JaggedTensor([feature]))

    model = fvdbnn.SparseConv3d(in_channels=i_ch, out_channels=o_ch).to(device)
    model.backend = backend

    model.eval()

    def run_model():
        model(example_inputs)

    # benchmark(run_model)
    benchmark.pedantic(run_model, iterations=10, rounds = 20)