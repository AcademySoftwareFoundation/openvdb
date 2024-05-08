import torch
import fvdb
import fvdb.nn as fvnn
import time
import point_cloud_utils as pcu
import numpy as np
from typing import Optional
from pathlib import Path

# from pycg import exp

from networks.fvnn_net import UNetModel as UNetModelFVDB
from networks.spnn_net import UNetModel as UNetModelSPNN

from torch.profiler import profile, record_function, ProfilerActivity
from fvdb import GridBatch
from fvdb.nn import VDBTensor


def normalize_pts(xyz: np.ndarray):
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_scale = np.max(xyz_max - xyz_min) * 1.2
    return (xyz - xyz_center) / xyz_scale + 0.5


def get_dragon_test_data(batch_size: int = 4, resolution: int = 128):
    dragon_pts = pcu.load_mesh_v("data/dragon.ply")
    voxel_size = 1 / resolution
    dragon_pts = normalize_pts(dragon_pts)
    dragon_pts = torch.from_numpy(dragon_pts).to(device).to(torch.float32)
    grid_list = []
    for _ in range(batch_size):
        grid_list.append(dragon_pts)
    grid = fvdb.sparse_grid_from_points(
            fvdb.JaggedTensor(grid_list), voxel_sizes=voxel_size, origins=[voxel_size / 2.] * 3)
    vdb_tensor = fvnn.VDBTensor(grid, grid.ijk.float())
    return vdb_tensor

def get_av_test_data():
    ijk = np.genfromtxt("../efty-conv/ijk-csv/15265053588821562107_60_000_80_000.zarr_crop002.csv", delimiter=",", dtype=np.int32)
    ijk = torch.from_numpy(ijk).to(device)
    grid = fvdb.sparse_grid_from_ijk(ijk)
    vdb_tensor = fvnn.VDBTensor(grid, grid.ijk.float())
    return vdb_tensor

base_channel = 32

device = "cuda"
torch.cuda.init()

is_fvdb = True
my_model = (UNetModelFVDB if is_fvdb else UNetModelSPNN)(
    in_channels=3,
    model_channels=base_channel,
    num_res_blocks=2,
    channel_mult=(1, 2, 2, 4)
).to(device)
fvnn.SparseConv3d.backend = "halo"

layer_counter = 0

if True:
    old_forward = fvnn.SparseConv3d.forward
    def forward(
        self,
        input: VDBTensor,
        out_grid: Optional[GridBatch] = None,
    ) -> VDBTensor:
        global layer_counter
        before_halo = torch.cuda.Event(enable_timing=True)
        after_halo = torch.cuda.Event(enable_timing=True)

        before_halo.record()
        res = old_forward(self, input, out_grid)
        after_halo.record()

        torch.cuda.synchronize()

        elapsed_time = before_halo.elapsed_time(after_halo)
        print(f"Layer {layer_counter} ({self.in_channels} -> {self.out_channels} K={self.kernel_size}): {elapsed_time:.3f} ms")

        # file_name = f"Layer-{layer_counter}-Input-{self.in_channels}-Output-{self.out_channels}-Kernel-{self.kernel_size}.csv"
        # save_path = Path("../efty-conv/layer-csv/") / file_name
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # np.savetxt(save_path, input.grid.ijk.jdata.cpu().numpy(), delimiter=",", fmt='%i')

        layer_counter += 1
        return res
    fvnn.SparseConv3d.forward = forward


# total param
total_param = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
print(f"Running {'FVDB UNet' if is_fvdb else 'SPNN UNet'} with {total_param / 1e6:.2f}M parameters")

# in_feature = get_dragon_test_data(batch_size=4)
in_feature = get_av_test_data()
timesteps = torch.randint(0, 1000, (in_feature.batch_size,), device=device) # B

if not is_fvdb:
    from torchsparse import SparseTensor
    coords = torch.cat([in_feature.grid.ijk.jdata, in_feature.feature.jidx[:, None]], dim=1)
    in_feature = SparseTensor(in_feature.feature.jdata, coords)

total_time = 0
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
for _ in range(100):
    torch.cuda.synchronize()
    start = time.time()
    # with torch.no_grad():
    # with exp.pt_profile_named("main", "1.json"):
    # with record_function("model_inference"):
    layer_counter = 0
    output = my_model(in_feature, timesteps)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time: {end - start}")
    total_time += (end - start)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(f"Average Time: {total_time / 100}")
