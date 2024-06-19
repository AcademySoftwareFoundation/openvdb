import time
import torch
import torch.nn as nn
from fvdb import sparse_grid_from_points, JaggedTensor
from fvdb.nn import VDBTensor


class CUDATorchGroupNorm(nn.GroupNorm):
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.feature.jdata.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"

        num_batches = input.grid.grid_count

        all_feats = []
        for b in range(num_batches):
            feat = input.feature[b].jdata
            if feat.size(0) == 0:
                all_feats.append(feat)
            else:
                feat = feat.transpose(0, 1).reshape(1, num_channels, -1)
                feat = super().forward(feat)
                feat = feat.reshape(num_channels, -1).transpose(0, 1).contiguous()
                all_feats.append(feat)

        return VDBTensor(input.grid, JaggedTensor(all_feats))


class NativeTorchGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.empty(num_channels))
            self.bias = nn.Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

    def _forward_single_batch(self, x: torch.Tensor) -> torch.Tensor:
        num_voxels, _ = x.size()
        x = x.transpose(0, 1).reshape(self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.reshape(self.num_channels, num_voxels).transpose(0, 1).contiguous()

        if self.affine:
            x = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        return x

    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.feature.jdata.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"

        num_batches = input.grid.grid_count

        all_feats = []
        for b in range(num_batches):
            feat = input.feature[b].jdata
            # detect if feat is empty
            if feat.size(0) == 0:
                all_feats.append(feat)
                continue

            feat = self._forward_single_batch(feat)
            all_feats.append(feat)

        out_feature = JaggedTensor(all_feats)
        return VDBTensor(input.grid, out_feature)


def print_memory():
    print("Allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1024 / 1024 / 1024),
          "Max allocated: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))


def get_time():
    torch.cuda.synchronize()
    return time.time()


if __name__ == "__main__":
    pts = JaggedTensor([torch.randn(10000000, 3, device="cuda") for _ in range(4)])
    grid = sparse_grid_from_points(pts, voxel_sizes=0.01)
    in_feat = grid.jagged_like(torch.randn(grid.total_voxels, 16, device="cuda", requires_grad=True))
    in_vdb_tensor = VDBTensor(grid, in_feat)

    native_gn = NativeTorchGroupNorm(4, 16).cuda()
    torch_gn = CUDATorchGroupNorm(4, 16).cuda()

    print("--- Native Torch ---")
    torch.cuda.reset_peak_memory_stats()
    print_memory()
    time0 = get_time()
    native_out = native_gn(in_vdb_tensor)
    print("Time: %.2f" % (get_time() - time0))
    print_memory()
    time1 = get_time()
    native_out.feature.jdata.sum().backward()
    print("Time: %.2f" % (get_time() - time1))
    print_memory()
    del native_out
    in_vdb_tensor.feature.jdata.grad = None

    print("--- CUDA Torch ---")
    torch.cuda.reset_peak_memory_stats()
    print_memory()
    time0 = get_time()
    torch_out = torch_gn(in_vdb_tensor)
    print("Time: %.2f" % (get_time() - time0))
    print_memory()
    time1 = get_time()
    torch_out.feature.jdata.sum().backward()
    print("Time: %.2f" % (get_time() - time1))
    print_memory()
    del torch_out
    in_vdb_tensor.feature.jdata.grad = None
