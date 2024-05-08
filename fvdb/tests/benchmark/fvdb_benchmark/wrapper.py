from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import record_function

import torchsparse
import torchsparse.nn as spnn
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse import SparseTensor
from torchsparse.nn import functional as spF

import fvdb
import fvdb.nn as fvnn

from fvdb_benchmark.utils import encode_range_name


class TorchsparseMaxPool(torch.nn.Module):
    """Max pooling layer for torchsparse"""

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        out_coords = spF.spdownsample(
            input.coords, stride=2, kernel_size=2, downsample_mode="minkowski")
        offsets = get_kernel_offsets(2, 1, 1, device=input.feats.device)

        references = spF.sphash(input.coords)
        queries = spF.sphash(out_coords * 2, offsets)
        results = spF.sphashquery(queries, references)

        in_feature = torch.cat([
            torch.full((1, input.feats.size(1)), fill_value=-1000.0,
                    device=input.feats.device, dtype=input.feats.dtype), input.feats
        ], dim=0)
        out_feature = torch.max(in_feature[results + 1], dim=0)[0]       # [8, N, C] -> [N, C]

        output = SparseTensor(feats=out_feature, coords=out_coords,
                            stride=tuple(input.stride[k] * 2 for k in range(3)))
        output._caches = input._caches
        output._caches.cmaps.setdefault(output.stride, output.coords)

        return output


class TorchsparseUpsampleGenerative(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input, mask):
        in_stride = input.stride
        out_stride = tuple([in_stride[k] // 2 for k in range(3)])

        in_coords = input.coords[mask] * 2
        out_coords = []
        out_feature = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    out_coords.append(in_coords + torch.tensor(
                        [[0, i, j, k]], device=in_coords.device, dtype=in_coords.dtype))
                    out_feature.append(input.feats[mask])
        out_coords = torch.cat(out_coords, dim=0)
        out_feature = torch.cat(out_feature, dim=0)

        output = SparseTensor(feats=out_feature, coords=out_coords, stride=out_stride)
        return output


class TorchsparseUpsample(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        in_stride = input.stride
        out_stride = tuple([in_stride[k] // 2 for k in range(3)])

        in_coords = input.coords
        out_coords = input._caches.cmaps[out_stride]
        if isinstance(out_coords, tuple):
            out_coords = out_coords[0]

        in_feature = input.feats
        out_in_pos = torch.div(out_coords[:, 1:4], 2, rounding_mode='floor')
        out_in_coord = torch.cat([out_coords[:, :1], out_in_pos], dim=1).int()

        queries = spF.sphash(out_in_coord)
        references = spF.sphash(in_coords)
        results = spF.sphashquery(queries, references)
        assert torch.all(results >= 0)
        out_feature = in_feature[results]

        output = SparseTensor(feats=out_feature, coords=out_coords, stride=out_stride)
        output._caches = input._caches
        return output


class Wrapper:
    """Wrapper for different backends with profiling support."""

    @classmethod
    def record_runtime(cls, module, name: str):
        # Register class as a module in fvdb.nn
        old_forward = module.forward
        def _forward(self, *args, **kwargs):
            with record_function(name):
                return old_forward(self, *args, **kwargs)
        module.forward = _forward
        return module

    def __init__(self, backend: str = 'ts') -> None:
        assert backend in ['ts', 'fvdb']
        self.backend = backend

    def conv3d(self,
               name: str,
               in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               dilation: int = 1,
               padding: int = 0,
               bias: bool = False,
               transpose: bool = False):
        if self.backend == 'ts':
            return self.record_runtime(spnn.Conv3d(
                    in_channels, out_channels, kernel_size, stride,
                    padding, dilation, bias, transpose
                ),
                encode_range_name(name, 'ts', {"I": in_channels, "O": out_channels})
            )

        elif self.backend == 'fvdb':
            assert dilation == 1, "fvdb does not support dilation"
            assert padding == 0, "fvdb does not support padding"
            return self.record_runtime(fvnn.SparseConv3d(
                    in_channels, out_channels, kernel_size, stride, bias, transpose
                ),
                encode_range_name(name, 'fvdb', {"I": in_channels, "O": out_channels})
            )

        else:
            raise NotImplementedError

    def bn(self, num_features: int,
           *,
           eps: float = 1e-5,
           momentum: float = 0.1):
        if self.backend == 'ts':
            return self.record_runtime(spnn.BatchNorm(
                    num_features, eps, momentum
                ),
                encode_range_name('bn', 'ts', {"F": num_features})
            )

        elif self.backend == 'fvdb':
            return self.record_runtime(fvnn.BatchNorm(
                    num_features, eps, momentum
                ),
                encode_range_name('bn', 'fvdb', {"F": num_features})
            )

        else:
            raise NotImplementedError

    def relu(self, inplace: bool = True):
        if self.backend == 'ts':
            return self.record_runtime(spnn.ReLU(inplace),
                encode_range_name('relu', 'ts', {})
            )

        elif self.backend == 'fvdb':
            return self.record_runtime(fvnn.ReLU(inplace),
                encode_range_name('relu', 'fvdb', {})
            )

        else:
            raise NotImplementedError

    def cat(self, *args):
        range_name = encode_range_name('cat', self.backend, {})
        with record_function(range_name):
            if self.backend == 'ts':
                return torchsparse.cat(*args)

            elif self.backend == 'fvdb':
                return fvnn.cat(*args, dim=1)

            else:
                raise NotImplementedError

    def maxpool(self, name: str, stride: int):
        if self.backend == 'ts':
            return self.record_runtime(TorchsparseMaxPool(stride),
                encode_range_name(name, 'ts', {"S": stride})
            )

        elif self.backend == 'fvdb':
            return self.record_runtime(fvnn.MaxPool(stride),
                encode_range_name(name, 'fvdb', {"S": stride})
            )

        else:
            raise NotImplementedError

    def upsample_generative(self, name: str, scale_factor: int):
        if self.backend == 'ts':
            assert scale_factor == 2, "Only support scale factor 2"
            return self.record_runtime(TorchsparseUpsampleGenerative(),
                encode_range_name(name, 'ts', {"S": scale_factor})
            )

        elif self.backend == 'fvdb':
            return self.record_runtime(fvnn.UpsamplingNearest(scale_factor),
                encode_range_name(name, 'fvdb', {"S": scale_factor})
            )

        else:
            raise NotImplementedError

    def upsample(self, name: str, scale_factor: int):
        if self.backend == 'ts':
            assert scale_factor == 2, "Only support scale factor 2"
            return self.record_runtime(TorchsparseUpsample(),
                encode_range_name(name, 'ts', {"S": scale_factor})
            )

        elif self.backend == 'fvdb':
            return self.record_runtime(fvnn.UpsamplingNearest(scale_factor),
                encode_range_name(name, 'fvdb', {"S": scale_factor})
            )

        else:
            raise NotImplementedError

    def get_coords(self, x):
        if self.backend == 'ts':
            return x.coords[:, 1:4]

        elif self.backend == 'fvdb':
            return x.grid.ijk.jdata

        else:
            raise NotImplementedError

    def get_mask(self, x, coords: fvdb.GridBatch):
        assert isinstance(coords, fvdb.GridBatch)
        cur_coords = self.get_coords(x)
        return coords.ijk_to_index(cur_coords).jdata != -1

    def sequential(self, *args):
        if self.backend == 'ts':
            return nn.Sequential(*args)

        elif self.backend == 'fvdb':

            class SequentialFVDB(nn.Sequential):
                def forward(self, input, ref_fine_data: Optional[fvnn.VDBTensor] = None):
                    for module in self:
                        if module._get_name() == "UpsamplingNearest":
                            input = module(input, ref_fine_data=ref_fine_data)
                        else:
                            input = module(input)
                    return input

            return SequentialFVDB(*args)

        else:
            raise NotImplementedError
