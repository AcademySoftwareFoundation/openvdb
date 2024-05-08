from typing import Optional, Any, Union, List
from dataclasses import dataclass

import torch

import fvdb
from fvdb import GridBatch, SparseConvPackInfo, JaggedTensor


@dataclass
class VDBTensor:
    """
    A VDBTensor is a thin wrapper around a GridBatch and its corresponding feature JaggedTensor, conceptually denoting a batch of
    sparse tensors along with its topology.
    It works as the input and output arguments of fvdb's neural network layers.
    One can simply construct a VDBTensor from a GridBatch and a JaggedTensor, or from a dense tensor using from_dense().
    """

    grid: GridBatch
    feature: JaggedTensor

    # Only stores the kernel map that operates on this grid, reasons being:
    #   1) A usual network seldom re-uses computation for down-up-sampling. This saves memory.
    #   2) This keeps the implementation simple and the kmap transparent.
    kmap: Optional[SparseConvPackInfo] = None

    def __post_init__(self):
        if not isinstance(self.grid, GridBatch):
            raise TypeError("grid should be of type GridBatch")
        if not isinstance(self.feature, JaggedTensor):
            raise TypeError("feature should be of type JaggedTensor")
        if self.grid.grid_count != len(self.feature.joffsets):
            raise ValueError("grid and feature should have the same batch size")
        if self.grid.total_voxels != self.feature.jdata.size(0):
            raise ValueError("grid and feature should have the same total voxel count")
        if self.kmap is not None:
            if not (self.same_grid(self.kmap.source_grid, self.grid) and
                    self.same_grid(self.kmap.target_grid, self.grid) and
                    self.kmap.stride == (1, 1, 1)):
                raise ValueError("kmap should operate on the same grid as this tensor")

    @staticmethod
    def same_grid(grid_a: GridBatch, grid_b: GridBatch):
        return grid_a.address == grid_b.address and grid_a.grid_count == grid_b.grid_count

    def type(self, arg0: torch.dtype):
        return VDBTensor(self.grid, self.feature.type(arg0))

    def cpu(self):
        return VDBTensor(self.grid.to('cpu'), self.feature.cpu())

    def cuda(self):
        return VDBTensor(self.grid.to('cuda'), self.feature.cuda())

    def to(self, device: Any):
        return VDBTensor(self.grid.to(device), self.feature.to(device))

    def detach(self):
        return VDBTensor(self.grid, self.feature.detach(), self.kmap)

    @staticmethod
    def _feature_ops(op, other: List[Union["VDBTensor", JaggedTensor, Any]]):
        pivot_tensor = [o for o in other if isinstance(o, VDBTensor)][0]
        raw_features = []
        for o in other:
            if isinstance(o, VDBTensor):
                assert VDBTensor.same_grid(o.grid, pivot_tensor.grid), "All VDBTensors should have the same grid"
                raw_features.append(o.feature.jdata)
            elif isinstance(o, JaggedTensor):
                assert pivot_tensor.total_voxels == o.jdata.size(0), "All tensors should have the same voxels"
                assert pivot_tensor.grid.grid_count == len(o.joffsets), "All tensors should have the same batch size"
                raw_features.append(o.jdata)
            else:
                raw_features.append(o)
        raw_features = op(*raw_features)
        return VDBTensor(pivot_tensor.grid, pivot_tensor.feature.jagged_like(raw_features), pivot_tensor.kmap)

    def __add__(self, other):
        return self._feature_ops(torch.add, [self, other])

    def __sub__(self, other):
        return self._feature_ops(torch.sub, [self, other])

    def __mul__(self, other):
        return self._feature_ops(torch.mul, [self, other])

    def __truediv__(self, other):
        return self._feature_ops(torch.div, [self, other])

    def __pow__(self, other):
        return self._feature_ops(torch.pow, [self, other])

    def __and__(self, other):
        return self._feature_ops(torch.logical_and, [self, other])

    def __or__(self, other):
        return self._feature_ops(torch.logical_or, [self, other])

    def __xor__(self, other):
        return self._feature_ops(torch.logical_xor, [self, other])

    @staticmethod
    def cat(tensors: List[Union["VDBTensor", JaggedTensor, torch.Tensor]], dim: int = 0) -> "VDBTensor":
        assert len(tensors) > 0, "At least one tensor should be provided"
        if dim == 0:
            assert all(isinstance(t, VDBTensor) for t in tensors), "All tensors should be of type VDBTensor"
            new_grid = fvdb.cat([t.grid for t in tensors]) # type: ignore
            new_feature = new_grid.jagged_like(torch.cat([t.feature.jdata for t in tensors])) # type: ignore
            return VDBTensor(new_grid, new_feature)
        else:
            return VDBTensor._feature_ops(lambda *t: torch.cat(t, dim=dim), tensors)

    @staticmethod
    def from_dense(dense_feature: torch.Tensor, ijk_min=None, origins=None, voxel_sizes=None) -> "VDBTensor":
        if origins is None:
            origins = [0.0] * 3
        if voxel_sizes is None:
            voxel_sizes = [1.0] * 3
        if ijk_min is None:
            ijk_min = [0, 0, 0]
        grid = fvdb.sparse_grid_from_dense(
            dense_feature.size(0), dense_feature.size()[1:4], ijk_min=ijk_min,
            voxel_sizes=voxel_sizes, origins=origins, device=dense_feature.device
        )
        # Note: this would map dense_feature[0, 0, 0] to grid[ijk_min]
        feature = grid.read_from_dense(dense_feature.contiguous(), dense_origins=ijk_min)
        return VDBTensor(grid, feature)

    def to_dense(self) -> torch.Tensor:
        # This would map grid.ijk.min() to dense_feature[0, 0, 0]
        return self.grid.read_into_dense(self.feature)

    def clear_cache(self):
        self.kmap = None

    @property
    def jdata(self):
        return self.feature.jdata

    @property
    def jidx(self):
        return self.feature.jidx

    @property
    def grid_count(self):
        return self.grid.grid_count

    batch_size = grid_count

    @property
    def total_voxels(self):
        return self.grid.total_voxels

    @property
    def device(self):
        return self.feature.device
