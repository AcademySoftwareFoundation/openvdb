from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import fvdb
from fvdb.nn import VDBTensor, SparseConv3d
from torchsparse import SparseTensor


class BaseConfig(ABC):
    in_channels: int
    dataset_paths: list[str]
    baselines: list[str] = ['ts', 'fvdb::igemm_mode1', 'fvdb::cutlass']

    def get_aux_inputs(self, vdb_tensor: VDBTensor) -> dict:
        return {}

    def post_measure(self, baseline_input, baseline_out):
        pass

    def make_model(self, baseline: str) -> nn.Module:
        if baseline.startswith("fvdb"):
            baseline, fvdb_method = baseline.split("::")
            SparseConv3d.backend = fvdb_method
        return self._make_model(baseline)

    @abstractmethod
    def _make_model(self, baseline: str) -> nn.Module:
        pass

    def to_baseline_input(self, vdb_tensor: VDBTensor, baseline: str):
        if baseline == "ts":
            return SparseTensor(vdb_tensor.feature.jdata.clone(), torch.cat([
                vdb_tensor.grid.ijk.jidx[:, None].int(),
                vdb_tensor.grid.ijk.jdata.int(),
            ], dim=1))
        elif baseline.startswith("fvdb"):
            return vdb_tensor
        else:
            raise NotImplementedError


class SingleConvConfig(BaseConfig):
    in_channels: int = 32

    def post_measure(self, baseline_input, baseline_out):
        if isinstance(baseline_input, VDBTensor):
            baseline_input.kmap = baseline_out.kmap


class GridBuildingConfig(BaseConfig):
    in_channels: int = 32

    def get_aux_inputs(self, vdb_tensor: VDBTensor) -> dict:
        if vdb_tensor.device != torch.device('cuda'):
            vdb_tensor = vdb_tensor.to('cuda')
        coord = vdb_tensor.grid.ijk
        feats = vdb_tensor.feature.jdata
        return {'coords': coord, 'feats': feats}

    def to_baseline_input(self, vdb_tensor: VDBTensor, baseline: str):
        return None


class XCubeConfig(BaseConfig):
    in_channels: int = 32
    dataset_paths: list[str] = [str(Path(__file__).parent.parent / 'data' / 'kc-256')]

    def get_aux_inputs(self, vdb_tensor: VDBTensor) -> dict:
        coord = vdb_tensor.grid.ijk.jdata
        gt_coords = {0: coord}
        for layer_idx in range(1, 4):
            coords = torch.div(gt_coords[layer_idx - 1], 2).floor().long()
            coords = torch.unique(coords, dim=0)
            gt_coords[layer_idx] = coords

        return {'gt_coords': {
            layer_idx: fvdb.sparse_grid_from_ijk(coords)
            for (layer_idx, coords) in gt_coords.items()
        }}

    def _make_model(self, baseline: str) -> nn.Module:
        from fvdb_benchmark.model.xcube import XCubeVAE
        return XCubeVAE(
            in_channels=self.in_channels,
            backend=baseline,
            num_blocks=4,
            f_maps=32,
            order='cr'
        )


class KITTISegmentationConfig(BaseConfig):
    in_channels: int = 4
    dataset_paths: list[str] = [str(Path(__file__).parent.parent / 'data' / 'kitti')]

    def _make_model(self, baseline: str) -> nn.Module:
        from fvdb_benchmark.model.minkunet import MinkUNet
        return MinkUNet(
            backend=baseline,
            in_channels=self.in_channels,
            cr=1.0, num_classes=19
        )


all_configs = {
    'single_conv': SingleConvConfig,
    'grid_building': GridBuildingConfig,
    'xcube': XCubeConfig,
    'kitti_segmentation': KITTISegmentationConfig,
}
