# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Type

import torch

import fvdb
import fvdb.nn

from .blocks import BasicConvolutionBlock, BasicDeconvolutionBlock, ResidualBlock


class fVDBSyncBatchNorm(torch.nn.Module):
    r"""Applies Batch Normalization over a VDBTensor, synchronized across GPUs."""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
    ):
        torch.nn.Module.__init__(self)
        self.bn = torch.nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            process_group=process_group,
        )

    def forward(self, input: fvdb.nn.VDBTensor) -> fvdb.nn.VDBTensor:
        num_channels = input.data.jdata.size(1)
        assert (
            num_channels == self.bn.num_features
        ), "Input feature should have the same number of self.channels as BatchNorm"
        result_data = self.bn(input.data.jdata)
        return fvdb.nn.VDBTensor(input.grid, input.grid.jagged_like(result_data), input.kmap)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert
        :attr:`fvdb.nn.BatchNorm` layer in the model to
        :attr:`fVDBSyncBatchNorm` layer.

        Args:
            module (nn.Module): containing module
            process_group (optional): process group to scope synchronization,
            default is the whole world

        Returns:
            The original module with the converted
            :attr:`fvdb.nn.BatchNorm` layer

        Example::

            >>> # Network with fvdb.nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            fvdb.nn.BatchNorm(100)
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, fvdb.nn.BatchNorm):
            module_output = fVDBSyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,  # type: ignore
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.bn.weight = module.weight
                    module_output.bn.bias = module.bias
            module_output.bn.running_mean = module.running_mean
            module_output.bn.running_var = module.running_var
            module_output.bn.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.bn.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class MaskPLSEncoderDecoder(torch.nn.Module):

    channels = [32, 32, 64, 128, 256, 256, 128, 96, 96]

    def __init__(
        self,
        input_dim: int = 4,
        stem_blocks: int = 1,
        output_feature_levels: List[int] = [3],
        conv_deconv_non_lin: Type = fvdb.nn.ReLU,
        bn_momentum: float = 0.02,
    ):
        super().__init__()
        self.output_feature_levels = output_feature_levels
        down_res_blocks = [2, 3, 4, 6]

        self.stem = [
            fvdb.nn.SparseConv3d(input_dim, self.channels[0], kernel_size=3),
            fvdb.nn.BatchNorm(self.channels[0], momentum=bn_momentum),
            fvdb.nn.ReLU(inplace=True),
        ]
        for _ in range(1, stem_blocks):
            self.stem.extend(
                [
                    fvdb.nn.SparseConv3d(self.channels[0], self.channels[0], kernel_size=3),
                    fvdb.nn.BatchNorm(self.channels[0], momentum=bn_momentum),
                    fvdb.nn.ReLU(inplace=True),
                ]
            )
        self.stem = torch.nn.Sequential(*self.stem)

        self.stage1 = [
            BasicConvolutionBlock(
                self.channels[0], self.channels[0], ks=2, stride=2, bn_mom=bn_momentum, non_lin=conv_deconv_non_lin
            ),
            ResidualBlock(self.channels[0], self.channels[1], ks=3, bn_mom=bn_momentum),
        ]
        self.stage1.extend(
            [
                ResidualBlock(self.channels[1], self.channels[1], ks=3, bn_mom=bn_momentum)
                for _ in range(1, down_res_blocks[0])
            ]
        )
        self.stage1 = torch.nn.Sequential(*self.stage1)

        self.stage2 = [
            BasicConvolutionBlock(
                self.channels[1], self.channels[1], ks=2, stride=2, bn_mom=bn_momentum, non_lin=conv_deconv_non_lin
            ),
            ResidualBlock(self.channels[1], self.channels[2], ks=3, bn_mom=bn_momentum),
        ]
        self.stage2.extend(
            [
                ResidualBlock(self.channels[2], self.channels[2], ks=3, bn_mom=bn_momentum)
                for _ in range(1, down_res_blocks[1])
            ]
        )
        self.stage2 = torch.nn.Sequential(*self.stage2)

        self.stage3 = [
            BasicConvolutionBlock(
                self.channels[2], self.channels[2], ks=2, stride=2, bn_mom=bn_momentum, non_lin=conv_deconv_non_lin
            ),
            ResidualBlock(self.channels[2], self.channels[3], ks=3, bn_mom=bn_momentum),
        ]
        self.stage3.extend(
            [
                ResidualBlock(self.channels[3], self.channels[3], ks=3, bn_mom=bn_momentum)
                for _ in range(1, down_res_blocks[2])
            ]
        )
        self.stage3 = torch.nn.Sequential(*self.stage3)

        self.stage4 = [
            BasicConvolutionBlock(
                self.channels[3], self.channels[3], ks=2, stride=2, bn_mom=bn_momentum, non_lin=conv_deconv_non_lin
            ),
            ResidualBlock(self.channels[3], self.channels[4], ks=3, bn_mom=bn_momentum),
        ]
        self.stage4.extend(
            [
                ResidualBlock(self.channels[4], self.channels[4], ks=3, bn_mom=bn_momentum)
                for _ in range(1, down_res_blocks[3])
            ]
        )
        self.stage4 = torch.nn.Sequential(*self.stage4)

        self.up1 = torch.nn.ModuleList(
            [
                BasicDeconvolutionBlock(
                    self.channels[4],
                    self.channels[5],
                    ks=2,
                    stride=2,
                    bn_mom=bn_momentum,
                ),
                torch.nn.Sequential(
                    ResidualBlock(self.channels[5] + self.channels[3], self.channels[5], ks=3, bn_mom=bn_momentum),
                    ResidualBlock(self.channels[5], self.channels[5], ks=3, bn_mom=bn_momentum),
                ),
            ]
        )

        self.up2 = torch.nn.ModuleList(
            [
                BasicDeconvolutionBlock(
                    self.channels[5],
                    self.channels[6],
                    ks=2,
                    stride=2,
                    bn_mom=bn_momentum,
                ),
                torch.nn.Sequential(
                    ResidualBlock(self.channels[6] + self.channels[2], self.channels[6], ks=3, bn_mom=bn_momentum),
                    ResidualBlock(self.channels[6], self.channels[6], ks=3, bn_mom=bn_momentum),
                ),
            ]
        )

        self.up3 = torch.nn.ModuleList(
            [
                BasicDeconvolutionBlock(
                    self.channels[6],
                    self.channels[7],
                    ks=2,
                    stride=2,
                    bn_mom=bn_momentum,
                ),
                torch.nn.Sequential(
                    ResidualBlock(self.channels[7] + self.channels[1], self.channels[7], ks=3, bn_mom=bn_momentum),
                    ResidualBlock(self.channels[7], self.channels[7], ks=3, bn_mom=bn_momentum),
                ),
            ]
        )

        self.up4 = torch.nn.ModuleList(
            [
                BasicDeconvolutionBlock(
                    self.channels[7],
                    self.channels[8],
                    ks=2,
                    stride=2,
                    bn_mom=bn_momentum,
                ),
                torch.nn.Sequential(
                    ResidualBlock(self.channels[8] + self.channels[0], self.channels[8], ks=3, bn_mom=bn_momentum),
                    ResidualBlock(self.channels[8], self.channels[8], ks=3, bn_mom=bn_momentum),
                ),
            ]
        )

        levels = [self.channels[-i] for i in range(4, 0, -1)]

        # conv mask projection
        self.mask_feat = fvdb.nn.SparseConv3d(
            self.channels[-1],
            self.channels[-1],
            kernel_size=3,
            stride=1,
        )

        self.out_bnorm = torch.nn.ModuleList([torch.nn.Sequential() for _ in levels])

    def forward(self, x) -> List[fvdb.nn.VDBTensor]:

        sparse_input = x["vdbtensor"]

        x0 = self.stem(sparse_input)  # type: ignore
        x1 = self.stage1(x0)  # type: ignore
        x2 = self.stage2(x1)  # type: ignore
        x3 = self.stage3(x2)  # type: ignore
        x4 = self.stage4(x3)  # type: ignore

        y1 = self.up1[0](x4, out_grid=x3.grid)
        y1 = fvdb.jcat([y1, x3], dim=1)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1, out_grid=x2.grid)
        y2 = fvdb.jcat([y2, x2], dim=1)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2, out_grid=x1.grid)
        y3 = fvdb.jcat([y3, x1], dim=1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3, out_grid=x0.grid)
        y4 = fvdb.jcat([y4, x0], dim=1)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]

        feat_levels = self.output_feature_levels + [3]

        out_feats = [out_feats[feats] for feats in feat_levels]

        out_feats[-1] = self.mask_feat(out_feats[-1])

        # batch norm
        out_feats = [bn(feat) for feat, bn in zip(out_feats, self.out_bnorm)]

        return out_feats
