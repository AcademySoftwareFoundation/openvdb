# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Type

import torch

import fvdb
import fvdb.nn

from .blocks import BasicConvolutionBlock, BasicDeconvolutionBlock, ResidualBlock


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
