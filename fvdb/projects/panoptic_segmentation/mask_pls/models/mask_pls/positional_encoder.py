# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, feat_size, dataset_extent):
        super().__init__()
        self.feat_size = feat_size
        self.dataset_extent = torch.tensor(dataset_extent)

        self.max_freq = 10000
        self.dimensionality = 3
        self.num_bands = math.floor(feat_size / self.dimensionality / 2)
        self.base = 2
        self.activated = True

        pad = feat_size - self.num_bands * 2 * self.dimensionality
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, coors):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        if not self.activated:
            return torch.zeros(
                coors.shape[0],
                coors.shape[1],
                self.feat_size,
            ).to(coors.device)

        x = coors.clone()  # B, N, 3

        x = x / self.dataset_extent.to(x.device)  # B, N, 3

        x = x.unsqueeze(-1)  # B, N, 3 -> B, N, 3, 1
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), ...)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # B, N, 3, 2
        x = x.flatten(2)  # B, N, 6
        enc = self.zero_pad(x)  # B, N, feat_dim
        return enc
