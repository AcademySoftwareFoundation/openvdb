# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Optional

import torch
import torch.nn.functional as F


def pad_batch(feats, coors=None, additional_feats: Optional[List] = None):
    """
    From a list of multi-level features create a list of batched tensors with
    features padded to the max number of points in the batch.

    returns:
        feats: List of batched feature Tensors per feature level
        coors: List of batched coordinate Tensors per feature level
        pad_masks: List of batched bool Tensors indicating padding
    """
    # get max number of points in the batch for each feature level
    maxs = [max([level.shape[0] for level in batch]) for batch in feats]
    # pad and batch each feature level in a single Tensor
    if coors is not None:
        coors = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch]) for i, batch in enumerate(coors)
        ]
    pad_masks = [
        torch.stack([F.pad(torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]), value=1).bool() for f in batch])
        for i, batch in enumerate(feats)
    ]
    feats = [torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch]) for i, batch in enumerate(feats)]
    if additional_feats is not None:
        additional_feats = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(additional_feats)
        ]

        return feats, coors, pad_masks, additional_feats
    return feats, coors, pad_masks
