# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn

import fvdb
import fvdb.nn

from .backbone import MaskPLSEncoderDecoder, fVDBSyncBatchNorm
from .blocks import MLP
from .decoder import MaskedTransformerDecoder
from .utils import pad_batch


class MaskPLS(torch.nn.Module):
    class DecoderInputMode(Enum):
        XYZ = "xyz"
        GRID = "grid"

    def __init__(
        self,
        num_classes: int,
        dataset_extent: Tuple[float, float, float],
        decoder_input_mode: DecoderInputMode = DecoderInputMode.GRID,
        decoder_num_queries: int = 100,
        segmentation_only=False,
    ) -> None:
        """
        Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving
        https://github.com/PRBonn/MaskPLS
        Args:
            num_classes (int): Number of classes for segmentation.
            dataset_extent (Tuple[float, float, float]): The magnitude of the spatial extents of the dataset.
            decoder_input_mode (DecoderInputMode, optional): Mode for decoder input. Defaults to DecoderInputMode.GRID.
            decoder_num_queries (int, optional): Number of queries for the decoder. Defaults to 100.
            segmentation_only (bool, optional): If True, only segmentation is performed, masked decoder not used. Defaults to False.
        Returns:
            None
        """
        super().__init__()
        self.decoder_input_mode = decoder_input_mode
        self.segmentation_only = segmentation_only

        backbone = MaskPLSEncoderDecoder(output_feature_levels=[3])
        self.backbone = fVDBSyncBatchNorm.convert_sync_batchnorm(backbone)

        self.sem_head = (
            fvdb.nn.Linear(self.backbone.channels[-1], num_classes)
            if self.decoder_input_mode == MaskPLS.DecoderInputMode.GRID
            else torch.nn.Linear(self.backbone.channels[-1], num_classes)
        )

        self.semantic_embedding_distil = False
        if self.semantic_embedding_distil:
            semantic_embedding_hidden_dims = [512, 1024, 768]
            self.sem_embed = MLP(
                self.backbone.channels[-1],
                semantic_embedding_hidden_dims[:-1],
                semantic_embedding_hidden_dims[-1],
                use_fvdb=(self.decoder_input_mode == MaskPLS.DecoderInputMode.GRID),
            )

        if not self.segmentation_only:
            self.decoder = MaskedTransformerDecoder(
                num_classes, dataset_extent, backbone_channels=self.backbone.channels, num_queries=decoder_num_queries
            )

    def forward(self, x: Dict):
        outputs = {}

        ###### Backbone ######
        out_feats_grids = self.backbone(x)
        # out_feats_grids is a List[fvdb.nn.VDBTensor]
        #    where each VDBTensor corresponds to the `ouput_feature_levels`
        #    plus 1 additional entry which is the last/full-resolution feature level run through the conv mask projection

        ###### v2p ######
        # NOTE: Matching MaskPLS paper which performs v2p before sem_head
        #    In SAL, features are at voxel centers throughout, so we provide an option to try either
        if self.decoder_input_mode == MaskPLS.DecoderInputMode.XYZ:
            # If decoder inputs are the original points, we need to sample the features in the grid and pad them for form
            #    a minibatch for the semantic head and decoder
            xyz = x["xyz"]
            feats = [feats_grid.sample_trilinear(xyz).unbind() for feats_grid in out_feats_grids]

            # pad batch
            feats, coords, pad_masks = pad_batch(feats, [xyz.unbind() for _ in feats])  # type: ignore
        else:
            feats = out_feats_grids

        logits = [self.sem_head(feats[-1])]

        if self.semantic_embedding_distil:
            logits_sem_embed_grid = self.sem_embed(feats[-1])

        if self.decoder_input_mode == MaskPLS.DecoderInputMode.GRID:
            # produce a padded batch for the decoder and loss
            coords = [feat.grid.grid_to_world(feat.ijk.float()).unbind() for feat in out_feats_grids]
            feats = [feat.data.unbind() for feat in out_feats_grids]
            logits = [ls.data.unbind() for ls in logits]
            feats, coords, pad_masks, logits = pad_batch(feats, coords, additional_feats=logits)  # type: ignore

        ###### Decoder ######
        if self.segmentation_only:
            padding = pad_masks.pop()
        else:
            outputs, padding = self.decoder(feats, coords, pad_masks)

        outputs["bb_sem_logits"] = logits[0]
        outputs["bb_sem_embed_logits"] = None if not self.semantic_embedding_distil else logits_sem_embed_grid
        outputs["padding"] = padding

        return outputs
