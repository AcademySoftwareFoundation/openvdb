#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from enum import Enum
from pathlib import Path
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
import tqdm
import tyro
from data import (
    E57Dataset,
    SemanticKITTIDataset,
    SemanticSegmentationDatasetCollation,
    fVDBSemanticSegmentationDatasetCollation,
)
from models import MaskPLS
from models.mask_pls.loss import SemLoss


class DatasetType(Enum):
    E57 = "e57"
    SemanticKITTI = "semanticKITTI"


def main(
    dataset_type: DatasetType,
    dataset_path: Path,
    dataset_spatial_normalization: Tuple[float, float, float],
    batch_size: int = 4,
    decoder_input_mode: MaskPLS.DecoderInputMode = MaskPLS.DecoderInputMode.GRID,
    decoder_num_queries: int = 100,
):
    """Example project training a MaskPLS model for panoptic segmentation of LiDAR.

    Args:
        dataset_type (DatasetType): Type of dataset to use (E57 or SemanticKITTI).
            The SemanticKITTI dataset is expected to be in the standard SemanticKITTI format.
            E57 is expected to be a directory containing E57 files and currently 'ground truth' labels are implemented as random classes
        dataset_path (Path): Path to the dataset directory
        dataset_spatial_normalization (Tuple[float, float, float]): Normalization factors for spatial coordinates of the points.
            This should be set to the magnitude of the maximum spatial extent of the dataset in each dimension.
            (i.e. if the dataset's points are in the range [-80, 100] in x, [-50, 20] in y, and [-10, 4] in z, this should be (100, 50, 10))
        batch_size (int, optional): Batch size for training.
        decoder_input_mode (MaskPLS.DecoderInputMode, optional): Input mode for the spatial input to the decoder.
            GRID corresponds to the grid centers of the sparse grid, POINTS corresponds to the original points. Using GRID can reduce memory usage whereas the original MaskPLS paper used POINTS.
        decoder_num_queries (int, optional): Number of queries to use for the decoder. More queries will increase memory usage.

    """
    #### Dataset ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_type == DatasetType.E57:
        dataset = E57Dataset(dataset_path)
    elif dataset_type == DatasetType.SemanticKITTI:
        dataset = SemanticKITTIDataset(dataset_path, split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=SemanticSegmentationDatasetCollation(),
    )

    fvdb_collate = fVDBSemanticSegmentationDatasetCollation(device=device)

    #### Model/Loss/Optimizer ####
    model = MaskPLS(
        dataset.num_classes,
        dataset_spatial_normalization,
        segmentation_only=False,
        decoder_input_mode=decoder_input_mode,
        decoder_num_queries=decoder_num_queries,
    ).to(device)

    ignore_class = dataset.ignore_classes[0]

    backbone_loss_sem = SemLoss(
        ignore_class,
        sem_distil=False,
        loss_scales={"loss_backbone_sem_ce": 2.0, "loss_backbone_sem_lov": 6.0},
        input_mode=decoder_input_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,
        weight_decay=0.0001,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[45000, 55000],
        gamma=0.1,
    )

    #### Training ####
    with tqdm.tqdm(dataloader) as pbar:
        network_time = 0.0
        for i, batch in enumerate(pbar):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            batch = fvdb_collate(batch)

            optimizer.zero_grad()

            outputs = model(batch)

            loss_dict = backbone_loss_sem(outputs, batch)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            end.record()

            torch.cuda.synchronize()
            network_time += start.elapsed_time(end) / 1000
            pbar.set_postfix(loss=f"{loss.item():.4f}", network_time=f"{network_time/(i+1):.2f}s/it")

        scheduler.step()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
