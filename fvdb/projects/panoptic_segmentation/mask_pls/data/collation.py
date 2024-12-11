# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Dict

import torch

import fvdb


class SemanticSegmentationDatasetCollation:
    """
    A data collation class for semantic segmentation datasets.
    This class handles the collation of various data components used in semantic
    segmentation tasks, including point cloud coordinates, labels, and auxiliary data.
    Attributes:
        keys (list): List of strings representing the expected data keys:
            - "xyz": Point cloud coordinates
            - "voxel_size": Voxel dimensions
            - "semantic_labels": Semantic class labels
            - "instance_labels": Instance segmentation labels
            - "intensity": Point intensity values
            - "semantic_embeddings": Semantic feature embeddings
            - "cam_images": Camera images
    Methods:
        __call__(data): Collates the input data into a dictionary format.
    Args:
        data: A sequence of data samples to be collated.
    Returns:
        Dict: A dictionary where each key corresponds to a list of the respective data components.
    """

    def __init__(self):
        self.keys = [
            "xyz",
            "voxel_size",
            "semantic_labels",
            "instance_labels",
            "intensity",
            "semantic_embeddings",
            "cam_images",
        ]

    def __call__(self, data) -> Dict:
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}


class fVDBSemanticSegmentationDatasetCollation:
    """
    A data collation class for semantic segmentation datasets using fVDB.
    This class handles the conversion of point cloud data into VDBTensors.
    Parameters
    ----------
    device : torch.device, optional
        The device on which to perform computations (default is "cuda:0")
    Methods
    -------
    __call__(data: dict) -> dict:
        Transforms input point cloud data into VDBTensors.
        Parameters:
            data (dict): Dictionary containing:
                - xyz: List of point coordinates in world space
                - voxel_size: List of voxel dimensions
                - intensity: List of intensity values for points
        Returns:
            dict: Original dictionary updated with:
                - xyz: JaggedTensor of point coordinates
                - vdbtensor: VDBTensor containing the structured volumetric data
    """

    def __init__(self, device=torch.device("cuda:0")):
        super().__init__()
        self.device = device

    def __call__(self, data):
        # xyz world space point positions
        data["xyz"] = fvdb.JaggedTensor([torch.tensor(c, device=self.device) for c in data["xyz"]])

        grid = fvdb.gridbatch_from_points(data["xyz"], voxel_sizes=[n.tolist() for n in data["voxel_size"]])

        # get mapping of the coordinates to the grid for feature mapping
        coord_ijks = grid.world_to_grid(data["xyz"]).round().int()
        inv_idx = grid.ijk_to_inv_index(coord_ijks, cumulative=True)

        # assert(torch.all(grid.ijk.jdata == coord_ijks.jdata[inv_idx.jdata]))

        jfeats = torch.cat([torch.tensor(f, device=self.device).unsqueeze(-1) for f in data["intensity"]])
        jfeats = grid.jagged_like(jfeats[inv_idx.jdata])

        jfeats = fvdb.jcat([grid.ijk.float(), jfeats], dim=1)

        vdbtensor = fvdb.nn.VDBTensor(grid, jfeats)

        data["vdbtensor"] = vdbtensor

        return data
