# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pye57
from torch.utils.data import Dataset

__all__ = [
    "E57Dataset",
]


class E57Dataset(Dataset):
    logger = logging.getLogger("E57Dataset")

    def __init__(
        self,
        directory: Path,
        cam_image_names: List = [
            "image_2",
        ],
        voxel_size: Union[float, np.ndarray] = 0.1,
    ):
        """This class handles E57 files and associated camera images for point cloud processing.
        Args:
            directory (Path): Path to the directory containing E57 files
            cam_image_names (List, optional): List of camera image folder names. Defaults to ["image_2"]
            voxel_size (Union[float, np.ndarray], optional): Size of voxels for point cloud discretization.
                Can be a single float or array. Defaults to 0.1
        """
        self._directory = directory
        self._cam_image_names = cam_image_names
        self._voxel_size = voxel_size

        # Hardcoded for now
        self.num_classes = 20
        self.ignore_classes = [0]

        self._e57_files = list(directory.glob("*.e57"))
        self.logger.info(f"Found {len(self._e57_files)} e57 files in {directory}")

    def get_directory(self):
        return self._directory

    def __len__(self):
        return len(self._e57_files)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range {len(self)}")

        e57_file = self._e57_files[idx]
        self.logger.debug(f"Loading {e57_file}")

        with pye57.E57(str(e57_file)) as e57:
            scan = e57.read_scan(0, intensity=True, colors=False, transform=True)
            xyz = np.stack([scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]], axis=-1)

            intensity = scan["intensity"]

            # color = np.stack([scan["colorRed"], scan["colorGreen"], scan["colorBlue"]], axis=-1)

        # TODO:  Determine a better way to set the voxel size.  Should this be a user param?  Dataset metadata?
        voxel_size = np.array(self._voxel_size)
        if voxel_size.size not in [1, 3]:
            raise ValueError(f"Invalid voxel size: {voxel_size}, must be a scalar or 3-element array")
        if voxel_size.size == 1:
            voxel_size = voxel_size.repeat(3)

        # instance labels just as random values as placeholder
        semantic_labels = np.random.randint(0, self.num_classes, xyz[:, 0].shape, dtype=np.uint32)

        # TODO: Color data, image data, etc.
        # xyz, voxel_size, semantic_labels, instance_labels, intensity, semantic_embeddings, cam_images
        self.logger.debug(f"Returning data for {e57_file}")
        return (xyz, voxel_size, semantic_labels, None, intensity, None, None)
