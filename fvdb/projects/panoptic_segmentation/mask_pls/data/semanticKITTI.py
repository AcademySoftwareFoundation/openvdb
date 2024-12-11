# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import itertools
import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "SemanticKITTIDataset",
    "SemanticKITTIDataItem",
    "SemanticKITTIDataSequence",
]


class SemanticKITTIDataItem:
    def __init__(
        self,
        parent_sequence: SemanticKITTIDataSequence,
        velodyne_file: Path,
        calibration_data: NamedTuple,
        cam_image_names: List = [],
        label_file: Optional[Path] = None,
    ) -> None:
        """This class represents a single frame from the SemanticKITTI dataset, containing LiDAR point cloud,
        optional semantic labels, calibration data, and associated camera images.

        Args:
            parent_sequence (SemanticKITTIDataSequence): Parent sequence object containing this frame
            velodyne_file (Path): Path to the velodyne point cloud file
            calibration_data (NamedTuple): Calibration data for sensors
            cam_image_names (List, optional): List of camera names to load images from. Defaults to empty list.
            label_file (Optional[Path], optional): Path to semantic label file. Defaults to None.

        Raises:
            FileNotFoundError: If velodyne file, label file, or any camera image file does not exist
        """
        self._parent_sequence = parent_sequence
        if not velodyne_file.exists():
            raise FileNotFoundError(f"Velodyne file {velodyne_file} not found")
        self._velodyne_file = velodyne_file
        if label_file and not label_file.exists():
            raise FileNotFoundError(f"Label file {label_file} not found")
        self._label_file = label_file

        self._calib = calibration_data
        self._cam_image_names = cam_image_names

        self._cam_image_files = []
        for cam_name in cam_image_names:
            self._cam_image_files.append(
                self._velodyne_file.parent.parent / cam_name / self._velodyne_file.with_suffix(".png").name
            )
            if not self._cam_image_files[-1].exists():
                raise FileNotFoundError(f"Image file {self._cam_image_files[-1]} not found")

    def __repr__(self):
        return f"SemanticKITTIDataItem({self._velodyne_file})"

    def get_parent_sequence(self) -> SemanticKITTIDataSequence:
        return self._parent_sequence

    def has_labels(self) -> bool:
        return self._label_file is not None

    def has_semantic_embeddings(self) -> bool:
        if self._label_file:
            if self._label_file.with_suffix(".pth").exists():
                return True
        return False

    def get_semantic_embeddings(self) -> Dict:
        if self.has_semantic_embeddings():
            semantic_embeddings = torch.load(self._label_file.with_suffix(".pth"))
            semantic_embeddings["ins"] = semantic_embeddings["ins"].numpy()
            semantic_embeddings["ins"] = 10 + ((semantic_embeddings["ins"] << 16) & 0xFFFF0000)

            embeddings_norm = semantic_embeddings["embeds"].norm(dim=-1, keepdim=True)
            semantic_embeddings["embeds"] /= embeddings_norm
            return semantic_embeddings

        else:
            raise ValueError(f"No semantic embeddings found for {self}")

    def get_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_labels():
            raise ValueError(f"No labels found for {self}")
        else:
            label_data = np.fromfile(self._label_file, dtype=np.uint32)

            semantic_labels = label_data & 0xFFFF
            instance_labels = label_data >> 16
            return semantic_labels, instance_labels

    def get_points_intensities(self) -> Tuple[np.ndarray, np.ndarray]:
        # points
        point_data = np.fromfile(self._velodyne_file, dtype=np.float32)
        point_data = point_data.reshape((-1, 4))

        xyz = point_data[:, :3]

        # intensities
        intensity = point_data[:, 3]
        if len(intensity.shape) == 2:
            intensity = np.squeeze(intensity)
        return xyz, intensity

    def get_cam_images(self, xyz: np.ndarray) -> Dict:
        # load images
        #
        cam_images = {
            "cam_names": self._cam_image_files,
            "imgs": [],
            "points_in_img": [],
            "points_in_img_indices": [],
        }
        if len(self._cam_image_files) > 0:
            imgs = [Image.open(f) for f in self._cam_image_files]
            cam_images["imgs"] = [np.array(img) for img in imgs]

        for cam_name, img in zip(self._cam_image_names, cam_images["imgs"]):
            xform_lidar_to_cam = getattr(self._calib, f"Tr_{cam_name}")
            # add ones to the end of the points
            xyz = np.hstack((xyz[:, :3], np.ones((xyz.shape[0], 1))))
            xyz_camera_space = np.dot(xyz, xform_lidar_to_cam.T)
            valid_points_in_img = xyz_camera_space[:, 2] > 0

            points_in_img = getattr(self._calib, f"P_{cam_name}").dot(xyz_camera_space.T).T
            points_in_img /= points_in_img[:, 2].reshape(-1, 1)
            points_in_img = points_in_img.astype(int)[:, :2]

            # valid points in image
            width, height, _ = img.shape
            valid_points_in_img = np.logical_and(valid_points_in_img, 0 <= points_in_img[:, 0])
            valid_points_in_img = np.logical_and(valid_points_in_img, points_in_img[:, 0] < height)
            valid_points_in_img = np.logical_and(valid_points_in_img, 0 <= points_in_img[:, 1])
            valid_points_in_img = np.logical_and(valid_points_in_img, points_in_img[:, 1] < width)

            points_in_img = points_in_img[valid_points_in_img]
            points_in_img_indices = np.where(valid_points_in_img)[0]

            cam_images["points_in_img"].append(points_in_img)
            cam_images["points_in_img_indices"].append(points_in_img_indices)
        return cam_images

    def get_points_intensities_cam_images(self, index):
        xyz, intensity = self.get_points_intensities()
        cam_images = self.get_cam_images(xyz)
        return xyz, intensity, cam_images


class SemanticKITTIDataSequence:
    def __init__(
        self,
        parent_dataset: SemanticKITTIDataset,
        sequence_directory: Path,
        cam_image_names: List = [],
        labels_name: Optional[str] = "",
    ) -> None:
        """SemanticKITTIDataSequence represents a sequence sub-directory in the SemanticKITTI dataset.

        Args:
            parent_dataset (SemanticKITTIDataset): The parent dataset instance this sequence belongs to
            sequence_directory (Path): Path to the sequence directory containing velodyne data, labels etc.
            cam_image_names (List, optional): List of camera image names to load. Defaults to empty list.
            labels_name (Optional[str], optional): Name of the labels directory. Defaults to empty string.

        Returns:
            None
        """
        self._parent = parent_dataset
        self._sequence_directory = sequence_directory
        self._velodyne = sequence_directory / "velodyne"
        self._labels = (sequence_directory / labels_name) if labels_name else None

        self._calib_path = sequence_directory / "calib.txt"

        self._cam_image_names = cam_image_names

        self.setup()

    def __len__(self):
        return len(self._data_items)

    def get_parent_dataset(self):
        return self._parent

    def setup(self):
        # read calibration data from calib.txt
        self.read_calib()

        # populate all the inidividual data items
        def populate_data_items():
            for velodyne_file in sorted(self._velodyne.glob("*.bin")):
                label_file = None
                if self._labels:
                    label_file = self._labels / velodyne_file.name.replace("bin", "label")
                yield SemanticKITTIDataItem(
                    self,
                    velodyne_file,
                    calibration_data=self.calib,
                    label_file=label_file,
                    cam_image_names=self._cam_image_names,
                )

        self._data_items = list(populate_data_items())

    def read_calib(self) -> None:
        """Load and compute intrinsic and extrinsic calibration parameters."""
        with self._calib_path.open("r") as f:
            lines = f.readlines()

        filedata = {}
        for line in lines:
            key, value = line.strip().split(":", 1)
            filedata[key] = np.array([float(x) for x in value.split()])

        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata["P0"], (3, 4))
        P_rect_10 = np.reshape(filedata["P1"], (3, 4))
        P_rect_20 = np.reshape(filedata["P2"], (3, 4))
        P_rect_30 = np.reshape(filedata["P3"], (3, 4))

        data["P_rect_00"] = P_rect_00
        data["P_rect_10"] = P_rect_10
        data["P_rect_20"] = P_rect_20
        data["P_rect_30"] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data["T_cam0_velo"] = np.reshape(filedata["Tr"], (3, 4))
        data["T_cam0_velo"] = np.vstack([data["T_cam0_velo"], [0, 0, 0, 1]])
        data["T_cam1_velo"] = T1.dot(data["T_cam0_velo"])
        data["T_cam2_velo"] = T2.dot(data["T_cam0_velo"])
        data["T_cam3_velo"] = T3.dot(data["T_cam0_velo"])

        # Compute the camera intrinsics
        data["K_cam0"] = P_rect_00[0:3, 0:3]
        data["K_cam1"] = P_rect_10[0:3, 0:3]
        data["K_cam2"] = P_rect_20[0:3, 0:3]
        data["K_cam3"] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data["T_cam0_velo"]).dot(p_cam)
        p_velo1 = np.linalg.inv(data["T_cam1_velo"]).dot(p_cam)
        p_velo2 = np.linalg.inv(data["T_cam2_velo"]).dot(p_cam)
        p_velo3 = np.linalg.inv(data["T_cam3_velo"]).dot(p_cam)

        data["b_gray"] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data["b_rgb"] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

        data["P_image_0"] = data["P_rect_00"]
        data["P_image_1"] = data["P_rect_10"]
        data["P_image_2"] = data["P_rect_20"]
        data["P_image_3"] = data["P_rect_30"]

        data["Tr_image_0"] = data["T_cam0_velo"]
        data["Tr_image_1"] = data["T_cam0_velo"]
        data["Tr_image_2"] = data["T_cam0_velo"]
        data["Tr_image_3"] = data["T_cam0_velo"]
        # data["Tr_image_1"] = data["T_cam1_velo"]
        # data["Tr_image_2"] = data["T_cam2_velo"]
        # data["Tr_image_3"] = data["T_cam3_velo"]

        data["K_image_0"] = data["K_cam0"]
        data["K_image_1"] = data["K_cam1"]
        data["K_image_2"] = data["K_cam2"]
        data["K_image_3"] = data["K_cam3"]

        self.calib = namedtuple("CalibData", data.keys())(*data.values())

    def get_data_item(self, idx):
        return self._data_items[idx]


class SemanticKITTIDataset(Dataset):
    logger = logging.getLogger("SemanticKITTIDataset")

    def __init__(
        self,
        directory: Path,
        format: str = "KITTI",
        labels_name="labels",
        cam_image_names: List = [
            "image_2",
        ],
        split: str = "train",
        camera_frustum_filter: bool = False,
        unlabelled_filter: bool = False,
        voxel_size: Union[float, np.ndarray] = 0.1,
    ):
        """SemanticKITTI dataset.

        Args:
            directory (Path): Root directory path containing dataset.
            format (str, optional): Dataset format. Defaults to "KITTI".
            labels_name (str, optional): Name of labels directory. Defaults to "labels".
            cam_image_names (List, optional): List of camera image directory names. Defaults to ["image_2"].
            split (str, optional): Dataset split ('train', 'val', 'test') as defined in KITTI dataset config. Defaults to "train".
            camera_frustum_filter (bool, optional): Whether to filter points outside camera frustum. Defaults to False.
            unlabelled_filter (bool, optional): Whether to filter unlabelled points. Defaults to False.
            voxel_size (Union[float, np.ndarray], optional): Size of voxels for point cloud discretization. Defaults to 0.1.

        Raises:
            ValueError: If more than one or no yaml configuration file is found in the dataset directory.
        """
        self._root_directory = directory
        self._format = format
        self._labels_name = labels_name
        self._cam_image_names = cam_image_names
        self._split = split
        self._camera_frustum_filter = camera_frustum_filter
        self._unlabelled_filter = unlabelled_filter
        self._voxel_size = voxel_size

        # load yaml config in dataset directory
        yaml_files = list(directory.glob("*.yaml"))
        if len(yaml_files) != 1:
            raise ValueError(f"Expected exactly one yaml file in {directory}")
        self._config = yaml.safe_load(yaml_files[0].open())

        self.setup()

    def get_directory(self) -> Path:
        return self._root_directory

    @property
    def config(self) -> Dict:
        return self._config

    def setup(self):
        # setup from config
        self.num_classes = len(self.config["learning_map_inv"])
        self.ignore_classes = [cl for cl, ignored in self.config["learning_ignore"].items() if ignored]
        self.include_classes = [cl for cl, ignored in self.config["learning_ignore"].items() if not ignored]

        self.logger.info(f"Setting up '{self._split}' split of dataset {self._root_directory}")
        # setup the sequences for this split
        if self._split == "all":
            self._data_sequences = [
                SemanticKITTIDataSequence(
                    self,
                    self._root_directory / "sequences" / f"{seq:02}",
                    cam_image_names=self._cam_image_names,
                    labels_name=self._labels_name,
                )
                for seq in list(itertools.chain(*self.config["split"]))
            ]
        else:
            self._data_sequences = [
                SemanticKITTIDataSequence(
                    self,
                    self._root_directory / "sequences" / f"{seq:02}",
                    cam_image_names=self._cam_image_names,
                    labels_name=self._labels_name,
                )
                for seq in self.config["split"][
                    self._split
                ]  # "split" in the config is a dictionary containing a list of sequence names for each split
            ]
        self.logger.info(f"Found {len(self._data_sequences)} sequences")

    def __len__(self) -> int:
        return sum(len(seq) for seq in self._data_sequences)

    def __getitem__(
        self, idx
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None], Dict]:
        # TODO: Implement negative indexing

        # find the sequence that contains the item
        data_item = None
        for seq in self._data_sequences:
            if idx < len(seq):
                data_item = seq.get_data_item(idx)
                break
            idx -= len(seq)
        if data_item is None:
            raise IndexError("Index out of bounds")
        (
            xyz,
            intensity,
            cam_images,
        ) = data_item.get_points_intensities_cam_images(idx)

        # TODO:  Determine a better way to set the voxel size.  Should this be a user param?  Dataset metadata?
        voxel_size = np.array(self._voxel_size)
        if voxel_size.size not in [1, 3]:
            raise ValueError(f"Invalid voxel size: {voxel_size}, must be a scalar or 3-element array")
        if voxel_size.size == 1:
            voxel_size = voxel_size.repeat(3)

        if data_item.has_labels:
            semantic_labels, instance_labels = data_item.get_labels()
        else:
            semantic_labels = np.zeros_like(xyz[:, 0], dtype=np.uint32)
            instance_labels = np.zeros_like(xyz[:, 0], dtype=np.uint32)

        assert semantic_labels.shape[0] == instance_labels.shape[0] == xyz.shape[0], (
            f"Number of points ({xyz.shape[0]}) and " f"labels {semantic_labels.shape[0]} not the same for {self} "
        )

        semantic_embeddings = None
        if data_item.has_semantic_embeddings():
            semantic_embeddings = data_item.get_semantic_embeddings()

        # things masks: semantic labels of thing instances
        instance_semantic_labels = np.where(instance_labels == 0, 0, semantic_labels)

        # semantic + instance value
        instance_labels = instance_semantic_labels + ((instance_labels << 16) & 0xFFFF0000)
        semantic_labels = np.vectorize(self.config["learning_map"].__getitem__)(semantic_labels)

        # filter points outside of camera frustum
        if self._camera_frustum_filter:
            cam_frustum_indices = np.unique(np.concatenate(cam_images["points_in_img_indices"]))

            xyz = xyz[cam_frustum_indices]
            semantic_labels = semantic_labels[cam_frustum_indices]
            instance_labels = instance_labels[cam_frustum_indices]
            intensity = intensity[cam_frustum_indices]

            points_in_img = cam_images["points_in_img"]
            points_in_img_indices = cam_images["points_in_img_indices"]

            for i in range(len(points_in_img)):
                _, intersec_indices_1, intersec_indices_2 = np.intersect1d(
                    points_in_img_indices[i],
                    cam_frustum_indices,
                    assume_unique=True,
                    return_indices=True,
                )

                points_in_img[i] = points_in_img[i][intersec_indices_1]
                points_in_img_indices[i] = intersec_indices_2

        # filter unlabelled points
        if self._unlabelled_filter:
            labelled_indices = np.where(semantic_labels != 0)
            xyz = xyz[labelled_indices]
            semantic_labels = semantic_labels[labelled_indices]
            instance_labels = instance_labels[labelled_indices]
            intensity = intensity[labelled_indices]

        return (xyz, voxel_size, semantic_labels, instance_labels, intensity, semantic_embeddings, cam_images)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)

    def test_SemanticKITTIDataset(semanticKITTIPath: Path):
        dataset = SemanticKITTIDataset(semanticKITTIPath, split="train")
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset config: {dataset.config}")
        dataset_iter = iter(dataset)
        print(f"Dataset item: {next(dataset_iter)}")
        print(f"Dataset item: {next(dataset_iter)}")

    tyro.cli(test_SemanticKITTIDataset)
