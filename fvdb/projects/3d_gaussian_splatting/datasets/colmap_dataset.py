# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.utils.data
from pyproj import Transformer

from ._colmap_utils import Camera, SceneManager


class ColmapCameraMetadata:
    def __init__(self, cam: Camera, image_downsample_factor: int = 1):

        img_w, img_h = (cam.width // image_downsample_factor, cam.height // image_downsample_factor)

        # camera intrinsics assuming a perspective projection model
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        proj_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        proj_mat[:2, :] /= image_downsample_factor

        # If not using perspective projection, store the distortion parameters and camera type
        # so we can apply undistortion later
        distortion_params = ColmapCameraMetadata._distortion_params_from_camera_type(cam)

        if len(distortion_params) != 0:
            undistorted_proj_mat, undistort_roi = cv2.getOptimalNewCameraMatrix(
                proj_mat, distortion_params, (img_w, img_h), 0
            )
            undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(
                proj_mat, distortion_params, None, undistorted_proj_mat, (img_w, img_h), cv2.CV_32FC1  # type: ignore
            )

            self.projection_matrix = undistorted_proj_mat
            self.undistort_roi = undistort_roi
            self.undistort_map_x = undistort_map_x
            self.undistort_map_y = undistort_map_y
        else:
            self.projection_matrix = proj_mat
            self.undistort_map_x = None
            self.undistort_map_y = None
            self.undistort_roi = None

        self.fx = self.projection_matrix[0, 0]
        self.fy = self.projection_matrix[1, 1]
        self.cx = self.projection_matrix[0, 2]
        self.cy = self.projection_matrix[1, 2]
        self.fovx = ColmapCameraMetadata._focal2fov(self.fx, img_w)
        self.fovy = ColmapCameraMetadata._focal2fov(self.fy, img_h)
        self.width = img_w
        self.height = img_h
        self.camera_type = cam.camera_type
        self.aspect = self.width / self.height

    @staticmethod
    def _focal2fov(focal: float, pixels: float) -> float:
        return 2 * np.arctan(pixels / (2 * focal))

    @staticmethod
    def _distortion_params_from_camera_type(cam: Camera) -> np.ndarray:
        """
        Get distotion model parameters (to use with cv2.initUndistortRectifyMap) from the specified camera type.
        We store these so we can distort images from non pinhole camera models and use a pinhole camera model.
        """
        if cam.camera_type == 0 or cam.camera_type == "SIMPLE_PINHOLE":
            return np.empty(0, dtype=np.float32)
        elif cam.camera_type == 1 or cam.camera_type == "PINHOLE":
            return np.empty(0, dtype=np.float32)
        elif cam.camera_type == 2 or cam.camera_type == "SIMPLE_RADIAL":
            return np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
        elif cam.camera_type == 3 or cam.camera_type == "RADIAL":
            return np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
        elif cam.camera_type == 4 or cam.camera_type == "OPENCV":
            return np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
        elif cam.camera_type == 5 or cam.camera_type == "OPENCV_FISHEYE":
            raise ValueError("Fisheye not currently supported")
            return np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
        else:
            raise ValueError(f"Unknown camera type {cam.camera_type}")

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.undistort_map_x is not None and self.undistort_map_y is not None:
            image_remap = cv2.remap(image, self.undistort_map_x, self.undistort_map_y, interpolation=cv2.INTER_LINEAR)
            assert self.undistort_roi is not None
            x, y, w, h = self.undistort_roi
            return image_remap[y : y + h, x : x + w]
        else:
            return image


class ColmapImageMetadata:
    def __init__(
        self,
        world_to_cam_mat: np.ndarray,
        cam_to_world_mat: np.ndarray,
        camera_metadata: ColmapCameraMetadata,
        camera_id: int,
        image_path: str,
        mask_path: str,
        point_indices: np.ndarray,
    ):
        self.world_to_cam_mat = world_to_cam_mat
        self.cam_to_world_mat = cam_to_world_mat
        self.camera_id = camera_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.point_indices = point_indices
        self.camera_metadata = camera_metadata

    @property
    def lookat(self):
        """Get the camera lookat vector."""
        return self.cam_to_world_mat[:3, 2]

    @property
    def origin(self):
        """Get the camera origin."""
        return self.cam_to_world_mat[:3, 3]

    @property
    def up(self):
        """Get the camera up vector."""
        return -self.cam_to_world_mat[:3, 1]

    @property
    def right(self):
        """Get the camera right vector."""
        return self.cam_to_world_mat[:3, 0]


class ColmapScene:
    """
    Class representing a COLMAP scene which consists of a set of posed images taken from a set of cameras along with
    a set of 3D points in the scene. The scene is normalized to a canonical coordinate system using one of several
    heuristics.
    """

    def __init__(self, dataset_path: str, image_downsample_factor: int = 1, normalization_type: str = "pca"):
        self.dataset_path = dataset_path
        self.image_downsample_factor = image_downsample_factor
        self.normalization_type = normalization_type
        self.utm_proj4 = None

        valid_normalization_types = {"none", "pca", "ecef2enu", "similarity"}
        if normalization_type not in valid_normalization_types:
            raise ValueError(
                f"Unknown normalization type {normalization_type}. Must be one of {valid_normalization_types}"
            )

        colmap_dir = os.path.join(dataset_path, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(dataset_path, "sparse")
        if not os.path.exists(colmap_dir):
            raise FileNotFoundError(f"COLMAP directory {colmap_dir} does not exist.")

        mask_path = os.path.join(dataset_path, "image_masks")
        if not os.path.exists(mask_path):
            mask_path = None

        scene_manager = SceneManager(colmap_dir)
        scene_manager.load_cameras()
        scene_manager.load_images()
        scene_manager.load_points3D()

        points: np.ndarray = scene_manager.points3D  # type: ignore
        self.normalization_transform = self._compute_normalization_transform(scene_manager, normalization_type)
        self.points = ColmapScene._transform_point_cloud(self.normalization_transform, points).astype(
            np.float32
        )  # (num_points, 3)
        self.points_err = scene_manager.point3D_errors.astype(np.float32)  # type: ignore
        self.points_rgb = scene_manager.point3D_colors.astype(np.uint8)  # type: ignore

        self.cameras, self.images, self.scene_scale = self._compute_image_and_camera_metadata(
            dataset_path, scene_manager, self.normalization_transform, image_downsample_factor, mask_path
        )

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @staticmethod
    def _transform_point_cloud(matrix, points):
        """
        Transform points using an SE(3) matrix.

        Args:
            matrix: 4x4 SE(3) matrix
            points: Nx3 array of points

        Returns:
            transformed_points: An Nx3 array of transformed points
        """
        assert matrix.shape == (4, 4)
        assert len(points.shape) == 2 and points.shape[1] == 3
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    @staticmethod
    def _transform_cam_to_world_matrix(matrix, cam_to_world_mats):
        """
        Apply a SE(3) transformation defined as a 4x4 matrix to a set of camera -> world transformation matrices.

        Args:
            matrix: 4x4 SE(3) matrix
            cam_to_world_mats: Nx4x4 array of camera-to-world matrices

        Returns:
            transformed_cam_to_world_mats: An [N,4,4]-shaped array of transformed camera-to-world matrices
        """
        assert matrix.shape == (4, 4)
        assert len(cam_to_world_mats.shape) == 3 and cam_to_world_mats.shape[1:] == (4, 4)
        cam_to_world_mats = np.einsum("nij, ki -> nkj", cam_to_world_mats, matrix)
        scaling = np.linalg.norm(cam_to_world_mats[:, 0, :3], axis=1)
        cam_to_world_mats[:, :3, :3] = cam_to_world_mats[:, :3, :3] / scaling[:, None, None]
        return cam_to_world_mats

    @staticmethod
    def _geo_ecef2enu_normalization_transform(point_cloud, londeg, latdeg, xorigin, yorigin, zorigin):
        """
        Compute a transformation matrix that converts ECEF coordinates to ENU coordinates.

        Args:
            point_cloud: Nx3 array of points in ECEF coordinates
            londeg: Longitude in degrees
            latdeg: Latitude in degrees
            xorigin: X coordinate of the origin in ECEF coordinates
            yorigin: Y coordinate of the origin in ECEF coordinates
            zorigin: Z coordinate of the origin in ECEF coordinates

        Returns:
            transform: 4x4 transformation matrix
        """
        # ECEF to ENU rotation matrix
        lon = np.deg2rad(londeg)
        lat = np.deg2rad(latdeg)
        rot = np.array(
            [
                [-np.sin(lon), np.cos(lon), 0.0],
                [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
                [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
            ]
        )

        tvec = np.array([xorigin, yorigin, zorigin])
        # Create SE(3) matrix (4x4 transformation matrix)
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = -rot @ tvec

        return transform

    @staticmethod
    def _pca_normalization_transform(point_cloud):
        """
        Compute a transormation matrix that normalizes the scene using PCA on a set of input points

        Args:
            point_cloud: Nx3 array of points

        Returns:
            transform: 4x4 transformation matrix
        """
        # Compute centroid
        centroid = np.median(point_cloud, axis=0)

        # Translate point cloud to centroid
        translated_point_cloud = point_cloud - centroid

        # Compute covariance matrix
        covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
        # is the principal axis with the smallest eigenvalue.
        sort_indices = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, sort_indices]

        # Check orientation of eigenvectors. If the determinant of the eigenvectors is
        # negative, then we need to flip the sign of one of the eigenvectors.
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1

        # Create rotation matrix
        rotation_matrix = eigenvectors.T

        # Create SE(3) matrix (4x4 transformation matrix)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = -rotation_matrix @ centroid

        return transform

    @staticmethod
    def _camera_similarity_normalization_transform(c2w, strict_scaling=False, center_method="focus"):
        """
        Get a similarity transformation to normalize a scene given its camera -> world transformations

        Args:
            c2w: A set of camera -> world transformations [R|t] (N, 4, 4)
            strict_scaling: If set to true, use the maximum distance to any camera to rescale the scene
                            which may not be that robust. If false, use the median
            center_method: If set to 'focus' use the focus of the scene to center the cameras
                           If set to 'poses' use the center of the camera positions to center the cameras

        Returns:
            transform: A 4x4 normalization transform (4,4)
        """
        t = c2w[:, :3, 3]
        R = c2w[:, :3, :3]

        # (1) Rotate the world so that z+ is the up axis
        # we estimate the up axis by averaging the camera up axes
        ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
        world_up = np.mean(ups, axis=0)
        world_up /= np.linalg.norm(world_up)

        up_camspace = np.array([0.0, -1.0, 0.0])
        c = (up_camspace * world_up).sum()
        cross = np.cross(world_up, up_camspace)
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        if c > -1:
            R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
        else:
            # In the unlikely case the original data has y+ up axis,
            # rotate 180-deg about x axis
            R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        #  R_align = np.eye(3) # DEBUG
        R = R_align @ R
        fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
        t = (R_align @ t[..., None])[..., 0]

        # (2) Recenter the scene.
        if center_method == "focus":
            # find the closest point to the origin for each camera's center ray
            nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
            translate = -np.median(nearest, axis=0)
        elif center_method == "poses":
            # use center of the camera positions
            translate = -np.median(t, axis=0)
        else:
            raise ValueError(f"Unknown center_method {center_method}")

        transform = np.eye(4)
        transform[:3, 3] = translate
        transform[:3, :3] = R_align

        # (3) Rescale the scene using camera distances
        scale_fn = np.max if strict_scaling else np.median
        scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
        transform[:3, :] *= scale

        return transform

    @staticmethod
    def _compute_normalization_transform(scene_manager: SceneManager, normalization_type: str):
        """
        Computes an affine transformatrion matrix which normalizes a scene using one of several heuristics

        Args:
            scene_manager: SceneManager object containing camera and point data
            normalization_type: Type of normalization to apply. Options are "pca", "similarity", "ecef2enu", or "none".
        Returns:
            normalization_transform: 4x4 transformation matrix for normalizing the scene
        """
        points: np.ndarray = scene_manager.points3D  # type: ignore

        # Normalize the world space.
        if normalization_type == "pca":
            normalization_transform = ColmapScene._pca_normalization_transform(points)
        elif normalization_type == "ecef2enu":
            centroid = np.median(points, axis=0)
            tform_ecef2lonlat = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
            pt_lonlat = tform_ecef2lonlat.transform(centroid[0], centroid[1], centroid[2])

            normalization_transform = ColmapScene._geo_ecef2enu_normalization_transform(
                points, pt_lonlat[0], pt_lonlat[1], centroid[0], centroid[1], centroid[2]
            )
        elif normalization_type == "similarity":
            imdata = scene_manager.images
            world_to_cam_mats = []
            for k in imdata:
                world_to_cam_mats.append(imdata[k].world_to_cam_matrix())
            world_to_cam_mats = np.stack(world_to_cam_mats, axis=0)
            cam_to_world_mats = np.linalg.inv(world_to_cam_mats)
            normalization_transform = ColmapScene._camera_similarity_normalization_transform(cam_to_world_mats)
        elif normalization_type == "none":
            normalization_transform = np.eye(4)
        else:
            raise RuntimeError(f"Unknown normalization type {normalization_type}")

        return normalization_transform

    @staticmethod
    def _compute_image_and_camera_metadata(
        data_path: str,
        scene_manager: SceneManager,
        normalization_transform: np.ndarray,
        image_downsample_factor: int = 1,
        mask_path: str | None = None,
    ):
        imdata = scene_manager.images
        camdata = scene_manager.cameras

        # Loop over every image in the dataset and compute its view transform, camera id, and image name.
        # We populate numpy arrays for the transforms so we can invert using a vectorized call which is
        # much faster than inverting each image in a loop.
        world_to_cam_mats = []
        camera_ids = []
        image_filenames = []
        mask_paths = []
        loaded_cameras = dict()
        for k in imdata:
            im = imdata[k]
            world_to_cam_mats.append(im.world_to_cam_matrix())
            camera_ids.append(im.camera_id)
            image_filenames.append(im.name)

            if mask_path is not None:
                image_mask_path = os.path.join(mask_path, os.path.basename(im.name))
                if os.path.exists(image_mask_path):
                    mask_paths.append(image_mask_path)
                else:
                    raise FileNotFoundError("missing mask: " + image_mask_path)
            else:
                mask_paths.append("")

            if im.camera_id not in loaded_cameras:
                loaded_cameras[im.camera_id] = ColmapCameraMetadata(
                    camdata[im.camera_id], image_downsample_factor=image_downsample_factor
                )

        world_to_cam_mats = np.stack(world_to_cam_mats, axis=0)
        cam_to_world_mats = ColmapScene._transform_cam_to_world_matrix(
            normalization_transform, np.linalg.inv(world_to_cam_mats)
        )
        world_to_cam_mats = np.linalg.inv(cam_to_world_mats)

        # Most papers use train/test splits based on sorted images so sort the images here
        sort_indices = np.argsort(image_filenames)
        image_filenames = [image_filenames[i] for i in sort_indices]
        if len(mask_paths) > 0:
            mask_paths = [mask_paths[i] for i in sort_indices]
        world_to_cam_mats = world_to_cam_mats[sort_indices]
        cam_to_world_mats = cam_to_world_mats[sort_indices]
        camera_ids = [camera_ids[i] for i in sort_indices]

        # Get the absolute path of each (maybe rescaled) image in the dataset
        rescale_dir_suffix = f"_{image_downsample_factor}" if image_downsample_factor > 1 else ""
        colmap_images_path = os.path.join(data_path, "images")
        rescaled_images_path = os.path.join(data_path, "images" + rescale_dir_suffix)
        if not os.path.exists(rescaled_images_path):
            raise ValueError("Rescaled image directory does not exist.")
        # We allow you to include your own downsampled image directory e.g. images_4 for 4x downsampled.
        # The files in these may follow a different naming convention than the original images.
        # This is okay as long as they sort the same way e.g. images/image_001.jpg and images_4/im_001.jpg correspond
        # to the same image at different scales.
        # This code will match the files in the two directories by sorting them and using their sort order as
        # correspondence
        full_res_sorted_filenames = sorted(ColmapScene._get_rel_paths(colmap_images_path))
        rescaled_sorted_filenames = sorted(ColmapScene._get_rel_paths(rescaled_images_path))
        full_scale_to_rescaled = dict(zip(full_res_sorted_filenames, rescaled_sorted_filenames))
        image_paths = [os.path.join(rescaled_images_path, full_scale_to_rescaled[f]) for f in image_filenames]

        # Compute the set of 3D points visible in each image
        image_id_to_name = {v: k for k, v in scene_manager.name_to_image_id.items()}
        point_indices = dict()  # Map from image names to point indices
        image_id_to_name = {v: k for k, v in scene_manager.name_to_image_id.items()}
        # For each point, get the images that see it
        for point_id, data in scene_manager.point3D_id_to_images.items():
            # For each image that sees this point, add the index of the point
            # to a list of points corresponding to that image
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = scene_manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}

        loaded_images = [
            ColmapImageMetadata(
                world_to_cam_mat=world_to_cam_mats[i].copy(),
                cam_to_world_mat=cam_to_world_mats[i].copy(),
                camera_id=camera_ids[i],
                camera_metadata=loaded_cameras[camera_ids[i]],
                image_path=image_paths[i],
                mask_path=mask_paths[i],
                point_indices=point_indices[image_filenames[i]].copy(),
            )
            for i in range(len(image_filenames))
        ]

        # Calculate the maximum distance from the average point of the scene to any point
        # which defines a notion of scene scale
        camera_locations = cam_to_world_mats[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)

        # points: np.ndarray = scene_manager.points3D  # type: ignore
        # points_err: np.ndarray = scene_manager.point3D_errors.astype(np.float32)  # type: ignore
        # points_rgb: np.ndarray = scene_manager.point3D_colors.astype(np.uint8)  # type: ignore
        # point_indices = dict()

        # import polyscope as ps

        # ps.init()
        # cpts = ps.register_point_cloud("colmap_points", points=points, radius=0.00025)
        # if len(points_rgb) > 0:
        #     cpts.add_color_quantity("colors", points_rgb / 255.0, enabled=True)

        # for i, img in enumerate(loaded_images):
        #     if i % 80 != 0:
        #         continue
        #     cam = loaded_cameras[img.camera_id]
        #     intrinsics = ps.CameraIntrinsics(fov_vertical_deg=np.rad2deg(cam.fovy), aspect=cam.aspect)
        #     extrinsics = ps.CameraExtrinsics(root=img.origin, look_dir=img.lookat, up_dir=img.up)
        #     params = ps.CameraParameters(intrinsics, extrinsics)
        #     cam = ps.register_camera_view(f"cam_{i}", params)
        #     cam.set_widget_focal_length(0.0075)
        #     cam.set_widget_thickness(0.0075)
        #     visible_pts = points[img.point_indices]
        #     loaded_img = imageio.imread(img.image_path).astype(np.float32) / 255.0
        #     cam.add_color_image_quantity("image", loaded_img, enabled=True, show_in_camera_billboard=True)
        #     color = np.random.rand(3) / 2 + 0.3
        #     ps.register_point_cloud(f"cam_{i}_points", points=visible_pts, radius=0.0005, color=color)
        #     ps.register_point_cloud(f"cam_{i}_origin", points=img.origin[None], radius=0.001, color=color)

        # ps.set_ground_plane_mode("none")
        # ps.set_up_dir("neg_y_up")
        # ps.show()

        return loaded_cameras, loaded_images, scene_scale

    @staticmethod
    def _get_rel_paths(path_dir: str) -> List[str]:
        """Recursively get relative paths of files in a directory."""
        paths = []
        for dp, dn, fn in os.walk(path_dir):
            for f in fn:
                paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
        return paths

    @staticmethod
    def _old_normalization(camtoworlds, points=None):
        raise DeprecationWarning("This function is deprecated and will be removed in the future.")
        # TODO: Remove this
        T1 = ColmapScene._camera_similarity_normalization_transform(camtoworlds)
        camtoworlds = ColmapScene._transform_cam_to_world_matrix(T1, camtoworlds)
        if points is not None:
            points = ColmapScene._transform_point_cloud(T1, points)
            T2 = ColmapScene._pca_normalization_transform(points)
            camtoworlds = ColmapScene._transform_cam_to_world_matrix(T2, camtoworlds)
            points = ColmapScene._transform_point_cloud(T2, points)
            return camtoworlds, points, T2 @ T1
        else:
            return camtoworlds, T1


class ColmapDataset(torch.utils.data.Dataset):
    __colmap_scene_cache: Dict[str, ColmapScene] = dict()

    def __colmap_scene_unique_id(self):
        """
        Return a unique ID for this particular colmap scene
        """
        return f"COLMAP_SCENE_{self.dataset_path}_{self.image_downsample_factor}_{self.normalization_type}"

    def __init__(
        self,
        dataset_path: str,
        normalization_type: str = "pca",
        image_downsample_factor: int = 1,
        test_every: int = 100,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.dataset_path = dataset_path
        self.image_downsample_factor = image_downsample_factor
        self.normalization_type = normalization_type

        # Loading colmap scenes can be very slow. We will do it once and re-use the same scene for
        # train/test splits.
        scene_id = self.__colmap_scene_unique_id()
        if scene_id not in ColmapDataset.__colmap_scene_cache:
            ColmapDataset.__colmap_scene_cache[scene_id] = ColmapScene(
                dataset_path=dataset_path,
                normalization_type=normalization_type,
                image_downsample_factor=image_downsample_factor,
            )

        self.colmap_scene = ColmapDataset.__colmap_scene_cache[scene_id]
        self.test_every = test_every
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        indices = np.arange(self.colmap_scene.num_images)
        if self.split == "train":
            self.indices = indices[indices % self.test_every != 0]
        elif self.split == "test":
            self.indices = indices[indices % self.test_every == 0]
        elif self.split == "all":
            self.indices = indices
        else:
            raise ValueError(f"Split must be one of 'train', 'test', or 'all'. Got {self.split}.")

    @property
    def scene_scale(self) -> float:
        return self.colmap_scene.scene_scale

    @property
    def points(self) -> np.ndarray:
        return self.colmap_scene.points

    @property
    def points_rgb(self) -> np.ndarray:
        return self.colmap_scene.points_rgb

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]

        image_meta: ColmapImageMetadata = self.colmap_scene.images[index]
        camera_meta: ColmapCameraMetadata = image_meta.camera_metadata
        image = imageio.imread(image_meta.image_path)[..., :3]
        projection_mat = camera_meta.projection_matrix.copy()  # undistorted K
        image = camera_meta.undistort_image(image)
        cam_to_world_mat = image_meta.cam_to_world_mat.copy()
        world_to_cam_mat = image_meta.world_to_cam_mat.copy()

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            projection_mat[0, 2] -= x
            projection_mat[1, 2] -= y

        data = {
            "K": torch.from_numpy(projection_mat).float(),
            "camtoworld": torch.from_numpy(cam_to_world_mat).float(),
            "worldtocam": torch.from_numpy(world_to_cam_mat).float(),
            "image": image,
            "image_id": item,  # the index of the image in the dataset
            "image_path": image_meta.image_path,
        }

        # If you passed in masks, we'll set set these in the data dictionary
        if image_meta.mask_path != "":
            mask = imageio.imread(image_meta.mask_path)[..., :3]
            mask = mask < 127
            data["mask_path"] = image_meta.mask_path
            data["mask"] = mask

        # If you asked to load depths, we'll load the depths of visible colmap points
        if self.load_depths:
            # projected points to image plane to get depths
            points_world = self.colmap_scene.points[image_meta.point_indices]
            points_cam = (world_to_cam_mat[:3, :3] @ points_world.T + world_to_cam_mat[:3, 3:4]).T
            points_proj = (projection_mat @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
        return data


__all__ = ["ColmapDataset"]

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    dataset = ColmapDataset(
        dataset_path=args.data_dir, image_downsample_factor=args.factor, test_every=8, split="train", load_depths=True
    )
    print(f"Dataset: {len(dataset)} images.")

    imsize = None
    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):  # type: ignore
        image = data["image"].numpy().astype(np.uint8)
        # Make sure all images we write are the same size. We use the first image to determine the size of the video.
        # This is done because some images have slightly different sizes due to undistortion.
        imsize = image.shape if imsize is None else imsize
        if image.shape != imsize:
            new_image = np.zeros(imsize, dtype=np.uint8)
            new_image[: image.shape[0], : image.shape[1]] = image[: imsize[0], : imsize[1]]
            image = new_image
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:  # type: ignore
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
