# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Set

import imageio.v2 as imageio
import numpy as np
import tqdm
import tyro
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from shapely import get_coordinates
from shapely.geometry import Polygon as shp_poly

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datasets import ColmapDataset
from datasets.colmap_dataset import ColmapScene


def partition_points_xy_grid(points, point_inds, xchunks: int, ychunks: int, pad_percent=0.1):
    """
    Creates a grid in the x-y coordinate plane to use to split up colmap run

    Args:
        points: array of 3D points to split up [npoints, 3]
        point_inds: colmap point indicies [npoints]
        xchunks: how many times to split in x
        ychunks: how many times to split in y
        pad_percent: overlap each chunk by this percent

    Returns:
        cluster_inds: list of point_inds contained in each cluster
        grid_bounds: list of x,y,z bounds for each cluster (minx, miny, minz, maxx, maxy, maxz)
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    xv = np.linspace(xmin, xmax, xchunks + 1)

    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    yv = np.linspace(ymin, ymax, ychunks + 1)

    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])

    cluster_inds = []
    grid_bounds = []
    for i in range(xv.size - 1):
        lower_boundx = xv[i]
        upper_boundx = xv[i + 1]
        padx = (upper_boundx - lower_boundx) * pad_percent
        for j in range(yv.size - 1):
            lower_boundy = yv[j]
            upper_boundy = yv[j + 1]
            pady = (upper_boundy - lower_boundy) * pad_percent
            good_inds = np.logical_and.reduce(
                [
                    points[:, 0] > lower_boundx - padx,
                    points[:, 0] < upper_boundx + padx,
                    points[:, 1] > lower_boundy - pady,
                    points[:, 1] < upper_boundy + pady,
                    points[:, 2] > zmin,
                    points[:, 2] < zmax,
                ]
            )
            cluster_inds.append(set(point_inds[good_inds]))
            grid_bounds.append([lower_boundx, lower_boundy, zmin, upper_boundx, upper_boundy, zmax])

    return cluster_inds, grid_bounds


def pointids2imageids(colmap_data: ColmapDataset, point_ids: Set):
    """
    Find the image ids that see the give list of point ids

    Args:
        colmap_data: ColmapDataset object
        point_ids: list of point ids to use in finding image ids

    Returns:
        A list of images ids for each given point ids
    """
    im_ids = set()
    for im_id, im in enumerate(colmap_data.colmap_scene.images):
        im_point_ids = set(im.point_indices.tolist())
        if len(im_point_ids.intersection(point_ids)) > 0:
            im_ids.add(im_id)
    return list(im_ids)


def polygon_to_mask(polygon, image_size):
    """
    Generates a boolean mask from a polygon.

    Args:
        polygon: A list of (x, y) tuples representing the polygon vertices.
        image_size: A tuple (height, width) representing the size of the image.

    Returns:
        A 2D boolean NumPy array representing the mask.
    """

    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
    pixels = np.vstack((x.flatten(), y.flatten())).T
    mask = polygon.contains_points(pixels).reshape(image_size)
    return mask


def save_mask_file(
    cluster_index,
    imIds_per_cluster,
    points_per_cluster,
    all_points,
    nimages,
    all_img_sizes,
    all_intrinsics,
    all_cam2worlds,
    factor,
    all_image_names,
    write_path,
):
    """
    Helper function to allow for multiprocessing to pickle.
    """
    print("starting masks for cluster: " + str(cluster_index))
    image_ids = list(imIds_per_cluster[cluster_index])
    pnt_ids = np.array(list(points_per_cluster[cluster_index]))
    points = all_points[pnt_ids, :]

    # make masks based on extent of padded points, not smaller bounds so we train well ok pad region and clip is clean
    minx, miny, minz, maxx, maxy, maxz = np.concatenate((np.min(points, axis=0), np.max(points, axis=0))).tolist()
    # minx, miny, minz, maxx, maxy, maxz = bounds_per_cluster[cluster_index]
    cube_bounds = np.array(
        [
            [minx, miny, minz],
            [maxx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [minx, maxy, maxz],
            [maxx, maxy, maxz],
        ]
    )

    for im_index in tqdm.tqdm(range(nimages)):

        width, height = all_img_sizes[im_index]
        if not im_index in image_ids:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            camtoworlds = all_cam2worlds[im_index]
            w2c = np.linalg.inv(camtoworlds.squeeze())

            fx = all_intrinsics[im_index][0]
            cx = all_intrinsics[im_index][1]
            fy = all_intrinsics[im_index][2]
            cy = all_intrinsics[im_index][3]

            bounds_cam = ColmapScene._transform_point_cloud(w2c, cube_bounds)
            bounds_pix = []
            for i in range(bounds_cam.shape[0]):
                x, y, z = list(bounds_cam[i, :].squeeze())
                rz = 1.0 / z
                mean2d = [fx * x * rz + cx, fy * y * rz + cy]
                bounds_pix.append(mean2d)
            bounds_pix = np.array(bounds_pix)

            hull = ConvexHull(bounds_pix)
            bounds_pix = bounds_pix[hull.vertices]

            bounds_pix[bounds_pix < 0] = 0
            bounds_pix[:, 0] = np.clip(bounds_pix[:, 0], 0, width - 1)
            bounds_pix[:, 1] = np.clip(bounds_pix[:, 1], 0, height - 1)

            bounds_poly = shp_poly(bounds_pix)

            image_bounds = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]])
            bounds_image_poly = shp_poly(image_bounds)

            keep_polygon_shp = bounds_poly.intersection(bounds_image_poly)
            keep_polygon_coords = get_coordinates(keep_polygon_shp)
            keep_polygon_mat = Polygon(keep_polygon_coords)

            boolean_mask = polygon_to_mask(keep_polygon_mat, (height, width))
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[boolean_mask] = 255

        image_dir_suffix = f"_{factor}"
        image_name = all_image_names[im_index]

        mask_path = os.path.join(write_path, str(cluster_index).zfill(4) + "_images" + image_dir_suffix + "_masks")
        mask_fname = os.path.join(mask_path, image_name + ".png")
        os.makedirs(os.path.dirname(mask_fname), exist_ok=True)
        if os.path.exists(mask_fname):
            os.remove(mask_fname)

        imageio.imwrite(mask_fname, mask, format="jpg")


def main(data_path: str, factor: int, ncores: int, nx_splits: int, ny_splits: int, normalization_type: str):
    """
    Tool to split a colmap run at provided path into different x-y clusters
    Args:
        data_path: Path to the colmap run
        factor: image downsample factor
        ncores: number of cores to run during mask generation
        xchunks: how many times to split in x
        ychunks: how many times to split in y
        normalization_type: what kind of normalization to use when loading colmap dataset
    Outputs:
        Outputs data products into new directory inside data_path based on number of clusters:
            [cluster]_cluster_point_ids.csv: contains list of point ids for each cluster
            [cluster]_cluster_image_ids.csv: contains list of image ids for each cluster
            [cluster]_cluster_bounds.csv: contains the x,y,z bounds for each cluster
            [cluster]_images_[factor]_masks: directory containing image masks to use for each cluster

    """
    dataset: ColmapDataset = ColmapDataset(
        dataset_path=data_path,
        image_downsample_factor=factor,
        normalization_type=normalization_type,
        split="all",
        percentile_filter_points=[98, 98, 98, 98, 99, 99],
    )

    write_path = os.path.join(data_path, str(nx_splits * ny_splits) + "_cluster_splits")
    os.makedirs(write_path, exist_ok=True)
    print("created output directory", write_path)

    # print("percentile clean of points")
    # points_filter_map, points_filter_inds = filter_points_percentile(parser.points)
    # points_filtered = parser.points[points_filter_map,:]

    print("cluster sparse points")
    points_per_cluster, bounds_per_cluster = partition_points_xy_grid(
        dataset.points, dataset.point_indices, nx_splits, ny_splits
    )

    print("extract which images see cleaned points")
    imIds_per_cluster = [pointids2imageids(dataset, point_ids) for point_ids in points_per_cluster]

    print("write cluster data files")

    for i, point_ids in enumerate(points_per_cluster):
        cluster_name = str(i).zfill(4) + "_cluster_point_ids.csv"
        print(f"writing point ids for cluster: {i} to: {cluster_name}")

        output_path = os.path.join(write_path, cluster_name)
        with open(output_path, "w") as fp:
            fp.write("pointId\n")
            [fp.write(str(ptid) + "\n") for ptid in point_ids]

    for i, cluster_ids in enumerate(imIds_per_cluster):
        cluster_name = str(i).zfill(4) + "_cluster_image_ids.csv"
        print(f"writing image ids for cluster: {i} to: {cluster_name}")

        output_path = os.path.join(write_path, cluster_name)
        with open(output_path, "w") as fp:
            fp.write("imageId\n")
            [fp.write(str(imid) + "\n") for imid in cluster_ids]

    for i, bounds in enumerate(bounds_per_cluster):
        cluster_name = str(i).zfill(4) + "_cluster_bounds.csv"
        print(f"writing bounds for cluster: {i} to: {cluster_name}")

        output_path = os.path.join(write_path, cluster_name)
        with open(output_path, "w") as fp:
            fp.write("xmin, ymin, zmin, xmax, ymax, zmax\n")
            row = ",".join([str(item) for item in bounds])
            fp.write(row + "\n")

    print("create image masks")
    all_points = dataset.colmap_scene.points
    nimages = dataset.colmap_scene.num_images
    all_img_sizes = [
        (
            dataset.colmap_scene.images[ind].camera_metadata.width,
            dataset.colmap_scene.images[ind].camera_metadata.height,
        )
        for ind in range(nimages)
    ]
    all_intrinsics = [
        (
            dataset.colmap_scene.images[ind].camera_metadata.fx,
            dataset.colmap_scene.images[ind].camera_metadata.cx,
            dataset.colmap_scene.images[ind].camera_metadata.fy,
            dataset.colmap_scene.images[ind].camera_metadata.cy,
        )
        for ind in range(nimages)
    ]
    all_cam2worlds = [dataset.colmap_scene.images[ind].cam_to_world_mat for ind in range(nimages)]
    all_image_names = [dataset.colmap_scene.images[ind].image_name for ind in range(nimages)]

    partial_func = partial(
        save_mask_file,
        imIds_per_cluster=imIds_per_cluster,
        points_per_cluster=points_per_cluster,
        all_points=all_points,
        nimages=nimages,
        all_img_sizes=all_img_sizes,
        all_intrinsics=all_intrinsics,
        all_cam2worlds=all_cam2worlds,
        factor=factor,
        all_image_names=all_image_names,
        write_path=write_path,
    )

    cluster_ids = list(range(len(imIds_per_cluster)))
    pool = Pool(ncores)
    for _ in pool.imap_unordered(partial_func, cluster_ids):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    tyro.cli(main)
