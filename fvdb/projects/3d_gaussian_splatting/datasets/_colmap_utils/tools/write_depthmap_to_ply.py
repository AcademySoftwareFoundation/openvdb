# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import sys

sys.path.append("..")

import os

import imageio
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage.interpolation import zoom

from .. import SceneManager

# -------------------------------------------------------------------------------


def main(args):
    suffix = ".photometric.bin" if args.photometric else ".geometric.bin"

    image_file = os.path.join(args.dense_folder, "images", args.image_filename)
    depth_file = os.path.join(args.dense_folder, args.stereo_folder, "depth_maps", args.image_filename + suffix)
    if args.save_normals:
        normals_file = os.path.join(args.dense_folder, args.stereo_folder, "normal_maps", args.image_filename + suffix)

    # load camera intrinsics from the COLMAP reconstruction
    scene_manager = SceneManager(os.path.join(args.dense_folder, "sparse"))
    scene_manager.load_cameras()
    scene_manager.load_images()

    image_id, image = scene_manager.get_image_from_name(args.image_filename)
    camera = scene_manager.cameras[image.camera_id]
    rotation_camera_from_world = image.R()
    camera_center = image.C()

    # load image, depth map, and normal map
    image = imageio.imread(image_file)

    with open(depth_file, "rb") as fid:
        w = int("".join(iter(lambda: fid.read(1), "&")))
        h = int("".join(iter(lambda: fid.read(1), "&")))
        c = int("".join(iter(lambda: fid.read(1), "&")))
        depth_map = np.fromfile(fid, np.float32).reshape(h, w)
        if (h, w) != image.shape[:2]:
            depth_map = zoom(depth_map, (float(image.shape[0]) / h, float(image.shape[1]) / w), order=0)

    if args.save_normals:
        with open(normals_file, "rb") as fid:
            w = int("".join(iter(lambda: fid.read(1), "&")))
            h = int("".join(iter(lambda: fid.read(1), "&")))
            c = int("".join(iter(lambda: fid.read(1), "&")))
            normals = np.fromfile(fid, np.float32).reshape(c, h, w).transpose([1, 2, 0])
            if (h, w) != image.shape[:2]:
                normals = zoom(normals, (float(image.shape[0]) / h, float(image.shape[1]) / w, 1.0), order=0)

    if args.min_depth is not None:
        depth_map[depth_map < args.min_depth] = 0.0
    if args.max_depth is not None:
        depth_map[depth_map > args.max_depth] = 0.0

    # create 3D points
    # depth_map = np.minimum(depth_map, 100.)
    points3D = np.dstack(camera.get_image_grid() + [depth_map])
    points3D[:, :, :2] *= depth_map[:, :, np.newaxis]

    # save
    points3D = points3D.astype(np.float32).reshape(-1, 3)
    if args.save_normals:
        normals = normals.astype(np.float32).reshape(-1, 3)
    image = image.reshape(-1, 3)
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if args.world_space:
        points3D = points3D.dot(rotation_camera_from_world) + camera_center
        if args.save_normals:
            normals = normals.dot(rotation_camera_from_world)

    if args.save_normals:
        vertices = np.rec.fromarrays(
            tuple(points3D.T) + tuple(normals.T) + tuple(image.T), names="x,y,z,nx,ny,nz,red,green,blue"
        )
    else:
        vertices = np.rec.fromarrays(tuple(points3D.T) + tuple(image.T), names="x,y,z,red,green,blue")
    vertices = PlyElement.describe(vertices, "vertex")
    PlyData([vertices]).write(args.output_filename)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dense_folder", type=str)
    parser.add_argument("image_filename", type=str)
    parser.add_argument("output_filename", type=str)

    parser.add_argument(
        "--photometric", default=False, action="store_true", help="use photometric depthmap instead of geometric"
    )

    parser.add_argument(
        "--world_space",
        default=False,
        action="store_true",
        help="apply the camera->world extrinsic transformation to the result",
    )

    parser.add_argument(
        "--save_normals",
        default=False,
        action="store_true",
        help="load the estimated normal map and save as part of the PLY",
    )

    parser.add_argument(
        "--stereo_folder",
        type=str,
        default="stereo",
        help="folder in the dense workspace containing depth and normal maps",
    )

    parser.add_argument(
        "--min_depth", type=float, default=None, help="set pixels with depth less than this value to zero depth"
    )

    parser.add_argument(
        "--max_depth", type=float, default=None, help="set pixels with depth greater than this value to zero depth"
    )

    args = parser.parse_args()

    main(args)
