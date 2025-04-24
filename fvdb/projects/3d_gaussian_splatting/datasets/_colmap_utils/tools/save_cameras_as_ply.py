# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import sys

sys.path.append("..")

import numpy as np

from .. import SceneManager

# -------------------------------------------------------------------------------


# Saves the cameras as a mesh
#
# inputs:
# - ply_file: output file
# - images: ordered array of pycolmap Image objects
# - color: color string for the camera
# - scale: amount to shrink/grow the camera model
def save_camera_ply(ply_file, images, scale):
    points3D = scale * np.array(
        ((0.0, 0.0, 0.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0))
    )

    faces = np.array(((0, 2, 1), (0, 4, 2), (0, 3, 4), (0, 1, 3), (1, 2, 4), (1, 4, 3)))

    r = np.linspace(0, 255, len(images), dtype=np.uint8)
    g = 255 - r
    b = r - np.linspace(0, 128, len(images), dtype=np.uint8)
    color = np.column_stack((r, g, b))

    with open(ply_file, "w") as fid:
        print >> fid, "ply"
        print >> fid, "format ascii 1.0"
        print >> fid, "element vertex", len(points3D) * len(images)
        print >> fid, "property float x"
        print >> fid, "property float y"
        print >> fid, "property float z"
        print >> fid, "property uchar red"
        print >> fid, "property uchar green"
        print >> fid, "property uchar blue"
        print >> fid, "element face", len(faces) * len(images)
        print >> fid, "property list uchar int vertex_index"
        print >> fid, "end_header"

        for image, c in zip(images, color):
            for p3D in points3D.dot(image.R()) + image.C():
                print >> fid, p3D[0], p3D[1], p3D[2], c[0], c[1], c[2]

        for i in xrange(len(images)):
            for f in faces + len(points3D) * i:
                print >> fid, "3 {} {} {}".format(*f)


# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load_images()

    images = sorted(scene_manager.images.itervalues(), key=lambda image: image.name)

    save_camera_ply(args.output_file, images, args.scale)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Saves camera positions to a PLY for easy viewing outside "
        "of COLMAP. Currently, camera FoV is not reflected in the output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_file")

    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for the camera mesh.")

    args = parser.parse_args()

    main(args)
