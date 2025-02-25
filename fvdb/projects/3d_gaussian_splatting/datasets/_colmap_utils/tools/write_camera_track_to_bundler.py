# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import sys

sys.path.append("..")

from .. import SceneManager

# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load_cameras()
    scene_manager.load_images()

    if args.sort:
        images = sorted(scene_manager.images.itervalues(), key=lambda im: im.name)
    else:
        images = scene_manager.images.values()

    fid = open(args.output_file, "w")
    fid_filenames = open(args.output_file + ".list.txt", "w")

    print >> fid, "# Bundle file v0.3"
    print >> fid, len(images), 0

    for image in images:
        print >> fid_filenames, image.name
        camera = scene_manager.cameras[image.camera_id]
        print >> fid, 0.5 * (camera.fx + camera.fy), 0, 0
        R, t = image.R(), image.t
        print >> fid, R[0, 0], R[0, 1], R[0, 2]
        print >> fid, -R[1, 0], -R[1, 1], -R[1, 2]
        print >> fid, -R[2, 0], -R[2, 1], -R[2, 2]
        print >> fid, t[0], -t[1], -t[2]

    fid.close()
    fid_filenames.close()


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Saves the camera positions in the Bundler format. Note " "that 3D points are not saved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_file")

    parser.add_argument("--sort", default=False, action="store_true", help="sort the images by their filename")

    args = parser.parse_args()

    main(args)
