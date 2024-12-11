# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import itertools
import sys

sys.path.append("..")

from .. import Quaternion, SceneManager

# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load()

    with open(args.output_file, "w") as fid:
        fid.write("NVM_V3\n \n{:d}\n".format(len(scene_manager.images)))

        image_fmt_str = " {:.3f} " + 7 * "{:.7f} "
        for image_id, image in scene_manager.images.iteritems():
            camera = scene_manager.cameras[image.camera_id]
            f = 0.5 * (camera.fx + camera.fy)
            fid.write(args.image_name_prefix + image.name)
            fid.write(image_fmt_str.format(*((f,) + tuple(image.q.q) + tuple(image.C()))))
            if camera.distortion_func is None:
                fid.write("0 0\n")
            else:
                fid.write("{:.7f} 0\n".format(-camera.k1))

        image_id_to_idx = dict((image_id, i) for i, image_id in enumerate(scene_manager.images))

        fid.write("{:d}\n".format(len(scene_manager.points3D)))
        for i, point3D_id in enumerate(scene_manager.point3D_ids):
            fid.write("{:.7f} {:.7f} {:.7f} ".format(*scene_manager.points3D[i]))
            fid.write("{:d} {:d} {:d} ".format(*scene_manager.point3D_colors[i]))
            keypoints = [
                (image_id_to_idx[image_id], kp_idx) + tuple(scene_manager.images[image_id].points2D[kp_idx])
                for image_id, kp_idx in scene_manager.point3D_id_to_images[point3D_id]
            ]
            fid.write("{:d}".format(len(keypoints)))
            fid.write((len(keypoints) * " {:d} {:d} {:.3f} {:.3f}" + "\n").format(*itertools.chain(*keypoints)))


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Save a COLMAP reconstruction in the NVM format " "(http://ccwu.me/vsfm/doc.html#nvm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_file")

    parser.add_argument(
        "--image_name_prefix", type=str, default="", help="prefix image names with this string (e.g., 'images/')"
    )

    args = parser.parse_args()

    main(args)
