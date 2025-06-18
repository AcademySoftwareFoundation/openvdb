# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import sys

sys.path.append("..")

from .. import DualQuaternion, Image, SceneManager

# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load()

    image_ids = map(scene_manager.get_image_from_name, iter(lambda: sys.stdin.readline().strip(), ""))
    scene_manager.delete_images(image_ids)

    scene_manager.save(args.output_folder)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deletes images (filenames read from stdin) from a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")

    args = parser.parse_args()

    main(args)
