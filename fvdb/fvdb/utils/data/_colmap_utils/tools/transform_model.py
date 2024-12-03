import sys

sys.path.append("..")

import numpy as np

from .. import Quaternion, SceneManager

# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load()

    # expect each line of input corresponds to one row
    P = np.array([map(float, sys.stdin.readline().strip().split()) for _ in xrange(3)])

    scene_manager.points3D[:] = scene_manager.points3D.dot(P[:, :3].T) + P[:, 3]

    # get rotation without any global scaling (assuming isotropic scaling)
    scale = np.cbrt(np.linalg.det(P[:, :3]))
    q_old_from_new = ~Quaternion.FromR(P[:, :3] / scale)

    for image in scene_manager.images.itervalues():
        image.q *= q_old_from_new
        image.tvec = scale * image.tvec - image.R().dot(P[:, 3])

    scene_manager.save(args.output_folder)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply a 3x4 transformation matrix to a COLMAP model and "
        "save the result as a new model. Row-major input can be piped in from "
        "a file or entered via the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")

    args = parser.parse_args()

    main(args)
