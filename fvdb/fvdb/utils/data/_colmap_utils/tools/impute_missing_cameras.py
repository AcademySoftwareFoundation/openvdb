import sys

sys.path.append("..")

import numpy as np

from .. import DualQuaternion, Image, SceneManager

# -------------------------------------------------------------------------------

image_to_idx = lambda im: int(im.name[: im.name.rfind(".")])


# -------------------------------------------------------------------------------


def interpolate_linear(images, camera_id, file_format):
    if len(images) < 2:
        raise ValueError("Need at least two images for linear interpolation!")

    prev_image = images[0]
    prev_idx = image_to_idx(prev_image)
    prev_dq = DualQuaternion.FromQT(prev_image.q, prev_image.t)
    start = prev_idx

    new_images = []

    for image in images[1:]:
        curr_idx = image_to_idx(image)
        curr_dq = DualQuaternion.FromQT(image.q, image.t)
        T = curr_idx - prev_idx
        Tinv = 1.0 / T

        # like quaternions, dq(x) = -dq(x), so we'll need to pick the one more
        # appropriate for interpolation by taking -dq if the dot product of the
        # two q-vectors is negative
        if prev_dq.q0.dot(curr_dq.q0) < 0:
            curr_dq = -curr_dq

        for i in xrange(1, T):
            t = i * Tinv
            dq = t * prev_dq + (1.0 - t) * curr_dq
            q, t = dq.ToQT()
            new_images.append(Image(file_format.format(prev_idx + i), args.camera_id, q, t))

        prev_idx = curr_idx
        prev_dq = curr_dq

    return new_images


# -------------------------------------------------------------------------------


def interpolate_hermite(images, camera_id, file_format):
    if len(images) < 4:
        raise ValueError("Need at least four images for Hermite spline interpolation!")

    new_images = []

    # linear blending for the first frames
    T0 = image_to_idx(images[0])
    dq0 = DualQuaternion.FromQT(images[0].q, images[0].t)
    T1 = image_to_idx(images[1])
    dq1 = DualQuaternion.FromQT(images[1].q, images[1].t)

    if dq0.q0.dot(dq1.q0) < 0:
        dq1 = -dq1
    dT = 1.0 / float(T1 - T0)
    for j in xrange(1, T1 - T0):
        t = j * dT
        dq = ((1.0 - t) * dq0 + t * dq1).normalize()
        new_images.append(Image(file_format.format(T0 + j), camera_id, *dq.ToQT()))

    T2 = image_to_idx(images[2])
    dq2 = DualQuaternion.FromQT(images[2].q, images[2].t)
    if dq1.q0.dot(dq2.q0) < 0:
        dq2 = -dq2

    # Hermite spline interpolation of dual quaternions
    # pdfs.semanticscholar.org/05b1/8ede7f46c29c2722fed3376d277a1d286c55.pdf
    for i in xrange(1, len(images) - 2):
        T3 = image_to_idx(images[i + 2])
        dq3 = DualQuaternion.FromQT(images[i + 2].q, images[i + 2].t)
        if dq2.q0.dot(dq3.q0) < 0:
            dq3 = -dq3

        prev_duration = T1 - T0
        current_duration = T2 - T1
        next_duration = T3 - T2

        # approximate the derivatives at dq1 and dq2 using weighted central
        # differences
        dt1 = 1.0 / float(T2 - T0)
        dt2 = 1.0 / float(T3 - T1)

        m1 = (current_duration * dt1) * (dq2 - dq1) + (prev_duration * dt1) * (dq1 - dq0)
        m2 = (next_duration * dt2) * (dq3 - dq2) + (current_duration * dt2) * (dq2 - dq1)

        dT = 1.0 / float(current_duration)

        for j in xrange(1, current_duration):
            t = j * dT  # 0 to 1
            t2 = t * t  # t squared
            t3 = t2 * t  # t cubed

            # coefficients of the Hermite spline (a=>dq and b=>m)
            a1 = 2.0 * t3 - 3.0 * t2 + 1.0
            b1 = t3 - 2.0 * t2 + t
            a2 = -2.0 * t3 + 3.0 * t2
            b2 = t3 - t2

            dq = (a1 * dq1 + b1 * m1 + a2 * dq2 + b2 * m2).normalize()

            new_images.append(Image(file_format.format(T1 + j), camera_id, *dq.ToQT()))

        T0, T1, T2 = T1, T2, T3
        dq0, dq1, dq2 = dq1, dq2, dq3

    # linear blending for the last frames
    dT = 1.0 / float(T2 - T1)
    for j in xrange(1, T2 - T1):
        t = j * dT  # 0 to 1
        dq = ((1.0 - t) * dq1 + t * dq2).normalize()
        new_images.append(Image(file_format.format(T1 + j), camera_id, *dq.ToQT()))

    return new_images


# -------------------------------------------------------------------------------


def main(args):
    scene_manager = SceneManager(args.input_folder)
    scene_manager.load()

    images = sorted(scene_manager.images.itervalues(), key=image_to_idx)

    if args.method.lower() == "linear":
        new_images = interpolate_linear(images, args.camera_id, args.format)
    else:
        new_images = interpolate_hermite(images, args.camera_id, args.format)

    map(scene_manager.add_image, new_images)

    scene_manager.save(args.output_folder)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Given a reconstruction with ordered images *with integer "
        "filenames* like '000100.png', fill in missing camera positions for "
        "intermediate frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")

    parser.add_argument("--camera_id", type=int, default=1, help="camera id to use for the missing images")

    parser.add_argument("--format", type=str, default="{:06d}.png", help="filename format to use for added images")

    parser.add_argument(
        "--method", type=str.lower, choices=("linear", "hermite"), default="hermite", help="Pose imputation method"
    )

    args = parser.parse_args()

    main(args)
