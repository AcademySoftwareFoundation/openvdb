# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import glob
import os
import shutil
from functools import partial
from multiprocessing import Pool

import cv2
import tqdm
import tyro


def downsize_image(img_path, out_dir, downsize, out_ext, multicamera):
    """
    resize specified image
    Args:
        img_path: path to image file
        out_dir: root directory for downsized image
        downsize: factor to decrease image size by
        out_ext: extension of output image
        multicamera: flag to set if multiple cameras exist (one per sub folder)
    """
    basename = os.path.basename(img_path)

    if multicamera:
        dirname = os.path.basename(os.path.dirname(img_path))
        dir_out = os.path.join(out_dir, dirname)
    else:
        dir_out = out_dir

    os.makedirs(dir_out, exist_ok=True)

    img_path_out = os.path.join(dir_out, os.path.splitext(basename)[0] + out_ext)

    img = cv2.imread(img_path)
    oh, ow, _ = img.shape
    (rw, rh) = ow // downsize, oh // downsize  # same logic used in colmap parser
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(img_path_out, img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])


def main(images_dir: str, ext: str, out_ext: str, factor: int, cores: int = 4, multicamera: bool = False):
    """
    Script to resize a set of colmap images that aligns with the colmap parser logic.
    Args:
        images_dir: path to images
        ext: image extension, typically .jpg
        out_ext: desired output image format, typically .jpg
        factor: image downsample factor
        cores: how many cores to use during processing
        multicamera: flag if colmap images arranged into one camera per sub folder
    """
    out_dir = os.path.join(os.path.dirname(images_dir), "images_" + str(factor))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    all_images = glob.glob(os.path.join(images_dir, "**/*" + ext), recursive=True)

    partial_function = partial(
        downsize_image, out_dir=out_dir, downsize=factor, out_ext=out_ext, multicamera=multicamera
    )

    nimages = len(all_images)
    pool = Pool(cores)
    for _ in tqdm.tqdm(pool.imap_unordered(partial_function, all_images), total=nimages):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    tyro.cli(main)
