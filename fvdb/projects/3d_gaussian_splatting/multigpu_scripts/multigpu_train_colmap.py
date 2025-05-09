# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import os
import subprocess
from functools import partial
from multiprocessing import Pool

import tyro


# TODO allow user to config other training args
def run_cluster(
    cluster_id, data_path, factor, devices_per_cluster, max_steps, num_clusters, geo_normalize, results_path
):
    """
    Run train_colmap.py for specified cluster.
    Args:
        cluster_id: integer for desired cluster to run
        data_path: Path to colmap run
        factor: image downsample factor
        devices_per_cluster: list of devices assigned per cluster
        max_steps: max step to train cluster to
        num_clusters: total number of clusters, used to generate final output path
        geo_normalize: set if colmap run is in ECEF geo coordinates
        results_path: root directory to save results to
    Returns:
        dictionary of splats after removal of gaussians outside bounds
    """

    cmd = ["python", "projects/3d_gaussian_splatting/train_colmap.py"]
    cluster_id_str = str(cluster_id).zfill(4)

    args = [
        "--data-path",
        f"{data_path}",
        "--data_scale_factor",
        f"{factor}",
        "--device",
        f"{devices_per_cluster[cluster_id]}",
        "--cfg.crops-per-image",
        "1",
        "--use_every_n_as_test",
        "1000",
        "--use_masks",
        "--cfg.max_steps",
        f"{max_steps}",
        "--cfg.save_steps",
        str(max_steps),
        "--cfg.sh_degree",
        "1",
        "--disable_viewer",
        "--point_ids_split_path",
        os.path.join(data_path, f"{num_clusters}_cluster_splits", f"{cluster_id_str}_cluster_point_ids.csv"),
        "--image_ids_split_path",
        os.path.join(data_path, f"{num_clusters}_cluster_splits", f"{cluster_id_str}_cluster_image_ids.csv"),
        "--split_masks_path",
        os.path.join(data_path, f"{num_clusters}_cluster_splits", f"{cluster_id_str}_images_{factor}_masks/"),
        "--results_path",
        os.path.join(results_path, f"{cluster_id_str}_cluster/"),
    ]
    if geo_normalize:
        args.append("--normalize_ecef2enu")

    cmd.extend(args)
    print(" ".join(cmd))
    p = subprocess.Popen(cmd)
    p.wait()


def main(
    num_clusters: int,
    factor: int,
    max_steps: int,
    data_path: str,
    results_path: str,
    devices: list[str],
    geo_normalize: bool = False,
):
    """
    Script to run multiple training jobs across multiple gpus. Assuming that the clusters were generated
    by the multigpu_split_colmap.py script. This script will call the regular train_colmap.py script with
    a custom set of arg per cluster and place the per cluster training results in results_path.
    Args:
        num_clusters: total number of clusters
        factor: image downsample factor
        max_steps: how many steps to train each cluster to
        data_path: path to colmap run
        results_path: root path to place results in for each cluster run
        devices: list of gpu devices to run across (eg: cuda:0 cuda:1)
    """
    clusters = list(range(num_clusters))
    ngpus = len(devices)
    dev_cycle = itertools.cycle(devices)
    devices_per_cluster = [next(dev_cycle) for _ in range(num_clusters)]

    partial_function = partial(
        run_cluster,
        data_path=data_path,
        factor=factor,
        devices_per_cluster=devices_per_cluster,
        max_steps=max_steps,
        num_clusters=num_clusters,
        geo_normalize=geo_normalize,
        results_path=results_path,
    )

    pool = Pool(ngpus)
    for _ in pool.imap_unordered(partial_function, clusters):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    tyro.cli(main)
