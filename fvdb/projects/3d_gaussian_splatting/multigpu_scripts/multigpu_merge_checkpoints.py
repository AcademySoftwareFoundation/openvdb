# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import Tuple

import numpy as np
import torch
import tyro

from fvdb import GaussianSplat3d

np.set_printoptions(suppress=True)
import csv
import os
import sys

import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import apply_bounds, filter_splat_means, filter_splat_opacities, prune_large
from viz import CameraState, Viewer


def main(ckpt_num: str, num_clusters: int, bounds_dir: str, results_dir: str, dst_path: str, flip_z: bool = False):
    """
    Script to merge multiple runs from a multigpu training job. This assumes that the data was clustered with
    multigpu_split_colmap.py, and then trained with multi_gpu_train_colmap.py.
    Args:
        ckpt_num: which checkpoint number to merge
        num_clusters: total number of clusters
        bounds_dir: directory containing bounds csv per cluster
        results_path: directory container multigpu runs per cluster
        dst_path: save merged checkpoint to this path
    """
    bounds_dir = os.path.normpath(bounds_dir)
    results_dir = os.path.normpath(results_dir)
    dst_ply_path = os.path.join(os.path.normpath(dst_path), "merged.ply")
    dst_pt_path = os.path.join(os.path.normpath(dst_path), "merged.pt")

    checkpoint_paths = []
    bounds_split_paths = []
    for cluster_index in range(num_clusters):
        ind_str = str(cluster_index).zfill(4)
        checkpoint_paths.append(
            os.path.join(results_dir, f"{ind_str}_cluster/checkpoints/ckpt_{str(ckpt_num).zfill(4)}.pt")
        )
        bounds_split_paths.append(os.path.join(bounds_dir, f"{ind_str}_cluster_bounds.csv"))

    bounds_per_cluster = []
    for bounds_split_path in bounds_split_paths:
        with open(bounds_split_path, "r") as fp:
            reader = csv.reader(fp)
            hdr = next(reader)
            if len(hdr) != 6:
                raise ValueError("header must container 6 elements for bounds (xmin, ymin, zmin, xmax, ymax, zmax)")
            else:
                clip_bounds = np.array([float(item) for item in next(reader)])
                clip_bounds = torch.from_numpy(clip_bounds).float().to("cuda")
        bounds_per_cluster.append(clip_bounds)

    checkpoint = torch.load(checkpoint_paths[0], map_location="cuda")
    splats = checkpoint["splats"]
    # commenting out due to mrge issues
    # splats = apply_bounds(splats, bounds_per_cluster[0])
    checkpoint_paths.pop(0)
    bounds_per_cluster.pop(0)

    for cluster_index, fpath in enumerate(checkpoint_paths):
        curr_ckpt = torch.load(fpath, map_location="cuda")
        curr_splats = curr_ckpt["splats"]
        # commenting out due to mrge issues
        # curr_splats = apply_bounds(curr_splats, bounds_per_cluster[cluster_index])

        splats["means"] = torch.concatenate([splats["means"], curr_splats["means"]], dim=0)

        splats["logit_opacities"] = torch.concatenate(
            [splats["logit_opacities"], curr_splats["logit_opacities"]], dim=0
        )

        splats["quats"] = torch.concatenate([splats["quats"], curr_splats["quats"]], dim=0)

        splats["log_scales"] = torch.concatenate([splats["log_scales"], curr_splats["log_scales"]], dim=0)

        splats["sh0"] = torch.concatenate([splats["sh0"], curr_splats["sh0"]], dim=1)

        splats["shN"] = torch.concatenate([splats["shN"], curr_splats["shN"]], dim=1)

        splats["accumulated_gradient_step_counts_for_grad"] = torch.concatenate(
            [
                splats["accumulated_gradient_step_counts_for_grad"],
                curr_splats["accumulated_gradient_step_counts_for_grad"],
            ],
            dim=0,
        )

        splats["accumulated_mean_2d_gradient_norms_for_grad"] = torch.concatenate(
            [
                splats["accumulated_mean_2d_gradient_norms_for_grad"],
                curr_splats["accumulated_mean_2d_gradient_norms_for_grad"],
            ],
            dim=0,
        )

    nsplats = splats["means"].shape[0]
    print(f"Merged: {nsplats} Gaussians")

    splats = filter_splat_opacities(splats)
    nsplats = splats["means"].shape[0]
    print(f"{nsplats} Gaussians after pruning low opacity")

    splats = prune_large(splats)

    nsplats = splats["means"].shape[0]
    print(f"{nsplats} Gaussians after pruning large")

    splats = filter_splat_means(splats, [0.99, 0.99, 0.99, 0.99, 0.999, 0.98])

    nsplats = splats["means"].shape[0]
    print(f"{nsplats} Gaussians after percentile filtering")

    model = GaussianSplat3d.from_state_dict(splats)

    print("saving merged model to: " + dst_pt_path)
    data = {
        "splats": model.state_dict(),
    }
    torch.save(data, dst_pt_path)

    print("saving merged model to: " + dst_ply_path)
    model.save_ply(dst_ply_path)

    @torch.no_grad()
    def _viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)

        w2c = torch.linalg.inv(torch.from_numpy(c2w).float().to("cuda")).contiguous()
        K = torch.from_numpy(K).float().to("cuda")

        render_colors, _ = model.render_images(w2c[None], K[None], W, H, 0.01, 1e10)
        rgb = render_colors[0, ..., :3].cpu().numpy()
        return rgb

    server = viser.ViserServer(port=8080, verbose=False)
    if flip_z:
        server.scene.set_up_direction("-z")
    viewer = Viewer(
        server=server,
        render_fn=_viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
