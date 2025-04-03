# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
from viz import CameraState, Viewer

from fvdb import GaussianSplat3d
from fvdb.nn.gaussian_splatting import GaussianSplat3D

np.set_printoptions(suppress=True)
import viser

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location="cuda")
model = GaussianSplat3d.from_state_dict(checkpoint["splats"])


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
server.scene.set_up_direction("-z")
viewer = Viewer(
    server=server,
    render_fn=_viewer_render_fn,
    mode="training",
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(1000000)
