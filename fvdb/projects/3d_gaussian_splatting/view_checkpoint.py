# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
from viz import CameraState, Viewer

from fvdb.nn.gaussian_splatting import GaussianSplat3D

np.set_printoptions(suppress=True)
import viser

checkpoint_path = "/workspace/Data1/gettysburg/prerun/fvdb_small_glomap_run_output/checkpoints/ckpt_29999.pt"
checkpoint = torch.load(checkpoint_path, map_location="cuda")
model = GaussianSplat3D(torch.rand([8, 3]), torch.rand([8, 3])).to("cuda")
model.load_state_dict(checkpoint["splats"])


@torch.no_grad()
def _viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
    """Callable function for the viewer."""
    W, H = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)

    w2c = torch.linalg.inv(torch.from_numpy(c2w).float().to("cuda")).contiguous()
    K = torch.from_numpy(K).float().to("cuda")

    render_colors, _, _ = model(
        image_w=W,
        image_h=H,
        extrinsics_mats=w2c[None],
        intrinsics_mats=K[None],
        sh_degree=3,
        radius_clip=3.0,
        render_depth=False,
    )
    rgb = render_colors[0, ..., :3].cpu().numpy()
    return rgb


server = viser.ViserServer(port=8080, verbose=False)
# self.server.scene.set_up_direction("-z")
viewer = Viewer(
    server=server,
    render_fn=_viewer_render_fn,
    mode="training",
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(1000000)
