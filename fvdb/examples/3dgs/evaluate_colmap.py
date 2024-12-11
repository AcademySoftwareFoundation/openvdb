# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import time
from typing import Optional, Union

import torch
import tyro

from .train_colmap import Config, Runner


def evaluate(
    ckpt_path: str,
    data_path: str,
    data_scale_factor: int = 4,
    results_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    use_every_n_as_test: int = 8,
    disable_viewer: bool = False,
):
    cfg = Config()
    runner = Runner(cfg, data_path, data_scale_factor, results_path, device, use_every_n_as_test, disable_viewer)
    step = runner.load_checkpoint(ckpt_path, load_optimizer=False)
    runner.eval(step, "test")

    # Hang and let the viewer run after training is complete
    if not disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(evaluate)
