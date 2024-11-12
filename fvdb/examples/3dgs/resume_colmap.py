from typing import Optional, Union

import torch
import tyro

from .train_colmap import Config, Runner


def resume(
    ckpt_path: str,
    data_path: str,
    data_scale_factor: int = 4,
    results_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    use_every_n_as_test: int = 8,
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
):
    cfg = Config()
    runner = Runner(
        cfg,
        data_path,
        data_scale_factor,
        results_path,
        device,
        use_every_n_as_test,
        disable_viewer,
        log_tensorboard_every,
        log_images_to_tensorboard,
    )
    step = runner.load_checkpoint(ckpt_path)
    runner.train(start_step=step)


if __name__ == "__main__":
    tyro.cli(resume)
