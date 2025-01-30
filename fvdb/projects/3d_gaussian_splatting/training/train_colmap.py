# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import tyro
import viser
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from viz import CameraState, Viewer

from fvdb.nn import GaussianSplat3D
from fvdb.optim import GaussianSplatOptimizer
from fvdb.utils.data import ColmapDataset, ColmapParser


@dataclass
class Config:
    # Random seed
    seed: int = 42

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # If you're using very large images, run the forward pass on crops and accumulate gradients
    crops_per_image: int = 1

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [3_500, 7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_500, 7_000, 30_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    increase_sh_degree_every: int = 1000
    # Initial opacity of GS
    initial_opacity: float = 0.1
    # Initial scale of GS
    initial_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Which network to use for LPIPS loss
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10


def crop_image_batch(image: torch.Tensor, ncrops: int):
    """
    Generator to iterate a minibatch of images (B, H, W, C) into disjoint patches patches (B, H_patch, W_patch, C).
    We use this function when training on very large images so that we can accumulate gradients over
    crops of each image.

    Args:
        image: Image minibatch (B, H, W, C)
        ncrops: Number of chunks to split the image into (i.e. each crop will have shape (B, H/ncrops x W/ncrops, C).

    Yields: A crop of the input image and its coordinate
        image_patch: the patch with shape (B, H/ncrops, W/ncrops, C)
        crop: the crop coordinates (x, y, w, h),
        is_last: is true if this is the last crop in the iteration
    """
    h, w = image.shape[1:3]
    patch_w, patch_h = w // ncrops, h // ncrops
    patches = np.array(
        [
            [i * patch_w, j * patch_h, (i + 1) * patch_w, (j + 1) * patch_h]
            for i, j in itertools.product(range(ncrops), range(ncrops))
        ]
    )
    for patch_id in range(patches.shape[0]):
        x1, y1, x2, y2 = patches[patch_id]
        image_patch = image[:, y1:y2, x1:x2]

        crop = (x1, y1, (x2 - x1), (y2 - y1))
        assert (x2 - x1) == patch_w and (y2 - y1) == patch_h
        is_last = patch_id == (patches.shape[0] - 1)
        yield image_patch, crop, is_last


class Runner:
    """Engine for training and testing."""

    def save_checkpoint(self, step):
        if self.no_save:
            return
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellapsed_time": time.time() - self.train_start_time,
            "num_gaussians": self.model.num_gaussians,
        }
        checkpoint_path = f"{self.checkpoint_dir}/ckpt_{step:04d}.pt"
        self.logger.info(f"Save checkpoint at step {step} to path {checkpoint_path}. Stats: {stats}.")
        with open(
            f"{self.stats_dir}/train_step{step:04d}.json",
            "w",
        ) as f:
            json.dump(stats, f)
        data = {
            "step": step,
            "splats": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": vars(self.cfg),
        }
        torch.save(data, f"{self.checkpoint_dir}/ckpt_{step:04d}.pt")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["splats"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = Config(*checkpoint["config"])

        if not self.disable_viewer:
            self.viewer.lock.release_lock
            self.viewer.state.status = "rendering"
            self.viewer.update(checkpoint["step"], 0)

        return checkpoint["step"]

    def make_results_dir(self):
        if self.no_save:
            self.output_dir = None
            self.render_dir = None
            self.stats_dir = None
            self.checkpoint_dir = None
            self.tensorboard_dir = None
            return
        os.makedirs("results", exist_ok=True)
        results_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}" if self.results_path is None else self.results_path
        tenative_results_dir = os.path.join("results", results_name)
        # If for some reason you have multiple runs at the same second, add a number to the directory
        num_retries = 0
        while os.path.exists(tenative_results_dir) and num_retries < 10:
            results_name = f"run_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            tenative_results_dir = os.path.join("results", f"{results_name}_{num_retries}")
            num_retries += 1

        self.output_dir = tenative_results_dir
        os.makedirs(self.output_dir)
        self.render_dir = os.path.join(tenative_results_dir, "render")
        os.makedirs(self.render_dir)
        self.stats_dir = os.path.join(tenative_results_dir, "stats")
        os.makedirs(self.stats_dir)
        self.checkpoint_dir = os.path.join(tenative_results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)
        self.tensorboard_dir = os.path.join(tenative_results_dir, "tb")
        os.makedirs(self.tensorboard_dir)

        # Dump config to file
        with open(f"{self.output_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(self.cfg), f)

        self.logger.info(f"Saving results to {self.output_dir}")

    def __init__(
        self,
        cfg: Config,
        data_path: str,
        data_scale_factor: int = 4,
        results_path: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        use_every_n_as_test: int = 8,
        disable_viewer: bool = False,
        log_tensorboard_every: int = 100,
        log_images_to_tensorboard: bool = False,
        no_save: bool = False,
    ) -> None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.cfg = cfg
        self.disable_viewer = disable_viewer
        self.device = device
        self.log_tensorboard_every = log_tensorboard_every
        self.log_images_to_tensorboard = log_images_to_tensorboard
        self.results_path = results_path
        self.no_save = no_save

        self.logger = logging.getLogger(__name__)

        # Setup output directories.
        self.make_results_dir()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir) if not self.no_save else None

        # Load data: Training data should contain initial points and colors.
        self.parser = ColmapParser(
            data_dir=data_path,
            factor=data_scale_factor,
            normalize=True,
            test_every=use_every_n_as_test,
        )
        self.trainset = ColmapDataset(self.parser, split="train")
        self.valset = ColmapDataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1
        self.logger.info(f"Created dataset. Scene scale = {self.scene_scale}")

        # Initialize model
        self.model = GaussianSplat3D.from_colmap(
            self.parser,
            initial_covariance_scale=cfg.initial_scale,
            initial_opacity=cfg.initial_opacity,
            scene_size=self.scene_scale,
            sh_degree=cfg.sh_degree,
        ).to(device)
        self.logger.info(f"Model initialized with {self.model.num_gaussians} Gaussians")

        # Initialize optimizer
        self.optimizer = GaussianSplatOptimizer(self.model, mean_lr_decay_exponent=0.01 ** (1.0 / cfg.max_steps))

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.disable_viewer:
            self.server = viser.ViserServer(port=8080, verbose=False)
            self.viewer = Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def train(self, start_step: int = 0):
        # We keep cycling through every image in a random order until we reach
        # the specified number of optimization steps. We can't use itertools.cycle
        # because it caches each minibatch element in memory which can quickly
        # exhaust the amount of available RAM
        def cycle(dataloader):
            while True:
                for minibatch in dataloader:
                    yield minibatch

        trainloader = cycle(
            torch.utils.data.DataLoader(
                self.trainset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
        )

        # Training loop.
        self.train_start_time = time.time()
        pbar = tqdm.tqdm(range(start_step, self.cfg.max_steps))
        for step in pbar:
            if not self.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            minibatch = next(trainloader)
            cam_to_world_mats = minibatch["camtoworld"].to(self.device)  # [B, 4, 4]
            world_to_cam_mats = torch.linalg.inv(cam_to_world_mats).contiguous()  # [B, 4, 4]
            intrinsics_mats = minibatch["K"].to(self.device)  # [B, 3, 3]
            image = minibatch["image"]  # [B, H, W, 3]
            image_height, image_width = image.shape[1:3]
            num_pixels_in_minibatch = image.shape[0] * image.shape[1] * image.shape[2]

            # If you have very large images, you can iterate over disjoint crops and accumulate gradients
            # If cfg.crops_per_image is 1, then this just returns the image
            for pixels, crop, is_last in crop_image_batch(image, self.cfg.crops_per_image):
                # Actual pixels to compute the loss on, normalized to [0, 1]
                pixels = pixels.to(self.device) / 255.0  # [1, H, W, 3]

                # Progressively use higher spherical harmonic degree as we optimize
                sh_degree_to_use = min(step // self.cfg.increase_sh_degree_every, self.cfg.sh_degree)

                # Render an image from the gaussian splats
                # possibly using a crop of the full image
                renders, alphas, info = self.model(
                    image_w=image_width,
                    image_h=image_height,
                    extrinsics_mats=world_to_cam_mats,
                    intrinsics_mats=intrinsics_mats,
                    rasterize_mode="classic",
                    sh_degree=sh_degree_to_use,
                    image_crop=crop,
                    cache_info=True,
                )
                # If you specified depth rendering, grab the depth map as well
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                # If you want to add random background, we'll mix it in here
                if self.cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=self.device)
                    colors = colors + bkgd * (1.0 - alphas)

                # Image losses
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
                loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

                # Regularization losses
                if self.cfg.opacity_reg > 0.0:
                    loss = loss + self.cfg.opacity_reg * torch.abs(self.model.opacities).mean()
                if self.cfg.scale_reg > 0.0:
                    loss = loss + self.cfg.scale_reg * torch.abs(self.model.scales).mean()

                # If we're splitting into crops, accumulate gradients
                loss.backward(retain_graph=not is_last)

            # Log to tensorboard and update progress bar
            pbar.set_description(f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| ")
            if self.log_tensorboard_every > 0 and step % self.log_tensorboard_every == 0 and self.writer is not None:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", self.model.num_gaussians, step)
                self.writer.add_scalar("train/mem", mem, step)
                if self.log_images_to_tensorboard:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in self.cfg.save_steps] or step == self.cfg.max_steps - 1:
                self.save_checkpoint(step)

            # Update the model (update parameters and potentially grow/shrink the number of Gaussians)
            self.optimizer.step(info)
            self.optimizer.zero_grad(set_to_none=True)

            # Run evaluation every eval_steps
            if step in [i - 1 for i in self.cfg.eval_steps]:
                self.eval(step)

            # Update the viewer
            if not self.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = num_pixels_in_minibatch * num_train_steps_per_sec
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_pixels_in_minibatch)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        self.logger.info("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            cam_to_world_mats = data["camtoworld"].to(device)
            world_to_cam_mats = torch.linalg.inv(cam_to_world_mats).contiguous()
            intrinsics_mats = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            colors, _, _ = self.model(
                image_w=width,
                image_h=height,
                extrinsics_mats=world_to_cam_mats,
                intrinsics_mats=intrinsics_mats,
                rasterize_mode="classic",
                sh_degree=cfg.sh_degree,
                render_depth=False,
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            # depths = colors[..., -1:] / alphas.clamp(min=1e-10)
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # depths = depths / depths.max()
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            if not self.no_save:
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        self.logger.info(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {self.model.num_gaussians}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": self.model.num_gaussians,
        }
        if not self.no_save:
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
        # save stats to tensorboard
        if self.writer is not None:
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, img_wh: Tuple[int, int]):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.linalg.inv(torch.from_numpy(c2w).float().to(self.device)).contiguous()
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.model(
            image_w=W,
            image_h=H,
            extrinsics_mats=c2w[None],
            intrinsics_mats=K[None],
            sh_degree=self.cfg.sh_degree,
            radius_clip=3.0,
            render_depth=False,
        )
        rgb = render_colors[0, ..., :3].cpu().numpy()
        return rgb
        # depth = render_colors[0][..., -1:].cpu().numpy() / alphas[0].clamp(min=1e-10).cpu().numpy()
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        # return depth.repeat(3, axis=-1)


def train(
    data_path: str,
    cfg: Config = Config(),
    data_scale_factor: int = 4,
    results_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    use_every_n_as_test: int = 8,
    disable_viewer: bool = False,
    log_tensorboard_every: int = 100,
    log_images_to_tensorboard: bool = False,
    no_save: bool = False,
):
    logging.basicConfig(level=logging.INFO)
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
        no_save,
    )
    runner.train()
    if not disable_viewer:
        runner.logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(train)
