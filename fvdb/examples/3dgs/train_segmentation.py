# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import time
from dataclasses import dataclass
from typing import Union

import torch
import tqdm
import tyro
from torch.utils.data import DataLoader, Dataset

from fvdb.nn.gaussian_splatting import GaussianSplat3D


class SegmentationDataset(Dataset):
    def __init__(self):
        data = torch.load("segmentation_dataset.pt")

        self.images = data["images"]
        self.intrinsics = data["intrinsics"]
        self.cam_to_worlds = data["cam_to_worlds"]
        self.scales = data["scales"]
        self.mask_cdss = data["mask_cdfs"]
        self.mask_ids = data["mask_ids"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(
        #     self.images[idx].shape,
        #     self.intrinsics[idx].shape,
        #     self.cam_to_worlds[idx].shape,
        #     self.scales[idx].shape,
        #     self.mask_cdss[idx].shape,
        #     self.mask_ids[idx].shape,
        # )
        return {
            "image": self.images[idx],
            "intrinsics": self.intrinsics[idx],
            "cam_to_world": self.cam_to_worlds[idx],
            # "scale": self.scales[idx],
            # "mask_cdf": self.mask_cdss[idx],
            "mask_id": self.mask_ids[idx],
        }

    @staticmethod
    def _collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "intrinsics": torch.stack([b["intrinsics"] for b in batch]),
            "cam_to_world": torch.stack([b["cam_to_world"] for b in batch]),
            # "scale": torch.stack([b["scale"] for b in batch]),
            # "mask_cdf": torch.stack([b["mask_cdf"] for b in batch]),
            # "mask_id": torch.stack([b["mask_id"] for b in batch]),
        }


@dataclass
class Config:
    num_train_iters: int = 1000


class Runner:
    def __init__(self, cfg: Config, checkpoint_path: str, device: Union[str, torch.device] = "cuda"):
        self.config = cfg

        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.gs_model: GaussianSplat3D = GaussianSplat3D(torch.rand([8, 3]), torch.rand([8, 3])).to(device)
        self.gs_model.load_state_dict(checkpoint["splats"])

        feats = torch.randn(self.gs_model.num_gaussians, 1, 64, device=device)
        self.gs_model.set_spherical_harmonic_coeffs(feats)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        ).to(device)

        self.train_dataset = SegmentationDataset()

    def train(self, dataset):
        trainloader = itertools.cycle(
            DataLoader(
                self.train_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=self.train_dataset._collate_fn,
            )
        )

        self.train_start_time = time.time()
        pbar = tqdm.tqdm(range(self.config.num_train_iters))

        for step in pbar:
            minibatch = next(trainloader)

            img = minibatch["image"]
            intrinsics = minibatch["intrinsics"].to(self.device)
            cam_to_world = minibatch["cam_to_world"].to(self.device)
            world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

            img_h, img_w = img.shape[1], img.shape[2]
            print(img_h, img_w, img.shape)

            # Forward pass
            feats, alphas, info = self.gs_model(
                image_w=img_w, image_h=img_h, intrinsics_mats=intrinsics, extrinsics_mat=world_to_cam
            )

            # TODO (Francis): Don't use Pytorch caching allocator which causes massive fragmentation
            #                 This should use about half the memory
            del alphas, info
            torch.cuda.empty_cache()
            feats_ravel = feats.view(-1, 64)
            idx = torch.randperm(feats_ravel.shape[0])[:4096]
            feats_slice = feats.view(-1, 64)[idx]
            gfeats = self.mlp(feats_slice)
            loss = gfeats.sum()
            torch.cuda.empty_cache()

            time.sleep(2)


def main(checkpoint_path: str):
    cfg = Config()
    runner = Runner(cfg, checkpoint_path)
    runner.train(None)


if __name__ == "__main__":
    tyro.cli(main)
