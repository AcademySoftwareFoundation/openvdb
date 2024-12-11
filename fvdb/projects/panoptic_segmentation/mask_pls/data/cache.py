# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, original_dataset, cache_dir=tempfile.gettempdir()):
        self.original_dataset = original_dataset
        if cache_dir is None:
            raise ValueError("cache_dir cannot be None")
        cache_dir = (
            Path(cache_dir)
            / self.original_dataset.__class__.__name__
            / (str(self.original_dataset.get_directory()).lstrip("/"))
        )
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        self.cache_dir = cache_dir

    def __getitem__(self, index):
        if (self.cache_dir / f"{index}.pt").exists():
            return torch.load(
                f"{self.cache_dir}/{index}.pt",
            )

        item = self.original_dataset[index]
        torch.save(item, f"{self.cache_dir}/{index}.pt")
        return item

    def __len__(self):
        return len(self.original_dataset)

    def __getattr__(self, name):
        return getattr(self.original_dataset, name)
