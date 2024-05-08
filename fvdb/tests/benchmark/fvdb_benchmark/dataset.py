from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from fvdb import GridBatch, sparse_grid_from_ijk


class CoordsDataset(Dataset):
    """ Loading coordinates from the dataset """

    def __init__(self,
                 paths: List[str],
                 in_channels: int = 32,
                 pad_level: int = 0,
                 max_files: int = -1,
                 positive_only: bool = True):
        super().__init__()
        self.pad_level = pad_level
        self.in_channels = in_channels
        self.max_files = max_files
        self.positive_only = positive_only

        self.data_paths = []
        for path in paths:
            self.data_paths.extend(self._get_files(Path(path)))

    def _get_files(self, base_path: Path) -> List[Path]:
        if base_path.is_file():
            return [base_path]

        else:
            all_paths = []
            for pkl_file in base_path.glob("*.pkl"):
                all_paths.append(pkl_file)
            for csv_file in base_path.glob("*.csv"):
                all_paths.append(csv_file)
            for npy_file in base_path.glob("*.npy"):
                all_paths.append(npy_file)
            if self.max_files > 0:
                return all_paths[:self.max_files]
            return all_paths

    def __len__(self):
        return len(self.data_paths)

    def load_grid(self, path) -> GridBatch:
        if path.suffix == '.pkl':
            input_data = torch.load(path)
            input_points: GridBatch = input_data['grid']
        elif path.suffix == '.csv':
            input_data = np.loadtxt(path, delimiter=',').astype(int)
            input_points: GridBatch = sparse_grid_from_ijk(torch.from_numpy(input_data))
        elif path.suffix == '.npy':
            input_data = np.load(path).astype(int)
            input_points: GridBatch = sparse_grid_from_ijk(torch.from_numpy(input_data))
        else:
            raise NotImplementedError
        return input_points

    def __getitem__(self, index):
        current_path: Path = self.data_paths[index]
        input_grid: GridBatch = self.load_grid(current_path)

        # Pad data.
        if self.pad_level > 0:
            input_grid = sparse_grid_from_ijk(
                input_grid.ijk,
                pad_min=[-self.pad_level] * 3,
                pad_max=[self.pad_level] * 3
            )

        # Make sure the coordinates are positive.
        if self.positive_only:
            ijk = input_grid.ijk.jdata
            ijk -= ijk.min(0).values
            input_grid = sparse_grid_from_ijk(ijk)

        feature: torch.Tensor = torch.randn(input_grid.total_voxels, self.in_channels)

        return current_path.stem, input_grid, feature
