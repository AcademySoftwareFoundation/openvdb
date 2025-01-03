#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Script to download test GARFIELD dataset"""

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import gdown
import tyro


@dataclass
class DownloadData:
    dataset: str = "figurines"
    save_dir: Path = Path(os.getcwd() + "/data")

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_savedir: Path = self.save_dir / self.dataset

        if not dataset_savedir.exists():
            downloaded_filename = Path(self.dataset_download(self.dataset))
            if not downloaded_filename.exists():
                raise FileNotFoundError(f"Downloaded file {downloaded_filename} does not exist, something went wrong")
            self.unzip_download(downloaded_filename)

    def dataset_download(self, dataset) -> str:
        lerf_datasets_url = "https://drive.google.com/drive/folders/119bheSoSrgekkgQdSGa6E1jkNKC84HWL"
        files = gdown.download_folder(url=lerf_datasets_url, skip_download=True, quiet=True)
        download_target_file = self.save_dir / (self.dataset + ".zip")
        for file in files:
            if file.path == self.dataset + ".zip":
                downloaded_filename = gdown.download(id=file.id, output=str(download_target_file))
                break
        if not download_target_file.exists():
            raise FileNotFoundError(f"Dataset {self.dataset} not found in the Google Drive folder {lerf_datasets_url}")
        return downloaded_filename

    def unzip_download(self, downloaded_filename: Path):
        with zipfile.ZipFile(downloaded_filename, "r") as zip_ref:
            zip_ref.extractall(downloaded_filename.parent)
        downloaded_filename.unlink()
        print(f"{self.dataset} downloaded and unzipped succesfully.")


if __name__ == "__main__":
    tyro.cli(DownloadData).main()
