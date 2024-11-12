# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import logging
import site
import tempfile
import timeit
from pathlib import Path
from typing import List, Tuple, Union

import git
import git.repo
import numpy as np
import point_cloud_utils as pcu
import torch
from git.exc import InvalidGitRepositoryError

from fvdb import GridBatch


def _is_editable_install() -> bool:
    # check we're not in a site package
    module_path = Path(__file__).resolve()
    for site_path in site.getsitepackages():
        if str(module_path).startswith(site_path):
            return False
    # check if we're in the source directory
    module_dir = module_path.parent.parent.parent.parent
    return (module_dir / "setup.py").is_file()


def _get_local_repo_path() -> Path:
    if _is_editable_install():
        external_dir = Path(__file__).resolve().parent.parent.parent.parent / "external"
        if not external_dir.exists():
            external_dir.mkdir()
        local_repo_path = external_dir
    else:
        local_repo_path = Path(tempfile.gettempdir())

    local_repo_path = local_repo_path / "fvdb_example_data"
    return local_repo_path


def _clone_fvdb_example_data():
    def is_git_repo(repo_path: str):
        is_repo = False
        try:
            _ = git.repo.Repo(repo_path)
            is_repo = True
        except InvalidGitRepositoryError:
            is_repo = False

        return is_repo

    git_tag = "613c3a4e220eb45b9ae0271dca4808ab484ee134"
    git_url = "https://github.com/voxel-foundation/fvdb-example-data.git"

    repo_path = _get_local_repo_path()
    if repo_path.exists() and repo_path.is_dir():
        if is_git_repo(str(repo_path)):
            repo = git.repo.Repo(repo_path)
            repo.git.checkout(git_tag)
        else:
            raise ValueError(f"A path {repo_path} exists but is not a git repo")
    else:
        repo = git.repo.Repo.clone_from(git_url, repo_path)
        repo.git.checkout(git_tag)

    return repo_path, repo


def _get_fvdb_example_data_path():
    repo_path, _ = _clone_fvdb_example_data()
    return repo_path


def _get_md5_checksum(file_path: Path):
    md5_hash = hashlib.md5(open(file_path, "rb").read())
    return md5_hash.hexdigest()


def make_grid_from_points(pts: torch.Tensor, padding, vox_size, vox_origin) -> GridBatch:
    logging.info("Building GridBatch from points...")
    start = timeit.default_timer()
    grid = GridBatch(device=pts.device)
    grid.set_from_points(pts, [-padding] * 3, [padding] * 3, voxel_sizes=vox_size, origins=vox_origin)
    torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s")
    logging.info(f"GridBatch has {grid.total_voxels} voxels")

    return grid


def make_ray_grid(
    nrays: int,
    origin: Union[torch.Tensor, Tuple, List],
    minb=(-0.3, -0.3),
    maxb=(0.3, 0.3),
    device: Union[str, torch.device] = "cpu",
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ray_o = torch.tensor([origin] * nrays**2)

    ray_d = torch.from_numpy(
        np.stack(
            [a.ravel() for a in np.mgrid[minb[0] : maxb[0] : nrays * 1j, minb[1] : maxb[1] : nrays * 1j]]
            + [np.ones(nrays**2)],
            axis=-1,
        ).astype(np.float32)
    )
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


def load_pointcloud(
    data_path,
    skip_every=1,
    shuffle=False,
    device=torch.device("cuda"),
    dtype=torch.float32,
) -> torch.Tensor:
    logging.info(f"Loading pointlcoud {data_path}...")
    start = timeit.default_timer()
    pts = pcu.load_mesh_v(data_path)
    if shuffle:
        pts = pts[np.random.permutation(pts.shape[0])]
    pts = pts[::skip_every]
    logging.info(f"Done in {timeit.default_timer() - start}s")
    return torch.from_numpy(pts).to(device).to(dtype)


def load_mesh(
    data_path, expected_md5, skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32
) -> List[torch.Tensor]:
    if _get_md5_checksum(data_path) != expected_md5:
        raise ValueError(f"Checksum for {data_path} is incorrect, expected {expected_md5}")
    logging.info(f"Loading mesh {data_path}...")
    start = timeit.default_timer()
    if mode == "v":
        attrs = [pcu.load_mesh_v(data_path)]
    elif mode == "vf":
        attrs = pcu.load_mesh_vf(data_path)
    elif mode == "vn":
        attrs = pcu.load_mesh_vn(data_path)
    else:
        raise ValueError(f"Unsupported mode {mode}")
    for a in attrs:
        if a is None:
            raise ValueError(f"Failed to load mesh {data_path}, missing attributes")
    attrs = [torch.from_numpy(a[::skip_every]).to(device).to(dtype) for a in attrs]
    logging.info(f"Done in {timeit.default_timer() - start}s")

    return attrs


def load_dragon_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "dragon.ply"
    return load_mesh(
        data_path,
        expected_md5="0222e7d2147eebcb2eacdaf6263a9512",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_happy_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "happy.ply"
    return load_mesh(
        data_path,
        expected_md5="5cfe3c9c0b58bad9a77b47ae04454160",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_bunny_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "bunny.ply"
    return load_mesh(
        data_path,
        expected_md5="fe2f062a8e22b7dab895a1945c32cd58",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_1_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-1.ply"
    return load_mesh(
        data_path,
        expected_md5="969f91abdf00bad792ca2af347c58499",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_2_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-2.ply"
    return load_mesh(
        data_path,
        expected_md5="d4aa0dd4f4609ea1b19aca7d8618d22a",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_3_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-3.ply"
    return load_mesh(
        data_path,
        expected_md5="a058d534da71748167799db0351f21f4",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_4_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = _get_fvdb_example_data_path() / "meshes" / "car-mesh-4.ply"
    return load_mesh(
        data_path,
        expected_md5="6238478fcf1f963e38a95b52a1521b5d",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def plot_ray_segments(ray_o, ray_d, times, plot_every=1):
    import polyscope as ps

    for i in range(0, ray_o.shape[0], plot_every):
        t0s = times[i].jdata[:, 0].unsqueeze(-1)
        t1s = times[i].jdata[:, 1].unsqueeze(-1)
        roi = ray_o[i].unsqueeze(0)
        rdi = ray_d[i].unsqueeze(0)
        rp = torch.cat([roi + t0s * rdi, roi + t1s * rdi])
        re = torch.stack(
            [torch.arange(t0s.shape[0]), torch.arange(t0s.shape[0]) + t0s.shape[0]],
            dim=-1,
        )

        ray_segs = ps.register_curve_network(f"ray segments {i}", rp, re, radius=0.001)
        rv = torch.zeros(re.shape[0])
        rv[::2] = 1.0
        ray_segs.add_scalar_quantity(f"segment colors {i}", rv, defined_on="edges", enabled=True, cmap="jet")


__all__ = [
    "make_grid_from_points",
    "make_ray_grid",
    "load_pointcloud",
    "load_mesh",
    "load_dragon_mesh",
    "load_happy_mesh",
    "load_bunny_mesh",
    "load_car_1_mesh",
    "load_car_2_mesh",
    "plot_ray_segments",
]
