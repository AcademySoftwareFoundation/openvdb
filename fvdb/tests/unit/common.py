from typing import List, Tuple, Union

import numpy as np
import torch
import functools
from parameterized import parameterized

from fvdb import GridBatch, sparse_grid_from_dense

Vec3i = Union[torch.Tensor, np.ndarray, List[int], Tuple[int, int, int]]
Vec3d = Union[torch.Tensor, np.ndarray, List[float], Tuple[float, float, float]]
NumberOrVec3 = Union[Vec3i, Vec3d, int, float]
Vec3 = Union[Vec3i, Vec3d]


# Hack parameterized to use the function name and the expand parameters as the test name
test_expand = functools.partial(
    parameterized.expand,
    name_func=lambda f, n, p: f'{f.__name__}_{parameterized.to_safe_name("_".join(str(x) for x in p.args))}')


def sparse_grid_from_dense_cube(resolution: NumberOrVec3,
                                cube_min: Vec3d = (0., 0., 0.),
                                cube_max: Vec3d = (1., 1., 1.),
                                voxel_center: bool = False,
                                mutable: bool = False,
                                device: Union[torch.device, str] = 'cpu') -> GridBatch:
    def _coord3d_to_tensor(coord: Vec3, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        if not hasattr(coord, "__len__") or len(coord) != 3:
            raise ValueError("expected 3D coordinate")

        if isinstance(coord, torch.Tensor):
            return torch.tensor([c.item() for c in coord], dtype=dtype)
        else:
            return torch.tensor([c for c in coord], dtype=dtype)


    def _number_or_coord3d_to_tensor(coord_or_number: NumberOrVec3, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        if isinstance(coord_or_number, (float, int)):
            return torch.tensor([coord_or_number] * 3, dtype=dtype)

        return _coord3d_to_tensor(coord_or_number, dtype=dtype)

    resolution = _number_or_coord3d_to_tensor(resolution, dtype=torch.int32)
    if torch.is_floating_point(resolution):
        raise ValueError("size must have an integer type")

    if torch.any(resolution) <= 0:
        raise ValueError("Resolution must be positive")

    cube_min = _coord3d_to_tensor(cube_min)
    cube_max = _coord3d_to_tensor(cube_max)

    if torch.any(cube_max <= cube_min):
        raise ValueError("cube_max must be greater than cube_min in all axes")

    if voxel_center:
        voxel_size = (cube_max - cube_min) / (resolution.to(torch.float64) - 1.0)
        origin = cube_min
    else:
        voxel_size = (cube_max - cube_min) / resolution.to(torch.float64)
        origin = cube_min + 0.5 * voxel_size

    return sparse_grid_from_dense(1, resolution, voxel_sizes=voxel_size, origins=origin, device=str(device), mutable=mutable)

def random_drop_points_if_mutable(grid: GridBatch, drop_pct: float = 0.5):
    if grid.mutable:
        all_ijk = grid.ijk.jdata
        drop_mask = torch.rand(grid.total_voxels, device=grid.device) < drop_pct
        drop_ijk = all_ijk[drop_mask]
        grid.disable_ijk(drop_ijk)


def make_dense_grid_and_point_data(nvox, device, dtype, mutable):
    grid_origin = (0.0, 0.0, 0.0)
    voxel_size = 1.0 / (np.floor(0.5 * nvox) + 0.5)

    fvdb = GridBatch(mutable=mutable, device=device)

    target_vox = int(2 * np.floor(0.5 * nvox) + 1) ** 3
    target_corners = int(2 * np.floor(0.5 * nvox) + 2) ** 3

    p = (2.0 * torch.rand(1, 3) - 1.0).to(device).to(dtype)
    # print(target_vox, target_corners)
    while not fvdb.total_voxels == target_vox:
        p = (2.0 * torch.rand(10 * p.shape[0], 3) - 1.0).to(device)
        p = torch.clip(p, -1.0 + 0.25 * voxel_size, 1.0 - 0.25 * voxel_size).to(dtype)
        fvdb.set_from_points(p, [0, 0, 0], [0, 0, 0], voxel_size, grid_origin)
    # print(fvdb.total_voxels, int(2 * np.floor(0.5 * nvox) + 1) ** 3)
    # print(fvdb.num_corners(), int(2 * np.floor(0.5 * nvox) + 2) ** 3)

    fvdb_d = fvdb.dual_grid()
    assert fvdb_d.total_voxels == target_corners
    dual_corners_xyz = fvdb_d.grid_to_world(fvdb_d.ijk.float()).jdata
    assert torch.allclose(dual_corners_xyz.min(0)[0], -torch.ones(3).to(dual_corners_xyz))
    assert torch.allclose(dual_corners_xyz.max(0)[0], torch.ones(3).to(dual_corners_xyz))

    return fvdb, fvdb_d, p


def make_sparse_grid_and_point_data(device, dtype, include_boundary_points: bool = False, expand: int = 10, mutable: bool = False):
    p = torch.randn((100, 3), device=device, dtype=dtype)
    vox_size = 0.05
    grid = GridBatch(mutable=mutable, device=device)
    grid.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, [0.0] * 3)
    grid_d = grid.dual_grid()

    if not include_boundary_points:
        return grid, grid_d, p

    # Ensure some samples land inside and outside the voxel grid
    # We create samples by sampling in a radius roughly the size of a voxel around each
    # voxel center so that some points will land outside but only slightly
    found = False
    mask = torch.zeros(1)
    samples = torch.zeros(1)
    while not found:
        # do everything in double then case so fp16 samples are
        # as close as possible from double and float
        primal_pts = grid.grid_to_world(grid.ijk.double()).jdata
        samples = torch.cat([primal_pts] * expand, dim=0)
        samples += torch.randn_like(samples) * vox_size
        mask = grid.points_in_active_voxel(samples).jdata
        found = not (torch.all(mask) or torch.all(~mask))

    samples = samples.to(dtype)

    assert not torch.all(mask)
    assert not torch.all(~mask)

    return grid, grid_d, samples
