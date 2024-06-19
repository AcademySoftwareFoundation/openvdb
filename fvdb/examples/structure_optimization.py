import math
from pathlib import Path

import fvdb
import torch
import pyrender, trimesh
import numpy as np
import point_cloud_utils as pcu
import polyscope as ps


class Isometry:
    def __init__(self) -> None:
        self.matrix = np.eye(4)

    def __matmul__(self, other: 'Isometry') -> 'Isometry':
        result = Isometry()
        result.matrix = self.matrix @ other.matrix
        return result

    @staticmethod
    def from_matrix(matrix: np.ndarray, t_component: np.ndarray) -> 'Isometry':
        result = Isometry()
        result.matrix[:3, :3] = matrix
        result.matrix[:3, 3] = t_component
        return result

    @property
    def pyrender_pose(self):
        return self.matrix @ np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

    def get_rays(self, fx: float, fy: float, cx: float, cy: float, w: int, h: int):
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.flatten()
        y = y.flatten()

        ray_dir = np.array([
            (x - cx) / fx,
            (y - cy) / fy,
            np.ones_like(x)
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=0)

        ray_dir = self.matrix[:3, :3] @ ray_dir
        ray_dir = ray_dir.transpose(1, 0)
        ray_orig = self.matrix[:3, 3:4].repeat(ray_dir.shape[0], axis=1).T

        return ray_orig, ray_dir


def inv_sigmoid(x):
    return -math.log(1 / x - 1)


def get_points(n_views: int):
    """
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def get_views(n_views: int):
    views = []
    points = get_points(n_views)

    for i in range(points.shape[0]):
        # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(latitude), -math.sin(latitude)],
                        [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)],
                        [0, 1, 0],
                        [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)

        correction_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        correction_iso = Isometry.from_matrix(correction_mat, np.array([0, 0, 1]))

        views.append(Isometry.from_matrix(R, t_component=np.zeros(3)) @ correction_iso)

    return views


def render_opacity(grid: fvdb.GridBatch, feature: torch.Tensor, pose: Isometry=None, ray_orig=None, ray_dir=None):
    if pose is not None:
        ray_orig, ray_dir = pose.get_rays(camera_fx, camera_fy, camera_w / 2., camera_h / 2., camera_w, camera_h)
        ray_orig = torch.from_numpy(ray_orig).to(grid.device).float()
        ray_dir = torch.from_numpy(ray_dir).to(grid.device).float()
    else:
        assert ray_orig is not None and ray_dir is not None

    pack_info, voxel_inds, out_times = grid.voxels_along_rays(ray_orig, ray_dir, 128, 0.0)
    pack_info = pack_info.jdata
    out_times = out_times.jdata
    voxel_inds = grid.ijk_to_index(voxel_inds).jdata

    rgb, depth, opacity, _, _ = fvdb.utils.volume_render(
        sigmas=-torch.log(1 - feature[voxel_inds]),
        rgbs=torch.ones((voxel_inds.shape[0], 1), device=grid.device),
        deltaTs=torch.ones(voxel_inds.shape[0], device=grid.device),
        ts=out_times.mean(1),
        packInfo=pack_info, transmittanceThresh=0.0
    )

    if pose is None:
        return opacity

    opacity = opacity.reshape(camera_h, camera_w)
    return opacity


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    mesh_v, mesh_f = pcu.load_mesh_vf(str(base_path / "data/car-mesh-2.ply"))

    bb_min, bb_max = np.min(mesh_v, axis=0), np.max(mesh_v, axis=0)
    centers = (bb_min + bb_max) / 2
    total_size = (bb_max - bb_min).max()
    scale = total_size / 0.8
    mesh_v = (mesh_v - centers) / scale

    print("Mesh bound:", np.min(mesh_v, axis=0), np.max(mesh_v, axis=0))

    camera_fx = camera_fy = 512
    camera_h = camera_w = 512
    init_resolution = 96

    renderer = pyrender.OffscreenRenderer(camera_w, camera_h)

    render_masks = []
    virtual_views = get_views(100)
    for view in virtual_views:
        pr_scene = pyrender.Scene()
        pr_scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            vertices=mesh_v, faces=mesh_f), smooth=False))
        pr_scene.add(pyrender.IntrinsicsCamera(
            camera_fx, camera_fy, camera_w / 2., camera_h / 2.), pose=view.pyrender_pose
        )
        render_flag = pyrender.RenderFlags.DEPTH_ONLY
        mask = renderer.render(pr_scene, flags=render_flag) > 0.0
        render_masks.append(mask)

    # import matplotlib.pyplot as plt
    # for vid, mask in enumerate(render_masks[:12]):
    #     plt.subplot(2, 6, vid + 1)
    #     plt.imshow(mask)
    # plt.show()

    # Gather rays
    ray_orig, ray_dir = [], []
    ray_opacity = []
    for vid, vv in enumerate(virtual_views):
        ro, rd = vv.get_rays(camera_fx, camera_fy, camera_w / 2., camera_h / 2., camera_w, camera_h)
        ray_orig.append(ro)
        ray_dir.append(rd)
        ray_opacity.append(render_masks[vid].flatten())
    ray_orig = np.concatenate(ray_orig, axis=0)
    ray_dir = np.concatenate(ray_dir, axis=0)
    ray_opacity = np.concatenate(ray_opacity, axis=0)
    ray_orig = torch.from_numpy(ray_orig).to("cuda").float()
    ray_dir = torch.from_numpy(ray_dir).to("cuda").float()
    ray_opacity = torch.from_numpy(ray_opacity).to("cuda").float()

    # Build grid that spans from -0.5 to 0.5 (at voxel centers)
    grid = fvdb.sparse_grid_from_dense(num_grids=1, dense_dims=[init_resolution] * 3,
                                       voxel_sizes=[1.0 / (init_resolution - 1)] * 3,
                                       origins=[-0.5, -0.5, -0.5],
                                       device="cuda", mutable=True)
    alpha = torch.full((grid.total_voxels, ), inv_sigmoid(0.1), device=grid.device, requires_grad=True)

    optimizer = torch.optim.Adam([alpha], lr=1.0)
    for it in range(100):
        sub_inds = torch.randint(0, ray_orig.shape[0], (10000, ), device=grid.device)
        pd_opacity = render_opacity(grid, torch.sigmoid(alpha), ray_orig=ray_orig[sub_inds], ray_dir=ray_dir[sub_inds])
        gt_opacity = ray_opacity[sub_inds]
        loss = torch.mean(torch.abs(pd_opacity - gt_opacity))
        print("Iter", it, "Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it > 0 and it % 5 == 0:
            with torch.no_grad():
                # disable
                bad_mask = torch.sigmoid(alpha) < 0.1
                grid.disable_ijk(grid.ijk.r_masked_select(bad_mask))

                # random revive
                if it < 20:
                    enable_mask = torch.rand(grid.total_voxels, device=grid.device) < 0.01
                    grid.enable_ijk(grid.ijk.r_masked_select(enable_mask))

                grid_mesh = pcu.voxel_grid_geometry(
                    grid.ijk_enabled.jdata.cpu().numpy(), grid.voxel_sizes[0].cpu().numpy(), gap_fraction=0.1)
                ps.init()
                ps.remove_all_structures()
                ps.register_surface_mesh("grid_1", grid_mesh[0], grid_mesh[1], enabled=True)
                ps.show()
