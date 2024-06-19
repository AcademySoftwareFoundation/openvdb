# Volume Rendering

In this example we replace `nerfacc`'s acceleration structure with fVDB and hence scale to unbounded scenes:

```python
import json
import math
import os
from typing import Optional, Tuple, Union

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import torch
import tqdm

from fvdb import GridBatch
from fvdb.utils import volume_render

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class _TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))

# SH
MAX_SH_BASIS = 10
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    SH_C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    SH_C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def speherical_harmonics(deg: int, sh: torch.Tensor, dirs: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    # sh is a tensor of shape [N, C, (deg+1)**2]
    # dirs is a tensor of shape [N, 3]
    assert 0 <= deg <= 4
    assert (deg + 1) ** 2 == sh.shape[-1]
    # C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result



def compute_psnr(rgb_gt: torch.Tensor, rgb_est: torch.Tensor) ->  torch.Tensor:
    x = torch.mean((rgb_gt - rgb_est)**2)
    return -10. * torch.log10(x)


def nerf_matrix_to_ngp(pose: np.ndarray, scale: float = 0.33, offset: Union[tuple, list, torch.Tensor] = (0, 0, 0)) -> np.ndarray:
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],], dtype=np.float32)

    return new_pose


def get_rays(pose: torch.Tensor, intrinsic: torch.Tensor, H: int, W: int, depth: Optional[torch.Tensor] = None) -> TensorTriple:
    fx, fy, cx, cy = intrinsic

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device='cpu'), torch.linspace(0, H-1, H, device='cpu'), indexing='ij')
    i = i.t().reshape([1, H*W]).expand([1, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([1, H*W]) + 0.5
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.cat((xs.reshape(-1,1), ys.reshape(-1,1), zs.reshape(-1,1)), dim=-1)

    # compute distances
    if depth is not None:
        dist = torch.norm(directions * depth[:,None], dim=-1, keepdim=True)
    else:
        dist = torch.empty([])

    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = (pose[:3,:3] @ directions.transpose(0,1)).transpose(0,1)

    rays_o = pose[:3, 3] # [3]
    rays_o = rays_o[None, :].expand_as(rays_d) # [N, 3]

    return rays_o.squeeze(), rays_d.squeeze(), dist.squeeze()


class NeRFDataset:
    def __init__(self, root_path: str = 'data/lego/', scale: float = 1.0, num_rays: int = 4096, mode: str = 'train'):
        super().__init__()

        self.root_path  = root_path
        self.scale = scale
        self.num_rays = num_rays
        self.mode = mode

        with open(os.path.join(self.root_path, f'transforms_{self.mode}.json'), 'r', encoding='utf-8') as f:
            transform = json.load(f)

        # read images
        frames = transform["frames"]
        self.n_frames = len(frames)

        # Read the intrinsics
        image = imageio.imread(os.path.join(self.root_path, frames[0]['file_path'] + '.png')) # [H, W, 3] o [H, W, 4]
        self.H, self.W  = image.shape[:2]
        fl_x = fl_y = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        cx = (transform['cx']) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy']) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.rays = []
        self.rgbs = []
        self.depths = []
        self.poses = []
        self.pc = []
        self.pc_rgbs = []

        for f in tqdm.tqdm(frames, desc=f'Loading {self.mode} data'):
            f_path = os.path.join(self.root_path, f['file_path'] + '.png')
            pose = nerf_matrix_to_ngp(np.array(f['transform_matrix'], dtype=np.float32), scale=self.scale)
            image = imageio.imread(f_path) / 255.0 # [H, W, 3] o [H, W, 4]
            depth = None

            if self.mode == 'train':
                f_path_depth = os.path.join(self.root_path, f['file_path'] + '_depth.npy')
                depth = np.load(f_path_depth).reshape(-1)

            ray_o, ray_d, depth = get_rays(torch.from_numpy(pose), torch.from_numpy(self.intrinsics), self.H, self.W, depth)

            # Scale the depth
            depth_mask = depth < 1000
            depth *= scale

            rgbs = torch.from_numpy(image).reshape(self.H * self.W, -1)
            self.poses.append(pose)
            self.rays.append(torch.cat([ray_o, ray_d], 1))
            self.rgbs.append(rgbs)

            if self.mode == 'train':
                self.depths.append(depth)
                self.pc.append(ray_o[depth_mask, :3] + ray_d[depth_mask, :3] * depth[depth_mask,None])
                self.pc_rgbs.append(rgbs[depth_mask,:3])

        self.rays = torch.vstack(self.rays)
        self.rgbs = torch.vstack(self.rgbs)

        if self.mode == 'train':
            self.depths = torch.cat(self.depths) # Note that depth denotes the distance along the ray
            self.pc = torch.vstack(self.pc)
            self.pc_rgbs = torch.vstack(self.pc_rgbs)

    def get_point_cloud(self, downsample_ratio: float = 1.0, return_color: bool = False) -> Union[torch.Tensor, TensorPair]:
        if self.mode == 'train':
            assert isinstance(self.pc, torch.Tensor)
            if return_color:
                assert isinstance(self.pc_rgbs, torch.Tensor)
                dri = int(1 / downsample_ratio)
                pts = self.pc[::dri, :]
                rgb = self.pc_rgbs[::dri, :]
                return pts, rgb
            return self.pc[::int(1/downsample_ratio),:]
        else:
            raise ValueError('Only training data has depth information!')

    def __len__(self):
        if self.mode == 'train':
            return 1000
        else:
            return self.n_frames


    def __getitem__(self, idx):
        # raise an error to now iterate in infinity
        if idx >= len(self): raise IndexError

        if self.mode == 'train':
            assert isinstance(self.rays, torch.Tensor)
            idxs = np.random.choice(self.rays.shape[0], self.num_rays)
            return  {'rays_o': self.rays[idxs,:3],
                     'rays_d': self.rays[idxs,3:6],
                     'rgba': self.rgbs[idxs],
                     'depth': self.depths[idxs],
                     'idxs': idxs}

        else:
            # raise an error to now iterate in infinity
            if idx >= len(self): raise IndexError
            assert isinstance(self.rays, torch.Tensor)
            assert isinstance(self.rgbs, torch.Tensor)
            return  {'rays_o': self.rays[idx * self.W * self.H : (idx + 1) * self.W * self.H, :3],
                     'rays_d': self.rays[idx * self.W * self.H : (idx + 1) * self.W * self.H, 3:6],
                     'rgba': self.rgbs[idx * self.W * self.H : (idx + 1) * self.W * self.H,],
                     'depth': None}


def make_ray_grid(origin, nrays, minb=(-0.45, -0.45), maxb=(0.45, 0.45),
                  device: Union[str, torch.device]='cpu', dtype=torch.float32):

    ray_o = torch.tensor([origin] * nrays**2) #+ p.mean(0, keepdim=True)
    ray_d = torch.from_numpy(
        np.stack([a.ravel() for a in
                  np.mgrid[minb[0]:maxb[0]:nrays*1j,
                           minb[1]:maxb[1]:nrays*1j]] +
                  [np.ones(nrays**2)], axis=-1).astype(np.float32))
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


def evaluate_density_and_color(dual_grid: GridBatch, sh_features: torch.Tensor, o_features: torch.Tensor,
                               ray_d: torch.Tensor, pts: torch.Tensor) -> TensorPair:

    pt_features = dual_grid.sample_trilinear(pts, sh_features.view(sh_features.shape[0], -1)).jdata.view(pts.shape[0], 3, 9)
    pt_o_features = dual_grid.sample_trilinear(pts, o_features.unsqueeze(-1)).jdata.squeeze(-1)
    return _TruncExp.apply(pt_o_features), torch.sigmoid(speherical_harmonics(2, pt_features, ray_d))


def render(primal_grid: GridBatch, dual_grid: GridBatch,
           sh_features: torch.Tensor, o_features: torch.Tensor,
           ray_o: torch.Tensor, ray_d: torch.Tensor,
           tmin: torch.Tensor, tmax: torch.Tensor, step_size: float,
           t_threshold: float = 0.0, chunk: bool = False) -> TensorTriple:

    pack_info, ray_idx, ray_intervals = \
        primal_grid.uniform_ray_samples(ray_o, ray_d, tmin, tmax, step_size)

    ray_t = ray_intervals.jdata.mean(1)
    ray_delta_t = (ray_intervals.jdata[:, 1] - ray_intervals.jdata[:, 0]).contiguous()
    ray_pts = ray_o[ray_idx.jdata] + ray_t[:, None] * ray_d[ray_idx.jdata]

    if chunk:
        ray_density = []
        ray_color = []
        ray_d = ray_d[ray_idx.jdata]
        chunk_size = 400000

        for i in range(ray_d.shape[0]//chunk_size + 1):
            ray_density_chunk, ray_color_chunk = evaluate_density_and_color(dual_grid, sh_features, o_features,
                                                                            ray_d[i*chunk_size:(i+1)*chunk_size, :],
                                                                            ray_pts[i*chunk_size:(i+1)*chunk_size, :])

            ray_density.append(ray_density_chunk)
            ray_color.append(ray_color_chunk)

        ray_density = torch.cat(ray_density, 0)
        ray_color = torch.vstack(ray_color)
    else:
        ray_density, ray_color = evaluate_density_and_color(dual_grid, sh_features, o_features,
                                                            ray_d[ray_idx.jdata], ray_pts)

    # Do the volume rendering
    # print(ray_density.shape, ray_color.shape, ray_delta_t.shape, ray_t.shape, pack_info.jdata.shape)
    rgb, depth, opacity, _, _ = volume_render(ray_density, ray_color, ray_delta_t,
                                                         ray_t, pack_info.jdata, t_threshold)

    return rgb, depth, opacity[:, None]


def tv_loss(dual_grid: GridBatch, ijk: torch.Tensor, sh_features: torch.Tensor, o_features: torch.Tensor, res) -> TensorPair:
    nhood = dual_grid.neighbor_indexes(ijk, 1).jdata.view(-1, 3, 3, 3)
    n_up = nhood[:, 1, 0, 0]
    n_right = nhood[:, 0, 1, 0]
    n_front = nhood[:, 0, 0, 1]
    n_center = nhood[:, 0, 0, 0]

    mask = torch.logical_and(torch.logical_and(n_center != -1, n_up != -1), n_front != -1)
    fmask = mask.float()
    n_up_mask, n_right_mask, n_center_mask, n_front_mask = n_up[mask], n_right[mask], n_center[mask], n_front[mask]

    diff_up_sh = (sh_features[n_up_mask] - sh_features[n_center_mask]) / (256.0 / res)
    diff_right_sh = (sh_features[n_right_mask] - sh_features[n_center_mask]) / (256.0 / res)
    diff_front_sh = (sh_features[n_front_mask] - sh_features[n_center_mask]) / (256.0 / res)

    diff_up_o = (o_features[n_up] * fmask - o_features[n_center]) / (256.0 / res)
    diff_right_o = (o_features[n_right] * fmask- o_features[n_center]) / (256.0 / res)
    diff_front_o = (o_features[n_front] * fmask - o_features[n_center]) / (256.0 / res)

    tv_reg_sh = (diff_up_sh ** 2.0 + diff_right_sh ** 2.0 + diff_front_sh ** 2.0).sum(-1).sum(-1)
    tv_reg_o = (diff_up_o ** 2.0 + diff_right_o ** 2.0 + diff_front_o ** 2.0)
    return tv_reg_sh.mean(), tv_reg_o.mean()


def main():
    # Configuration parameters
    device = torch.device('cuda')
    dtype = torch.float32
    # scene_aabb = 1.0
    starting_resolution = 256
    resolution = starting_resolution
    vox_size = (1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
    vox_origin = (vox_size[0]/2, vox_size[1]/2, vox_size[2]/2)
    ray_step_size = math.sqrt(3) / 512
    rays_per_batch = 4096
    lr_o = 1e-1
    lr_sh = 1e-2

    plot_every = 2
    num_epochs = 30
    bg_color = (0.0, 0.0, 0.0)
    t_threshold = 1e-5

    # Create the dataset. Assumes there is a file <repository_root>/data/lego_test.h5
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/lego/")

    if not os.path.exists(data_path):
        data_url = "https://drive.google.com/drive/folders/1i6qMn-mnPwPEioiNIFMO8QJlTjU0dS1b?usp=share_link"
        raise RuntimeError(f"You need to download the data at {data_url} "
                           "into <repository_root>/data "
                           "in order to run this script")

    train_dataset = NeRFDataset(data_path, scale=0.33, num_rays=rays_per_batch, mode='train')
    test_dataset = NeRFDataset(data_path, scale=0.33, num_rays=rays_per_batch, mode='test')


    # Create a sparse grid used to support features and do ray queries
    print("Building grid...")
    primal_grid = GridBatch(device=device)
    primal_grid.set_from_dense_grid(1, [resolution]*3, [-resolution//2]*3, voxel_sizes=vox_size, voxel_origins=vox_origin)
    dual_grid = primal_grid # primal_grid.dual_grid()

    print("Done bulding the grid!")

    # Initialize features at the voxel centers
    xyz = primal_grid.ijk.jdata / (resolution / 2.0)
    print(xyz.min(0)[0], xyz.max(0)[0])

    sh_features = torch.stack([eval_sh_bases(9, xyz)]*3, dim=1)
    print(sh_features.shape)
    sh_features = sh_features.to(device=device, dtype=dtype)
    o_features = torch.rand(dual_grid.total_voxels)
    o_features = o_features.to(device=device, dtype=dtype)
    o_features.requires_grad = True
    sh_features.requires_grad = True

    # Init optimizer
    param_group = []
    param_group.append({'params': o_features, 'lr': lr_o })
    param_group.append({'params': sh_features, 'lr': lr_sh })
    # optimizer = torch.optim.Adam(param_group)
    optimizer = torch.optim.RMSprop(param_group)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=lr/30)


    print("Starting training!")
    ps.init()  # Initialize 3d plotting
    for epoch in tqdm.trange(num_epochs):
        if resolution <= starting_resolution:
            all_ijk = dual_grid.ijk.jdata
        else:
            all_ijk = None
        pbar = tqdm.tqdm(enumerate(train_dataset))  # type: ignore
        for _, batch in pbar: # type: ignore
            optimizer.zero_grad()
            ray_o, ray_d = batch['rays_o'].to(device=device, dtype=dtype), \
                           batch['rays_d'].to(device=device, dtype=dtype)
            tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
            tmax = torch.full_like(tmin, 1e10)

            # Render color and depth along rays
            rgb, depth, opacity = render(primal_grid, dual_grid, sh_features, o_features, ray_o, ray_d,
                                         tmin, tmax, ray_step_size, t_threshold=t_threshold)

            rgb = opacity * rgb + (1.0 - opacity) * torch.tensor(bg_color).to(rgb)[None, :]

            # RGB loss
            rgb_gt = batch['rgba'].to(rgb)[:,:3]
            loss_rgb = torch.nn.functional.mse_loss(rgb, rgb_gt) # torch.nn.functional.huber_loss(rgb, rgb_gt) / 5.

            # Depth loss
            # depth_gt = batch['depth'].to(rgb)
            # depth_mask = depth_gt < np.sqrt(3) # Mask out rays that miss the object
            # loss_depth = torch.nn.functional.l1_loss(depth[depth_mask], depth_gt[depth_mask]) / 100

            if resolution <= starting_resolution:
                assert all_ijk is not None
                random_ijk = all_ijk[torch.randperm(all_ijk.shape[0])[:int(.1*dual_grid.total_voxels)]]
                tv_reg_sh, tv_reg_o = tv_loss(dual_grid, random_ijk, sh_features, o_features, resolution)
                tv_reg = 1e-1 * tv_reg_sh + 1e-2 * tv_reg_o
            else:
                tv_reg_sh = torch.tensor([0.0]).to(device)
                tv_reg_o = torch.tensor([0.0]).to(device)
                tv_reg = torch.tensor([0.0]).to(device)
            # Total loss to minimize
            loss = loss_rgb + tv_reg #+ 0.1 * loss_depth

            loss.backward()
            optimizer.step()

            # Compute current PSNR
            psnr = compute_psnr(rgb, rgb_gt)

            # Log losses in tqdm progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                              "Loss RGB": f"{loss_rgb.item():.4f}",
                            #   "Loss Depth": f"{loss_depth.item():.4f}",
                              "Loss TV (sh)": f"{tv_reg_sh.item():.4f}",
                              "Loss TV (o)": f"{tv_reg_o.item():.4f}",
                              "PSNR": f"{psnr.item():.2f}"})

        # scheduler.step()

        if epoch % plot_every == 0:
            with torch.no_grad():
                torch.cuda.empty_cache()

                grid_res = 512
                ray_o, ray_d = make_ray_grid((0., 0.15, 1.2), grid_res, device=device, dtype=dtype)
                ray_d = - ray_d
                tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
                tmax = torch.full_like(tmin, 1e10)
                rgb, depth, opacity = render(primal_grid,dual_grid, sh_features, o_features, ray_o, ray_d,
                                             tmin, tmax, ray_step_size, t_threshold=t_threshold, chunk=True)
                rgb = opacity * rgb + (1.0 - opacity) * torch.tensor(bg_color).to(rgb)[None, :]

                rgb_img = rgb.clip(0.0, 1.0).detach().cpu().numpy().reshape([grid_res, grid_res, 3])
                depth_img = depth.detach().cpu().numpy().reshape([grid_res, grid_res])

                plt.figure()
                plt.imshow(rgb_img)
                plt.figure()
                plt.imshow(depth_img)
                plt.show()

                ray_v = torch.cat([ray_o, ray_o + ray_d*0.33]).cpu().numpy()
                ray_e = np.array([[i, i + ray_o.shape[0]] for i in range(ray_o.shape[0])])

                ps.register_curve_network("rays", ray_v, ray_e, radius=0.00002)
                vox_ijk = primal_grid.ijk.jdata
                vox_ctrs = primal_grid.grid_to_world(vox_ijk.to(dtype)).jdata
                vox_density, vox_color = evaluate_density_and_color(dual_grid, sh_features, o_features,
                                                                    torch.ones_like(vox_ctrs), vox_ctrs)

                # Subdivide
                if epoch > 0:
                    sh_features, sub_grid = dual_grid.subdivide(2, sh_features.view(sh_features.shape[0], -1), mask=vox_density > 0.25)
                    o_features, sub_grid = dual_grid.subdivide(2, o_features.unsqueeze(-1), mask=vox_density > 0.25)
                    o_features = o_features.jdata.squeeze(-1)
                    sh_features = sh_features.jdata.reshape(sh_features.r_shape[0], 3, -1)
                    sh_features.requires_grad = True
                    o_features.requires_grad = True
                    resolution *= 2.0
                    ray_step_size /= 2.0

                    print(f"Subdivided grid with {dual_grid.total_voxels} to {sub_grid.total_voxels}")
                    dual_grid = sub_grid
                    primal_grid = sub_grid

                camera_origins = []
                for pose in test_dataset.poses:
                    camera_origins.append(pose @ np.array([0.0, 0.0, 0.0, 1.0]))
                camera_origins = np.stack(camera_origins)[:, :3]
                ps.register_point_cloud("camera origins", camera_origins)

                v, e = primal_grid.viz_edge_network
                v, e = v.jdata, e.jdata
                ps.register_curve_network("grid", v.cpu(), e.cpu(), radius=0.0001)
                pc = ps.register_point_cloud("vox centers", vox_ctrs.cpu(),
                                            point_render_mode='quad')
                pc.add_scalar_quantity("density", vox_density.cpu(), enabled=True)
                pc.add_scalar_quantity("density thresh", (vox_density.cpu() < .25).float(), enabled=True)
                pc.add_color_quantity("rgb", vox_color.cpu(), enabled=False)
                ps.show()


    print("Starting testing!")
    pbar = tqdm.tqdm(enumerate(test_dataset))  # type: ignore
    psnr_test = []
    for _, batch in pbar: # type: ignore
        with torch.no_grad():
            ray_o, ray_d = batch['rays_o'].to(device=device, dtype=dtype), \
                           batch['rays_d'].to(device=device, dtype=dtype)
            tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
            tmax = torch.full_like(tmin, 1e10)
            rgb_gt = batch['rgba'].to(rgb)[:,:3]  # type: ignore

            # Render color and depth along rays
            rgb, depth, opacity = render(primal_grid, dual_grid, sh_features, o_features, ray_o, ray_d,
                                         tmin, tmax, ray_step_size, t_threshold=t_threshold, chunk=True)

            rgb = opacity * rgb + (1.0 - opacity) * torch.tensor(bg_color).to(rgb)[None, :]

            # Compute current PSNR
            psnr = compute_psnr(rgb, rgb_gt)
            psnr_test.append(psnr.item())
            # Log losses in tqdm progress bar
            pbar.set_postfix({"PSNR": f"{psnr.item():.2f}"})

    print(f"Mean PSNR on the test set across {len(test_dataset)} images: {torch.tensor(psnr_test).mean().item()} ")

if __name__ == "__main__":
    main()

```
