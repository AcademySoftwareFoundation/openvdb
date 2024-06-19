import gzip
import json
import math
import os
from typing import Optional, Tuple, Union

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import polyscope as ps
import torch
import tqdm
from torch import nn

from fvdb import GridBatch
from fvdb.utils import volume_render

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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

class ViewDependentMlp(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_hidden: int):
        super().__init__()
        fc_layers = [
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU()
        ]
        for i in range(num_hidden):
            if i == num_hidden - 1:
                fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                fc_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU()
                ])
        self.fc = nn.Sequential(*fc_layers)
        self.fc_clr = nn.Linear(hidden_dim + 3, 3)
        self.fc_density = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor, view_dirs: torch.Tensor) -> TensorPair:
        x = self.fc(features)
        density = self.softplus(self.fc_density(x)).squeeze(-1)
        color = torch.sigmoid(self.fc_clr(torch.cat([x, view_dirs], dim=-1)))

        return density, color


class _TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class DirectionalEmbedding(nn.Module):
    def __init__(self, num_frequency: int):
        super().__init__()

        self.num_frequency = num_frequency
        self.fncs = [torch.sin, torch.cos]
        self.out_channels = 3*(len(self.fncs)*self.num_frequency+1)
        self.freq_bands = 2**torch.linspace(0, num_frequency-1, num_frequency)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            for func in self.fncs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class ViewDependentMultiMlp(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int,  num_frequency: int):
        super().__init__()

        layers_color = [
            nn.Linear(16 + 2 * 3 + 2 * 6 * num_frequency, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        layers_density = [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        ]

        layers_color.append(nn.Linear(hidden_dim, 3))
        layers_density.append(nn.Linear(hidden_dim, 16))

        self.dir_embedder = DirectionalEmbedding(num_frequency)
        self.mlp_color = nn.Sequential(*layers_color)
        self.mlp_density = nn.Sequential(*layers_density)
        self.trunc_exp = _TruncExp.apply

    def forward(self, features: torch.Tensor, view_dirs: torch.Tensor, pts: torch.Tensor) -> TensorPair:
        h = self.mlp_density(features)
        density = self.trunc_exp(h[:,0])
        assert isinstance(density, torch.Tensor)
        color = torch.sigmoid(self.mlp_color(torch.cat([h, self.dir_embedder(view_dirs), self.dir_embedder(pts)], dim=-1)))

        return density, color


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


def evaluate_density_and_color(fvdb: GridBatch, features: torch.Tensor, ray_d: torch.Tensor,
                               mlp: nn.Module, pts: torch.Tensor) -> TensorPair:

    pt_features = fvdb.sample_trilinear(pts, features).jdata
    return mlp(pt_features, ray_d, pts)


def render(fvdb: GridBatch, fvdb_dual: GridBatch, features: torch.Tensor, mlp: nn.Module,
           ray_o: torch.Tensor, ray_d: torch.Tensor,
           tmin: torch.Tensor, tmax: torch.Tensor, step_size: float,
           t_threshold: float = 0.0, chunk: bool = False) -> TensorTriple:

    pack_info, ray_idx, ray_intervals = \
        fvdb.uniform_ray_samples(ray_o, ray_d, tmin, tmax, step_size)

    ray_t = ray_intervals.jdata.mean(1)
    ray_delta_t = (ray_intervals.jdata[:, 1] - ray_intervals.jdata[:, 0]).contiguous()
    ray_pts = ray_o[ray_idx.jdata] + ray_t[:, None] * ray_d[ray_idx.jdata]

    if chunk:
        ray_density = []
        ray_color = []
        ray_d = ray_d[ray_idx.jdata]
        chunk_size = 400000

        for i in range(ray_d.shape[0]//chunk_size + 1):
            ray_density_chunk, ray_color_chunk = evaluate_density_and_color(fvdb_dual, features, ray_d[i*chunk_size:(i+1)*chunk_size, :],
                                                            mlp, ray_pts[i*chunk_size:(i+1)*chunk_size, :])

            ray_density.append(ray_density_chunk)
            ray_color.append(ray_color_chunk)

        ray_density = torch.cat(ray_density, 0)
        ray_color = torch.vstack(ray_color)
    else:
         ray_density, ray_color = evaluate_density_and_color(fvdb_dual, features, ray_d[ray_idx.jdata],
                                                             mlp, ray_pts)

    # Do the volume rendering
    rgb, depth, opacity, ws, tot_samples = volume_render(ray_density, ray_color, ray_delta_t,
                                                         ray_t, pack_info.jdata, t_threshold)

    return rgb, depth, opacity[:, None]


def load_ingp_grid(ingp_path: str) -> torch.Tensor:
    def dilate3(r: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        r = (r | (r << 16)) & 0xFF0000FF
        r = (r | (r << 8)) & 0x0F00F00F
        r = (r | (r << 4)) & 0xC30C30C3
        r = (r | (r << 2)) & 0x49249249
        return r

    def morton3(xyz: np.ndarray) -> np.ndarray:
        return np.dot(dilate3(xyz), [1, 2, 4])

    ingp_data = msgpack.unpackb(gzip.open(ingp_path, 'rb').read())

    dgb = ingp_data['snapshot']['density_grid_binary']

    r = np.arange(128)
    zs, ys, xs = np.meshgrid(r, r, r,indexing='ij')
    grid = np.stack((xs,ys,zs),axis=-1)
    morton_grid = morton3(grid)
    dg = np.frombuffer(dgb, dtype=np.float16)
    density_grid = dg[morton_grid]
    keep_mask = density_grid > 0.01

    ijk = torch.from_numpy(np.stack([a[keep_mask] for a in (xs, ys, zs)], axis=-1))

    return ijk


def main():
    # Configuration parameters
    device = torch.device('cuda')
    dtype = torch.float32
    scene_aabb = 1.0
    resolution = 256
    vox_size = (scene_aabb / resolution, scene_aabb / resolution, scene_aabb / resolution)
    vox_origin = (vox_size[0]/2, vox_size[1]/2, vox_size[2]/2)
    ray_step_size = math.sqrt(3) * scene_aabb / 1024
    rays_per_batch = 1024 * 8
    lr = 0.001
    lr_features = 0.1
    feature_dim = 16
    hidden_dim = 128
    num_frequency = 4
    model_class = ViewDependentMultiMlp
    voxel_padding = 6
    plot_every = 5
    num_epochs = 30
    bg_color = (0., 0., 0.)
    t_threshold = 1e-4
    init_mode = "ingp"

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
    accpts_small = None
    if init_mode == "ingp":
        ingp_path = os.path.join(os.path.dirname(__file__), "..", "data/digger.ingp")
        feature_grid = GridBatch(device=device)
        ijk = load_ingp_grid(ingp_path) - 64
        feature_grid.set_from_ijk(ijk.to(device), voxel_sizes=1.0/128, origins=[1.0/128]*3)
        # feature_grid = grid_coafeature_gridrse.subdivided_grid(2)
    elif init_mode == "point_cloud":
        accpts_small = train_dataset.get_point_cloud(downsample_ratio=0.05)
        assert isinstance(accpts_small, torch.Tensor)
        accpts_small = accpts_small.cuda()
        feature_grid = GridBatch(device)
        feature_grid.set_from_points(accpts_small, [-voxel_padding]*3, [voxel_padding]*3, voxel_sizes=vox_size, origins=vox_origin)
    else:
        feature_grid = GridBatch(device=device)
        points = torch.stack(torch.meshgrid([torch.arange(resolution)]*3, indexing="xy"), dim=-1,).reshape(-1, 3).to(device) - resolution/2
        index = torch.LongTensor([1, 2, 0])
        points = points[:, index]
        feature_grid.set_from_ijk(points.to(dtype=torch.int32, device= device), [0,0,0], [0,0,0], voxel_sizes=vox_size, origins=vox_origin)

    fvdb_dual = feature_grid.dual_grid()

    print("Done bulding the FVDB!")

    # Initialize features at the voxel centers
    features = torch.randn(fvdb_dual.total_voxels, feature_dim) / np.sqrt(feature_dim)
    features = features.to(device=device, dtype=dtype)
    features.requires_grad = True

    # Create an MLP that maps features and view directions to density and color
    mlp = model_class(feature_dim, hidden_dim, num_frequency).to(device=device, dtype=dtype)

    # Init optimizer
    param_group = []
    param_group.append({'params': mlp.parameters(), 'lr': lr })
    param_group.append({'params': features, 'lr': lr_features })
    optimizer = torch.optim.Adam(param_group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=lr/30)

    print("Starting training!")
    ps.init()  # Initialize 3d plotting
    for epoch in tqdm.trange(num_epochs):
        pbar = tqdm.tqdm(enumerate(train_dataset))  # type: ignore
        for _, batch in pbar: # type: ignore
            optimizer.zero_grad()
            ray_o, ray_d = batch['rays_o'].to(device=device, dtype=dtype), \
                           batch['rays_d'].to(device=device, dtype=dtype)
            tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
            tmax = torch.full_like(tmin, 1e10)

            # Render color and depth along rays
            rgb, depth, opacity = render(feature_grid, fvdb_dual, features, mlp, ray_o, ray_d,
                                         tmin, tmax, ray_step_size, t_threshold=t_threshold)

            rgb = opacity * rgb + (1.0 - opacity) * torch.tensor(bg_color).to(rgb)[None, :]

            # RGB loss
            rgb_gt = batch['rgba'].to(rgb)[:,:3]
            loss_rgb = torch.nn.functional.huber_loss(rgb, rgb_gt) / 5.

            # Depth loss
            depth_gt = batch['depth'].to(rgb)
            depth_mask = depth_gt < np.sqrt(3) * scene_aabb # Mask out rays that miss the object
            loss_depth = torch.nn.functional.l1_loss(depth[depth_mask], depth_gt[depth_mask]) / 100

            # Total loss to minimize
            loss = loss_rgb # + 0.1 * loss_depth

            loss.backward()
            optimizer.step()

            # Compute current PSNR
            psnr = compute_psnr(rgb, rgb_gt)

            # Log losses in tqdm progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                              "Loss RGB": f"{loss_rgb.item():.4f}",
                              "Loss Depth": f"{loss_depth.item():.4f}",
                              "PSNR": f"{psnr.item():.2f}"})

        scheduler.step()

        if epoch % plot_every == 0:
            with torch.no_grad():
                torch.cuda.empty_cache()

                grid_res = 512
                ray_o, ray_d = make_ray_grid((0., 0.15, 1.2), grid_res, device=device, dtype=dtype)
                ray_d = - ray_d
                tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
                tmax = torch.full_like(tmin, 1e10)
                rgb, depth, opacity = render(feature_grid,fvdb_dual, features, mlp, ray_o, ray_d,
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

                ps.register_curve_network("rays", ray_v, ray_e)
                vox_ijk = feature_grid.ijk.jdata
                vox_ctrs = feature_grid.grid_to_world(vox_ijk.to(dtype)).jdata
                vox_density, vox_color = evaluate_density_and_color(fvdb_dual, features,
                                                                    torch.ones_like(vox_ctrs),
                                                                    mlp, vox_ctrs)

                camera_origins = []
                for pose in test_dataset.poses:
                    camera_origins.append(pose @ np.array([0.0, 0.0, 0.0, 1.0]))
                camera_origins = np.stack(camera_origins)[:, :3]
                ps.register_point_cloud("camera origins", camera_origins)

                if isinstance(accpts_small, torch.Tensor):
                    ps.register_point_cloud("pts", accpts_small.cpu(), radius=0.001)
                v, e = feature_grid.viz_edge_network
                v, e = v.jdata, e.jdata
                ps.register_curve_network("grid", v.cpu(), e.cpu(), radius=0.0001)
                pc = ps.register_point_cloud("vox centers", vox_ctrs.cpu(),
                                            point_render_mode='quad')
                pc.add_scalar_quantity("density", vox_density.cpu(), enabled=True)
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
            rgb, depth, opacity = render(feature_grid, fvdb_dual, features, mlp, ray_o, ray_d,
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

