
import os
import torch
from torch import nn
import tqdm
from typing import Union

from debug_nerf import NeRFDataset, ViewDependentMultiMlp, render, compute_psnr
from fvdb import SparseIndexGrid

def test_model(fvdb: SparseIndexGrid, fvdb_dual: SparseIndexGrid, features: torch.Tensor, mlp: nn.Module,
               device: Union[str, torch.device] = 'cuda', dtype = torch.float32) -> float:

    pbar = tqdm.tqdm(enumerate(test_dataset))
    psnr_test = []

    for _, batch in pbar:
        with torch.no_grad():
            ray_o, ray_d = batch['rays_o'].to(device=device, dtype=dtype), \
                            batch['rays_d'].to(device=device, dtype=dtype)
            tmin = torch.zeros(ray_o.shape[0]).to(ray_o)
            tmax = torch.full_like(tmin, 1e10)


            # Render color and depth along rays
            rgb, depth, opacity = render(fvdb, fvdb_dual, features, mlp, ray_o, ray_d,
                                            tmin, tmax, 0.0016914558667664816, t_threshold=1e-4, chunk=True)

            rgb = opacity * rgb + (1.0 - opacity) * torch.tensor((0.0,0.0,0.0)).to(rgb)[None, :]

            # Compute current PSNR
            rgb_gt = batch['rgba'].to(rgb)[:,:3]
            psnr = compute_psnr(rgb, rgb_gt)
            psnr_test.append(psnr.item())
            # Log losses in tqdm progress bar
            pbar.set_postfix({"PSNR": f"{psnr.item():.2f}"})

    return torch.tensor(psnr_test).mean().item()


before_compression_path = ['fvdb_nerf_fvdb.pth', 'fvdb_nerf_features.pth']
if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data/lego/")):
        data_url = "https://drive.google.com/drive/folders/1i6qMn-mnPwPEioiNIFMO8QJlTjU0dS1b?usp=share_link"
        raise RuntimeError(f"You need to download the data at {data_url} "
                           "into <repository_root>/data "
                           "in order to run this script")

test_dataset = NeRFDataset(os.path.join(os.path.dirname(__file__), "..", "data/lego/"), scale=0.33, num_rays=0, mode='test')
mlp = ViewDependentMultiMlp(16, 128, 4).to(device='cuda', dtype=torch.float32)

# Path to the models
path_before_compression = 'data/fvdb_nerf.pth'
path_after_compression = 'path/after/compression/'

# Load the model before compression

before_compression = torch.load(path_before_compression)
mlp.load_state_dict(before_compression['mlp'])
fvdb_before = before_compression['fvdb']
fvdb_before_dual = fvdb_before.dual_grid()
features_before = before_compression['features'].to('cuda')


psnr_before = test_model(fvdb_before, fvdb_before_dual, features_before, mlp)
print(f'PSNR before compression: {psnr_before}')


# Load the model after compression
after_compression = torch.load(path_after_compression)
mlp.load_state_dict(after_compression['mlp'])
fvdb_after = after_compression['fvdb']
fvdb_after_dual = fvdb_after.dual_grid()
features_after = after_compression['features'].to('cuda')


psnr_after = test_model(fvdb_after, fvdb_after_dual, features_after, mlp)
print(f'PSNR before compression: {psnr_after}')