import os

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
import tqdm
from fvdb import GridBatch


def prepare_sdf(data_path, npts, ng):
    print("Loading data...")
    v, f = pcu.load_mesh_vf(data_path)
    v -= v.min(0)
    v /= v.max()
    v -= 0.5 * v.max()
    n = pcu.estimate_mesh_vertex_normals(v, f)
    fid, bc = pcu.sample_mesh_poisson_disk(v, f, npts)
    pts = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    nms = pcu.interpolate_barycentric_coords(f, fid, bc, n)
    print("Done")

    print("Generating grid samples")
    gpts = np.stack([
        a.ravel() for a in
        np.mgrid[v.min(0)[0]*1.05:v.max(0)[0]*1.05:ng*1j,
                 v.min(0)[1]*1.05:v.max(0)[1]*1.05:ng*1j,
                 v.min(0)[2]*1.05:v.max(0)[2]*1.05:ng*1j]
    ], axis=-1).astype(pts.dtype)
    print("Done")

    print("Computing SDF")
    sdf, _, _ = pcu.signed_distance_to_mesh(gpts, v, f)
    print("Done")

    return pts, nms, gpts, sdf


def main():
    device = 'cuda'
    dtype = torch.float32
    vox_size = 0.005
    vox_origin = np.zeros(3)
    vox_pad = 1
    ng = 256
    npts = 10_000
    num_iters = 7500
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/lavalamp.ply")

    # Cache the sdf data so we don't have to recompute it every single time
    if not os.path.exists("sdf.npz"):
        pts, nms, gpts, sdf = prepare_sdf(data_path, npts, ng)
        np.savez("sdf.npz", pts=pts, nms=nms, gpts=gpts, sdf=sdf)
    else:
        dat = np.load("sdf.npz")
        pts, nms, gpts, sdf = dat['pts'], dat['nms'], dat['gpts'], dat['sdf']

    p, n = torch.from_numpy(pts).to(device).to(dtype), \
           torch.from_numpy(nms).to(device).to(dtype)

    index = GridBatch(device=device)

    index.set_from_points(p, [-vox_pad] * 3, [vox_pad] * 3, vox_size, vox_origin)
    dual_index = index.dual_grid()

    mask = index.points_in_active_voxel(torch.from_numpy(gpts).to(dtype).to(device)).jdata.cpu().numpy()
    vol_pts = torch.from_numpy(gpts[mask]).to(device=device, dtype=dtype)
    vol_sdf = torch.from_numpy(sdf[mask]).to(device=device, dtype=dtype).unsqueeze(-1)

    p, n = p.to(device), n.to(device)

    features = torch.randn(dual_index.total_voxels, 1).to(device).to(dtype)
    features.requires_grad = True

    optimizer = torch.optim.Adam([features], lr=1e-2)

    # This should converge to around 2e-10 loss
    pbar = tqdm.tqdm(range(num_iters))
    for _ in pbar:
        optimizer.zero_grad()
        vp_idx = torch.randperm(vol_pts.shape[0])
        vpts = vol_pts[vp_idx]
        vsdf = vol_sdf[vp_idx]

        samp_sdf = dual_index.sample_trilinear(vpts, features).jdata
        assert isinstance(samp_sdf, torch.Tensor)

        loss = torch.nn.functional.mse_loss(samp_sdf, vsdf)
        loss.backward()
        pbar.set_postfix({"Loss": loss.item()})
        optimizer.step()

    ps.init()
    pred_sdf = dual_index.sample_trilinear(vol_pts, features).jdata
    assert isinstance(pred_sdf, torch.Tensor)
    vol_pc = ps.register_point_cloud("pts", vol_pts.cpu().numpy())
    vol_pc.add_scalar_quantity("sdf_pred", pred_sdf.squeeze().detach().cpu().numpy())
    vol_pc.add_scalar_quantity("sdf_gt", vol_sdf.squeeze().detach().cpu().numpy())
    vol_pc.add_scalar_quantity("delta", (vol_sdf - pred_sdf).squeeze().abs().detach().cpu().numpy())
    ps.show()


if __name__ == "__main__":
    main()
