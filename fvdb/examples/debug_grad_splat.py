import os
import time
import tqdm
import point_cloud_utils as pcu
import polyscope as ps
import torch
from fvdb import SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32

    vox_size = 0.0025
    vox_origin = torch.zeros(3)
    N = 1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dragon.ply")
    pts, nms = pcu.load_mesh_vn(data_path)
    p, n = torch.from_numpy(pts[::N]).to(device).to(dtype), \
           torch.from_numpy(nms[::N]).to(device).to(dtype)

    grid = SparseIndexGrid(vox_size, vox_origin, device=device)

    grid.build_from_pointcloud(p, [-1, -1, -1], [1, 1, 1])
    dual_index = grid.dual_grid()

    print("Splatting into grid...")
    start = time.time()
    target_n = grid.splat_trilinear(p, n)
    assert isinstance(target_n, torch.Tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f"Done in {time.time() - start}s!")

    n = torch.randn_like(n)
    n.requires_grad = True

    print("Optimizing point normals to agree with splat...")
    optimizer = torch.optim.Adam([n], lr=1e-1)
    pbar = tqdm.tqdm(range(1000))
    for _ in pbar:
        optimizer.zero_grad()
        nsplat_i = grid.splat_trilinear(p, n)
        assert isinstance(nsplat_i, torch.Tensor)
        loss = torch.nn.functional.mse_loss(nsplat_i, target_n)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss:" : loss.item()})
    pred_n = grid.splat_trilinear(p, n)
    assert isinstance(pred_n, torch.Tensor)

    gp = grid.active_grid_coords()
    gd = dual_index.active_grid_coords()
    gp = grid.grid_to_world(gp.to(dtype))
    gd = dual_index.grid_to_world(gd.to(dtype))

    p, n = p.cpu(), n.detach().cpu()
    gp, gd = gp.cpu(), gd.cpu()
    pred_n, target_n = pred_n.detach().cpu(), target_n.cpu()

    ps.init()
    pcloud = ps.register_point_cloud("points", p, radius=0.00075)
    pcloud.add_vector_quantity("learned normals", n)
    grid_pts = ps.register_point_cloud("vox coords", gp, radius=0.0005)
    grid_pts.add_vector_quantity("predicted splatted normals", pred_n,
                                 enabled=True, length=0.05, radius=0.001)
    grid_pts.add_vector_quantity("target splatted normals", target_n,
                                 enabled=True, length=0.05, radius=0.001)
    ps.show()


if __name__ == "__main__":
    main()
