import polyscope as ps
import torch
from fvdb import SparseIndexGrid


def main():
    device = "cpu"
    dtype = torch.float64

    vox_size = 1.0
    vox_origin = [0.0, 0.0, 0.0]
    p = torch.Tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]).to(dtype).to(device)

    grid = SparseIndexGrid(vox_size, vox_origin, device=device)
    grid.build_from_pointcloud(p, [0, 0, 0], [0, 0, 0])
    features = torch.Tensor([[1],[0],[0],[0]]).to(dtype).to(device)
    print(features.dim())

    kernel = torch.rand(1,1,3,3,3).to(dtype).to(device)
    print(kernel.dim())
    kernel =torch.Tensor ([[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]).unsqueeze(0).unsqueeze(0).to(dtype).to(device)
    print(kernel.shape)
    identity_convolve_features = grid.sparse_conv_inplace(features, kernel)
    print(f"in features {str(features)}")
    print(f"kernel {str(kernel)}")
    print(f"output {str(identity_convolve_features)}")

    kernel = torch.where(kernel==0, 1., 0.).to(torch.float64)
    inverse_convolve_features = grid.sparse_conv_inplace(features, kernel)
    print(f"kernel {str(kernel)}")
    print(f"output {str(inverse_convolve_features)}")


    ps.init()
    ps.set_ground_plane_mode('none')
    grid_pts = ps.register_point_cloud("points", p, radius=vox_size/4)
    grid_pts.add_scalar_quantity("input features", features.flatten(), enabled=True)
    grid_pts.add_scalar_quantity("identity conv", identity_convolve_features.flatten(),) #enabled=True)
    grid_pts.add_scalar_quantity("inverse conv", inverse_convolve_features.flatten(),)

    ps.show()

if __name__ == "__main__":
    main()
