import torch
from argparse import ArgumentParser

import open3d as o3d
import numpy as np
import MinkowskiEngine as ME
import fvdb
import fvdb.nn as fvnn
import torch_scatter


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", type=str, help="path to the input ply file")
    parser.add_argument("--me", action="store_true", help="Use MinkowskiEngine")

    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.data)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    voxel_size = 0.02

    device = "cuda:0"

    if args.me:
        from examples.networks.perception.minkunet import MinkUNet34C
        model = MinkUNet34C(3, 20).to(device)
    else:
        from examples.networks.perception.fvdbunet import FVDBUNet34C
        model = FVDBUNet34C(3, 20).to(device)

    model.eval()

    if args.me:
        with torch.no_grad():
            # Feed-forward pass and get the prediction
            in_field = ME.TensorField(
                features=torch.from_numpy(colors).float() - 0.5,
                coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=device,
            )
            # Convert to a sparse tensor
            sinput = in_field.sparse()
            # Output sparse tensor
            soutput = model(sinput)
            # get the prediction on the input tensor field
            out_field = soutput.slice(in_field)
            logits = out_field.F

    else:
        coords = torch.from_numpy(coords).float().to(device)
        features = torch.from_numpy(colors).float().to(device) - 0.5

        with torch.no_grad():
            grid = fvdb.sparse_grid_from_points(coords, voxel_sizes=[voxel_size] * 3)

            # For each voxel, compute the average color of the points inside:
            #   First determine the voxel id of each point
            pts_vox_idx = grid.ijk_to_index(grid.world_to_grid(coords).jdata.round().long()).jdata
            pts_valid_mask = pts_vox_idx >= 0
            #   Then compute the average color of the points in each voxel
            features = fvdb.JaggedTensor([
                torch_scatter.scatter_mean(features[pts_valid_mask], pts_vox_idx[pts_valid_mask], dim=0, dim_size=grid.total_voxels)
            ])
            sinput = fvnn.VDBTensor(grid, features)
            # Get the prediction
            soutput = model(sinput)
            logits = soutput.jdata

    _, pred = logits.max(1)
    pred = pred.cpu().numpy()

    print(f"Done -- prediction is {pred}.")
