import os
import polyscope as ps
import tempfile
import torch
import subprocess

import fvdb


def save_random_batch(save_dir: str) -> str:
    p = fvdb.JaggedTensor(
        [
            torch.randn(10, 3),
            torch.randn(100, 3),
        ]
    )
    grid = fvdb.sparse_grid_from_points(
        p, voxel_sizes=[[0.1, 0.1, 0.1], [0.15, 0.15, 0.15]], origins=[0.0] * 3
    )

    # save the grid and features to a compressed nvdb file
    path = os.path.join(save_dir, "two_random_grids.nvdb")
    fvdb.save(path, grid, names=["taco1", "taco2"], compressed=True)

    return path


def save_random_batch_with_data(save_dir: str) -> str:
    p = fvdb.JaggedTensor(
        [
            torch.randn(10, 3),
            torch.randn(100, 3),
        ]
    )
    grid = fvdb.sparse_grid_from_points(
        p, voxel_sizes=[[0.1, 0.1, 0.1], [0.15, 0.15, 0.15]], origins=[0.0] * 3
    )

    # a single, scalar float feature per grid
    feats = fvdb.JaggedTensor([torch.randn(x, 1) for x in grid.num_voxels])

    # save the grid and features to a compressed nvdb file
    path = os.path.join(save_dir, "two_random_grids.nvdb")
    fvdb.save(path, grid, feats, names=["taco1", "taco2"], compressed=True)

    return path


def save_random_batch_with_vec3d_data(save_dir: str) -> str:
    p = fvdb.JaggedTensor(
        [
            torch.randn(10, 3),
            torch.randn(100, 3),
        ]
    )
    grid = fvdb.sparse_grid_from_points(
        p, voxel_sizes=[[0.1, 0.1, 0.1], [0.15, 0.15, 0.15]], origins=[0.0] * 3
    )

    # a 3-vector double feature per grid
    feats = fvdb.JaggedTensor([torch.randn(x, 3, dtype=torch.float64) for x in grid.num_voxels])

    # save the grid and features to a compressed nvdb file
    path = os.path.join(save_dir, "two_random_vec3d_grids.nvdb")
    fvdb.save(path, grid, feats, names=["taco1", "taco2"], compressed=True)

    return path


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_nvdb = save_random_batch(tmpdir)
        print("nanovdb_print of two random grids saved to: ", saved_nvdb)
        print(subprocess.check_output(["nanovdb_print", saved_nvdb]).decode("utf-8"))

        saved_nvdb = save_random_batch_with_data(tmpdir)
        print("nanovdb_print of two random grids of float data saved to: ", saved_nvdb)
        print(subprocess.check_output(["nanovdb_print", saved_nvdb]).decode("utf-8"))

        saved_nvdb = save_random_batch_with_vec3d_data(tmpdir)
        print("nanovdb_print of two random grids of vec3d data saved to: ", saved_nvdb)
        print(subprocess.check_output(["nanovdb_print", saved_nvdb]).decode("utf-8"))

        # Load the grid and features from the compressed nvdb file
        grid_batch, features, names = fvdb.load(saved_nvdb, device=torch.device("cuda:0"))
        print("Loaded grid batch total number of voxels: ", grid_batch.total_voxels)
        print("Loaded grid batch data type: %s, device: %s" % (features.dtype, features.device))
        print("\n")

        vdb_path = os.path.join(tmpdir, "two_random_grids.vdb")
        convert_cmd = "nanovdb_convert -v %s %s" % (saved_nvdb, vdb_path)
        print("nanovdb_convert our nvdb to vdb: ", convert_cmd)
        print(subprocess.check_output(convert_cmd.split()).decode("utf-8"))

        convert_cmd = "nanovdb_convert -v -f %s %s" % (  # -f flag forces overwriting existing file
            vdb_path,
            saved_nvdb,
        )
        print("nanovdb_convert roundtrip the vdb to nvdb: ", convert_cmd)
        print(subprocess.check_output(convert_cmd.split()).decode("utf-8"))

        # Load the nvdb file of the converted vdb
        grid_batch, features, names = fvdb.load(saved_nvdb, device=torch.device("cuda:0"))
        print("Loaded grid batch total number of voxels: ", grid_batch.total_voxels)
        print("Loaded grid batch data type: %s, device: %s" % (features.dtype, features.device))
        print("\n")

        # Load a smoke simulation grid and visualize it
        workdir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(workdir, "../data/smoke-blosc.nvdb")
        grid_in, data_in, name_in = fvdb.load(data_path)

        ps.init()

        points = grid_in.grid_to_world(grid_in.ijk.jdata.float()).jdata
        dual_grid_pts = ps.register_point_cloud(
            "primal centers", points.cpu().detach().numpy(), radius=0.001
        )
        dual_grid_pts.add_scalar_quantity(
            "grid value", data_in.jdata[:, 0].cpu().detach().numpy(), enabled=True
        )

        ps.show()
