# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np

from nanovdb_editor import Compiler, Compute, Editor, Raster


TEST_RASTER_TO_NANOVDB = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_NPZ = os.path.join(SCRIPT_DIR, "../data/splats.npz")
TEST_NANOVDB = os.path.join(SCRIPT_DIR, "../data/raster_test.nvdb")

if __name__ == "__main__":

    print(f"Current Process ID (PID): {os.getpid()}")

    compiler = Compiler()
    compiler.create_instance()

    compute = Compute(compiler)
    compute.device_interface().create_device_manager()
    compute.device_interface().create_device()

    device = compute.device_interface().create_device()
    raster = Raster(compute, device)

    editor = Editor(compute, compiler)

    def raster_func(filename_ptr):
        if isinstance(filename_ptr, bytes):
            _filename = filename_ptr.decode('utf-8')
        else:
            _filename = filename_ptr

        try:
            npz_array = np.load(TEST_NPZ)
        except FileNotFoundError:
            print(f"File '{_filename}' not found")
            return

        print(f"Rasterizing npz file '{_filename}'...")

        # Get data from npz file
        means_npz = npz_array["means"]
        opacities_npz = npz_array["opacities"]
        quaternions_npz = npz_array["quaternions"]
        scales_npz = npz_array["scales"]
        sh_npz = npz_array["sh"]

        point_count = means_npz.shape[0]

        # Create and fill position array
        means = np.zeros((point_count, 3), dtype=np.float32)
        means[:, 0] = means_npz[:, 0]
        means[:, 1] = means_npz[:, 1]
        means[:, 2] = means_npz[:, 2]

        quaternions = np.zeros((point_count, 4), dtype=np.float32)
        quaternions[:, 0] = quaternions_npz[:, 0]
        quaternions[:, 1] = quaternions_npz[:, 1]
        quaternions[:, 2] = quaternions_npz[:, 2]
        quaternions[:, 3] = quaternions_npz[:, 3]

        # Normalize quaternions
        magnitudes = np.sqrt(np.sum(quaternions**2, axis=1, keepdims=True))
        quaternions = quaternions / magnitudes

        scales = np.zeros((point_count, 3), dtype=np.float32)
        scales[:, 0] = scales_npz[:, 0]
        scales[:, 1] = scales_npz[:, 1]
        scales[:, 2] = scales_npz[:, 2]
        scales = np.exp(scales)

        # Calculate colors from spherical harmonics
        C_0 = 0.28209479177387814
        colors = np.zeros((point_count, 3), dtype=np.float32)
        colors[:, 0] = C_0 * sh_npz[:, 0, 0] + 0.5
        colors[:, 1] = C_0 * sh_npz[:, 0, 1] + 0.5
        colors[:, 2] = C_0 * sh_npz[:, 0, 2] + 0.5

        sh = np.zeros((point_count, 48), dtype=np.float32)
        for j in range(0, 16):
            for i in range(0, 3):
                sh[:, j * 3 + i] = sh_npz[:, j, i]
                sh[:, j * 3 + i] = sh_npz[:, j, i]
                sh[:, j * 3 + i] = sh_npz[:, j, i]

        opacities = np.zeros((point_count, 1), dtype=np.float32)
        opacities[:, 0] = opacities_npz

        # Apply sigmoid function to opacities
        opacities = 1.0 / (1.0 + np.exp(-opacities))

        def print_array_range(name, array):
            if array.ndim == 1:
                min_val = array.min()
                max_val = array.max()
                avg_val = array.mean()
                print(f"array({name}) channel(0) min({min_val}) "
                      f"max({max_val}) ave({avg_val})")
            else:
                channel_count = array.shape[1]
                for channel_idx in range(channel_count):
                    channel_data = array[:, channel_idx]
                    min_val = channel_data.min()
                    max_val = channel_data.max()
                    avg_val = channel_data.mean()
                    print(f"array({name}) channel({channel_idx}) "
                          f"min({min_val}) max({max_val}) ave({avg_val})")

        print_array_range("means", means)
        print_array_range("quats", quaternions)
        print_array_range("scales", scales)
        print_array_range("colors", colors)
        print_array_range("opacities", opacities)

        # Create compute arrays
        means_array = compute.create_array(means)
        quaternions_array = compute.create_array(quaternions)
        scales_array = compute.create_array(scales)
        colors_array = compute.create_array(colors)
        sh_array = compute.create_array(sh)
        opacities_array = compute.create_array(opacities)

        VOXEL_SIZE = 1.0 / 128.0

        nvdb_array = raster.raster_points(
            VOXEL_SIZE,
            means_array,
            quaternions_array,
            scales_array,
            colors_array,
            sh_array,
            opacities_array
        )

        editor.add_nanovdb(nvdb_array)

        compute.destroy_array(means_array)
        compute.destroy_array(quaternions_array)
        compute.destroy_array(scales_array)
        compute.destroy_array(colors_array)
        compute.destroy_array(sh_array)
        compute.destroy_array(opacities_array)

    if TEST_RASTER_TO_NANOVDB:
        raster_func(TEST_NPZ)
        compute.save_nanovdb(editor.get_nanovdb(), TEST_NANOVDB)

    editor.add_callable("Raster", raster_func)
    editor.show()
