# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path


def open_file(file_path):
    """
    Opens a file. If utf-8 decoding fails, try windows-1252.

    Args:
        file_path: Path of the file to open.

    Returns:
        The content of the file in an arbitrary format.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="windows-1252", errors="replace") as file:
            return file.read()


def write_file(file_path, content):
    """
    Writes a file. If utf-8 decoding fails, try windows-1252.

    Args:
        file_path: Path of the file to open.

    Returns:
        None.
    """
    try:
        with open(file_path, "w", encoding="utf-8", errors="replace") as file:
            file.write(content)
    except UnicodeDecodeError:
        with open(file_path, "w", encoding="windows-1252", errors="replace") as file:
            file.write(content)


def update_files(dir_path):
    """
    Updates the content of files ending in .h, .cuh, .cc, .cu, and .cpp
    to call the appropriate API as we update NanoVDB from version 32.6 to
    version 32.7. This includes changes in namespaces, function names, and
    include directories.

    Args:
        Directory path: will include files in downstream directories.

    Returns:
        None. Writes the contents of the file.
    """

    # List of file extensions to search for
    file_extensions = [".h", ".cuh", ".cc", ".cu", ".cpp"]

    nspace_dic = {
        "math": [
            "Ray",
            "DDA<",
            "HDDA",
            "Vec3<",
            "Vec4<",
            "BBox<",
            "ZeroCrossing",
            "TreeMarcher",
            "PointTreeMarcher",
            "BoxStencil<",
            "CurvatureStencil<",
            "GradStencil<",
            "WenoStencil<",
            "AlignUp",
            "Min",
            "Max",
            "Abs",
            "Clamp",
            "Sqrt",
            "Sign",
            "Maximum<",
            "Delta<",
            "RoundDown<",
            "pi<",
            "isApproxZero<",
            "Round<",
            "createSampler",
            "SampleFromVoxels<",
        ],
        "tools": [
            "createNanoGrid",
            "StatsMode",
            "createLevelSetSphere",
            "createFogVolumeSphere",
            "createFogVolumeSphere createFogVolumeSphere",
            "createFogVolumeTorus",
            "createLevelSetBox",
            "CreateNanoGrid",
            "updateGridStats",
            "evalChecksum",
            "validateChecksum",
            "checkGrid",
            "Extrema",
        ],
        "util": [
            "is_floating_point",
            "findLowestOn",
            "findHighestOn",
            "Range",
            "streq",
            "strcpy",
            "strcat",
            "empty(",
            "Split",
            "invoke",
            "forEach",
            "reduce",
            "prefixSum",
            "is_same",
            "is_specialization",
            "PtrAdd",
            "PtrDiff",
        ],
    }

    rename_dic = {
        # list from func4 in updateFiles.sh
        "nanovdb::build::": "nanovdb::tools::build::",
        "nanovdb::BBoxR": "nanovdb::Vec3dBBox",
        "nanovdb::BBox<nanovdb::Vec3d>": "nanovdb::Vec3dBbox",
        # scope and rename, i.e. list from func2 in updateFiles.sh
        "nanovdb::cudaCreateNodeManager": "nanovdb::cuda::createNodeManager",
        "nanovdb::cudaVoxelsToGrid": "nanovdb::cuda::voxelsToGrid",
        "nanovdb::cudaPointsToGrid": "nanovdb::cuda::pointsToGrid",
        "nanovdb::DitherLUT": "nanovdb::math::DitherLUT",
        "nanovdb::PackedRGBA8": "nanovdb::math::Rgba8",
        "nanovdb::Rgba8": "nanovdb::math::Rgba8",
        "nanovdb::CpuTimer": "nanovdb::util::Timer",
        "nanovdb::GpuTimer": "nanovdb::util::cuda::Timer",
        "nanovdb::CountOn": "nanovdb::util::countOn",
    }

    movdir_dic = {
        # list comes from func3 calls on updateFiles.sh
        "util/GridHandle.h": "GridHandle.h",
        "util/BuildGrid.h": "tools/GridBuilder.h",
        "util/GridBuilder.h": "tools/GridBuilder.h",
        "util/IO.h": "io/IO.h",
        "util/CSampleFromVoxels.h": "math/CSampleFromVoxels.h",
        "util/DitherLUT.h": "math/DitherLUT.h",
        "util/HDDA.h": "math/HDDA.h",
        "util/Ray.h": "math/Ray.h",
        "util/SampleFromVoxels.h": "math/SampleFromVoxels.h",
        "util/Stencils.h": "nanovdb/math/Stencils.h",
        "util/CreateNanoGrid.h": "tools/CreateNanoGrid.h",
        "util/Primitives.h": "tools/CreatePrimitives.h",
        "util/GridChecksum.h": "tools/GridChecksum.h",
        "util/GridStats.h": "tools/GridStats.h",
        "util/GridChecksum.h": "tools/GridChecksum.h",
        "util/GridValidator.h": "tools/GridValidator.h",
        "util/NanoToOpenVDB.h": "tools/NanoToOpenVDB.h",
        "util/cuda/CudaGridChecksum.cuh": "tools/cuda/CudaGridChecksum.cuh",
        "util/cuda/CudaGridStats.cuh": "tools/cuda/CudaGridStats.cuh",
        "util/cuda/CudaGridValidator.cuh": "tools/cuda/CudaGridValidator.cuh",
        "util/cuda/CudaIndexToGrid.cuh": "tools/cuda/CudaIndexToGrid.cuh",
        "util/cuda/CudaPointsToGrid.cuh": "tools/cuda/PointsToGrid.cuh",
        "util/cuda/CudaSignedFloodFill.cuh": "tools/cuda/CudaSignedFloodFill.cuh",
        "util/cuda/CudaDeviceBuffer.h": "cuda/DeviceBuffer.h",
        "util/cuda/CudaGridHandle.cuh": "cuda/GridHandle.cuh",
        "util/cuda/CudaUtils.h": "util/cuda/Util.h",
        "util/cuda/GpuTimer.h": "util/cuda/Timer.h",
    }

    # Iterate over files in the directory and its subdirectories
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                content = open_file(file_path)

                # Correspond to func1 $file in updateFiles.sh
                for key, vals in nspace_dic.items():
                    for val in vals:
                        old_word = "nanovdb::" + val
                        new_word = "nanovdb::" + key + "::" + val
                        content = content.replace(old_word, new_word)

                # Correspond to func4 and func2 in updateFiles.sh
                for key, val in rename_dic.items():
                    content = content.replace(key, val)

                # Correspond to func3 in updateFiles.sh
                for key, val in movdir_dic.items():
                    old_path = "<nanovdb/" + key + ">"
                    new_path = "<nanovdb/" + val + ">"
                    content = content.replace(old_path, new_path)

                write_file(file_path, content)

# Example use:
# To update all the files using NanoVDB in the current directory (and directories downstream):
# python ./nanovdb/nanovdb/cmd/updateFiles.py
# To update all the files using NanoVDB in a directory called foo (and directories downstream):
# python ./nanovdb/nanovdb/cmd/updateFiles.py -d /path/to/foo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Generation for USD")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=None,
        help="Path to directory containing .h, .cc, and .cu files using NanoVDB.",
    )

    args = parser.parse_args()
    dir_path = os.getcwd() if args.directory is None else Path(args.directory).resolve()

    update_files(dir_path)
