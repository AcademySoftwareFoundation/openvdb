// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file vbm_host_cuda.cpp

    \brief Test harness for the VoxelBlockManager (CUDA reference implementation).

    Generates a random sparse domain at a configurable occupancy level using a
    Morton-curve layout, builds a ValueOnIndex NanoVDB grid, constructs the
    VoxelBlockManager on the GPU, decodes the full inverse map (leafIndex,
    voxelOffset per active voxel), and validates the result on the host.

    Usage: vbm_host_cuda [ambient_voxels [occupancy]]
      ambient_voxels  Total universe of voxel positions  (default: 1048576)
      occupancy       Fraction of positions that are active in [0,1]  (default: 0.5)
*/

#include <nanovdb/NanoVDB.h>

#include <random>
#include <vector>
#include <iostream>
#include <stdexcept>

void runVBMCudaTest(const std::vector<nanovdb::Coord>& coords);

/// @brief Unpack one component of a Morton-encoded index into a coordinate.
/// Keeps every third bit of the input, then packs them into a contiguous integer.
static uint32_t coordinate_bitpack(uint32_t x)
{
    x &= 0x49249249;
    x |= (x >>  2); x &= 0xc30c30c3;
    x |= (x >>  4); x &= 0x0f00f00f;
    x |= (x >>  8); x &= 0xff0000ff;
    x |= (x >> 16); x &= 0x0000ffff;
    return x;
}

/// @brief Generate active voxel coordinates at the requested occupancy level.
/// Voxels are drawn uniformly at random from a Morton-curve layout over
/// ambient_voxels positions, giving spatially coherent 3D coordinates.
static std::vector<nanovdb::Coord>
generateDomain(int ambient_voxels, float occupancy, uint32_t seed = 42)
{
    const int target = (int)(occupancy * (float)ambient_voxels);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, ambient_voxels - 1);

    std::vector<bool> voxmap(ambient_voxels, false);
    int active = 0;
    while (active < target) {
        int i = dist(rng);
        if (!voxmap[i]) { voxmap[i] = true; ++active; }
    }

    std::vector<nanovdb::Coord> coords;
    coords.reserve(active);
    for (int i = 0; i < ambient_voxels; ++i) {
        if (voxmap[i]) {
            coords.emplace_back(
                (int)coordinate_bitpack( i         & 0x49249249),
                (int)coordinate_bitpack((i >>  1)  & 0x49249249),
                (int)coordinate_bitpack((i >>  2)  & 0x49249249));
        }
    }
    return coords;
}

int main(int argc, char** argv)
{
    try {
        int   ambient_voxels = 1024 * 1024;
        float occupancy      = 0.5f;

        if (argc > 1) ambient_voxels = std::stoi(argv[1]);
        if (argc > 2) occupancy      = std::stof(argv[2]);

        occupancy = std::max(0.0f, std::min(1.0f, occupancy));

        std::cout << "ambient_voxels = " << ambient_voxels << "\n"
                  << "occupancy      = " << occupancy      << "\n";

        auto coords = generateDomain(ambient_voxels, occupancy);
        std::cout << "Active voxels generated: " << coords.size() << "\n";

        runVBMCudaTest(coords);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"\n";
        return 1;
    }
    return 0;
}
