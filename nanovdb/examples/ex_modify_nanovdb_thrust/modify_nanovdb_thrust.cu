// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @brief This examples demonstrates how values in a NanpVDB grid can be
///        modified on the device. It depends on NanoVDB and CUDA thrust.

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

void scaleActiveVoxels(const nanovdb::FloatGrid *grid_d, uint64_t leafCount, float scale)
{
    using LeafT = nanovdb::NanoLeaf<float>;

    auto kernel = [grid_d, scale] __device__ (const uint64_t n) {
        LeafT *leaf_d = const_cast<LeafT*>(grid_d->tree().getNode<0>(n >> 9));
        const int i = n & 511;
        const float v = scale * leaf_d->getValue(i);
        if (leaf_d->isActive(i)) {
            leaf_d->setValueOnly(i, v);// only possible execution divergence
        }
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

int main()
{
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto handle = nanovdb::createLevelSetSphere<float, nanovdb::CudaDeviceBuffer>(100.0f);

        handle.deviceUpload(0, false); // Copy the NanoVDB grid to the GPU asynchronously

        auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !grid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        std::cerr << "Value before scaling = " << grid->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;

        scaleActiveVoxels(deviceGrid, grid->tree().nodeCount(0), 2.0f);

        handle.deviceDownload(0, true); // Copy the NanoVDB grid to the CPU synchronously

        std::cerr << "Value after scaling  = " << grid->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}