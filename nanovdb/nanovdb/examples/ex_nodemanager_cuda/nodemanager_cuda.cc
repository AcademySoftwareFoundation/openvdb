// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/tools/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/cuda/HandleStorage.h> // host-includable: cuda::copyTo transfers grids without any kernel
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/NodeManager.h>

extern "C" void launch_kernels(const nanovdb::NodeManager<float>*,// device NaodeManager
                               const nanovdb::NodeManager<float>*,// host NodeManager
                               cudaStream_t stream);

extern nanovdb::NodeManagerHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>>
uploadNodeManager(const nanovdb::NanoGrid<float>* d_grid, cudaStream_t stream); // constructs a NodeManager for a device grid

/// @brief This examples depends on OpenVDB, NanoVDB and CUDA.
int main()
{
    using SrcGridT = openvdb::FloatGrid;
    try {
        cudaStream_t stream; // stream that orders the transfers and the kernels below
        cudaStreamCreate(&stream);
        {

        // Create an OpenVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcGrid = openvdb::tools::createLevelSetSphere<SrcGridT>(100.0f, openvdb::Vec3f(0.0f), 1.0f);

        // Converts the OpenVDB to NanoVDB and returns a GridHandle that uses CUDA for memory management.
        auto gridHandle = nanovdb::tools::createNanoGrid<SrcGridT, float>(*srcGrid);
        auto deviceGridHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(gridHandle, stream); // deep-copy the grid to the GPU
        auto* grid = gridHandle.grid<float>(); // a (raw) pointer to the grid on the CPU
        auto* deviceGrid = deviceGridHandle.deviceGrid<float>(); // and its deep copy on the GPU
        if (!deviceGrid || !grid) {
            throw std::runtime_error("GridHandle did not contain a grid with value type float");
        }

        auto nodeHandle = nanovdb::createNodeManager<float>(*grid); // host NodeManager over the host grid
        auto *nodeMgr = nodeHandle.template mgr<float>();
        auto nodeHandle2 = uploadNodeManager(deviceGrid, stream); // device NodeManager constructed for the device grid
        auto *deviceNodeMgr = nodeHandle2.template deviceMgr<float>();
        if (!deviceNodeMgr || !nodeMgr) {
            throw std::runtime_error("NodeManagerHandle did not contain a grid with value type float");
        }

        launch_kernels(deviceNodeMgr, nodeMgr, stream); // Call a host method to print a grid value on both the CPU and GPU
        cudaStreamSynchronize(stream); // the kernels must finish before the device handles (whose buffers free on this stream) go away
        }
        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}