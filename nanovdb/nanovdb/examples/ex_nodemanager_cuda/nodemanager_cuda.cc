// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/tools/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/NodeManager.h>

extern "C" void launch_kernels(const nanovdb::NodeManager<float>*,// device NaodeManager
                               const nanovdb::NodeManager<float>*,// host NodeManager
                               cudaStream_t stream);

extern "C" void cudaCreateNodeManager(const nanovdb::NanoGrid<float>*,// device grid
                                      nanovdb::NodeManagerHandle<nanovdb::cuda::DeviceBuffer>*);// Handle to device NodeManager

/// @brief This examples depends on OpenVDB, NanoVDB and CUDA.
int main()
{
    using SrcGridT = openvdb::FloatGrid;
    using BufferT = nanovdb::cuda::DeviceBuffer;
    try {
        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        // Create an OpenVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcGrid = openvdb::tools::createLevelSetSphere<SrcGridT>(100.0f, openvdb::Vec3f(0.0f), 1.0f);

        // Converts the OpenVDB to NanoVDB and returns a GridHandle that uses CUDA for memory management.
        auto gridHandle = nanovdb::tools::createNanoGrid<SrcGridT, float, BufferT>(*srcGrid);
        gridHandle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously
        auto* grid = gridHandle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = gridHandle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
        if (!deviceGrid || !grid) {
            throw std::runtime_error("GridHandle did not contain a grid with value type float");
        }

        auto nodeHandle = nanovdb::createNodeManager<float, BufferT>(*grid);
        auto *nodeMgr = nodeHandle.template mgr<float>();
#if 0// this approach copies a NodeManager from host to device
        nodeHandle.deviceUpload(deviceGrid, stream, false);
        auto *deviceNodeMgr = nodeHandle.template deviceMgr<float>();
#else// the approach below constructs a new NodeManager directly for a device grid
        nanovdb::NodeManagerHandle<BufferT> nodeHandle2;
        cudaCreateNodeManager(deviceGrid, &nodeHandle2);
        auto *deviceNodeMgr = nodeHandle2.template deviceMgr<float>();
#endif
        if (!deviceNodeMgr || !nodeMgr) {
            throw std::runtime_error("NodeManagerHandle did not contain a grid with value type float");
        }

        launch_kernels(deviceNodeMgr, nodeMgr, stream); // Call a host method to print a grid value on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}