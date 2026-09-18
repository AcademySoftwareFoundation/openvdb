// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Efty Sifakis
///
/// @file BenchmarkIO.cpp
///
/// @brief Implements the OpenVDB/NanoVDB bridge methods of the Benchmark
///        singleton: grid conversion, sidecar allocation, host/device data
///        transfer, and correctness-comparison utilities.
///
///        No CUDA device code is present in this file; it compiles as a
///        standard C++ translation unit alongside Benchmark.cu.

// the following files are from OpenVDB
#include <openvdb/math/Operators.h>
#include <openvdb/util/CpuTimer.h>

#include "Stencils.h"
#include "Benchmark.h"

// the following files are from NanoVDB
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/HostBuffer.h>          // for CPU-only temporary grid conversions

Benchmark::
Benchmark()
{}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void
Benchmark::
copyHostOpenVDBToDeviceNanoVDB(OpenVDBGridT& grid, GridHandleT& handle, const bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Converting OpenVDB (from CPU host) to NanoVDB/IndexGrid (on GPU)");

    // HostBuffer: CPU-only temporary; the grid is constructed on the host and then
    // explicitly copied into a UnifiedBuffer below.  Future CUDA/CPU specializations
    // of this function may collapse the two allocations (CPU path) or switch
    // deviceGridBuffer to DeviceBuffer (CUDA path).
    auto hostHandle = nanovdb::tools::createNanoGrid<OpenVDBGridT, BuildT, nanovdb::HostBuffer>(
        grid,
        0u,    // Don't copy data channel
        false, // No stats
        false, // No tiles
        1      // Verbose mode
    );

    BufferT deviceGridBuffer = BufferT::create(hostHandle.size());
    cudaCheck(cudaMemcpy(deviceGridBuffer.deviceData(), hostHandle.data(), hostHandle.size(), cudaMemcpyHostToDevice));
    handle.reset(); // TODO: remove after memory leak in move constructor is fixed
    handle = nanovdb::GridHandle<BufferT>(std::move(deviceGridBuffer));

    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void
Benchmark::
printGridDiagnostics(GridHandleT& handle)
{
    auto deviceGrid = handle.deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU grid found in printGridDiagnostics()");

    auto valueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(deviceGrid);
    auto treeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(deviceGrid);
    auto gridSize = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getGridSize(deviceGrid);
    auto indexBBox = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(deviceGrid, treeData);

    std::cout << "======= Grid info =======" << std::endl;
    std::cout << "Allocated values         : " << valueCount << std::endl;
    std::cout << "Active voxels            : " << treeData.mVoxelCount << std::endl;
    auto minCorner = indexBBox.min(), maxCorner = indexBBox.max();
    std::cout << "Index-space bounding box : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
              << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
    std::cout << "Leaf nodes               : " << treeData.mNodeCount[0] << std::endl;
    std::cout << "Lower internal nodes     : " << treeData.mNodeCount[1] << std::endl;
    std::cout << "Upper internal nodes     : " << treeData.mNodeCount[2] << std::endl;
    std::cout << "Leaf-level occupancy     : "
              << 100.f * (float)(treeData.mVoxelCount)/(float)(treeData.mNodeCount[0] * 512)
              << "%" << std::endl;
    std::cout << "Memory usage             : " << gridSize << " bytes" << std::endl;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

bool
Benchmark::
compareHostOpenVDBToDeviceNanoVDB(OpenVDBGridT& grid, GridHandleT& handle)
{
    // HostBuffer: CPU-only scratch for comparison; never reaches the GPU.
    auto openHandle = nanovdb::tools::createNanoGrid<OpenVDBGridT, BuildT, nanovdb::HostBuffer>(
        grid,
        0u,    // Don't copy data channel
        false, // No stats
        false, // No tiles
        1      // Verbose mode
    );

    auto result = std::memcmp(openHandle.data(), handle.data(), handle.size());

    return (result == 0);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// NOTE: initializeGPUSidecarAndBackgroundValue moved out of this file -- it is core simulation
// infrastructure (sidecar allocation), not OpenVDB conversion/validation. Its dual-platform
// specializations now live in Benchmark.cu (<CUDA>) and Benchmark.cpp (<CPU>).

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void
Benchmark::
copyOpenVDBDataToNanoVDBSidecar(OpenVDBGridT& grid, GridHandleT& handle, BufferT& buffer, bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Copying OpenVDB data to NanoVDB/IndexGrid sidecar");

    const auto hostGrid = static_cast<const GridT*>(handle.data());
    auto nanoAcc = hostGrid->getAccessor();
    auto hostData = static_cast<ValueType*>(buffer.data());
    if (!hostData)
        throw std::runtime_error("Uninitialized sidecar in copyOpenVDBDataToNanoVDBSidecar()");

    uint64_t activeVoxelCount = 0;
    for (auto iter = grid.tree().cbeginValueOn(); iter; ++iter, ++activeVoxelCount) {
        const ValueType& value = *iter;
        const auto openCoord = iter.getCoord();
        const nanovdb::Coord nanoCoord(openCoord.x(), openCoord.y(), openCoord.z());
        const auto index = nanoAcc.getValue(nanoCoord);
        if (!index) throw std::runtime_error("OpenVDB voxel not found in IndexGrid, in copyOpenVDBDataToNanoVDBSidecar()");
        hostData[index] = value; }
    if (activeVoxelCount != hostGrid->activeVoxelCount())
        throw std::runtime_error("Active voxel mismatch between OpenVDB/IndexGrid, in copyOpenVDBDataToNanoVDBSidecar()");

    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void
Benchmark::
copyNanoVDBSidecarToOpenVDBData(OpenVDBGridT& grid, GridHandleT& handle, BufferT& buffer, bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Copying NanoVDB/IndexGrid sidecar to OpenVDB data");

    cudaCheck(cudaStreamSynchronize(0)); // ensure any prior GPU writes to handle/buffer are visible to the CPU
    const auto hostGrid = static_cast<const GridT*>(handle.data());
    auto nanoAcc = hostGrid->getAccessor();
    const auto hostData = static_cast<const ValueType*>(buffer.data());
    if (!hostData)
        throw std::runtime_error("Uninitialized sidecar in copyNanoVDBSidecarToOpenVDBData()");

    uint64_t activeVoxelCount = 0;
    for (auto iter = grid.tree().beginValueOn(); iter; ++iter, ++activeVoxelCount) {
        const auto openCoord = iter.getCoord();
        const nanovdb::Coord nanoCoord(openCoord.x(), openCoord.y(), openCoord.z());
        const auto index = nanoAcc.getValue(nanoCoord);
        if (!index) throw std::runtime_error("OpenVDB voxel not found in IndexGrid, in copyNanoVDBSidecarToOpenVDBData()");
        iter.setValue(hostData[index]); }
    if (activeVoxelCount != hostGrid->activeVoxelCount())
        throw std::runtime_error("Active voxel mismatch between OpenVDB/IndexGrid, in copyNanoVDBSidecarToOpenVDBData()");

    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

bool
Benchmark::
compareOpenVDBDataToNanoVDBSidecar(const OpenVDBGridT& grid, GridHandleT& handle, BufferT& buffer, ValueType tolerance)
{
    cudaCheck(cudaStreamSynchronize(0)); // ensure any prior GPU writes to handle/buffer are visible to the CPU
    const auto hostGrid = static_cast<const GridT*>(handle.data());
    auto nanoAcc = hostGrid->getAccessor();
    const auto hostData = static_cast<const ValueType*>(buffer.data());
    if (!hostData)
        throw std::runtime_error("Uninitialized sidecar in compareOpenVDBDataToNanoVDBSidecar()");
    
    ValueType maxDiff = std::abs(hostData[0]-grid.background());    
    uint64_t activeVoxelCount = 0;    
    for (auto iter = grid.tree().cbeginValueOn(); iter; ++iter, ++activeVoxelCount) {
        const ValueType& value = *iter;
        const auto openCoord = iter.getCoord();
        const nanovdb::Coord nanoCoord(openCoord.x(), openCoord.y(), openCoord.z());
        const auto index = nanoAcc.getValue(nanoCoord);
        if (!index) throw std::runtime_error("OpenVDB voxel not found in IndexGrid, in compareOpenVDBDataToNanoVDBSidecar()");
        maxDiff = std::max(std::abs(hostData[index]-value),maxDiff); }
    if (activeVoxelCount != hostGrid->activeVoxelCount())
        throw std::runtime_error("Active voxel mismatch between OpenVDB/IndexGrid, in compareOpenVDBDataToNanoVDBSidecar()");

    if (maxDiff > tolerance) {
        std::cout << "Difference observed between OpenVDB/IndexGrid sidecars : " << maxDiff << std::endl;
        return false; }
    return true;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Benchmark&
Benchmark::
getInstance()
{
    static Benchmark sInstance;
    return sInstance;
}

bool
Benchmark::
compareStencil(OpenVDBStencilT& stencil, const OpenCoordT& openCoord, BufferT& buffer)
{
    auto &instance = getInstance();

    auto hostGrid = instance.mHandle.grid<BuildT>();
    if (!hostGrid) throw std::logic_error("No CPU grid found in Benchmark::initBuffers()");
    auto hostData = static_cast<ValueType*>(buffer.data());
    if (!hostData) throw std::logic_error("No CPU buffer found in Benchmark::compareOpenToNanoVDB()");
    const nanovdb::Coord nanoCoord(openCoord.x(), openCoord.y(), openCoord.z());
    auto acc = hostGrid->getAccessor();

    using StencilT = nanovdb::math::WenoStencil<GridT>;
    
    uint64_t stencilIndices[StencilT::SIZE];
    stencilIndices[nanovdb::math::WenoPt< 0, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 1, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 1, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 1, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 1, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0, 1>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0, 1));
    stencilIndices[nanovdb::math::WenoPt<-1, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy(-1, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0,-1, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0,-1, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0,-1>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0,-1));
    stencilIndices[nanovdb::math::WenoPt< 2, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 2, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 2, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 2, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0, 2>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0, 2));
    stencilIndices[nanovdb::math::WenoPt<-2, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy(-2, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0,-2, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0,-2, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0,-2>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0,-2));
    stencilIndices[nanovdb::math::WenoPt< 3, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 3, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 3, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 3, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0, 3>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0, 3));
    stencilIndices[nanovdb::math::WenoPt<-3, 0, 0>::idx] = acc.getValue(nanoCoord.offsetBy(-3, 0, 0));
    stencilIndices[nanovdb::math::WenoPt< 0,-3, 0>::idx] = acc.getValue(nanoCoord.offsetBy( 0,-3, 0));
    stencilIndices[nanovdb::math::WenoPt< 0, 0,-3>::idx] = acc.getValue(nanoCoord.offsetBy( 0, 0,-3));

    ValueType stencilValues[StencilT::SIZE];
    for (int i = 0; i < StencilT::SIZE; i++)
    stencilValues[i] = hostData[stencilIndices[i]];
    if (!stencilIndices[nanovdb::math::WenoPt< 1, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 1, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 1, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 1, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0, 1>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0, 1>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt<-1, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt<-1, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0,-1, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0,-1, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0,-1>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0,-1>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]);

    if (!stencilIndices[nanovdb::math::WenoPt< 2, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 2, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 1, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 2, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 2, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 1, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0, 2>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0, 2>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 1>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt<-2, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt<-2, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt<-1, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0,-2, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0,-2, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0,-1, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0,-2>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0,-2>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0,-1>::idx]);

    if (!stencilIndices[nanovdb::math::WenoPt< 3, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 3, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 2, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 3, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 3, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 2, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0, 3>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0, 3>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0, 2>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt<-3, 0, 0>::idx]) stencilValues[nanovdb::math::WenoPt<-3, 0, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt<-2, 0, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0,-3, 0>::idx]) stencilValues[nanovdb::math::WenoPt< 0,-3, 0>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0,-2, 0>::idx]);
    if (!stencilIndices[nanovdb::math::WenoPt< 0, 0,-3>::idx]) stencilValues[nanovdb::math::WenoPt< 0, 0,-3>::idx] *= nanovdb::math::Sign(stencilValues[nanovdb::math::WenoPt< 0, 0,-2>::idx]);

    bool isSame = true;
    if (stencil.getValue< 0, 0, 0>() != stencilValues[nanovdb::math::WenoPt< 0, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 1, 0, 0>() != stencilValues[nanovdb::math::WenoPt< 1, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 1, 0>() != stencilValues[nanovdb::math::WenoPt< 0, 1, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0, 1>() != stencilValues[nanovdb::math::WenoPt< 0, 0, 1>::idx]) isSame = false;
    if (stencil.getValue<-1, 0, 0>() != stencilValues[nanovdb::math::WenoPt<-1, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0,-1, 0>() != stencilValues[nanovdb::math::WenoPt< 0,-1, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0,-1>() != stencilValues[nanovdb::math::WenoPt< 0, 0,-1>::idx]) isSame = false;

    if (stencil.getValue< 2, 0, 0>() != stencilValues[nanovdb::math::WenoPt< 2, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 2, 0>() != stencilValues[nanovdb::math::WenoPt< 0, 2, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0, 2>() != stencilValues[nanovdb::math::WenoPt< 0, 0, 2>::idx]) isSame = false;
    if (stencil.getValue<-2, 0, 0>() != stencilValues[nanovdb::math::WenoPt<-2, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0,-2, 0>() != stencilValues[nanovdb::math::WenoPt< 0,-2, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0,-2>() != stencilValues[nanovdb::math::WenoPt< 0, 0,-2>::idx]) isSame = false;

    if (stencil.getValue< 3, 0, 0>() != stencilValues[nanovdb::math::WenoPt< 3, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 3, 0>() != stencilValues[nanovdb::math::WenoPt< 0, 3, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0, 3>() != stencilValues[nanovdb::math::WenoPt< 0, 0, 3>::idx]) isSame = false;
    if (stencil.getValue<-3, 0, 0>() != stencilValues[nanovdb::math::WenoPt<-3, 0, 0>::idx]) isSame = false;
    if (stencil.getValue< 0,-3, 0>() != stencilValues[nanovdb::math::WenoPt< 0,-3, 0>::idx]) isSame = false;
    if (stencil.getValue< 0, 0,-3>() != stencilValues[nanovdb::math::WenoPt< 0, 0,-3>::idx]) isSame = false;

    using OpenVDBGradientT = typename openvdb::math::ISGradientNormSqrd<openvdb::math::HJWENO5_BIAS>;
    auto openGradient = OpenVDBGradientT::result(stencil);
    
    using NanoVDBGradientT = nanovdb::math::WenoStencil<nanovdb::FloatGrid>;
    auto nanoGradient = NanoVDBGradientT::normSqGrad(stencilValues, 1.0, 1.0);

    if (fabs(openGradient-nanoGradient)>1e-3) isSame = false;

    return isSame;
}
