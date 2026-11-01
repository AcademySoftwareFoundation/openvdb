// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/cuda/Buffer.h>

#include <vector>

/// @brief Run PropagateSign on a device grid and its SDF sidecar, and copy the invert masks and the root-tile
///        origins to the host. The mask vectors come sized by the caller, one mask per node of each level.
template<typename BuildT>
void mainPropagateSign(
    const nanovdb::NanoGrid<BuildT> *deviceGrid,
    const float *deviceSdf,
    std::vector<nanovdb::Mask<3>> &leafMasks,
    std::vector<nanovdb::Mask<4>> &lowerMasks,
    std::vector<nanovdb::Mask<5>> &upperMasks,
    std::vector<nanovdb::Coord> &rootOrigins)
{
    // The caller allocates the masks, which propagate() fills
    nanovdb::cuda::Buffer<nanovdb::Mask<3>> deviceLeafMasks(cudaStream_t(0), leafMasks.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Mask<4>> deviceLowerMasks(cudaStream_t(0), lowerMasks.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Mask<5>> deviceUpperMasks(cudaStream_t(0), upperMasks.size(), nanovdb::cuda::noInit);

    nanovdb::tools::cuda::PropagateSign<BuildT> propagator(deviceGrid, deviceSdf);
    propagator.setVerbose(1);
    auto [deviceRootOrigins, hasInvertRootTiles] =
        propagator.propagate(deviceLeafMasks.data(), deviceLowerMasks.data(), deviceUpperMasks.data());

    cudaCheck(cudaMemcpy(leafMasks.data(), deviceLeafMasks.data(), deviceLeafMasks.size_bytes(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(lowerMasks.data(), deviceLowerMasks.data(), deviceLowerMasks.size_bytes(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(upperMasks.data(), deviceUpperMasks.data(), deviceUpperMasks.size_bytes(), cudaMemcpyDeviceToHost));
    rootOrigins.resize(deviceRootOrigins.size());
    if (hasInvertRootTiles)
        cudaCheck(cudaMemcpy(rootOrigins.data(), deviceRootOrigins.data(), deviceRootOrigins.size_bytes(), cudaMemcpyDeviceToHost));
}

template
void mainPropagateSign(
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGrid,
    const float *deviceSdf,
    std::vector<nanovdb::Mask<3>> &leafMasks,
    std::vector<nanovdb::Mask<4>> &lowerMasks,
    std::vector<nanovdb::Mask<5>> &upperMasks,
    std::vector<nanovdb::Coord> &rootOrigins
);
