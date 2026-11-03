// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/PartitionGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/cuda/GridHandle.cuh> // for cuda::copyTo

#include <iostream>
#include <vector>

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    std::vector<T> tmpBuffer(elem_count);
    cudaCheck(cudaMemcpy(tmpBuffer.data(), deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elem_count; ++i) if (tmpBuffer[i] != hostBuffer[i]) return false;
    return true;
}

template<typename BuildT>
void mainPartitionGrid(
    nanovdb::NanoGrid<BuildT> *deviceGrid,
    nanovdb::NanoGrid<BuildT> *indexGrid,
    const std::vector<nanovdb::NanoGrid<BuildT>*> &segments)
{
    using BufferT = nanovdb::cuda::Buffer<std::byte>;

    // Label the connected components of the active voxels
    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(deviceGrid);
    auto [labels, componentCount] = cc.getVoxelLabelsAndCount();
    std::vector<uint32_t> hostLabels(labels.size());
    cudaCheck(cudaMemcpy(hostLabels.data(), labels.data(), labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Connected components: " << componentCount << " (OpenVDB: " << segments.size() << ")" << std::endl;

    // Split the grid into one grid per component
    nanovdb::tools::cuda::PartitionGrid<BuildT> partitioner(deviceGrid, labels.data(), componentCount);
    partitioner.setChecksum(nanovdb::CheckMode::Default);
    partitioner.setVerbose(1);
    auto outputs = partitioner.template getHandles<BufferT>();

    // Check each output against PruneGrid with the mask of its component, and against the OpenVDB segment that
    // holds a voxel of that component
    bool pruneCorrect = true, openvdbCorrect = outputs.size() == segments.size();
    const auto *leaves = indexGrid->tree().getFirstLeaf();
    const uint32_t leafCount = indexGrid->tree().nodeCount(0);
    for (uint32_t l = 0; l < componentCount; ++l) {
        std::vector<nanovdb::Mask<3>> mask(leafCount);
        nanovdb::Coord voxel;
        for (uint32_t i = 0; i < leafCount; ++i)
            for (auto it = leaves[i].beginValueOn(); it; ++it)
                if (hostLabels[*it] == l) { mask[i].setOn(it.pos()); voxel = it.getCoord(); }
        const char *output = static_cast<const char*>(outputs[l].deviceData());

        nanovdb::cuda::Buffer<nanovdb::Mask<3>> deviceMask(cudaStream_t(0), leafCount, nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(deviceMask.data(), mask.data(), deviceMask.size_bytes(), cudaMemcpyHostToDevice));
        nanovdb::tools::cuda::PruneGrid<BuildT> pruner(deviceGrid, deviceMask.data());
        pruner.setChecksum(nanovdb::CheckMode::Default);
        auto pruned = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(pruner.template getHandle<BufferT>());
        pruneCorrect &= outputs[l].bufferSize() == pruned.bufferSize() &&
                        bufferCheck(output, static_cast<const char*>(pruned.data()), pruned.bufferSize());

        const nanovdb::NanoGrid<BuildT> *segment = nullptr;
        for (auto *s : segments) if (s->tree().isActive(voxel)) segment = s;
        openvdbCorrect &= segment && outputs[l].bufferSize() == segment->gridSize() &&
                          bufferCheck(output, reinterpret_cast<const char*>(segment), segment->gridSize());
    }

    std::cout << "Result of PartitionGrid " << (pruneCorrect ? "check out CORRECT" : "compares INCORRECT")
              << " against PruneGrid" << std::endl;
    std::cout << "Result of PartitionGrid " << (openvdbCorrect ? "check out CORRECT" : "compares INCORRECT")
              << " against OpenVDB segmentActiveVoxels" << std::endl;
}

template
void mainPartitionGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGrid,
    const std::vector<nanovdb::NanoGrid<nanovdb::ValueOnIndex>*> &segments
);
