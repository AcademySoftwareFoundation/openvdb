// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/DilateGrid.h>
#include <nanovdb/tools/PruneGrid.h>
#include <nanovdb/util/cuda/Injection.cuh>

template<typename T>
bool bufferCheck(const T* deviceBuffer, const T* hostBuffer, size_t elem_count) {
    T* tmpBuffer = new T[elem_count];
    cudaCheck(cudaMemcpy(tmpBuffer, deviceBuffer, elem_count * sizeof(T), cudaMemcpyDeviceToHost));
    bool same = true;
    for (int i=0; same && i< elem_count; ++i) { same = (tmpBuffer[i] == hostBuffer[i]); }
    delete [] tmpBuffer;
    return same;
}

template<typename BuildT>
void mainDilateGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize dilator
    nanovdb::tools::DilateGrid<BuildT> dilator( deviceGridOriginal );
    dilator.setOperation(nanovdb::tools::morphology::NearestNeighbors(nnType));
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(1);

    auto handle = dilator.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()))
        std::cout << "Result of DilateGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of DilateGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    dilator.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire dilation after warmstart");
        auto dummyHandle = dilator.getHandle();
        gpuTimer.stop();
    }

    // --- Prune round-trip: recover the original grid by pruning the dilated grid back to the
    //     original topology, supplied as a leaf-mask sidecar. ---

    // Inject the original topology as a per-(dilated-)leaf mask sidecar. This injection kernel is
    // an example helper (not part of PruneGrid), so it stays a CUDA kernel for now; the sidecar is
    // a UnifiedBuffer so the host PruneGrid can read it.
    uint32_t dstLeafCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dstGrid).mNodeCount[0];
    nanovdb::cuda::UnifiedBuffer dstLeafMaskBuffer;
    nanovdb::Mask<3>* dstLeafMasks = nullptr;
    if (dstLeafCount) {
        dstLeafMaskBuffer = nanovdb::cuda::UnifiedBuffer::create(std::size_t(dstLeafCount) * sizeof(nanovdb::Mask<3>));
        dstLeafMasks = static_cast<nanovdb::Mask<3>*>(dstLeafMaskBuffer.deviceData());
        if (!dstLeafMasks) throw std::runtime_error("No buffer for dstLeafMask");
    }

    const unsigned int numThreads = 128;
    auto numBlocks = [numThreads] (unsigned int n) {return (n + numThreads - 1) / numThreads;};
    gpuTimer.start("Injecting un-dilated topology as a pruning mask");
    if (dstLeafCount)
        nanovdb::util::cuda::lambdaKernel<<<numBlocks(dstLeafCount), numThreads>>>(dstLeafCount,
            nanovdb::util::cuda::InjectGridMaskFunctor<BuildT>(),
            deviceGridOriginal, dstGrid, dstLeafMasks );
    cudaCheck(cudaDeviceSynchronize());
    gpuTimer.stop();

    // Initialize pruner (host operator; reads dstGrid + dstLeafMasks host-side via managed memory)
    nanovdb::tools::PruneGrid<BuildT> pruner( dstGrid, dstLeafMasks );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(1);

    auto prunedHandle = pruner.getHandle();
    auto prunedGrid = prunedHandle.template deviceGrid<BuildT>();

    // Check for correctness (the pruned grid should recover the original)
    if (bufferCheck((char*)prunedGrid, (char*)indexGridOriginal->data(), indexGridOriginal->gridSize()))
        std::cout << "Result of PruneGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of PruneGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    pruner.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire pruning after warmstart");
        auto dummyHandle = pruner.getHandle();
        gpuTimer.stop();
    }
}

template
void mainDilateGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);
