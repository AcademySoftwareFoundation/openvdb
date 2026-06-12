// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <nanovdb/tools/DilateGrid.h>
#include <nanovdb/tools/PruneGrid.h>
#include <nanovdb/util/Injection.h>

template<typename BuildT>
void mainDilateGrid(
    nanovdb::NanoGrid<BuildT> *srcGrid,          // original (un-dilated) grid; also the prune reference
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize dilator
    nanovdb::tools::DilateGrid<BuildT> dilator( srcGrid );
    dilator.setOperation(nanovdb::tools::morphology::NearestNeighbors(nnType));
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(1);

    auto handle = dilator.getHandle();
    auto dstGrid = handle.template grid<BuildT>();   // host grid (HostBuffer)

    // Check for correctness (host vs host)
    if (std::memcmp((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()) == 0)
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

    // Inject the original topology as a per-(dilated-)leaf mask sidecar (host).
    uint32_t dstLeafCount = dstGrid->tree().nodeCount(0);
    nanovdb::HostBuffer dstLeafMaskBuffer;
    nanovdb::Mask<3>* dstLeafMasks = nullptr;
    if (dstLeafCount) {
        dstLeafMaskBuffer = nanovdb::HostBuffer::create(std::size_t(dstLeafCount) * sizeof(nanovdb::Mask<3>));
        dstLeafMasks = static_cast<nanovdb::Mask<3>*>(dstLeafMaskBuffer.data());
        if (!dstLeafMasks) throw std::runtime_error("No buffer for dstLeafMask");
    }

    gpuTimer.start("Injecting un-dilated topology as a pruning mask");
    if (dstLeafCount)
        nanovdb::util::injectGridMask<BuildT>(srcGrid, dstGrid, dstLeafMasks, dstLeafCount);
    gpuTimer.stop();

    // Initialize pruner (host operator)
    nanovdb::tools::PruneGrid<BuildT> pruner( dstGrid, dstLeafMasks );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(1);

    auto prunedHandle = pruner.getHandle();
    auto prunedGrid = prunedHandle.template grid<BuildT>();   // host grid (HostBuffer)

    // Check for correctness (the pruned grid should recover the original)
    if (std::memcmp((char*)prunedGrid, (char*)srcGrid->data(), srcGrid->gridSize()) == 0)
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
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);
