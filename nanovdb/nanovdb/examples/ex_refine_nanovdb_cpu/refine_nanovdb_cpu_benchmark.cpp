// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <nanovdb/tools/RefineGrid.h>

template<typename BuildT>
void mainRefineGrid(
    nanovdb::NanoGrid<BuildT> *srcGrid,          // original (un-refined) grid
    nanovdb::NanoGrid<BuildT> *indexGridRefined,
    uint32_t benchmark_iters)
{
    nanovdb::util::Timer cpuTimer;

    // Initialize refiner
    nanovdb::tools::RefineGrid<BuildT> refiner( srcGrid );
    refiner.setChecksum(nanovdb::CheckMode::Default);
    refiner.setVerbose(1);

    auto handle = refiner.getHandle();
    auto dstGrid = handle.template grid<BuildT>();   // host grid (HostBuffer)

    // Check for correctness (host vs host)
    if (std::memcmp((char*)dstGrid, (char*)indexGridRefined->data(), indexGridRefined->gridSize()) == 0)
        std::cout << "Result of RefineGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of RefineGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    refiner.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        cpuTimer.start("Re-running entire refinement after warmstart");
        auto dummyHandle = refiner.getHandle();
        cpuTimer.stop();
    }
}

template
void mainRefineGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridRefined,
    uint32_t benchmark_iters
);
