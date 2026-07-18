// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <nanovdb/tools/CoarsenGrid.h>

template<typename BuildT>
void mainCoarsenGrid(
    nanovdb::NanoGrid<BuildT> *srcGrid,           // original (un-coarsened) grid
    nanovdb::NanoGrid<BuildT> *indexGridCoarsened,
    uint32_t benchmark_iters)
{
    nanovdb::util::Timer cpuTimer;

    // Initialize coarsener
    nanovdb::tools::CoarsenGrid<BuildT> coarsener( srcGrid );
    coarsener.setChecksum(nanovdb::CheckMode::Default);
    coarsener.setVerbose(1);

    auto handle = coarsener.getHandle();
    auto dstGrid = handle.template grid<BuildT>();   // host grid (HostBuffer)

    // Check for correctness (host vs host)
    if (std::memcmp((char*)dstGrid, (char*)indexGridCoarsened->data(), indexGridCoarsened->gridSize()) == 0)
        std::cout << "Result of CoarsenGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of CoarsenGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    coarsener.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        cpuTimer.start("Re-running entire coarsening after warmstart");
        auto dummyHandle = coarsener.getHandle();
        cpuTimer.stop();
    }
}

template
void mainCoarsenGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridCoarsened,
    uint32_t benchmark_iters
);
