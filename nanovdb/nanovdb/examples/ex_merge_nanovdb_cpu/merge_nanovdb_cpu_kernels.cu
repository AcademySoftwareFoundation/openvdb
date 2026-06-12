// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <nanovdb/tools/MergeGrids.h>

template<typename BuildT>
void mainMergeGrids(
    nanovdb::NanoGrid<BuildT> *srcGrid1,
    nanovdb::NanoGrid<BuildT> *srcGrid2,
    nanovdb::NanoGrid<BuildT> *dstReferenceGrid,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize converter
    nanovdb::tools::MergeGrids<BuildT> converter( srcGrid1, srcGrid2 );
    converter.setChecksum(nanovdb::CheckMode::Default);
    converter.setVerbose(1);

    auto handle = converter.getHandle();
    auto dstGrid = handle.template grid<BuildT>();   // host grid (HostBuffer)

    // Check for correctness (host vs host -- both grids are host-resident)
    if (std::memcmp((char*)dstGrid, (char*)dstReferenceGrid->data(), dstReferenceGrid->gridSize()) == 0)
        std::cout << "Result of MergeGrids check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of MergeGrids compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    converter.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire merge after warmstart");
        auto handle2 = converter.getHandle();
        gpuTimer.stop();
    }
}

template
void mainMergeGrids(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid1,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *srcGrid2,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *dstReferenceGrid,
    uint32_t benchmark_iters);
