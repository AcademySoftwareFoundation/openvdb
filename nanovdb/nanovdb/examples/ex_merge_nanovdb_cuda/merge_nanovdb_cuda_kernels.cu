// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/MergeGrids.cuh>

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
void mainMergeGrids(
    nanovdb::NanoGrid<BuildT> *deviceSrcGrid1,
    nanovdb::NanoGrid<BuildT> *deviceSrcGrid2,
    nanovdb::NanoGrid<BuildT> *deviceDstReferenceGrid,
    nanovdb::NanoGrid<BuildT> *hostSrcGrid1,
    nanovdb::NanoGrid<BuildT> *hostSrcGrid2,
    nanovdb::NanoGrid<BuildT> *hostDstReferenceGrid,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize converter
    nanovdb::tools::cuda::MergeGrids<BuildT> converter( deviceSrcGrid1, deviceSrcGrid2 );
    converter.setChecksum(nanovdb::CheckMode::Default);
    converter.setVerbose(1);

    auto handle = converter.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)hostDstReferenceGrid->data(), hostDstReferenceGrid->gridSize()))
        std::cout << "Result of MergeGrids check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of MergeGrids compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    converter.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire dilation after warmstart");
        auto handle2 = converter.getHandle();
        gpuTimer.stop();
    }
}

template
void mainMergeGrids(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceSrcGrid1,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceSrcGrid2,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceDstReferenceGrid,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *hostSrcGrid1,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *hostSrcGrid2,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *hostDstReferenceGrid,
    uint32_t benchmark_iters);
