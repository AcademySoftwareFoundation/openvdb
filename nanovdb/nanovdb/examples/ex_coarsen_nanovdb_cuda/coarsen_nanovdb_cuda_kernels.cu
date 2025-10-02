// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/CoarsenGrid.cuh>

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
void mainCoarsenGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridCoarsened,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridCoarsened,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize coarsener
    nanovdb::tools::cuda::CoarsenGrid<BuildT> coarsener( deviceGridOriginal );
    coarsener.setChecksum(nanovdb::CheckMode::Default);
    coarsener.setVerbose(1);

    auto handle = coarsener.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)indexGridCoarsened->data(), indexGridCoarsened->gridSize()))
        std::cout << "Result of CoarsenGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of CoarsenGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    coarsener.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire coarsening after warmstart");
        auto dummyHandle = coarsener.getHandle();
        gpuTimer.stop();
    }
}

template
void mainCoarsenGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridCoarsened,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridCoarsened,
    uint32_t benchmark_iters
);
