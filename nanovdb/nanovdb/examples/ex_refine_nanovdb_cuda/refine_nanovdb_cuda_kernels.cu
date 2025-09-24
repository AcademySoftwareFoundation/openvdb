// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/RefineGrid.cuh>

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
void mainRefineGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridRefined,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridRefined,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize refiner
    nanovdb::tools::cuda::RefineGrid<BuildT> refiner( deviceGridOriginal );
    refiner.setChecksum(nanovdb::CheckMode::Default);
    refiner.setVerbose(1);

    auto handle = refiner.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)indexGridRefined->data(), indexGridRefined->gridSize()))
        std::cout << "Result of RefineGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of RefineGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    refiner.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire refinment after warmstart");
        auto dummyHandle = refiner.getHandle();
        gpuTimer.stop();
    }
}

template
void mainRefineGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridRefined,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridRefined,
    uint32_t benchmark_iters
);
