// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/DilateGrid.cuh>

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
    nanovdb::NanoGrid<BuildT> *deviceGridDilated,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters)
{
    nanovdb::util::cuda::Timer gpuTimer;

    // Initialize converter
    nanovdb::tools::cuda::DilateGrid<BuildT> converter( deviceGridOriginal );
    converter.setOperation(nanovdb::tools::morphology::NearestNeighbors(nnType));
    converter.setChecksum(nanovdb::CheckMode::Default);
    converter.setVerbose(1);

    auto handle = converter.getHandle();
    auto dstGrid = handle.template deviceGrid<BuildT>();

    // Check for correctness
    if (bufferCheck((char*)dstGrid, (char*)indexGridDilated->data(), indexGridDilated->gridSize()))
        std::cout << "Result of DilateGrid check out CORRECT against reference" << std::endl;
    else
        std::cout << "Result of DilateGrid compares INCORRECT against reference" << std::endl;

    // Re-run warm-started iterations
    converter.setVerbose(0);
    for (int i = 0; i < benchmark_iters; i++) {
        gpuTimer.start("Re-running entire dilation after warmstart");
        auto handle2 = converter.getHandle();
        gpuTimer.stop();
    }
}

template
void mainDilateGrid(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGridDilated,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridOriginal,
    nanovdb::NanoGrid<nanovdb::ValueOnIndex> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);
