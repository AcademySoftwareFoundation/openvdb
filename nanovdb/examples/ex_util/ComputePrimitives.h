// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ComputePrimitives.h
/// @brief A collection of parallel compute primitives

#pragma once

#if defined(NANOVDB_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(NANOVDB_USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <utility>
#include <tuple>


// forward compatibility for C++14 Standard Library
namespace cxx14 {
template<std::size_t...>
struct index_sequence
{
};

template<std::size_t N, std::size_t... Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...>
{
};

template<std::size_t... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...>
{
};
} // namespace cxx14

#if defined(__CUDACC__)

static inline bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        return false;
    }
    return true;
}

#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static inline void checkErrorCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        exit(1);
    }
}

#define NANOVDB_CUDA_CHECK_ERROR(result, file, line) checkErrorCUDA(result, file, line)

#endif

template<typename Fn, typename... Args>
class ApplyFunc
{
public:
    ApplyFunc(int count, int blockSize, const Fn& fn, Args... args)
        : mCount(count)
        , mBlockSize(blockSize)
        , mArgs(args...)
        , mFunc(fn)
    {
    }

    template<std::size_t... Is>
    void call(int start, int end, cxx14::index_sequence<Is...>) const
    {
        mFunc(start, end, std::get<Is>(mArgs)...);
    }

    void operator()(int i) const
    {
        int start = i * mBlockSize;
        int end = i * mBlockSize + mBlockSize;
        if (end > mCount)
            end = mCount;
        call(start, end, cxx14::make_index_sequence<sizeof...(Args)>());
    }

#if defined(NANOVDB_USE_TBB)
    void operator()(const tbb::blocked_range<int>& r) const
    {
        int start = r.begin();
        int end = r.end();
        if (end > mCount)
            end = mCount;
        call(start, end, cxx14::make_index_sequence<sizeof...(Args)>());
    }
#endif

private:
    int                 mCount;
    int                 mBlockSize;
    Fn                  mFunc;
    std::tuple<Args...> mArgs;
};

#if defined(__CUDACC__)

template<int WorkPerThread, typename FnT, typename... Args>
__global__ void parallelForKernel(int numItems, FnT f, Args... args)
{
    for (int j=0;j<WorkPerThread;++j)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x + j * blockDim.x * gridDim.x;
        if (i < numItems)
            f(i, i + 1, args...);
    }
}

#endif

inline void computeSync(bool useCuda, const char* file, int line)
{
#if defined(__CUDACC__)
    if (useCuda) {
        NANOVDB_CUDA_CHECK_ERROR(cudaDeviceSynchronize(), file, line);
    }
#endif
}

inline void computeFill(bool useCuda, void* data, uint8_t value, size_t size)
{
    if (useCuda) {
#if defined(__CUDACC__)
        cudaMemset(data, value, size);
#endif
    } else {
        std::memset(data, value, size);
    }
}

template<typename FunctorT, typename... Args>
inline void computeForEach(bool useCuda, int numItems, int blockSize, const char* file, int line, const FunctorT& op, Args... args)
{
    if (numItems == 0)
        return;

    if (useCuda) {
#if defined(__CUDACC__)
        static const int WorkPerThread = 1;
        int blockCount = ((numItems/WorkPerThread) + (blockSize - 1)) / blockSize;
        parallelForKernel<WorkPerThread, FunctorT, Args...><<<blockCount, blockSize, 0, 0>>>(numItems, op, args...);
        NANOVDB_CUDA_CHECK_ERROR(cudaGetLastError(), file, line);
#endif
    } else {
#if defined(NANOVDB_USE_TBB)
        tbb::blocked_range<int> range(0, numItems, blockSize);
        tbb::parallel_for(range, ApplyFunc<FunctorT, Args...>(numItems, blockSize, op, args...));
#else
        for (int i = 0; i < numItems; ++i)
            op(i, i + 1, args...);
#endif
    }
}

inline void computeDownload(bool useCuda, void* dst, const void* src, size_t size)
{
    if (useCuda) {
#if defined(__CUDACC__)
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
#endif
    } else {
        std::memcpy(dst, src, size);
    }
}

inline void computeCopy(bool useCuda, void* dst, const void* src, size_t size)
{
    if (useCuda) {
#if defined(__CUDACC__)
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
#endif
    } else {
        std::memcpy(dst, src, size);
    }
}
