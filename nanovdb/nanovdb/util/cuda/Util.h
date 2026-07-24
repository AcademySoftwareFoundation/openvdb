// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/cuda/Util.h

    \author Ken Museth

    \date December 20, 2023

    \brief Cuda specific utility functions
*/

#ifndef NANOVDB_UTIL_CUDA_UTIL_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_UTIL_H_HAS_BEEN_INCLUDED

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <nanovdb/util/Util.h> // for stderr and NANOVDB_ASSERT

// change 1 -> 0 to only perform asserts during debug builds
#if 1 || defined(DEBUG) || defined(_DEBUG)
    static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
    {
        if (code != cudaSuccess) {
            fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
            //fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    static inline void ptrAssert(const void* ptr, const char* msg, const char* file, int line, bool abort = true)
    {
        if (ptr == nullptr) {
            fprintf(stderr, "NULL pointer error: %s %s %d\n", msg, file, line);
            if (abort) exit(1);
        } else if (uint64_t(ptr) % 32) {
            fprintf(stderr, "Pointer misalignment error: %s %s %d\n", msg, file, line);
            if (abort) exit(1);
        }
    }
#else
    static inline void gpuAssert(cudaError_t, const char*, int, bool = true){}
    static inline void ptrAssert(void*, const char*, const char*, int, bool = true){}
#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define cudaCheck(ans) \
    { \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

#define checkPtr(ptr, msg) \
    { \
        ptrAssert((ptr), (msg), __FILE__, __LINE__); \
    }

#define cudaSync() \
    { \
        cudaCheck(cudaDeviceSynchronize()); \
    }

#define cudaCheckError() \
    { \
        cudaCheck(cudaGetLastError()); \
    }

namespace nanovdb {// =========================================================

namespace util::cuda {// ======================================================

//#define NANOVDB_USE_SYNC_CUDA_MALLOC
// cudaMallocAsync and cudaFreeAsync were introduced in CUDA 11.2, and even on newer
// toolkits a device may not expose stream-ordered memory pools at runtime — fractional
// vGPU configurations commonly disable them, making cudaMallocAsync fail with
// cudaErrorNotSupported. The wrappers below therefore behave in two modes: when
// CUDA < 11.2 or NANOVDB_USE_SYNC_CUDA_MALLOC is defined they map to synchronous
// cudaMalloc/cudaFree; otherwise they use cudaMallocAsync/cudaFreeAsync and, on a
// device that lacks memory pools, fail with an actionable diagnostic rather than
// silently substituting synchronous allocation (which would make an async resource
// misrepresent its own semantics — the choice of a synchronous backend belongs to the
// caller, not to a hidden runtime fallback). Callers select synchronous allocation by
// defining NANOVDB_USE_SYNC_CUDA_MALLOC, which -- because the wrappers are inline
// functions -- must be defined identically in every translation unit to avoid violating
// the one-definition rule.

/// @brief Returns true if @c device supports stream-ordered memory pools, i.e.
///        cudaMallocAsync/cudaFreeAsync. Queried once per process for all
///        devices and cached; out-of-range device ids return false.
inline bool memoryPoolsSupported(int device)
{
#if (CUDART_VERSION < 11020)
    (void)device;
    return false;
#else
    static const auto supported = [] {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 0) count = 0;
        std::vector<char> s(static_cast<size_t>(count), 0);
        for (int i = 0; i < count; ++i) {
            int attr = 0;
            if (cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, i) == cudaSuccess)
                s[static_cast<size_t>(i)] = char(attr != 0);
        }
        return s;
    }();
    return device >= 0 && static_cast<size_t>(device) < supported.size() && supported[static_cast<size_t>(device)] != 0;
#endif
}

#if (CUDART_VERSION < 11020) || defined(NANOVDB_USE_SYNC_CUDA_MALLOC) // 11.2 introduced cudaMallocAsync and cudaFreeAsync

/// @brief Wrapper forced to synchronous cudaMalloc; see the mode comment above.
/// @param d_ptr Device pointer to allocated device memory
/// @param size  Number of bytes to allocate
/// @param dummy The stream establishing the stream ordering contract and the memory pool to allocate from (ignored)
/// @return Cuda error code
inline cudaError_t mallocAsync(void** d_ptr, size_t size, cudaStream_t){return cudaMalloc(d_ptr, size);}

/// @brief Wrapper forced to synchronous cudaFree; see the mode comment above.
/// @param d_ptr Device pointer that will be freed
/// @param dummy The stream establishing the stream ordering promise (ignored)
/// @return Cuda error code
inline cudaError_t freeAsync(void* d_ptr, cudaStream_t){return cudaFree(d_ptr);}

#else

/// @brief Wrapper that calls cudaMallocAsync. On a device without stream-ordered
///        memory pools it emits an actionable diagnostic and returns
///        cudaErrorNotSupported rather than silently allocating synchronously.
/// @param d_ptr Device pointer to allocated device memory
/// @param size  Number of bytes to allocate
/// @param stream The stream establishing the stream ordering contract and the memory pool to allocate from
/// @return Cuda error code
inline cudaError_t mallocAsync(void** d_ptr, size_t size, cudaStream_t stream)
{
    int device = 0;
    if (const cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;
    if (!memoryPoolsSupported(device)) {
        fprintf(stderr,
                "NanoVDB: device %d does not support stream-ordered CUDA memory pools required by "
                "cudaMallocAsync. Define NANOVDB_USE_SYNC_CUDA_MALLOC to allocate synchronously with "
                "cudaMalloc/cudaFree instead.\n",
                device);
        return cudaErrorNotSupported;
    }
    return cudaMallocAsync(d_ptr, size, stream);
}

/// @brief Wrapper that calls cudaFreeAsync. Mirrors mallocAsync's pool check so a
///        device without memory pools reports cudaErrorNotSupported rather than
///        calling cudaFreeAsync where it cannot succeed.
/// @param d_ptr Device pointer that will be freed
/// @param stream The stream establishing the stream ordering promise
/// @return Cuda error code
inline cudaError_t freeAsync(void* d_ptr, cudaStream_t stream)
{
    int device = 0;
    if (const cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) return err;
    if (!memoryPoolsSupported(device)) return cudaErrorNotSupported;
    return cudaFreeAsync(d_ptr, stream);
}

#endif

/// @brief Returns the device ID associated with the specified pointer
/// @note  If @c ptr points to host memory (only) the return ID is either cudaInvalidDeviceId = -2 or cudaCpuDeviceId = -1
inline int ptrToDevice(void *ptr)
{
    cudaPointerAttributes ptrAtt;
    cudaCheck(cudaPointerGetAttributes(&ptrAtt, ptr));
    return ptrAtt.device;
}

/// @brief Returns the ID of the current device
inline int currentDevice()
{
    int current = cudaInvalidDeviceId;
    cudaCheck(cudaGetDevice(&current));
    assert(current != cudaInvalidDeviceId);
    return current;
}

/// @brief Returns the number of devices with compute capability greater or equal to 1.0 that are available for execution
inline int deviceCount()
{
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

/// @brief Print information about a specific device
/// @param device device ID for which information will be printed
/// @param preMsg optional message printed before the device information
/// @param file   Optional file stream to print to, e.g. stderr or stdout
inline void printDevInfo(int device, const char *preMsg = nullptr, std::FILE* file = stderr)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (preMsg) fprintf(file, "%s ", preMsg);
    fprintf(file,"GPU #%d, named \"%s\", compute capability %d.%d, %zu GB of VRAM\n",
            device, prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 30);
}

/// @brief Simple (naive) implementation of a unique device pointer
///        using stream ordered memory allocation and deallocation.
/// @tparam T Type of the device pointer
template <typename T>
class unique_ptr
{
    T           *mPtr;// pointer to stream ordered memory allocation
    cudaStream_t mStream;
public:
    unique_ptr(size_t count = 0, cudaStream_t stream = 0) : mPtr(nullptr), mStream(stream)
    {
        if (count>0) cudaCheck(mallocAsync((void**)&mPtr, count*sizeof(T), stream));
    }
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr(unique_ptr&& other) : mPtr(other.mPtr), mStream(other.mStream)
    {
        other.mPtr = nullptr;
    }
    ~unique_ptr()
    {
        if (mPtr) cudaCheck(freeAsync(mPtr, mStream));
    }
    unique_ptr& operator=(const unique_ptr&) = delete;
    unique_ptr& operator=(unique_ptr&& rhs) noexcept
    {
        mPtr = rhs.mPtr;
        mStream = rhs.mStream;
        rhs.mPtr = nullptr;
        return *this;
    }
    void reset() {
        if (mPtr) {
            cudaCheck(freeAsync(mPtr, mStream));
            mPtr = nullptr;
        }
    }
    T* get()                 const {return mPtr;}
    explicit operator bool() const {return mPtr != nullptr;}
};// util::cuda::unique_ptr

/// @brief Computes the number of blocks per grid given the problem size and number of threads per block
/// @param numItems Problem size
/// @param threadsPerBlock Number of threads per block (second CUDA launch parameter)
/// @return number of blocks per grid (first CUDA launch parameter)
/// @note CUDA launch parameters: kernel<<< blocksPerGrid, threadsPerBlock, sharedMemSize, streamID>>>
inline size_t blocksPerGrid(size_t numItems, size_t threadsPerBlock)
{
    NANOVDB_ASSERT(numItems > 0 && threadsPerBlock >= 32 && threadsPerBlock % 32 == 0);
    return (numItems + threadsPerBlock - 1) / threadsPerBlock;
}

// CUDA 13.0 changes cudaMemPrefetchAsync and cudaMemPrefetch to use a cudaMemLocation as an argument as
// opposed to an integer device id. This function provides compatibility by returning the corresponding
// location in CUDA 13.0 and above while passing through the device in earlier versions.
#if (CUDART_VERSION < 13000)
/// @brief Compatbility wrapper for cudaMemAdvise/cudaMemAdvise
inline cudaError_t memAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
    return cudaMemAdvise(devPtr, count, advice, device);
}

/// @brief Compatbility wrapper for cudaMemPrefetchAsync/cudaMemPrefetchAsync
inline cudaError_t memPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    return cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
}
#else
/// @brief Helper function that converts a device id to a cudaMemLocation
/// @param device Integer device id
/// @return cudaMemLocation corresponding to the device id
inline cudaMemLocation deviceToLocation(int device) {
    if (device < cudaCpuDeviceId) {
        return {cudaMemLocationTypeInvalid, device};
    } else if (device == cudaCpuDeviceId) {
        return {cudaMemLocationTypeHost, device};
    } else {
        return {cudaMemLocationTypeDevice, device};
    }
}

/// @brief Compatbility wrapper for cudaMemAdvise/cudaMemAdvise
inline cudaError_t memAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
    return cudaMemAdvise(devPtr, count, advice, deviceToLocation(device));
}

/// @brief Compatbility wrapper for cudaMemPrefetchAsync/cudaMemPrefetchAsync
inline cudaError_t memPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    return cudaMemPrefetchAsync(devPtr, count, deviceToLocation(dstDevice), 0u, stream);
}
#endif

#if defined(__CUDACC__)// the following functions only run on the GPU!

/// @brief Cuda kernel that launches device lambda functions
/// @param numItems Problem size
template<typename Func, typename... Args>
__global__ void lambdaKernel(const size_t numItems, Func func, Args... args)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    func(tid, args...);
}// util::cuda::lambdaKernel

/// @brief Cuda kernel that launches device lambda functions with a tid offset
/// @param numItems Problem size
/// @param offset Offset for thread id
template<typename Func, typename... Args>
__global__ void offsetLambdaKernel(size_t numItems, unsigned int offset, Func func, Args... args)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    func(tid + offset, args...);
}// util::cuda::offsetLambdaKernel

/// @brief Cuda kernel that launches device operator functors with arbitrary arguments
template<class Operator, typename... Args>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void operatorKernel(
    Args... args)
{
    Operator op;
    op( args... );
}

/// @brief Cuda kernel that launches a pre-constructed device operator functor with arbitrary arguments.
///        Unlike operatorKernel, the operator is passed by value (copied at launch) rather than
///        default-constructed on the device, allowing functors with data members.
template<class Operator, typename... Args>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void operatorKernelInstance(Operator op, Args... args)
{
    op( args... );
}

/// @brief Cuda kernel that launches device operator functors with arbitrary arguments, using dynamic shared memory
template<class Operator, typename... Args>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void operatorKernelDynamic(Args... args)
{
    extern __shared__ char smem_buf[];
    Operator op;
    op( args..., smem_buf );
}

/// @brief Wrapper for launching a device operator that leverages dynamic shared memory, with a specified size
/// @code
/// struct MyFunctor
/// {
///     // These are passed to __launch_bounds__
///     static constexpr int MaxThreadsPerBlock = <nThreads>
///     static constexpr int MinBlocksPerMultiprocessor = 1;
///
///     struct SharedStorage {
///         // Include whatever is needed in smem
///     };
///
///     __device__
///     void operator()(Args ... myArgs, char smem_buf[])
///     { ... }
/// };
///
/// dynamicSharedMemoryLauncher<MyFunctor>(nBlocks, sizeof(typename MyFunctor::SharedStorage), myArgs...);
/// // smem_buff of size sizeof(MyFunctor::SharedStorage) will be automatically passed along
/// @endcode
template<class Operator, typename... Args>
void dynamicSharedMemoryLauncher(const size_t numItems, const size_t smem_size, cudaStream_t stream, Args... args)
{
    cudaCheck(cudaFuncSetAttribute(operatorKernelDynamic<Operator, Args...>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size));
    operatorKernelDynamic<Operator>
        <<<numItems, Operator::MaxThreadsPerBlock, smem_size, stream>>>( args ... );
}

#endif// __CUDACC__

}// namespace util::cuda ============================================================

}// namespace nanovdb ===============================================================

#if defined(__CUDACC__)// the following functions only run on the GPU!
template<typename Func, typename... Args>
[[deprecated("Use nanovdb::cuda::lambdaKernel instead")]]
__global__ void cudaLambdaKernel(const size_t numItems, Func func, Args... args)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    func(tid, args...);
}
#endif// __CUDACC__

#endif// NANOVDB_UTIL_CUDA_UTIL_H_HAS_BEEN_INCLUDED
