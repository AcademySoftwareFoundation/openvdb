// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef NANOVDB_CUDA_UTILS_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_UTILS_H_HAS_BEEN_INCLUDED

#include <cuda.h>
#include <cuda_runtime_api.h>

//#if defined(DEBUG) || defined(_DEBUG)
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
        } else if (uint64_t(ptr) % NANOVDB_DATA_ALIGNMENT) {
            fprintf(stderr, "Pointer misalignment error: %s %s %d\n", msg, file, line);
            if (abort) exit(1);
        }
    }
//#else
//    static inline void gpuAssert(cudaError_t, const char*, int, bool = true){}
//    static inline void ptrAssert(void*, const char*, const char*, int, bool = true){}
//#endif

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

#if defined(__CUDACC__)// the following functions only run on the GPU!

// --- Wrapper for launching lambda kernels
template<typename Func, typename... Args>
__global__ void cudaLambdaKernel(const size_t numItems, Func func, Args... args)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    func(tid, args...);
}

/// @brief Copy characters from @c src to @c dst on the device.
/// @param dst pointer to the character array to write to.
/// @param src pointer to the null-terminated character string to copy from.
/// @return pointer to the character array being written to.
/// @note Emulates the behaviour of std::strcpy.
__device__ inline char* cudaStrcpy(char *dst, const char *src)
{
    char *p = dst;
    do {*p++ = *src;} while(*src++);
    return dst;
}

/// @brief Appends a copy of the character string pointed to by @c src to
///        the end of the character string pointed to by @c dst on the device.
/// @param dst pointer to the null-terminated byte string to append to.
/// @param src pointer to the null-terminated byte string to copy from.
/// @return pointer to the character array being appended to.
/// @note Emulates the behaviour of std::strcat.
__device__ inline char* cudaStrcat(char *dst, const char *src)
{
    char *p = dst;
    while (*p) ++p;
    cudaStrcpy(p, src);
    return dst;
}

/// @brief Compares two null-terminated byte strings lexicographically on the device.
/// @param lhs pointer to the null-terminated byte strings to compare
/// @param rhs pointer to the null-terminated byte strings to compare
/// @return Negative value if @c lhs appears before @c rhs in lexicographical order.
///         Zero if @c lhs and @c rhs compare equal. Positive value if @c lhs appears
///         after @c rhs in lexicographical order.
__device__ inline int cudaStrcmp(const char *lhs, const char *rhs)
{
    while(*lhs && (*lhs == *rhs)){
        lhs++;
        rhs++;
    }
    return *(const unsigned char*)lhs - *(const unsigned char*)rhs;// zero if lhs == rhs
}

/// @brief Test if two null-terminated byte strings are the same
/// @param lhs pointer to the null-terminated byte strings to compare
/// @param rhs pointer to the null-terminated byte strings to compare
/// @return true if the two c-strings are identical
__device__ inline bool cudaStrEq(const char *lhs, const char *rhs)
{
    return cudaStrcmp(lhs, rhs) == 0;
}

#endif// __CUDACC__

#endif// NANOVDB_CUDA_UTILS_H_HAS_BEEN_INCLUDED