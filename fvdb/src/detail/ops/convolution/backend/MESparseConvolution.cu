#ifndef GPU_CONVOLUTION
#define GPU_CONVOLUTION

#include <iostream>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cublas_v2.h>

#define CUDA_CHECK(condition)                                                  \
  /* Code block avoids redefinition of cudaError_t error */                    \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(error));                     \
    }                                                                          \
  }

#define CUBLAS_CHECK(condition)                                                \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(cublasGetErrorString(status));                  \
    }                                                                          \
  }

constexpr uint32_t MAX_GRID = 65535;

inline int GET_BLOCKS(const uint32_t N, const uint32_t THREADS) {
  return std::max((N + THREADS - 1) / THREADS, uint32_t(1));
}

namespace fvdb {
namespace detail {
namespace ops {

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

// TODO(cfujitsang): use at::cuda::blas::gemm
template <typename Dtype>
void gpu_gemm(const cublasOperation_t transA, const cublasOperation_t transB,
              const int M, const int N, const int K, const at::opmath_type<Dtype> alpha,
              const Dtype *A, const Dtype *B, const at::opmath_type<Dtype> beta, Dtype *C);

template <>
void gpu_gemm<at::Half>(const cublasOperation_t transA, const cublasOperation_t transB,
                        const int M, const int N, const int K,
                        const float alpha, const at::Half *A,
                        const at::Half *B, const float beta, at::Half *C) {
  // Note that cublas follows (column-major) fortran order.
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  int lda = (transA == CUBLAS_OP_N) ? K : M;
  int ldb = (transB == CUBLAS_OP_N) ? N : K;
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  if (!at::globalContext().allowFP16ReductionCuBLAS()) {
    cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }

  CUBLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
  CUBLAS_CHECK(cublasGemmEx(
      handle, transB, transA, N, M, K, &alpha, B, CUDA_R_16F, ldb,
      A, CUDA_R_16F, lda, &beta, C, CUDA_R_16F, N, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void gpu_gemm<float>(const cublasOperation_t transA, const cublasOperation_t transB,
                     const int M, const int N, const int K, const float alpha, const float *A,
                     const float *B, const float beta, float *C) {
  // Note that cublas follows (column-major) fortran order.
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  int lda = (transA == CUBLAS_OP_N) ? K : M;
  int ldb = (transB == CUBLAS_OP_N) ? N : K;
  CUBLAS_CHECK(cublasSgemm(handle, transB, transA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, N));
}

template <>
void gpu_gemm<double>(const cublasOperation_t transA, const cublasOperation_t transB,
                      const int M, const int N, const int K, const double alpha, const double *A,
                      const double *B, const double beta, double *C) {
  // Note that cublas follows fortran order.
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  int lda = (transA == CUBLAS_OP_N) ? K : M;
  int ldb = (transB == CUBLAS_OP_N) ? N : K;
  CUBLAS_CHECK(cublasDgemm(handle, transB, transA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, N));
}

template <typename Dtype, typename Itype>
__global__ void __shared_copy_kernel_map(Dtype *__restrict__ dst,
                                         const Dtype *__restrict__ const src,
                                         const Itype *__restrict__ const map,
                                         const Itype nthreads,
                                         const Itype length) {
  // cchoy: cache map and benchmark.
  extern __shared__ unsigned int smap[];
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const Itype src_index = i / length;
  const Itype length_index = i % length;
  const Itype block_rem = (blockIdx.x * blockDim.x) % length;
  const Itype smap_index = (threadIdx.x + block_rem) / length;
  if ((threadIdx.x == 0 || (threadIdx.x + block_rem) % length == 0) &&
      i < nthreads)
    smap[smap_index] = map[src_index];
  __syncthreads();
  if (i < nthreads) {
    dst[i] = src[smap[smap_index] * length + length_index];
  }
}


template <typename Dtype, typename Itype>
void shared_copy_kernel_map(Dtype *dst, const Dtype *const src,
                            const Itype *const map, const Itype nthreads,
                            const Itype length) {
  constexpr Itype MAX_THREADS = 512;
  if (MAX_THREADS >= length) {
    __shared_copy_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(MAX_THREADS, length) * sizeof(unsigned int)>>>(
            dst, src, map, nthreads, length);
  } else {
    __shared_copy_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(length, MAX_THREADS) * sizeof(unsigned int)>>>(
            dst, src, map, nthreads, length);
  }
}

template <typename Dtype, typename Itype>
__global__ void
__shared_accumulate_kernel_map(Dtype *__restrict__ dst,
                               const Dtype *__restrict__ const src,
                               const Itype *__restrict__ const map,
                               const Itype nthreads, const Itype length) {
  // cchoy: cache map and benchmark.
  extern __shared__ unsigned int smap[];
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const Itype src_index = i / length;
  const Itype length_index = i % length;
  const Itype block_rem = (blockIdx.x * blockDim.x) % length;
  const Itype smap_index = (threadIdx.x + block_rem) / length;
  if ((threadIdx.x == 0 || (threadIdx.x + block_rem) % length == 0) &&
      i < nthreads)
    smap[smap_index] = map[src_index];
  __syncthreads();
  if (i < nthreads)
    atomicAdd(&dst[smap[smap_index] * length + length_index], src[i]);
}

template <typename Dtype, typename Itype>
void shared_accumulate_kernel_map(Dtype *dst, const Dtype *const src,
                                  const Itype *const map, const Itype nthreads,
                                  const Itype length) {
  constexpr Itype MAX_THREADS = 512;
  if (MAX_THREADS >= length)
    __shared_accumulate_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(MAX_THREADS, length) * sizeof(unsigned int)>>>(
            dst, src, map, nthreads, length);
  else
    __shared_accumulate_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(length, MAX_THREADS) * sizeof(unsigned int)>>>(
            dst, src, map, nthreads, length);
}

template <typename Dtype, typename Itype>
__global__ void
add_mapped_output_tr(const size_t n, const Dtype *__restrict__ in_feat,
                     const size_t in_nchannel, Dtype *__restrict__ out_feat,
                     const size_t out_nchannel, const Itype *__restrict__ map) {
  extern __shared__ Itype map_index[];
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. y is for rows, x is for columns.
  const int x = blockDim.x * bx + tx;
  const int y = blockDim.y * by + ty;

  if (x < n && ty == 0)
    map_index[tx] = map[x];

  __syncthreads();

  if (x < n && y < out_nchannel) {
    atomicAdd(&out_feat[map_index[tx] * out_nchannel + y],
              in_feat[y * in_nchannel + x]);
  }
}

namespace ConvolutionMode {
enum Type {
  DEFAULT,
  DIRECT_GEMM,
  COPY_GEMM,
};
}

namespace MinkowskiAlgorithm {
enum Mode { DEFAULT = 0, MEMORY_EFFICIENT = 1, SPEED_OPTIMIZED = 2 };
}

bool check_direct_gemm_forward(MinkowskiAlgorithm::Mode const algo_index, //
                               ConvolutionMode::Type const &convolution_mode,
                               long const sA, long const sB, long const N) {
  if ((convolution_mode == ConvolutionMode::DIRECT_GEMM) ||
      (algo_index == MinkowskiAlgorithm::MEMORY_EFFICIENT))
    return true;
  if (convolution_mode == ConvolutionMode::COPY_GEMM)
    return false;
  if (sA == 32 && sB == 64 and N <= 490537) return true;
  if (sB <= 40) {
    if (sB <= 20) {
      return true;
    } else {
      if (N <= 295625) {
        return true;
      } else {
        return (sA <= 12);
      }
    }
  } else {
    if (sA <= 20)
      return true;
    else {
      if (N <= 74556) {
        return (sB <= 112);
      } else {
        return false;
      }
    }
  }
}

bool check_direct_gemm_backward(MinkowskiAlgorithm::Mode const algo_index, //
                                ConvolutionMode::Type const &convolution_mode,
                                long const sA, long const sB, long const N) {
  if ((convolution_mode == ConvolutionMode::DIRECT_GEMM) ||
      (algo_index == MinkowskiAlgorithm::MEMORY_EFFICIENT))
    return true;
  if (convolution_mode == ConvolutionMode::COPY_GEMM)
    return false;
  if (sA == 32 && sB == 64 and N <= 490537) return true;
  if (sB <= 40) {
    if (sA <= 20)
      return true;
    else {
      if (N <= 490540) {
        return true;
      } else {
        return (sA <= 12);
      }
    }
  } else {
    if (sA <= 20) {
      return true;
    } else {
      if (N <= 30612) {
        return (sB <= 160);
      } else {
        return false;
      }
    }
  }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul(const Dtype *__restrict__ A, const int wA, const int hA, //
       const Dtype *__restrict__ B, const int wB, const int hB, //
       Dtype *__restrict__ C,                                   //
       const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < wB)
    atomicAdd(&C[wB * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
 * wA is A's width and wB is B's width
 *
 *                +---+
 *                |B^T|
 *            +-------+
 *            |   |   |
 *            | A | C |
 *            |   |   |
 *            |   |   |
 * +------------------+
 * |    D^T   | E |
 * +----------+---+
 *
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul2(const Dtype *__restrict__ A, const int wA, const int hA, //
        const Dtype *__restrict__ B, const int wB, const int hB, //
        const Dtype *__restrict__ D, const int wD, const int hD, //
        Dtype *__restrict__ C, Dtype *__restrict__ E,
        const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. y is for rows, x is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;
  Dtype Esub = 0;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ Dtype BTs[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Ds used to
  // store the sub-matrix of D
  __shared__ Dtype DTs[BLOCK_SIZE][BLOCK_SIZE];

  // For Ds = D^T[...:..., ...:...], use the transposed grid dimension for A
  DTs[ty][tx] = (x < wD && y < hD) ? D[wD * in_row + x] : static_cast<Dtype>(0.);

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * out_row + s + tx] : static_cast<Dtype>(0.);

    // Transposed kernel
    BTs[ty][tx] = ((s + ty) < wB && x < hB) ? B[wB * x + s + ty] : static_cast<Dtype>(0.);

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * BTs[k][tx];
    }

    // For Esub, reset to 0
    Esub = 0;
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Esub += DTs[k][ty] * As[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();

    // For the E matrix which requires accmulation of multiple blocks, use
    // atomic addition. This can be replaced with a more sophisticaed
    // reduction algorithm.
    if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
      atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < hB)
    atomicAdd(&C[hB * in_row + x], Csub);
}
//template <typename Dtype, typename Itype>
//void ConvolutionForwardKernelGPU(
//    const Dtype *d_in_feat,                      //
//    const uint32_t in_nchannel,  //
//    Dtype *d_out_feat,                           //
//    const uint32_t out_nchannel, //
//    Dtype *d_kernel,
//    const gpu_kernel_map<Itype> &kernel_map,
//    const uint32_t in_nrows,      //
//    const uint32_t out_nrows,     //
//    const MinkowskiAlgorithm::Mode algo_index,    //
//    const ConvolutionMode::Type convolution_mode, //
//    cublasHandle_t cuhandle, cudaStream_t stream) {
//
//  size_t n_active_in_volume, shared_mem_size;
//
//  if (check_direct_gemm_forward(algo_index, convolution_mode,
//                                in_nchannel, out_nchannel, in_nrows)) {
//    // Define the shared memory size
//    if ((in_nchannel > 16 && out_nchannel > 16 &&
//         in_nchannel * out_nchannel >= 512) ||
//        (in_nchannel > 24 && out_nchannel > 24))
//      shared_mem_size = 32;
//    else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
//      shared_mem_size = 24;
//    else if ((in_nchannel > 8 && out_nchannel > 8) ||
//             (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
//      shared_mem_size = 16;
//    else
//      shared_mem_size = 8;
//
//    dim3 threads(shared_mem_size, shared_mem_size);
//
//    // Iterate through each spatial kernel and get indices for in_map and
//    // out_map
//    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
//      auto const k = it->first;
//      n_active_in_volume = kernel_map.size(k);
//      if (n_active_in_volume == 0)
//        continue;
//
//      size_t const num_grid =
//          (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
//      size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
//      size_t const step = (n_active_in_volume + num_div - 1) / num_div;
//
//      for (size_t s = 0; s < num_div; s++) {
//        size_t const offset = step * s;
//        size_t const remainder = n_active_in_volume - offset;
//        size_t const curr_num_active = remainder < step ? remainder : step;
//        dim3 const grid((out_nchannel + threads.x - 1) / threads.x,
//                        (curr_num_active + threads.y - 1) / threads.y);
//
//        switch (shared_mem_size) {
//        case 32:
//          matmul<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
//              d_in_feat, in_nchannel, curr_num_active,
//              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
//              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
//              kernel_map.out_maps.begin(k) + offset);
//          break;
//        case 24:
//          matmul<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
//              d_in_feat, in_nchannel, curr_num_active,
//              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
//              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
//              kernel_map.out_maps.begin(k) + offset);
//          break;
//        case 16:
//          matmul<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
//              d_in_feat, in_nchannel, curr_num_active,
//              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
//              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
//              kernel_map.out_maps.begin(k) + offset);
//          break;
//        case 8:
//          matmul<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
//              d_in_feat, in_nchannel, curr_num_active,
//              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
//              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
//              kernel_map.out_maps.begin(k) + offset);
//          break;
//        }
//      }
//      C10_CUDA_KERNEL_LAUNCH_CHECK();
//    }
//  } else { // copy gemm
//    Itype const max_numel = kernel_map.max_size();
//    Dtype *mapped_in_feat = reinterpret_cast<Dtype*>(
//        c10::cuda::CUDACachingAllocator::raw_alloc(max_numel * in_nchannel * sizeof(Dtype)));
//    Dtype *mapped_out_feat = reinterpret_cast<Dtype *>(
//        c10::cuda::CUDACachingAllocator::raw_alloc(max_numel * out_nchannel * sizeof(Dtype)));
//
//    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
//      auto const k = it->first;
//      n_active_in_volume = kernel_map.size(k);
//      if (n_active_in_volume == 0)
//        continue;
//
//      shared_copy_kernel_map<Dtype, Itype>(
//          // mapped_in_feat,
//          mapped_in_feat, d_in_feat, kernel_map.in_maps.begin(k),
//          n_active_in_volume * in_nchannel, in_nchannel);
//
//      gpu_gemm<Dtype>(cuhandle, CUBLAS_OP_N, CUBLAS_OP_N,
//                      n_active_in_volume,                        // M
//                      out_nchannel,                              // N
//                      in_nchannel,                               // K
//                      1,                                         // alpha
//                      mapped_in_feat,                            // A
//                      &d_kernel[k * in_nchannel * out_nchannel], // B
//                      0,                                         // beta
//                      mapped_out_feat                            // C
//      );
///*
//      int lda = K
//      int ldb = N;
//
//      at::cuda::blas::gemm<Dtype>(
//          'n', 'n', n_active_in_volume, out_nchannel, in_nchannel,
//          1, mapped_in_feat,
//*/
//      shared_accumulate_kernel_map<Dtype, Itype>(
//          d_out_feat, mapped_out_feat, kernel_map.out_maps.begin(k),
//          n_active_in_volume * out_nchannel, out_nchannel);
//    }
//
//    c10::cuda::CUDACachingAllocator::raw_delete(mapped_in_feat);
//    c10::cuda::CUDACachingAllocator::raw_delete(mapped_out_feat);
//  }
//  CUDA_CHECK(cudaStreamSynchronize(stream));
//}
//
//template void
//ConvolutionForwardKernelGPU<float, uint32_t>(
//    const float *d_in_feat,
//    const uint32_t in_nchannel,
//    float *d_out_feat,
//    const uint32_t out_nchannel,
//    float *d_kernel,
//    const gpu_kernel_map<uint32_t> &kernel_map,
//    const uint32_t in_nrows,
//    const uint32_t out_nrows,
//    const MinkowskiAlgorithm::Mode algo_index,
//    const ConvolutionMode::Type convolution_mode,
//    cublasHandle_t cuhandle,
//    cudaStream_t stream);
//
//template void
//ConvolutionForwardKernelGPU<double, uint32_t>(
//    const double *d_in_feat,
//    const uint32_t in_nchannel,
//    double *d_out_feat,
//    const uint32_t out_nchannel,
//    double *d_kernel,
//    const gpu_kernel_map<uint32_t> &kernel_map,
//    const uint32_t in_nrows,
//    const uint32_t out_nrows,
//    const MinkowskiAlgorithm::Mode algo_index,
//    const ConvolutionMode::Type convolution_mode,
//    cublasHandle_t cuhandle,
//    cudaStream_t stream);


void dispatchMESparseConvolutionKernelMapGrad(at::Tensor in_feat,
                                              at::Tensor grad_in_feat,
                                              at::Tensor grad_out_feat,
                                              at::Tensor kernel,
                                              at::Tensor grad_kernel,
                                              at::Tensor neighbor_map,
                                              at::Tensor neighbor_offset,
                                              const bool transpose) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();
  const auto full_in_map = neighbor_map.index({torch::indexing::Slice(), int(transpose)}).contiguous();
  const auto full_out_map = neighbor_map.index({torch::indexing::Slice(), int(!transpose)}).contiguous();
  bool is_half = in_feat.scalar_type() == at::ScalarType::Half;
  int in_nrows = in_feat.size(0);
  int in_nchannel = in_feat.size(1);
  int out_nrows = grad_out_feat.size(0);
  int out_nchannel = kernel.size(-1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    in_feat.scalar_type(), "convolution_backward_cuda", ([&] {
  //using scalar_t = float;
  using Dtype = scalar_t;
  using Itype = uint32_t;

  const scalar_t *d_in_feat = in_feat.data_ptr<scalar_t>();
  scalar_t *d_grad_in_feat = grad_in_feat.data_ptr<scalar_t>();
  const scalar_t *d_grad_out_feat = grad_out_feat.data_ptr<scalar_t>();
  scalar_t const *d_kernel = kernel.data_ptr<scalar_t>();
  scalar_t* d_grad_kernel = grad_kernel.data_ptr<scalar_t>();


  int kernel_volume = kernel.size(0);
  size_t n_active_in_volume, shared_mem_size;
  // Define the shared memory size
  if ((in_nchannel > 16 && out_nchannel > 16 &&
       in_nchannel * out_nchannel >= 512) ||
      (in_nchannel % 32 == 0 && out_nchannel % 32 == 0))
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if ((in_nchannel > 8 && out_nchannel > 8) ||
           (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
    shared_mem_size = 16;
  else
    shared_mem_size = 8;
  int cur_offset = 0;
  if (!check_direct_gemm_backward(
          MinkowskiAlgorithm::Mode::DEFAULT, ConvolutionMode::Type::DEFAULT,
	  in_nchannel, out_nchannel, in_nrows)) {
    // find max size
    size_t max_active = *std::max_element(
        neighbor_offset.data_ptr<int>(),
        neighbor_offset.data_ptr<int>() + kernel_volume);

    size_t in_buffer_size = max_active * in_nchannel * sizeof(scalar_t);
    size_t out_buffer_size = max_active * out_nchannel * sizeof(scalar_t);
    scalar_t *d_input_buffer = reinterpret_cast<scalar_t*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(in_buffer_size));
    scalar_t *d_output_buffer = reinterpret_cast<scalar_t*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(out_buffer_size));

    dim3 threads(32, shared_mem_size);
    for (int k = 0; k < kernel_volume; ++k) {
      n_active_in_volume = neighbor_offset.data_ptr<int>()[k];
      if (n_active_in_volume == 0)
        continue;
      const Itype* d_in_map = reinterpret_cast<Itype*>(full_in_map.data_ptr<int>()) + cur_offset;
      const Itype* d_out_map = reinterpret_cast<Itype*>(full_out_map.data_ptr<int>()) + cur_offset;
      shared_copy_kernel_map<Dtype, Itype>(
        d_output_buffer, d_grad_out_feat, d_out_map,
        n_active_in_volume * out_nchannel, out_nchannel);
      gpu_gemm<Dtype>(CUBLAS_OP_N, CUBLAS_OP_T,
                      in_nchannel,                               // M
                      n_active_in_volume,                        // N
                      out_nchannel,                              // K
                      1,                                         // alpha
                      &d_kernel[k * in_nchannel * out_nchannel], // A
                      d_output_buffer,                           // B
                      0,                                         // beta
                      d_input_buffer                             // C
      );
      // Accumulate gradients back to the input grad feat
      // Put it back to the correct index
      dim3 const grid_tr(GET_BLOCKS(n_active_in_volume, threads.x),
                         GET_BLOCKS(in_nchannel, threads.y));
      add_mapped_output_tr<Dtype, Itype>
          <<<grid_tr, threads, threads.x * sizeof(Itype), stream>>>(
              n_active_in_volume,
              d_input_buffer,              // In
              n_active_in_volume,          // In channel
              d_grad_in_feat, in_nchannel, // Out
              d_in_map);                   // Out channel

      // Compute gradient for kernel
      // Copy features to the buffer
      dim3 const grid_in(GET_BLOCKS(n_active_in_volume, threads.x),
                         GET_BLOCKS(in_nchannel, threads.y));
      shared_copy_kernel_map<Dtype, Itype>(
          d_input_buffer, d_in_feat, d_in_map, n_active_in_volume * in_nchannel,
          in_nchannel);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      gpu_gemm<Dtype>(CUBLAS_OP_T, CUBLAS_OP_N,
                      in_nchannel,                                   // M
                      out_nchannel,                                  // N
                      n_active_in_volume,                            // K
                      1,                                             // alpha
                      d_input_buffer,                                // A
                      d_output_buffer,                               // B
                      1,                                             // beta
                      &d_grad_kernel[k * in_nchannel * out_nchannel] // C
      );
      CUDA_CHECK(cudaStreamSynchronize(0));
      cur_offset += n_active_in_volume;
    }
    c10::cuda::CUDACachingAllocator::raw_delete(d_input_buffer);
    c10::cuda::CUDACachingAllocator::raw_delete(d_output_buffer);
  } else {
    dim3 threads(shared_mem_size, shared_mem_size);
    for (int k = 0; k < kernel_volume; ++k) {
      n_active_in_volume = neighbor_offset.data_ptr<int>()[k];
      if (n_active_in_volume == 0)
        continue;

      size_t const num_grid =
          (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
      size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
      size_t const step = (n_active_in_volume + num_div - 1) / num_div;

      const Itype* d_in_map = reinterpret_cast<Itype*>(full_in_map.data_ptr<int>()) + cur_offset;
      const Itype* d_out_map = reinterpret_cast<Itype*>(full_out_map.data_ptr<int>()) + cur_offset;
      for (int s = 0; s < num_div; s++) {
        size_t const offset = step * s;
        size_t const remainder = n_active_in_volume - offset;
        size_t const curr_num_active = remainder < step ? remainder : step;
        dim3 const grid((in_nchannel + threads.x - 1) / threads.x,
                        (curr_num_active + threads.y - 1) / threads.y);
        switch (shared_mem_size) {
        case 32:
          matmul2<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              d_in_map + offset,
              d_out_map + offset);
          break;
        case 24:
          matmul2<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              d_in_map + offset,
              d_out_map + offset);
          break;
        case 16:
          matmul2<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              d_in_map + offset,
              d_out_map + offset);
          break;
        case 8:
          matmul2<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              d_in_map + offset,
              d_out_map + offset);
          break;
        }
      }
      CUDA_CHECK(cudaGetLastError());
      cur_offset += n_active_in_volume;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  }));
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // end GPU_CONVOLUTION

