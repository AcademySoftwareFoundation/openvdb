#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>
#include <mma.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/convolution/backend/ConvOps.h"

#define COALESCED_MEMORY_ACCESS_VARIANT

namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType>
__global__ __launch_bounds__(1024)     // Hinting maximum threads per block during launch to optimize register usage.
void stencilConvHaloKernel(int kM, int kN, int numLeaves,
                           BatchGridAccessor<GridType> gridAcc,
                           TorchRAcc32<float, 2> inFeatures,
                           TorchRAcc32<float, 7> stencil,
                           TorchRAcc32<float, 2> outFeatures) {

// While 700 (Volta) already supports TensorCore, it does not support TF32.
// 800 (Ampere) supports both TensorCore and TF32.
#if __CUDA_ARCH__ >= 800

    int tid = threadIdx.x;
    int leafIdx = blockIdx.x % numLeaves;
    int nIdx = (blockIdx.x / numLeaves) % kN;
    int mIdx = blockIdx.x / numLeaves / kN;

    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    // Constants: TensorCore GEMM shape MK x KN = MN
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 8;
    using MatrixAType = float (&)[M][K];
    using MatrixBType = float (&)[K][N];
    using MatrixCType = float (&)[M][N];

    // Constants: Input and output dimension multiples
    static constexpr int Di = 8;
    static constexpr int Do = 16;

    using HaloBufferType = float (&)[10][10][10][Di];
    using DenseOutputBufferType = float (&)[8][8][8][Do];

    const int64_t batchIdx = gridAcc.leafBatchIndex(leafIdx);
    const int64_t localLeafIdx = leafIdx - gridAcc.leafOffset(batchIdx);
    const int64_t baseOffset = gridAcc.voxelOffset(batchIdx);

    const nanovdb::NanoGrid<GridType>* deviceGrid = gridAcc.grid(batchIdx);
    const LeafNodeType& leaf = deviceGrid->tree().template getFirstNode<0>()[localLeafIdx];
    const nanovdb::Coord origin = leaf.origin();
    auto deviceGridAcc = deviceGrid->getAccessor();

    // Shared memory buffer (re-used by both 10x10x10 of size Di, or 8x8x8 of size Do)
    __shared__ float sBufferRaw [8192];
    HaloBufferType sHaloBuffer = reinterpret_cast<HaloBufferType>(sBufferRaw[0]);

    // Dense gathering of input features 10x10x10 = 1000
    if (tid < 1000) {
        int di = ((tid/100) % 10) - 1;
        int dj = ((tid/10) % 10) - 1;
        int dk = (tid % 10) - 1;

        auto coord = origin.offsetBy(di,dj,dk);

        if (deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
            auto offset = deviceGridAcc.getValue(coord) - 1 + baseOffset;
            for (int s = 0; s < Di; s++) {
                int tDim = s + mIdx * Di;
                sHaloBuffer[0][0][tid][s] = tDim < inFeatures.size(1) ? inFeatures[offset][tDim] : 0.0f;
            }
        }
        else {
            for (int s = 0; s < Di; s++)
                sHaloBuffer[0][0][tid][s] = 0.0f;
        }
    }

    __syncthreads();

    // Spoke tensor is a 8x8x8 subset of the 10x10x10 halo buffer
    __shared__ float spokeStencil [8][8][8][Di];

    // Depulicated k dimension
    int i  = (threadIdx.x >> 7) & 0x7;
    int J  = (threadIdx.x >> 4) & 0x6; // rounded down to "even" j
#ifndef COALESCED_MEMORY_ACCESS_VARIANT
    int jj = (threadIdx.x >> 4) & 0x1; // remainder; j = J + jj
    int k  = (threadIdx.x >> 1) & 0x7;
#else
    int threadWarpID = threadIdx.x & 0x1f;
#endif

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // For all kernel positions, add result to C.
    //  Note: each 2-thread (totalling 512, achieved by k) is responsible for each location in the output.
    //   each of the 32 [16]x2-threads collaborate within to do the matrix multiplication.
    for (int di = 0; di <= 2; di++)
    for (int dj = 0; dj <= 2; dj++)
    for (int dk = 0; dk <= 2; dk++)
    {

#ifdef COALESCED_MEMORY_ACCESS_VARIANT
        // Coalesced memory access pattern
        spokeStencil[i][J  ][0][threadWarpID] = sHaloBuffer[i+di][J+dj  ][dk  ][threadWarpID]; // jj = 0; k = 0
        spokeStencil[i][J  ][4][threadWarpID] = sHaloBuffer[i+di][J+dj  ][dk+4][threadWarpID]; // jj = 0; k = 4
        spokeStencil[i][J+1][0][threadWarpID] = sHaloBuffer[i+di][J+dj+1][dk  ][threadWarpID]; // jj = 1; k = 0
        spokeStencil[i][J+1][4][threadWarpID] = sHaloBuffer[i+di][J+dj+1][dk+4][threadWarpID]; // jj = 1; k = 4
#else
        // Reference version
        for (int inDim = 0; inDim < Di; inDim++)
            spokeStencil[i][J+jj][k][inDim] = mHaloBuffer[i+di][J+dj+jj][dk+k][inDim];
        __syncthreads();
#endif

        // Declare the fragments
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;

        MatrixAType matrixA = reinterpret_cast<MatrixAType>(spokeStencil[i][J][0][0]);
        MatrixBType matrixB = reinterpret_cast<MatrixBType>(stencil[di][dj][dk][mIdx][nIdx][0][0]);

        nvcuda::wmma::load_matrix_sync(a_frag, &matrixA[0][0], K);
        nvcuda::wmma::load_matrix_sync(b_frag, &matrixB[0][0], N); // b is row-major, hence the stride

#if 1
#pragma unroll
        for (int t = 0; t < a_frag.num_elements; t++)
            a_frag.x[t] =  nvcuda::wmma::__float_to_tf32(a_frag.x[t]);

#pragma unroll
        for (int t = 0; t < b_frag.num_elements; t++)
            b_frag.x[t] =  nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
#endif

        // Perform the matrix multiplication using TensorCore (tf32) and accumulation
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }

    DenseOutputBufferType sOutputBuffer = reinterpret_cast<DenseOutputBufferType>(sBufferRaw[0]);
    MatrixCType matrixC = reinterpret_cast<MatrixCType>(sOutputBuffer[i][J][0][0]);
    __syncthreads();

    nvcuda::wmma::store_matrix_sync(&matrixC[0][0], c_frag, N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // Sparse commit to 8x8x8 output.
    if (threadIdx.x < 512) {
        int ti = (threadIdx.x >> 6 ) & 0x7;
        int tj = (threadIdx.x >> 3 ) & 0x7;
        int tk = threadIdx.x & 0x7;

        auto coord = origin.offsetBy(ti, tj, tk);

        if (deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
            auto offset = deviceGridAcc.getValue(coord) - 1 + baseOffset;
            for (int s = 0; s < Do; s++) {
                int tDim = s + nIdx * Do;
                if (tDim < outFeatures.size(1))
                    gpuAtomicAddNoReturn(&outFeatures[offset][tDim], sOutputBuffer[ti][tj][tk][s]);
                    // outFeatures[offset][tDim] += sOutputBuffer[ti][tj][tk][s];
            }
        }
    }

#endif // __CUDA_ARCH__ >= 800

}

template <typename GridType>
__global__ __launch_bounds__(256)
void stencilConvHaloLargeDepthKernel(int kM, int kN, int numLeaves,
                                     BatchGridAccessor<GridType> gridAcc,
                                     TorchRAcc32<float, 2> inFeatures,
                                     TorchRAcc32<float, 7> stencil,
                                     TorchRAcc32<float, 2> outFeatures) {

// While 700 (Volta) already supports TensorCore, it does not support TF32.
// 800 (Ampere) supports both TensorCore and TF32.
#if __CUDA_ARCH__ >= 800

    const int tid = threadIdx.x;
    const int Bk = (blockIdx.x & 0x1) << 2;
    const int Bj = ((blockIdx.x >> 1) & 0x3) << 1;
    const int Bi = ((blockIdx.x >> 3) & 0x3) << 1;
    const int restIdx = blockIdx.x >> 5;
    const int leafIdx = restIdx % numLeaves;
    const int nIdx = (restIdx / numLeaves) % kN;
    const int mIdx = restIdx / numLeaves / kN;

    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    // Constants: Input and output dimension multiples
    static constexpr int Di = 64;
    static constexpr int Do = 128;

    const int64_t batchIdx = gridAcc.leafBatchIndex(leafIdx);
    const int64_t localLeafIdx = leafIdx - gridAcc.leafOffset(batchIdx);
    const int64_t baseOffset = gridAcc.voxelOffset(batchIdx);

    const nanovdb::NanoGrid<GridType>* deviceGrid = gridAcc.grid(batchIdx);
    const LeafNodeType& leaf = deviceGrid->tree().template getFirstNode<0>()[localLeafIdx];
    const nanovdb::Coord origin = leaf.origin();
    auto deviceGridAcc = deviceGrid->getAccessor();

    // Check if the current brick is active (this will ignore disabling status)
    const auto& valueMask = leaf.valueMask();
    uint64_t activeMask = valueMask.words()[Bi] | valueMask.words()[Bi + 1];    // 8x8 slice
    activeMask &= (0xffffUL << (Bj << 3));
    activeMask &= (0xf0f0f0f0f0f0f0fUL << Bk);
    if (!activeMask)
        return;

    // Shared memory buffers
    __shared__ float sHaloBuffer[4][4][6][Di];
    __shared__ float sSpokeStencil[2][2][4][Di];
    __shared__ float sOutputBuffer[2][2][4][Do];

    // Gathering data from input features (collectively)
    const int II = (tid >> 6) & 0x3;    // First 2 bits
    const int E = tid & 0x3f;           // Last 6 bits = 64 input channels
    int tDim = E + mIdx * Di;
    for (int jj = 0; jj < 4; ++jj) {
        for (int kk = 0; kk < 6; ++kk) {
            auto coord = origin.offsetBy(Bi + II - 1, Bj + jj - 1, Bk + kk - 1);
            if (tDim < inFeatures.size(1) && deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
                auto offset = deviceGridAcc.getValue(coord) - 1 + baseOffset;
                sHaloBuffer[II][jj][kk][E] = inFeatures[offset][tDim];
            }
            else {
                sHaloBuffer[II][jj][kk][E] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Preparation of GEMM
    using a_frag_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8,
                        nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8,
                        nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // For all kernel positions, add result to C.
    //  A [16 x (8x8)] x B [(8x8) x (8x16)] = C [16 x (8x16)]
    // Block wise: [1 x 8] x [8 (in-block) x 8 (out-block)] = [1 x 8]

    // A list of each 'block' of matrix A (8 x 16 x 8 = 8 x 32 x 4)
    __shared__ float sFragBuffer[8][32][4];

    for (int di = 0; di <= 2; di++)
    for (int dj = 0; dj <= 2; dj++)
    for (int dk = 0; dk <= 2; dk++)
    {
        // Copy data from Halo to the spoke stencil (using 2x2x4x64/256=4 iterations)
        for (int b = 0; b < 2 * 2 * 4 * Di; b += 256) {
            int eid = b + tid;
            int ii = (eid >> 9) & 0x1;
            int jj = (eid >> 8) & 0x1;
            int kk = (eid >> 6) & 0x3;
            int ee = eid & 0x3f;
            sSpokeStencil[ii][jj][kk][ee] = sHaloBuffer[di + ii][dj + jj][dk + kk][ee];
        }
        __syncthreads();

        // Build all 8 blocks of matrix A from spoke -- each block is assigned to each warp.
        //  TODO: Get rid of sSpokeStencil and read from sHaloBuffer to sFragBuffer directly??
        {
            int inBlockIdx = tid >> 5;

            a_frag_t& a_frag = *reinterpret_cast<a_frag_t*>(sFragBuffer[inBlockIdx][tid & 0x1f]);
            nvcuda::wmma::load_matrix_sync(a_frag, &sSpokeStencil[0][0][0][inBlockIdx << 3], 64);
        #pragma unroll
            for (int t = 0; t < a_frag.num_elements; t++)
                a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
            __syncthreads();
        }

        // For each warp (inBlock), perform 8 times of GEMM to obtain the corresponding outBlocks
        for (int b = 0; b < 8 * 8 * 32; b += 256) {
            int eid = tid + b;
            int eWarpIdx = eid >> 5;
            int inBlockIdx = (eWarpIdx >> 3) & 0x7;
            int outBlockIdx = eWarpIdx & 0x7;

            a_frag_t& a_frag = *reinterpret_cast<a_frag_t*>(sFragBuffer[inBlockIdx][tid & 0x1f]);
            nvcuda::wmma::load_matrix_sync(b_frag, &stencil[di][dj][dk][mIdx][nIdx][inBlockIdx << 3][outBlockIdx << 4], 128);
        #pragma unroll
            for (int t = 0; t < b_frag.num_elements; t++)
                b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
            __syncthreads();

            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    // Store the result to the output buffer
    nvcuda::wmma::store_matrix_sync(
        &sOutputBuffer[0][0][0][(tid >> 5) << 4], c_frag, 128, nvcuda::wmma::mem_row_major);

    // Sparse commit
    __syncthreads();
    const int warpI     = (tid >> 8) & 0x1;
    const int warpJ     = (tid >> 7) & 0x1;
    const int warpK     = (tid >> 5) & 0x3;
    const int laneID    = tid & 0x1f;
    const int elementID = laneID | ((tid & 0x200) >> 4);

    // NB: this if fixed for 256 threads/block
    const int elementsPerSM = 32;
    const int xSpanPerSM = 1;
    const int ySpanPerSM = 2;

#pragma unroll
    for (int xOffset = 0; xOffset < 2; xOffset += xSpanPerSM)
    for (int yOffset = 0; yOffset < 2; yOffset += ySpanPerSM) {
        const auto coord = origin.offsetBy(Bi + warpI + xOffset, Bj + warpJ + yOffset, Bk + warpK);

        if (deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
            auto offset = deviceGridAcc.getValue(coord) - 1 + baseOffset;

    #pragma unroll
            for (int elementOffset = 0; elementOffset < Do; elementOffset += elementsPerSM) {
                int s = elementID + elementOffset;
                int tDim = s + nIdx * Do;
                if (tDim < outFeatures.size(1))
                    // gpuAtomicAddNoReturn(&outFeatures[offset][tDim], sOutputBuffer[warpI + xOffset][warpJ + yOffset][warpK][s]);
                    outFeatures[offset][tDim] = sOutputBuffer[warpI + xOffset][warpJ + yOffset][warpK][s];
            }
        }
    }

#endif // __CUDA_ARCH__ >= 800
}

template <>
torch::Tensor dispatchSparseConvolutionHalo<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                          const torch::Tensor& inFeatures,
                                                          const torch::Tensor& kernel,
                                                          int variant) {

    // Check compute capability
    {
        int device_id = inFeatures.device().index();
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        int computeCapability = deviceProp.major * 100 + deviceProp.minor * 10;
        TORCH_CHECK(computeCapability >= 800, "SparseConvolutionHalo requires Ampere (compute capability >= 800)!");
    }

    const auto numLeaves = batchHdl.totalLeaves();

    // Constants: I/O dimension multiples
    const int Di = (variant == 8) ? 8 : 64;
    const int Do = (variant == 8) ? 16 : 128;

    // Output features
    const int outC = kernel.size(4), inC = kernel.size(3);
    auto outFeatures = torch::zeros({inFeatures.size(0), outC}, inFeatures.options());

    // Pad kernel: [3, 3, 3, I, O] -> [3, 3, 3, MxDi, NxDo] -> [3, 3, 3, M, N, Di, Do]
    const int M = (inC + Di - 1) / Di;
    const int N = (outC + Do - 1) / Do;

    torch::Tensor paddedKernel = kernel;
    if (M * Di != inC || N * Do != outC) {
        paddedKernel = torch::zeros({3, 3, 3, M * Di, N * Do}, kernel.options());
        paddedKernel.slice(3, 0, inC).slice(4, 0, outC) = kernel;
    }
    paddedKernel = paddedKernel.view({3, 3, 3, M, Di, N, Do});
    paddedKernel = paddedKernel.permute({0, 1, 2, 3, 5, 4, 6}).contiguous();

    // Launch kernels for each M x N x leaf.
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        auto gridAccessor = batchHdl.deviceAccessor<GridType>();

        if (variant == 8) {
            stencilConvHaloKernel<<<M * N * numLeaves, 1024>>>(
                M, N, numLeaves, gridAccessor,
                inFeatures.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                paddedKernel.packed_accessor32<float, 7, torch::RestrictPtrTraits>(),
                outFeatures.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
            );
        } else {
            stencilConvHaloLargeDepthKernel<<<M * N * numLeaves * 32, 256>>>(
                M, N, numLeaves, gridAccessor,
                inFeatures.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                paddedKernel.packed_accessor32<float, 7, torch::RestrictPtrTraits>(),
                outFeatures.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
            );
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return outFeatures;

}


template <>
torch::Tensor dispatchSparseConvolutionHalo<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                         const torch::Tensor& inFeatures,
                                                         const torch::Tensor& kernel,
                                                         int variant) {
    TORCH_CHECK(false, "CPU not supported for SparseConvolutionHalo yet!");
}


} // namespace ops
} // namespace detail
} // namespace fvdb

