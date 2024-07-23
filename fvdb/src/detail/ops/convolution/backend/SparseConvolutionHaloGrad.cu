// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>
#include <mma.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/convolution/backend/ConvOps.h"

#define COALESCED_MEMORY_ACCESS_VARIANT

namespace fvdb {
namespace detail {
namespace ops {

struct GradStencilFunctor {
    static constexpr int Di = 8;
    static constexpr int logDi = 3;
    static constexpr int Do = 16;

    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int nWarps = MaxThreadsPerBlock / 32;

    struct SharedStorage {
        float inputHaloBuffer[10][10][10][Di];
        float inputSpokeBuffer[4][8][8][2][Di];
        float gradOutputLeafBuffer[8][8][8][Do];
        float gradStencil[2][Di][Do];   // For 2 spokes at a time
        float warpMatrixC[nWarps][16][16];
    };

    template <typename GridType>
    __device__ void operator()(int kM, int kN, int numLeaves,
                             BatchGridAccessor<GridType> gridAcc,
                             TorchRAcc64<float, 2> inFeatures,
                             TorchRAcc64<float, 2> gradOutFeatures,
                             TorchRAcc64<float, 5> gradStencil,
                             char* smemBuffer) {
// While 700 (Volta) already supports TensorCore, it does not support TF32.
// 800 (Ampere) supports both TensorCore and TF32.
#if __CUDA_ARCH__ >= 800

    using MatrixAType = float (&)[8][16];
    using MatrixBType = float (&)[8][16];
    using MatrixCType = float (&)[16][16];
    using MatrixCAltType = float (&)[2][8][16];

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smemBuffer);
    int tid = threadIdx.x;
    int leafIdx = blockIdx.x % numLeaves;
    int nIdx = (blockIdx.x / numLeaves) % kN;
    int mIdx = blockIdx.x / numLeaves / kN;

    // Shared memory buffer
    auto& sInputHaloBuffer = storage.inputHaloBuffer;
    auto& sInputSpokeBuffer = storage.inputSpokeBuffer;
    auto& sGradOutputLeafBuffer = storage.gradOutputLeafBuffer;
    auto& sGradStencil = storage.gradStencil;
    auto sWarpMatrixC = storage.warpMatrixC;

    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    const int64_t batchIdx = gridAcc.leafBatchIndex(leafIdx);
    const int64_t localLeafIdx = leafIdx - gridAcc.leafOffset(batchIdx);
    const int64_t baseOffset = gridAcc.voxelOffset(batchIdx);

    const nanovdb::NanoGrid<GridType>* deviceGrid = gridAcc.grid(batchIdx);
    const LeafNodeType& leaf = deviceGrid->tree().template getFirstNode<0>()[localLeafIdx];
    const nanovdb::Coord origin = leaf.origin();
    auto deviceGridAcc = deviceGrid->getAccessor();

    // Dense gathering of 10x10x10 input features and 8x8x8 grad output features
    // We don't have 1000 threads so have to iterate a bit.
    for (int b = 0; b < 1000; b += blockDim.x) {
        int idx = tid + b;
        if (idx < 1000) {
            int di = ((idx/100) % 10) - 1;
            int dj = ((idx/10) % 10) - 1;
            int dk = (idx % 10) - 1;

            auto coord = origin.offsetBy(di,dj,dk);
            bool inLeaf = (di >= 0 && di < 8 && dj >= 0 && dj < 8 && dk >= 0 && dk < 8);

            if (deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
                auto offset = deviceGridAcc.getValue(coord) - 1 + baseOffset;
                for (int s = 0; s < Di; s++) {
                    int tDim = s + mIdx * Di;
                    sInputHaloBuffer[di+1][dj+1][dk+1][s] = tDim < inFeatures.size(1) ? inFeatures[offset][tDim] : 0.0f;
                }
                for (int s = 0; (s < Do) && inLeaf; s++) {
                    int tDim = s + nIdx * Do;
                    sGradOutputLeafBuffer[di][dj][dk][s] = tDim < gradOutFeatures.size(1) ? gradOutFeatures[offset][tDim] : 0.0f;
                }
            }
            else {
                for (int s = 0; s < Di; s++)
                    sInputHaloBuffer[di+1][dj+1][dk+1][s] = 0.0f;
                for (int s = 0; (s < Do) && inLeaf; s++)
                    sGradOutputLeafBuffer[di][dj][dk][s] = 0.0f;
            }
        }
    }

    __syncthreads();

    int outDim = tid & 0xf;
    int inDim = (tid >> 4) & 0x7;
    int warpId = threadIdx.x >> 5;
    int threadWarpID = threadIdx.x & 0x1f;
    int spokeHalf = (tid >> 7) & 0x1;

    int fullInDim = inDim + mIdx * Di;
    int fullOutDim = outDim + nIdx * Do;
    int fullWithinDim = fullInDim < inFeatures.size(1) && fullOutDim < gradOutFeatures.size(1);

    for (int spokeId = 0; spokeId < 27; spokeId += 2) {
        sGradStencil[0][0][tid] = 0.;
        __syncthreads();

        int di = spokeId / 9 - 1;
        int dj = ((spokeId / 3) % 3) - 1;
        int dk = (spokeId % 3) - 1;
        int diNext = (((spokeId+1) / 9) % 3) - 1;
        int djNext = (((spokeId+1) / 3) % 3) - 1;
        int dkNext = ((spokeId+1) % 3) - 1;

        // iterate through the 64 1x1x8 sticks in a leaf,
        // using as many passes necessary with the available threads/warps
        for (int stickOffset = 0; stickOffset < 64; stickOffset += nWarps) {
            int sid = stickOffset + warpId;
            int i = sid >> 3;
            int I = i & 0x3;
            int j = sid & 0x7;
            // TODO: This technically should work both for Di <= 16 (and power of two)
            // but has only been tested for Di = 8. Must test in other cases
            for (int elementOffset = 0; elementOffset < 8*Di; elementOffset += 32) {
                int z = (threadWarpID + elementOffset) >> logDi;
                int e = (threadWarpID + elementOffset) & (Di-1);
                sInputSpokeBuffer[I][j][z][0][e] =
                    sInputHaloBuffer[i+di+1][j+dj+1][dk+1][threadWarpID+elementOffset];
                sInputSpokeBuffer[I][j][z][1][e] =
                    sInputHaloBuffer[i+diNext+1][j+djNext+1][dkNext+1][threadWarpID+elementOffset];
            }

            MatrixAType matrixA = reinterpret_cast<MatrixAType>(sInputSpokeBuffer[I][j][0][0][0]);
            MatrixBType matrixB = reinterpret_cast<MatrixBType>(sGradOutputLeafBuffer[i][j][0][0]);
            MatrixCType matrixC = reinterpret_cast<MatrixCType>(sWarpMatrixC[warpId][0][0]);

            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag;
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

            // Declare the fragments
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;

            nvcuda::wmma::load_matrix_sync(a_frag, &matrixA[0][0], 16);
            nvcuda::wmma::load_matrix_sync(b_frag, &matrixB[0][0], 16);

#if 1
#pragma unroll
            for (int t = 0; t < a_frag.num_elements; t++)
                a_frag.x[t] =  nvcuda::wmma::__float_to_tf32(a_frag.x[t]);

#pragma unroll
            for (int t = 0; t < b_frag.num_elements; t++)
                b_frag.x[t] =  nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
#endif

            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            nvcuda::wmma::store_matrix_sync(&matrixC[0][0], c_frag, 16, nvcuda::wmma::mem_row_major);

            __syncthreads();

            for (int w = 0; w < nWarps; w++) {
                MatrixCAltType matrixCalt = reinterpret_cast<MatrixCAltType>(sWarpMatrixC[w][0][0]);
                sGradStencil[0][0][tid] += matrixCalt[0][0][tid];
            }
        }

        __syncthreads();

        // Do 2 spokes at a time
        if (spokeId < 26 && fullWithinDim)
            gpuAtomicAddNoReturn(
                &gradStencil[di+1][dj+1][dk+1+spokeHalf][fullInDim][fullOutDim],
                sGradStencil[0][0][tid]
            );

    }

    // Just the last (single) spoke remaining
    if (tid < 128 && fullWithinDim)
        gpuAtomicAddNoReturn(&gradStencil[2][2][2][fullInDim][fullOutDim], sGradStencil[0][inDim][outDim]);

    __syncthreads();

#endif // __CUDA_ARCH__ >= 800

    } // operator()

};

template <typename GridType>
__global__ __launch_bounds__(GradStencilFunctor::MaxThreadsPerBlock)
void stencilConvHaloGradKernel(int kM, int kN, int numLeaves,
                           BatchGridAccessor<GridType> gridAcc,
                           TorchRAcc64<float, 2> inFeatures,
                           TorchRAcc64<float, 2> gradOutFeatures,
                           TorchRAcc64<float, 5> gradStencil) {
    extern __shared__ char smemBuffer[];
    GradStencilFunctor()(kM, kN, numLeaves, gridAcc, inFeatures, gradOutFeatures, gradStencil, smemBuffer);
}

template <>
torch::Tensor dispatchSparseConvolutionHaloGrad<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                              const torch::Tensor& inFeatures,
                                                              const torch::Tensor& gradOutFeatures) {

    // Check compute capability
    {
        int device_id = inFeatures.device().index();
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        int computeCapability = deviceProp.major * 100 + deviceProp.minor * 10;
        TORCH_CHECK(computeCapability >= 800, "SparseConvolutionHalo requires Ampere (compute capability >= 800)!");
    }

    const auto numLeaves = batchHdl.totalLeaves();

    // Kernel Grad size: [3, 3, 3, I, O]
    const int outC = gradOutFeatures.size(1), inC = inFeatures.size(1);
    auto gradStencil = torch::zeros({3, 3, 3, inC, outC}, inFeatures.options());
    const int M = (inC + GradStencilFunctor::Di - 1) / GradStencilFunctor::Di;
    const int N = (outC + GradStencilFunctor::Do - 1) / GradStencilFunctor::Do;

    constexpr size_t smemSize = sizeof(typename GradStencilFunctor::SharedStorage);

    // Launch kernels for each M x N x leaf.
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        auto gridAccessor = batchHdl.deviceAccessor<GridType>();
        cudaFuncSetAttribute(
            stencilConvHaloGradKernel<GridType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemSize);

        stencilConvHaloGradKernel<GridType><<<M * N * numLeaves, GradStencilFunctor::MaxThreadsPerBlock, smemSize>>>(
            M, N, numLeaves, gridAccessor,
            inFeatures.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            gradOutFeatures.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            gradStencil.packed_accessor64<float, 5, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return gradStencil;
}


template <>
torch::Tensor dispatchSparseConvolutionHaloGrad<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                             const torch::Tensor& inFeatures,
                                                             const torch::Tensor& gradOutFeatures) {
    TORCH_CHECK(false, "CPU not supported for SparseConvolutionHalo yet!");
}


} // namespace ops
} // namespace detail
} // namespace fvdb

