// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "ConvOps.h"

#include <detail/utils/AccessorHelpers.cuh>

#include <c10/cuda/CUDAException.h>

#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

namespace fvdb {
namespace detail {
namespace ops {

using namespace cute;

struct ConvolutionFunctorV1 {
    static constexpr int Di                 = 128; // input channels (C)
    static constexpr int Do                 = 128; // ouptut channels (K)
    static constexpr int maxIndicesPerBlock = 64;  // (linear NZPQ)

    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    struct SharedStorage {
        float inputMatrix[maxIndicesPerBlock][Di];
        float outputMatrixSpoke[maxIndicesPerBlock][Do];
        float outputMatrixBlock[maxIndicesPerBlock][Do];
    };

    using InputMatrixType  = float (&)[][Di];
    using OutputMatrixType = float (&)[][Do];

    using SmemLayoutAtom = Layout<Shape<Int<maxIndicesPerBlock>, Int<Di>>, Stride<Int<Di>, _1>>;
    using SmemCopyAtom   = Copy_Atom<DefaultCopy, tfloat32_t>;

    void __device__
    operator()(float *deviceInputTensor,
               float *deviceOutputTensor,
               float *deviceStencil,
               int numVoxels,
               int32_t *deviceSpokeIndicesFlattenedOffset,
               int32_t *deviceSpokeInputGlobalIndicesFlattenedData,
               int32_t *deviceSpokeOutputLocalOffsetsRelativeToBlockFlattenedData,
               char *smem_buf) {
        int b                   = blockIdx.x;
        int tid                 = threadIdx.x;
        SharedStorage &storage  = *reinterpret_cast<SharedStorage *>(smem_buf);
        auto &inputMatrix       = storage.inputMatrix;
        auto &outputMatrixSpoke = storage.outputMatrixSpoke;
        auto &outputMatrixBlock = storage.outputMatrixBlock;

        Tensor deviceSpokeIndicesFlattenedOffsetTensor =
            make_tensor(make_gmem_ptr(deviceSpokeIndicesFlattenedOffset),
                        make_layout(make_shape(gridDim.x, make_shape(_3{}, _3{}, _3{})),
                                    make_stride(Int<27>{}, make_shape(_1{}, _3{}, _9{}))));
        Tensor localDeviceSpokeIndicesFlattenedOffsetTensor =
            deviceSpokeIndicesFlattenedOffsetTensor(blockIdx.x, _);

        using StencilLayout   = decltype(make_ordered_layout(
            Shape<Shape<_3, _3, _3>, Int<Di>, Int<Do>>{}, tuple<tuple<_2, _3, _4>, _0, _1>{}));
        Tensor stencil_tensor = make_tensor(make_gmem_ptr(deviceStencil), StencilLayout{});

        InputMatrixType inputTensor   = reinterpret_cast<InputMatrixType>(*deviceInputTensor);
        OutputMatrixType outputTensor = reinterpret_cast<OutputMatrixType>(*deviceOutputTensor);

        for (int e = 0; e < maxIndicesPerBlock; e++) {
            outputMatrixBlock[e][tid] = 0.;
        }

        __syncthreads();

// for every spoke
#pragma nounroll
        for (int k_tile_iter = 0; k_tile_iter < size<0>(stencil_tensor); ++k_tile_iter) {
            auto stencil_slice = stencil_tensor(k_tile_iter, _, _);
            const auto spokeGlobalIndicesBegin =
                localDeviceSpokeIndicesFlattenedOffsetTensor(k_tile_iter);
            const auto spokeIndicesCount =
                localDeviceSpokeIndicesFlattenedOffsetTensor(k_tile_iter + 1) -
                spokeGlobalIndicesBegin;
            const auto spokeInputGlobalIndices =
                &deviceSpokeInputGlobalIndicesFlattenedData[spokeGlobalIndicesBegin];
            const auto spokeOutputLocalOffsetsRelativeToBlock =
                &deviceSpokeOutputLocalOffsetsRelativeToBlockFlattenedData[spokeGlobalIndicesBegin];

            // if there is no work to be done in this spoke, skip this filter application
            if (spokeIndicesCount == 0) {
                continue;
            }

            Tensor sA = make_tensor(make_smem_ptr(&inputMatrix[0][0]),
                                    tile_to_shape(SmemLayoutAtom{}, make_shape(_64{}, _128{})));

            auto tid_k = tid % 32;
            auto tid_m = tid / 32;
            // Iterate 16 times.
            for (int e = tid_m; e < maxIndicesPerBlock; e += 4) {
                auto sA_128b = recast<float4>(sA);
                if (e < spokeIndicesCount) {
                    auto inputIndex = spokeInputGlobalIndices[e];
                    if (inputIndex < 0) { // Won't happen if not locally densified
                        sA_128b(e, tid_k) = float4{0, 0, 0, 0};
                        continue;
                    }
                    Tensor input_tensor_slice = make_tensor(
                        make_gmem_ptr(reinterpret_cast<float4 *>(&inputTensor[inputIndex])),
                        make_layout(make_shape(_32{})));
                    sA_128b(e, tid_k) = input_tensor_slice(tid_k);
                } else {
                    sA_128b(e, tid_k) = float4{0, 0, 0, 0};
                }
            }
            __syncthreads();

            if (spokeIndicesCount <= 16) {
                using _M = _16;  // == maxIndicesPerBlock
                using _N = _128; // == Do
                using _K = _128; // == Di
                using TiledMma =
                    TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>, Layout<Shape<_2, _2, _1>>>;

                Tensor sA = make_tensor(make_smem_ptr(&inputMatrix[0][0]),
                                        make_layout(make_shape(_M{}, _K{}), GenRowMajor{}));
                Tensor sB = stencil_slice;
                Tensor sC = make_tensor(make_smem_ptr(&outputMatrixSpoke[0][0]),
                                        make_layout(make_shape(_M{}, _N{}), GenRowMajor{}));

                TiledMma tiled_mma;
                Tensor accum = partition_fragment_C(tiled_mma, Shape<_M, _N>{});

                auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

                Tensor tCsA = thr_mma.partition_A(sA);
                Tensor tCsB = thr_mma.partition_B(sB);
                Tensor tCrA = thr_mma.partition_fragment_A(sA);
                Tensor tCrB = thr_mma.partition_fragment_B(sB);

                copy(tCsA, tCrA);
                copy(tCsB, tCrB);

                __syncthreads();

                gemm(tiled_mma, accum, tCrA, tCrB, accum);

                Tensor tCsC = thr_mma.partition_C(sC);
                copy(accum, tCsC);
            } else if (spokeIndicesCount > 16 && spokeIndicesCount <= 32) {
                using _M = _32;  // == maxIndicesPerBlock
                using _N = _128; // == Do
                using _K = _128; // == Di
                using TiledMma =
                    TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>, Layout<Shape<_2, _2, _1>>>;

                Tensor sA = make_tensor(make_smem_ptr(&inputMatrix[0][0]),
                                        make_layout(make_shape(_M{}, _K{}), GenRowMajor{}));
                Tensor sB = stencil_slice;
                Tensor sC = make_tensor(make_smem_ptr(&outputMatrixSpoke[0][0]),
                                        make_layout(make_shape(_M{}, _N{}), GenRowMajor{}));

                TiledMma tiled_mma;
                Tensor accum = partition_fragment_C(tiled_mma, Shape<_M, _N>{});

                auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

                Tensor tCsA = thr_mma.partition_A(sA);
                Tensor tCsB = thr_mma.partition_B(sB);
                Tensor tCrA = thr_mma.partition_fragment_A(sA);
                Tensor tCrB = thr_mma.partition_fragment_B(sB);

                copy(tCsA, tCrA);
                copy(tCsB, tCrB);

                __syncthreads();

                gemm(tiled_mma, accum, tCrA, tCrB, accum);

                Tensor tCsC = thr_mma.partition_C(sC);
                copy(accum, tCsC);
            } else if (spokeIndicesCount <= 64) {
                using _M = _64;  // == maxIndicesPerBlock
                using _N = _128; // == Do
                using _K = _128; // == Di
                using TiledMma =
                    TiledMMA<MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>, Layout<Shape<_2, _2, _1>>>;

                Tensor sA = make_tensor(make_smem_ptr(&inputMatrix[0][0]),
                                        make_layout(make_shape(_M{}, _K{}), GenRowMajor{}));
                Tensor sB = stencil_slice;
                Tensor sC = make_tensor(make_smem_ptr(&outputMatrixSpoke[0][0]),
                                        make_layout(make_shape(_M{}, _N{}), GenRowMajor{}));

                TiledMma tiled_mma;
                Tensor accum = partition_fragment_C(tiled_mma, Shape<_M, _N>{});

                auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

                Tensor tCsA = thr_mma.partition_A(sA);
                Tensor tCsB = thr_mma.partition_B(sB);
                Tensor tCrA = thr_mma.partition_fragment_A(sA);
                Tensor tCrB = thr_mma.partition_fragment_B(sB);

                copy(tCsA, tCrA);
                copy(tCsB, tCrB);

                __syncthreads();

                gemm(tiled_mma, accum, tCrA, tCrB, accum);

                Tensor tCsC = thr_mma.partition_C(sC);
                copy(accum, tCsC);
            }

            __syncthreads();

            for (int e = 0; e < spokeIndicesCount; e++) {
                const auto blockOutputIndex = spokeOutputLocalOffsetsRelativeToBlock[e];
                if (blockOutputIndex < 0) { // Won't happen if not locally densified
                    continue;
                }
                outputMatrixBlock[blockOutputIndex][tid] += outputMatrixSpoke[e][tid];
            }
        }

        __syncthreads();

        const auto rStart = b * maxIndicesPerBlock;
        const auto rEnd   = min((b + 1) * maxIndicesPerBlock, numVoxels);
        for (int e = 0; e < (rEnd - rStart); e++) {
            outputTensor[e + rStart][tid] = outputMatrixBlock[e][tid];
        }
    }
};

template <class Operator>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor) void kernel_entrypoint(
    float *deviceInputTensor,
    float *deviceOutputTensor,
    float *deviceStencil,
    int numVoxels,
    int32_t *deviceSpokeIndicesFlattenedOffset,
    int32_t *deviceSpokeInputGlobalIndicesFlattenedData,
    int32_t *deviceSpokeOutputLocalOffsetsRelativeToBlockFlattenedData) {
    extern __shared__ char smem_buf[];
    Operator op;
    op(deviceInputTensor,
       deviceOutputTensor,
       deviceStencil,
       numVoxels,
       deviceSpokeIndicesFlattenedOffset,
       deviceSpokeInputGlobalIndicesFlattenedData,
       deviceSpokeOutputLocalOffsetsRelativeToBlockFlattenedData,
       smem_buf);
}

template <>
torch::Tensor
dispatchSparseConvolutionLggs<torch::kCUDA>(
    const torch::Tensor &inFeatures,
    const torch::Tensor &kernel,
    const torch::Tensor &spokeIndicesFlattenedOffset,
    const torch::Tensor &spokeInputGlobalIndicesFlattenedData,
    const torch::Tensor &spokeOutputLocalOffsetsRelativeToBlockFlattenedData) {
    using Op = ConvolutionFunctorV1;

    // Assuming kernel is reshaped from [Do, Di, D, H, W] to [D*H*W, Di, Do]
    const int inC = kernel.size(1), outC = kernel.size(2);
    TORCH_CHECK(inC == 128 && outC == 128, "ConvolutionFunctorV1 only supports 128x128 kernels");

    torch::Tensor outFeatures = torch::empty({inFeatures.size(0), outC}, inFeatures.options());

    constexpr size_t smem_size = sizeof(typename Op::SharedStorage);
    cudaFuncSetAttribute(
        kernel_entrypoint<Op>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int blockCount = (spokeIndicesFlattenedOffset.size(0) - 1) / 27;
    kernel_entrypoint<Op><<<size_t(blockCount), Op::MaxThreadsPerBlock, smem_size>>>(
        inFeatures.data_ptr<float>(),
        outFeatures.data_ptr<float>(),
        kernel.data_ptr<float>(),
        inFeatures.size(0),
        spokeIndicesFlattenedOffset.data_ptr<int>(),
        spokeInputGlobalIndicesFlattenedData.data_ptr<int>(),
        spokeOutputLocalOffsetsRelativeToBlockFlattenedData.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return outFeatures;
}

template <>
torch::Tensor
dispatchSparseConvolutionLggs<torch::kCPU>(
    const torch::Tensor &inFeatures,
    const torch::Tensor &kernel,
    const torch::Tensor &spokeIndicesFlattenedOffset,
    const torch::Tensor &spokeInputGlobalIndicesFlattenedData,
    const torch::Tensor &spokeOutputLocalOffsetsRelativeToBlockFlattenedData) {
    TORCH_CHECK(false, "SparseConvolutionLggs is not implemented for CPU");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
