// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_CONVOPS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_CONVOPS_H

#include <Types.h>
#include <detail/GridBatchImpl.h>

#include <torch/extension.h>

#ifdef ENABLE_CONV_PRAGMA_MESSAGE
#define PRAGMA_MESSAGE(x) _Pragma(#x)
#else
#define PRAGMA_MESSAGE(x)
#endif

namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType>
void dispatchSparseConvolutionKernelMap(torch::Tensor inFeat,
                                        torch::Tensor outFeat,
                                        torch::Tensor kernel,
                                        torch::Tensor nbMap,
                                        torch::Tensor nbSizes,
                                        bool transpose,
                                        bool middleAcceleration);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionImplicitGEMM(torch::Tensor inFeat,
                                                    torch::Tensor kernel,
                                                    torch::Tensor outInMap,
                                                    int numOutFeats,
                                                    int numOutChannels,
                                                    bool allow_tf32,
                                                    bool allow_fp16);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionImplicitGEMMSorted(torch::Tensor inFeat,
                                                          torch::Tensor kernel,
                                                          torch::Tensor outInMap,
                                                          torch::Tensor reducedMask,
                                                          torch::Tensor reorderLoc,
                                                          int numOutFeats,
                                                          int numOutChannels,
                                                          bool allow_tf32,
                                                          bool allow_fp16);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionImplicitGEMMGrad(torch::Tensor inFeat,
                                                        torch::Tensor kernel,
                                                        torch::Tensor outInMap,
                                                        const int splitKIters,
                                                        bool allow_tf32,
                                                        bool allow_fp16);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionImplicitGEMMGradSorted(torch::Tensor inFeat,
                                                              torch::Tensor kernel,
                                                              torch::Tensor outInMap,
                                                              torch::Tensor reducedMask,
                                                              torch::Tensor reorderLoc,
                                                              const int splitKIters,
                                                              bool allow_tf32,
                                                              bool allow_fp16);

void dispatchMESparseConvolutionKernelMapGrad(at::Tensor in_feat,
                                              at::Tensor grad_in_feat,
                                              at::Tensor grad_out_feat,
                                              at::Tensor kernel,
                                              at::Tensor grad_kernel,
                                              at::Tensor neighbor_map,
                                              at::Tensor neighbor_offset,
                                              const bool transpose);

template <c10::DeviceType>
void dispatchSparseConvolutionKernelMapGrad(torch::Tensor inFeat,
                                            torch::Tensor gradInFeat,
                                            torch::Tensor gradOutFeat,
                                            torch::Tensor kernel,
                                            torch::Tensor gradKernel,
                                            torch::Tensor nbMap,
                                            torch::Tensor nbSizes,
                                            bool transpose);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionHalo(const GridBatchImpl &grid,
                                            const torch::Tensor &inFeat,
                                            const torch::Tensor &kernel,
                                            int variant);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionHaloGrad(const GridBatchImpl &batchHdl,
                                                const torch::Tensor &inFeatures,
                                                const torch::Tensor &gradOutFeatures);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionCutlass(const torch::Tensor &inFeatures,
                                               const torch::Tensor &kernel,
                                               const torch::Tensor &haloIndexBuffer,
                                               const torch::Tensor &outputIndexBuffer,
                                               bool benchmark);

template <c10::DeviceType>
torch::Tensor dispatchSparseConvolutionLggs(
    const torch::Tensor &inFeatures,
    const torch::Tensor &kernel,
    const torch::Tensor &spokeIndicesFlattenedOffset,
    const torch::Tensor &spokeInputGlobalIndicesFlattenedData,
    const torch::Tensor &spokeOutputLocalOffsetsRelativeToBlockFlattenedData);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_BACKEND_CONVOPS_H
