// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_SPARSECONVPACKINFO_H
#define FVDB_SPARSECONVPACKINFO_H

#include "GridBatch.h"

namespace fvdb {

enum ConvPackBackend {
    GATHER_SCATTER,
    IGEMM,
    CUTLASS,
    LGGS,
};

class SparseConvPackInfo : torch::CustomClassHolder {
    // #IO: Number of input-output pairs
    // #O-P: Number of output voxels, padded to multiple of 128
    // #I-P: Number of input voxels, padded to multiple of 128
    // K: Kernel volume count
    // S: Split count

    bool mGSUseME = false;
    std::optional<torch::Tensor>
        mGSNeighborMap;   // [#IO, 2] (int32), GATHER_SCATTER, GATHER_SCATTER(me)
    std::optional<torch::Tensor>
        mGSNeighborSizes; // [#IO, 2] (int32), GATHER_SCATTER, GATHER_SCATTER(me)

    bool mIGEMMUseTF32 = false;
    std::optional<torch::Tensor> mIGEMMOutInMap;          // [#O-P, K] (int32), IGEMM, IGEMM(sorted)
    std::optional<torch::Tensor> mIGEMMReorderLoc;        // [S, #O-P] (int32), IGEMM(sorted)
    std::optional<torch::Tensor> mIGEMMSortedMask;        // [S, #O-P] (int32), IGEMM(sorted)
    std::optional<torch::Tensor> mIGEMMReducedSortedMask; // [S, #O-P//128] (int32), IGEMM(sorted)
    std::optional<torch::Tensor> mIGEMMReoderOutInMap;    // [#O-P, K] (int32), IGEMM(sorted)

    std::optional<torch::Tensor>
        mIGEMMOutInMapBwd;        // [#I-P, K] (int32), IGEMM, IGEMM(sorted, training)
    std::optional<torch::Tensor>
        mIGEMMReorderLocBwd;      // [S, #I-P] (int32), IGEMM, IGEMM(sorted, training)
    std::optional<torch::Tensor>
        mIGEMMSortedMaskBwdW;     // [S, #I-P//x] (int32), IGEMM, IGEMM(sorted, training)
    std::optional<torch::Tensor>
        mIGEMMSortedMaskBwdD;     // [S, #I-P//y] (int32), IGEMM, IGEMM(sorted, training)
    std::optional<torch::Tensor>
        mIGEMMReorderOutInMapBwd; // [#I-P, K] (int32), IGEMM, IGEMM(sorted, training)

    bool mCUTLASSBenchmark = false;
    std::optional<torch::Tensor>
        mCUTLASSHaloIndexBuffer;   // [#active_brick, 6, 4, 4] (int32), CUTLASS
    std::optional<torch::Tensor>
        mCUTLASSOutputIndexBuffer; // [#active_brick, 4, 2, 2] (int32), CUTLASS

    std::optional<torch::Tensor> mLGGSSpokeIndicesFlattenedOffset; // 1D array. (int32), LGGS
    std::optional<torch::Tensor>
        mLGGSSpokeInputGlobalIndicesFlattenedData;                 // 1D array. (int32), LGGS
    std::optional<torch::Tensor>
        mLGGSSpokeOutputLocalOffsetsRelativeToBlockFlattenedData;  // 1D array. (int32), LGGS

    Vec3iOrScalar mStride;
    Vec3iOrScalar mKernelSize;

    GridBatch mSourceGrid;
    GridBatch mTargetGrid;

  public:
    SparseConvPackInfo
    to(const torch::Device &device) const {
        SparseConvPackInfo copy = *this;
        if (copy.mGSNeighborMap) {
            copy.mGSNeighborMap = copy.mGSNeighborMap.value().to(device);
        }
        if (copy.mGSNeighborSizes) {
            copy.mGSNeighborSizes = copy.mGSNeighborSizes.value().to(device);
        }
        if (copy.mIGEMMOutInMap) {
            copy.mIGEMMOutInMap = copy.mIGEMMOutInMap.value().to(device);
        }
        if (copy.mIGEMMReorderLoc) {
            copy.mIGEMMReorderLoc = copy.mIGEMMReorderLoc.value().to(device);
        }
        if (copy.mIGEMMSortedMask) {
            copy.mIGEMMSortedMask = copy.mIGEMMSortedMask.value().to(device);
        }
        if (copy.mIGEMMReducedSortedMask) {
            copy.mIGEMMReducedSortedMask = copy.mIGEMMReducedSortedMask.value().to(device);
        }
        if (copy.mIGEMMReoderOutInMap) {
            copy.mIGEMMReoderOutInMap = copy.mIGEMMReoderOutInMap.value().to(device);
        }
        if (copy.mIGEMMOutInMapBwd) {
            copy.mIGEMMOutInMapBwd = copy.mIGEMMOutInMapBwd.value().to(device);
        }
        if (copy.mIGEMMReorderLocBwd) {
            copy.mIGEMMReorderLocBwd = copy.mIGEMMReorderLocBwd.value().to(device);
        }
        if (copy.mIGEMMSortedMaskBwdW) {
            copy.mIGEMMSortedMaskBwdW = copy.mIGEMMSortedMaskBwdW.value().to(device);
        }
        if (copy.mIGEMMSortedMaskBwdD) {
            copy.mIGEMMSortedMaskBwdD = copy.mIGEMMSortedMaskBwdD.value().to(device);
        }
        if (copy.mIGEMMReorderOutInMapBwd) {
            copy.mIGEMMReorderOutInMapBwd = copy.mIGEMMReorderOutInMapBwd.value().to(device);
        }
        if (copy.mCUTLASSHaloIndexBuffer) {
            copy.mCUTLASSHaloIndexBuffer = copy.mCUTLASSHaloIndexBuffer.value().to(device);
        }
        if (copy.mCUTLASSOutputIndexBuffer) {
            copy.mCUTLASSOutputIndexBuffer = copy.mCUTLASSOutputIndexBuffer.value().to(device);
        }
        if (copy.mLGGSSpokeIndicesFlattenedOffset) {
            copy.mLGGSSpokeIndicesFlattenedOffset =
                copy.mLGGSSpokeIndicesFlattenedOffset.value().to(device);
        }
        if (copy.mLGGSSpokeInputGlobalIndicesFlattenedData) {
            copy.mLGGSSpokeInputGlobalIndicesFlattenedData =
                copy.mLGGSSpokeInputGlobalIndicesFlattenedData.value().to(device);
        }
        if (copy.mLGGSSpokeOutputLocalOffsetsRelativeToBlockFlattenedData) {
            copy.mLGGSSpokeOutputLocalOffsetsRelativeToBlockFlattenedData =
                copy.mLGGSSpokeOutputLocalOffsetsRelativeToBlockFlattenedData.value().to(device);
        }
        copy.mSourceGrid = copy.mSourceGrid.to(device);
        copy.mTargetGrid = copy.mTargetGrid.to(device);

        return copy;
    }

    SparseConvPackInfo
    to(const std::string &device_string) const {
        torch::Device device(device_string);
        if (device.is_cuda() && !device.has_index()) {
            device.set_index(c10::cuda::current_device());
        }
        return to(device);
    }

    SparseConvPackInfo
    cuda() const {
        return to(torch::kCUDA);
    }

    SparseConvPackInfo
    cpu() const {
        return to(torch::kCPU);
    }

    const std::optional<torch::Tensor>
    neighborMap() const {
        return mGSNeighborMap;
    }
    const std::optional<torch::Tensor>
    neighborSizes() const {
        return mGSNeighborSizes;
    }
    const bool
    useME() const {
        return mGSUseME;
    }

    const std::optional<torch::Tensor>
    outInMap() const {
        return mIGEMMOutInMap;
    }
    const std::optional<torch::Tensor>
    reorderLoc() const {
        return mIGEMMReorderLoc;
    }
    const std::optional<torch::Tensor>
    sortedMask() const {
        return mIGEMMSortedMask;
    }
    const std::optional<torch::Tensor>
    reducedSortedMask() const {
        return mIGEMMReducedSortedMask;
    }
    const std::optional<torch::Tensor>
    reoderOutInMap() const {
        return mIGEMMReoderOutInMap;
    }
    const bool
    useTF32() const {
        return mIGEMMUseTF32;
    }

    const std::optional<torch::Tensor>
    outInMapBwd() const {
        return mIGEMMOutInMapBwd;
    }
    const std::optional<torch::Tensor>
    reorderLocBwd() const {
        return mIGEMMReorderLocBwd;
    }
    const std::optional<torch::Tensor>
    sortedMaskBwdW() const {
        return mIGEMMSortedMaskBwdW;
    }
    const std::optional<torch::Tensor>
    sortedMaskBwdD() const {
        return mIGEMMSortedMaskBwdD;
    }
    const std::optional<torch::Tensor>
    reorderOutInMapBwd() const {
        return mIGEMMReorderOutInMapBwd;
    }

    const std::optional<torch::Tensor>
    haloIndexBuffer() const {
        return mCUTLASSHaloIndexBuffer;
    }
    const std::optional<torch::Tensor>
    outputIndexBuffer() const {
        return mCUTLASSOutputIndexBuffer;
    }
    const bool
    benchmark() const {
        return mCUTLASSBenchmark;
    }

    const std::optional<torch::Tensor>
    blockKernelRanges() const {
        return mLGGSSpokeIndicesFlattenedOffset;
    }
    const std::optional<torch::Tensor>
    blockKernelInIdx() const {
        return mLGGSSpokeInputGlobalIndicesFlattenedData;
    }
    const std::optional<torch::Tensor>
    blockKernelRelOutIdx() const {
        return mLGGSSpokeOutputLocalOffsetsRelativeToBlockFlattenedData;
    }

    const Vec3iOrScalar
    stride() const {
        return mStride;
    }
    const Vec3iOrScalar
    kernelSize() const {
        return mKernelSize;
    }

    GridBatch
    targetGrid() const {
        return mTargetGrid;
    }
    GridBatch
    sourceGrid() const {
        return mSourceGrid;
    }

    SparseConvPackInfo(Vec3iOrScalar kernelsize,
                       Vec3iOrScalar stride,
                       GridBatch src,
                       std::optional<GridBatch> maybeTarget);

    SparseConvPackInfo transposed() const;

    // Will not rebuild if already built
    void buildGatherScatter(bool use_me = false);
    void buildImplicitGEMM(
        bool sorted, int splitMaskNum, bool training, int splitMaskNumBwd, bool use_tf32 = false);
    void buildCutlass(bool benchmark = false);
    void buildLGGS();

    JaggedTensor sparseConv3d(const JaggedTensor &input,
                              const torch::Tensor &weights,
                              ConvPackBackend backend = ConvPackBackend::GATHER_SCATTER) const;
    JaggedTensor
    sparseTransposeConv3d(const JaggedTensor &input,
                          const torch::Tensor &weights,
                          ConvPackBackend backend = ConvPackBackend::GATHER_SCATTER) const;
};

} // namespace fvdb

#endif // FVDB_SPARSECONVPACKINFO_H
