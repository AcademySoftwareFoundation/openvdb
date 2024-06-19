#pragma once

#include "GridBatch.h"


namespace fvdb {

enum ConvPackBackend {
    GATHER_SCATTER,
    IGEMM,
    CUTLASS
};

class SparseConvPackInfo : torch::CustomClassHolder {

    // #IO: Number of input-output pairs
    // #O-P: Number of output voxels, padded to multiple of 128
    // #I-P: Number of input voxels, padded to multiple of 128
    // K: Kernel volume count
    // S: Split count

    bool mGSUseME = false;
    torch::optional<torch::Tensor> mGSNeighborMap;          // [#IO, 2] (int32), GATHER_SCATTER, GATHER_SCATTER(me)
    torch::optional<torch::Tensor> mGSNeighborSizes;        // [#IO, 2] (int32), GATHER_SCATTER, GATHER_SCATTER(me)

    bool mIGEMMUseTF32 = false;
    torch::optional<torch::Tensor> mIGEMMOutInMap;          // [#O-P, K] (int32), IGEMM, IGEMM(sorted)
    torch::optional<torch::Tensor> mIGEMMReorderLoc;        // [S, #O-P] (int32), IGEMM(sorted)
    torch::optional<torch::Tensor> mIGEMMSortedMask;        // [S, #O-P] (int32), IGEMM(sorted)
    torch::optional<torch::Tensor> mIGEMMReducedSortedMask; // [S, #O-P//128] (int32), IGEMM(sorted)
    torch::optional<torch::Tensor> mIGEMMReoderOutInMap;    // [#O-P, K] (int32), IGEMM(sorted)

    torch::optional<torch::Tensor> mIGEMMOutInMapBwd;       // [#I-P, K] (int32), IGEMM, IGEMM(sorted, training)
    torch::optional<torch::Tensor> mIGEMMReorderLocBwd;     // [S, #I-P] (int32), IGEMM, IGEMM(sorted, training)
    torch::optional<torch::Tensor> mIGEMMSortedMaskBwdW;    // [S, #I-P//x] (int32), IGEMM, IGEMM(sorted, training)
    torch::optional<torch::Tensor> mIGEMMSortedMaskBwdD;    // [S, #I-P//y] (int32), IGEMM, IGEMM(sorted, training)
    torch::optional<torch::Tensor> mIGEMMReorderOutInMapBwd; // [#I-P, K] (int32), IGEMM, IGEMM(sorted, training)

    bool mCUTLASSBenchmark = false;
    torch::optional<torch::Tensor> mCUTLASSHaloIndexBuffer;   // [#active_brick, 6, 4, 4] (int32), CUTLASS
    torch::optional<torch::Tensor> mCUTLASSOutputIndexBuffer; // [#active_brick, 4, 2, 2] (int32), CUTLASS

    Vec3iOrScalar mStride;
    Vec3iOrScalar mKernelSize;

    GridBatch mSourceGrid;
    GridBatch mTargetGrid;

public:
    const torch::optional<torch::Tensor> neighborMap() const { return mGSNeighborMap; }
    const torch::optional<torch::Tensor> neighborSizes() const { return mGSNeighborSizes; }
    const bool useME() const { return mGSUseME; }

    const torch::optional<torch::Tensor> outInMap() const { return mIGEMMOutInMap; }
    const torch::optional<torch::Tensor> reorderLoc() const { return mIGEMMReorderLoc; }
    const torch::optional<torch::Tensor> sortedMask() const { return mIGEMMSortedMask; }
    const torch::optional<torch::Tensor> reducedSortedMask() const { return mIGEMMReducedSortedMask; }
    const torch::optional<torch::Tensor> reoderOutInMap() const { return mIGEMMReoderOutInMap; }
    const bool useTF32() const { return mIGEMMUseTF32; }

    const torch::optional<torch::Tensor> outInMapBwd() const { return mIGEMMOutInMapBwd; }
    const torch::optional<torch::Tensor> reorderLocBwd() const { return mIGEMMReorderLocBwd; }
    const torch::optional<torch::Tensor> sortedMaskBwdW() const { return mIGEMMSortedMaskBwdW; }
    const torch::optional<torch::Tensor> sortedMaskBwdD() const { return mIGEMMSortedMaskBwdD; }
    const torch::optional<torch::Tensor> reorderOutInMapBwd() const { return mIGEMMReorderOutInMapBwd; }

    const torch::optional<torch::Tensor> haloIndexBuffer() const { return mCUTLASSHaloIndexBuffer; }
    const torch::optional<torch::Tensor> outputIndexBuffer() const { return mCUTLASSOutputIndexBuffer; }
    const bool benchmark() const { return mCUTLASSBenchmark; }

    const Vec3iOrScalar stride() const { return mStride; }
    const Vec3iOrScalar kernelSize() const { return mKernelSize; }

    GridBatch targetGrid() const { return mTargetGrid; }
    GridBatch sourceGrid() const { return mSourceGrid; }

    SparseConvPackInfo(Vec3iOrScalar kernelsize, Vec3iOrScalar stride, GridBatch src,
                       torch::optional<GridBatch> maybeTarget);

    SparseConvPackInfo transposed() const;

    // Will not rebuild if already built
    void buildGatherScatter(bool use_me = false);
    void buildImplicitGEMM(bool sorted, int splitMaskNum, bool training, int splitMaskNumBwd, bool use_tf32 = false);
    void buildCutlass(bool benchmark = false);

    JaggedTensor sparseConv3d(const JaggedTensor& input, const torch::Tensor& weights, ConvPackBackend backend = ConvPackBackend::GATHER_SCATTER) const;
    JaggedTensor sparseTransposeConv3d(const JaggedTensor& input, const torch::Tensor& weights, ConvPackBackend backend = ConvPackBackend::GATHER_SCATTER) const;
};


}
