#include "SparseConvPackInfo.h"

#include "detail/ops/Ops.h"
#include "detail/ops/convolution/pack_info/PackInfoOps.h"
#include "detail/autograd/Autograd.h"


namespace fvdb {

SparseConvPackInfo::SparseConvPackInfo(Vec3iOrScalar kernelsize, Vec3iOrScalar stride, GridBatch srcGrid,
                                       torch::optional<GridBatch> maybeTarget) {
    TORCH_CHECK(Vec3iOrScalar(0).value() < kernelsize.value(), "Expect kernel size to be larger than {0,0,0}, but got " + kernelsize.toString() + ".");
    TORCH_CHECK(Vec3iOrScalar(0).value() < stride.value(), "Expect stride to be larger than 0, but got " + stride.toString() + ".");

    GridBatch targetGrid;
    if (!maybeTarget.has_value()) {
        if (stride.value() == Vec3iOrScalar(1).value()) {
            targetGrid = srcGrid;
        } else {
            targetGrid = srcGrid.conv_grid(kernelsize, stride);
        }
    } else {
        targetGrid = maybeTarget.value();
    }

    TORCH_CHECK(srcGrid.is_mutable() == targetGrid.is_mutable(), "Source and target grids must both be mutable or immutable");
    TORCH_CHECK(srcGrid.device() == targetGrid.device(), "Source and target grids must both be on the same device");
    TORCH_CHECK(srcGrid.device() == targetGrid.device(), "Device should match between this grid and target grid.");
    TORCH_CHECK(!(kernelsize.value() == Vec3iOrScalar(1).value() && stride.value() == Vec3iOrScalar(1).value()), "1x1 conv does not need kernel map to be built!");

    mStride = stride;
    mKernelSize = kernelsize;
    mTargetGrid = targetGrid;
    mSourceGrid = srcGrid;
}

void SparseConvPackInfo::buildGatherScatter(bool use_me) {
    if (mGSNeighborMap.has_value() && mGSNeighborSizes.has_value()) {
        TORCH_CHECK(mGSUseME == use_me, "Gather scatter is already built with different use_me value");
        return;
    }

    int kernelVolume = mKernelSize.value().x() * mKernelSize.value().y() * mKernelSize.value().z();

    torch::Tensor kmap = torch::full(
        {mTargetGrid.total_voxels(), kernelVolume}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(mTargetGrid.device()));

    FVDB_DISPATCH_KERNEL_DEVICE(mSourceGrid.device(), [&]() {
        detail::ops::dispatchConvolutionKernelMap<DeviceTag>(
            *mSourceGrid.impl(), *mTargetGrid.impl(), kmap, mKernelSize, mStride);
    });
    kmap = kmap.t();
    torch::Tensor kmask = kmap != -1;
    torch::Tensor nbsizes = torch::sum(kmask, -1);
    torch::Tensor nbmap = torch::nonzero(kmask).contiguous();

    torch::Tensor indices = nbmap.index({torch::indexing::Slice(), 0}) * kmap.size(1) + \
        nbmap.index({torch::indexing::Slice(), 1});
    nbmap.index_put_({torch::indexing::Slice(), 0}, kmap.reshape({-1}).index({indices}));
    mGSNeighborMap = nbmap.to(torch::kInt32);
    mGSNeighborSizes = nbsizes.to(torch::kInt32);
    mGSUseME = use_me;
}

void SparseConvPackInfo::buildImplicitGEMM(bool sorted, int splitMaskNum, bool training, int splitMaskNumBwd, bool use_tf32) {
    if (mIGEMMOutInMap.has_value()) {
        if (mIGEMMReorderLoc.has_value()) {
            TORCH_CHECK(mIGEMMReorderLoc->size(0) == splitMaskNum,
                        "Implicit GEMM is already built with different splitMaskNum value");
        }
        TORCH_CHECK(mIGEMMOutInMapBwd.has_value() == training,
                    "Implicit GEMM is already built with different training flag");
        return;
    }

    int kernelVolume = mKernelSize.value().x() * mKernelSize.value().y() * mKernelSize.value().z();

    int outInMapSize = (mTargetGrid.total_voxels() + 128 - 1) / 128 * 128;
    mIGEMMOutInMap = torch::full(
        {outInMapSize, kernelVolume}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(mTargetGrid.device()));
    mIGEMMUseTF32 = use_tf32;

    // Note: This could also be converted from GSNeighbourMap if exists
    FVDB_DISPATCH_KERNEL_DEVICE(mSourceGrid.device(), [&]() {
        detail::ops::dispatchConvolutionKernelMap<DeviceTag>(
            *mSourceGrid.impl(), *mTargetGrid.impl(), mIGEMMOutInMap.value(), mKernelSize, mStride);
    });

    if (sorted) {
        TORCH_CHECK(mSourceGrid.device().is_cuda(), "Implicit GEMM with sorted kernel map is only supported on CUDA");
        torch::Tensor bitmask = detail::ops::dispatchBitmaskFromOutInMap<torch::kCUDA>(
            mIGEMMOutInMap.value(), splitMaskNum, mTargetGrid.total_voxels());
        auto ret = torch::sort(bitmask, -1L, true);
        mIGEMMSortedMask = std::get<0>(ret);    // Mainly used for transpose.
        mIGEMMReorderLoc = std::get<1>(ret).to(torch::kInt32);
        mIGEMMReoderOutInMap = detail::ops::dispatchReorderOutInMap<torch::kCUDA>(
            mIGEMMOutInMap.value(), mIGEMMReorderLoc.value());
        mIGEMMReducedSortedMask = detail::ops::dispatchReduceMask<torch::kCUDA>(
            mIGEMMSortedMask.value(), 128);
    }

    if (training) {
        int outInMapTSize = (mSourceGrid.total_voxels() + 128 - 1) / 128 * 128;
        mIGEMMOutInMapBwd = torch::full(
            {outInMapTSize, kernelVolume}, -1,
            torch::TensorOptions().dtype(torch::kInt32).device(mSourceGrid.device()));
        detail::ops::dispatchTransposeOutInMap<torch::kCUDA>(
            mIGEMMOutInMap.value(), mIGEMMOutInMapBwd.value());
        torch::Tensor bitmask = detail::ops::dispatchBitmaskFromOutInMap<torch::kCUDA>(
            mIGEMMOutInMapBwd.value(), splitMaskNumBwd, mSourceGrid.total_voxels());
        auto ret = torch::sort(bitmask, -1L, true);
        torch::Tensor sortedMaskBwd = std::get<0>(ret);
        mIGEMMReorderLocBwd = std::get<1>(ret).to(torch::kInt32);
        mIGEMMReorderOutInMapBwd = detail::ops::dispatchReorderOutInMap<torch::kCUDA>(
            mIGEMMOutInMapBwd.value(), mIGEMMReorderLocBwd.value());
        mIGEMMSortedMaskBwdW = detail::ops::dispatchReduceMask<torch::kCUDA>(
            sortedMaskBwd, 64);
        mIGEMMSortedMaskBwdD = detail::ops::dispatchReduceMask<torch::kCUDA>(
            sortedMaskBwd, 128);
    }

}

SparseConvPackInfo SparseConvPackInfo::transposed() const {
    SparseConvPackInfo ret(mKernelSize, mStride, mSourceGrid, mTargetGrid);
    bool sorted = mIGEMMReorderLoc.has_value();
    bool training = mIGEMMOutInMapBwd.has_value();
    int splitMaskNum = mIGEMMReorderLoc.has_value() ? mIGEMMReorderLoc.value().size(0) : 1;

    int outInMapSize = (mSourceGrid.total_voxels() + 128 - 1) / 128 * 128;
    int kernelVolume = mKernelSize.value().x() * mKernelSize.value().y() * mKernelSize.value().z();

    ret.mIGEMMOutInMap = torch::full(
            {outInMapSize, kernelVolume}, -1,
            torch::TensorOptions().dtype(torch::kInt32).device(mSourceGrid.device()));
    detail::ops::dispatchTransposeOutInMap<torch::kCUDA>(
        mIGEMMOutInMap.value(), ret.mIGEMMOutInMap.value());

    if (sorted) {
        if (training) {
            ret.mIGEMMOutInMapBwd = mIGEMMOutInMap;
            ret.mIGEMMReorderOutInMapBwd = mIGEMMReoderOutInMap;
            ret.mIGEMMReorderLocBwd = mIGEMMReorderLoc;
            torch::Tensor sortedMaskBwd = mIGEMMSortedMask.value();
            ret.mIGEMMSortedMaskBwdW = detail::ops::dispatchReduceMask<torch::kCUDA>(
                sortedMaskBwd, 64);
            ret.mIGEMMSortedMaskBwdD = detail::ops::dispatchReduceMask<torch::kCUDA>(
                sortedMaskBwd, 128);
        }
        torch::Tensor bitmask = detail::ops::dispatchBitmaskFromOutInMap<torch::kCUDA>(
            ret.mIGEMMOutInMap.value(), splitMaskNum, mSourceGrid.total_voxels());
        auto rets = torch::sort(bitmask, -1L, true);
        ret.mIGEMMSortedMask = std::get<0>(rets);
        ret.mIGEMMReorderLoc = std::get<1>(rets).to(torch::kInt32);
        ret.mIGEMMReoderOutInMap = detail::ops::dispatchReorderOutInMap<torch::kCUDA>(
            ret.mIGEMMOutInMap.value(), ret.mIGEMMReorderLoc.value());
        ret.mIGEMMReducedSortedMask = detail::ops::dispatchReduceMask<torch::kCUDA>(
            ret.mIGEMMSortedMask.value(), 128);
    } else if (training) {
        int splitMaskNumBwd = mIGEMMReorderLocBwd.value().size(0);
        ret.mIGEMMOutInMapBwd = mIGEMMOutInMap;
        torch::Tensor bitmaskBwd = detail::ops::dispatchBitmaskFromOutInMap<torch::kCUDA>(
            ret.mIGEMMOutInMapBwd.value(), splitMaskNumBwd, mTargetGrid.total_voxels());
        auto rets = torch::sort(bitmaskBwd, -1L, true);
        torch::Tensor sortedMaskBwd = std::get<0>(rets);
        ret.mIGEMMReorderLocBwd = std::get<1>(rets).to(torch::kInt32);
        ret.mIGEMMReorderOutInMapBwd = detail::ops::dispatchReorderOutInMap<torch::kCUDA>(
            ret.mIGEMMOutInMapBwd.value(), ret.mIGEMMReorderLocBwd.value());
        ret.mIGEMMSortedMaskBwdW = detail::ops::dispatchReduceMask<torch::kCUDA>(
            sortedMaskBwd, 64);
        ret.mIGEMMSortedMaskBwdD = detail::ops::dispatchReduceMask<torch::kCUDA>(
            sortedMaskBwd, 128);
    }
    ret.mIGEMMUseTF32 = mIGEMMUseTF32;
    return ret;
}

void SparseConvPackInfo::buildCutlass(bool benchmark) {
    if (mCUTLASSHaloIndexBuffer.has_value()) {
        TORCH_CHECK(mCUTLASSBenchmark == benchmark, "Cutlass is already built with different benchmark flag");
        return;
    }
    std::vector<torch::Tensor> res = FVDB_DISPATCH_KERNEL_DEVICE(mSourceGrid.device(), [&]() {
        return detail::ops::dispatchBrickHaloBuffer<DeviceTag>(
            *mSourceGrid.impl(), benchmark);
    });
    mCUTLASSHaloIndexBuffer = res[1];
    mCUTLASSOutputIndexBuffer = res[2];
    mCUTLASSBenchmark = benchmark;
}

JaggedTensor SparseConvPackInfo::sparseConv3d(const JaggedTensor& input, const torch::Tensor& weights, ConvPackBackend backend) const {
    TORCH_CHECK_VALUE(input.batch_size()  == mSourceGrid.grid_count(), "Input batch size must match target grid batch size");
    TORCH_CHECK_VALUE(input.element_count() == mSourceGrid.total_voxels(), "Input element count must match target grid total voxels");

    if (backend == ConvPackBackend::GATHER_SCATTER) {
        auto ret = detail::autograd::SparseConvolutionKernelMap::apply(
            input.jdata(), weights, *this, false /* transposed */)[0];

        return mTargetGrid.impl()->jaggedTensor(ret, false);
    } else if (backend == ConvPackBackend::IGEMM) {
        auto ret = detail::autograd::SparseConvolutionImplicitGEMM::apply(
            input.jdata(), weights, *this, false /* transposed */)[0];

        return mTargetGrid.impl()->jaggedTensor(ret, false);
    } else if (backend == ConvPackBackend::CUTLASS) {
        // Re-shape kernel from [Do, Di, D, H, W] to [Do, D, H, W, Di].
        TORCH_CHECK(mCUTLASSHaloIndexBuffer.has_value() && mCUTLASSOutputIndexBuffer.has_value(),
                    "Cutlass buffer is not built");
        auto kernel = weights.permute({0, 4, 3, 2, 1}).contiguous();
        torch::Tensor out = FVDB_DISPATCH_KERNEL_DEVICE(mCUTLASSHaloIndexBuffer->device(), [&]() {
            return detail::ops::dispatchSparseConvolutionCutlass<DeviceTag>(
                input.jdata(), kernel,
                mCUTLASSHaloIndexBuffer.value(), mCUTLASSOutputIndexBuffer.value(),
                mCUTLASSBenchmark);
        });
        return mTargetGrid.impl()->jaggedTensor(out, false);
    } else {
        TORCH_CHECK(false, "Unknown backend");
    }

}

JaggedTensor SparseConvPackInfo::sparseTransposeConv3d(const JaggedTensor& input, const torch::Tensor& weights, ConvPackBackend backend) const {
    TORCH_CHECK_VALUE(input.batch_size()  == mTargetGrid.grid_count(), "Input batch size must match target grid batch size");
    TORCH_CHECK_VALUE(input.element_count() == mTargetGrid.total_voxels(), "Input element count must match target grid total voxels");

    if (backend == ConvPackBackend::GATHER_SCATTER) {
        auto ret = detail::autograd::SparseConvolutionKernelMap::apply(
            input.jdata(), weights, *this, true /* transposed */)[0];

        return mSourceGrid.impl()->jaggedTensor(ret, false);
    } else if (backend == ConvPackBackend::IGEMM) {
        auto ret = detail::autograd::SparseConvolutionImplicitGEMM::apply(
            input.jdata(), weights, *this, true /* transposed */)[0];

        return mSourceGrid.impl()->jaggedTensor(ret, false);
    } else if (backend == ConvPackBackend::CUTLASS) {
        TORCH_CHECK(false, "Cutlass does not support transpose convolution yet");
    } else {
        TORCH_CHECK(false, "Unknown backend");
    }

}


} // namespace fvdb
