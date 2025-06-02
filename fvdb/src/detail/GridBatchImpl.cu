// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GridBatchImpl.h"

#include <detail/ops/Ops.h>
#include <detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void
computeBatchOffsetsFromMetadata(
    uint32_t numGrids,
    fvdb::detail::GridBatchImpl::GridMetadata *perGridMetadata,
    torch::PackedTensorAccessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>
        outBatchOffsets) {
    if (numGrids == 0) {
        return;
    }
    outBatchOffsets[0] = 0;
    for (uint32_t i = 1; i < (numGrids + 1); i += 1) {
        outBatchOffsets[i] = outBatchOffsets[i - 1] + perGridMetadata[i - 1].mNumVoxels;
    }
}

// We use a helper with std::malloc rather than just using new because it makes
// clangd not crash on this file.
fvdb::detail::GridBatchImpl::GridMetadata *
allocateHostGridMetadata(int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");
    GridMetadata *ret =
        reinterpret_cast<GridMetadata *>(std::malloc(sizeof(GridMetadata) * batchSize));
    for (auto i = 0; i < batchSize; ++i) {
        ret[i] = GridMetadata();
    }
    return ret;
}

void
freeHostGridMetadata(fvdb::detail::GridBatchImpl::GridMetadata *hostGridMetadata) {
    std::free(hostGridMetadata);
}

fvdb::detail::GridBatchImpl::GridMetadata *
allocateDeviceGridMetadata(torch::Device device, int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    c10::cuda::CUDAGuard deviceGuard(device);
    auto wrapper                  = c10::cuda::getCurrentCUDAStream(device.index());
    const size_t metadataByteSize = sizeof(GridMetadata) * batchSize;
    auto data =
        c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(metadataByteSize, wrapper.stream());
    return static_cast<GridMetadata *>(data);
}

void
freeDeviceGridMetadata(torch::Device device,
                       fvdb::detail::GridBatchImpl::GridMetadata *deviceGridMetadata) {
    c10::cuda::CUDAGuard deviceGuard(device);
    c10::cuda::CUDACachingAllocator::raw_delete(deviceGridMetadata);
}

fvdb::detail::GridBatchImpl::GridMetadata *
allocateUnifiedMemoryGridMetadata(int64_t batchSize) {
    using GridMetadata = fvdb::detail::GridBatchImpl::GridMetadata;
    TORCH_CHECK(batchSize > 0, "Batch size must be greater than 0");

    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    auto data      = allocator->raw_allocate(sizeof(GridMetadata) * batchSize);
    return static_cast<GridMetadata *>(data);
}

void
freeUnifiedMemoryGridMetadata(
    fvdb::detail::GridBatchImpl::GridMetadata *unifiedMemoryGridMetadata) {
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    allocator->raw_deallocate(unifiedMemoryGridMetadata);
}

} // namespace

namespace fvdb {
namespace detail {

GridBatchImpl::GridBatchImpl(const torch::Device &device) {
    auto deviceTensorOptions = torch::TensorOptions().device(device);
    // TODO (Francis): No list-of-lists support for now, so we just assign an empty list of indices
    mLeafBatchIndices = torch::empty({0}, deviceTensorOptions.dtype(fvdb::JIdxScalarType));
    mBatchOffsets     = torch::zeros({1}, deviceTensorOptions.dtype(fvdb::JOffsetsScalarType));
    mListIndices      = torch::empty({0, 1}, deviceTensorOptions.dtype(fvdb::JLIdxScalarType));

    auto gridHdl = createEmptyGridHandle(device);
    mGridHdl     = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(gridHdl));

    mBatchMetadata.mIsContiguous = true;
}

GridBatchImpl::GridBatchImpl(const torch::Device &device,
                             const nanovdb::Vec3d &voxelSize,
                             const nanovdb::Vec3d &origin)
    : GridBatchImpl(createEmptyGridHandle(device), {voxelSize}, {origin}) {}

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                             const std::vector<nanovdb::Vec3d> &voxelSizes,
                             const std::vector<nanovdb::Vec3d> &voxelOrigins) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(),
                "Cannot create a batched grid handle from an empty grid handle");
    for (std::size_t i = 0; i < voxelSizes.size(); i += 1) {
        TORCH_CHECK_VALUE(voxelSizes[i][0] > 0 && voxelSizes[i][1] > 0 && voxelSizes[i][2] > 0,
                          "Voxel size must be greater than 0");
    }
    // TODO (Francis): No list-of-lists support for now, so we just pass an empty list of indices
    const torch::Tensor lidx = torch::empty(
        {0, 1},
        torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(gridHdl.buffer().device()));
    setGrid(std::move(gridHdl), lidx, voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                             const nanovdb::Vec3d &globalVoxelSize,
                             const nanovdb::Vec3d &globalVoxelOrigin) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(),
                "Cannot create a batched grid handle from an empty grid handle");
    TORCH_CHECK_VALUE(globalVoxelSize[0] > 0 && globalVoxelSize[1] > 0 && globalVoxelSize[2] > 0,
                      "Voxel size must be greater than 0");
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    for (size_t i = 0; i < gridHdl.gridCount(); ++i) {
        voxelSizes.push_back(globalVoxelSize);
        voxelOrigins.push_back(globalVoxelOrigin);
    }
    // TODO (Francis): No list-of-lists support for now, so we just pass an empty list of indices
    const torch::Tensor lidx = torch::empty(
        {0, 1},
        torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(gridHdl.buffer().device()));
    setGrid(std::move(gridHdl), lidx, voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::~GridBatchImpl() {
    const torch::Device device = mGridHdl->buffer().device();
    if (device.is_cpu() || device.is_cuda()) {
        freeHostGridMetadata(mHostGridMetadata);
        mHostGridMetadata = nullptr;
        if (device.is_cuda()) {
            freeDeviceGridMetadata(device, mDeviceGridMetadata);
            mDeviceGridMetadata = nullptr;
        }
    } else if (device.is_privateuseone()) {
        freeUnifiedMemoryGridMetadata(mDeviceGridMetadata);
        mHostGridMetadata   = nullptr;
        mDeviceGridMetadata = nullptr;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }
};

torch::Tensor
GridBatchImpl::worldToGridMatrix(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);

    torch::Tensor xformMat =
        torch::eye(4, torch::TensorOptions().device(device()).dtype(torch::kDouble));
    const VoxelCoordTransform &transform = primalTransform(bi);
    const nanovdb::Vec3d &scale          = transform.scale<double>();
    const nanovdb::Vec3d &translate      = transform.translate<double>();

    xformMat[0][0] = scale[0];
    xformMat[1][1] = scale[1];
    xformMat[2][2] = scale[2];

    xformMat[3][0] = translate[0];
    xformMat[3][1] = translate[1];
    xformMat[3][2] = translate[2];

    return xformMat;
}

void
GridBatchImpl::recomputeBatchOffsets() {
    mBatchOffsets = torch::empty(
        {batchSize() + 1}, torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(device()));
    if (device().is_cuda()) {
        computeBatchOffsetsFromMetadata<<<1, 1>>>(
            batchSize(),
            mDeviceGridMetadata,
            mBatchOffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto outBatchOffsets = mBatchOffsets.accessor<fvdb::JOffsetsType, 1>();
        outBatchOffsets[0]   = 0;
        for (int64_t i = 1; i < (batchSize() + 1); i += 1) {
            outBatchOffsets[i] = outBatchOffsets[i - 1] + mHostGridMetadata[i - 1].mNumVoxels;
        }
    }
}

template <typename Indexable>
c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const Indexable &idx, int64_t size) const {
    if (size == 0) {
        return c10::make_intrusive<GridBatchImpl>(device());
    }
    TORCH_CHECK(size >= 0,
                "Indexing with negative size is not supported (this should never happen)");
    c10::intrusive_ptr<GridBatchImpl> ret = c10::make_intrusive<GridBatchImpl>();
    ret->mGridHdl                         = mGridHdl;

    int64_t cumVoxels     = 0;
    int64_t cumLeaves     = 0;
    int64_t maxVoxels     = 0;
    uint32_t maxLeafCount = 0;
    int64_t count         = 0;
    nanovdb::CoordBBox totalBbox;

    std::vector<torch::Tensor> leafBatchIdxs;

    // If the grid we're creating a view over is contiguous, we inherit that
    bool isContiguous      = mBatchMetadata.mIsContiguous;
    ret->mHostGridMetadata = allocateHostGridMetadata(size);
    ret->mBatchSize        = size;
    for (int64_t i = 0; i < size; i += 1) {
        int64_t bi = idx[i];
        bi         = negativeToPositiveIndexWithRangecheck(bi);

        // If indices are not contiguous or the grid we're viewing is not contiguous, then we're
        // no longer contiguous
        isContiguous = isContiguous && (bi == count);

        const uint32_t numLeaves       = mHostGridMetadata[bi].mNumLeaves;
        const int64_t numVoxels        = mHostGridMetadata[bi].mNumVoxels;
        const nanovdb::CoordBBox &bbox = mHostGridMetadata[bi].mBBox;

        ret->mHostGridMetadata[count]            = mHostGridMetadata[bi];
        ret->mHostGridMetadata[count].mCumLeaves = cumLeaves;
        ret->mHostGridMetadata[count].mCumVoxels = cumVoxels;

        if (count == 0) {
            totalBbox = bbox;
        } else {
            totalBbox.expand(bbox);
        }
        cumLeaves += numLeaves;
        cumVoxels += numVoxels;
        maxVoxels    = std::max(maxVoxels, numVoxels);
        maxLeafCount = std::max(maxLeafCount, numLeaves);
        leafBatchIdxs.push_back(
            torch::full({numLeaves},
                        torch::Scalar(count),
                        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device())));
        count += 1;
    }

    // If all the indices were contiguous and the grid we're viewing is contiguous, then we're
    // contiguous
    ret->mBatchMetadata.mIsContiguous = isContiguous && (count == batchSize());
    ret->mBatchMetadata.mTotalLeaves  = cumLeaves;
    ret->mBatchMetadata.mTotalVoxels  = cumVoxels;
    ret->mBatchMetadata.mMaxVoxels    = maxVoxels;
    ret->mBatchMetadata.mMaxLeafCount = maxLeafCount;
    ret->mBatchMetadata.mTotalBBox    = totalBbox;

    if (leafBatchIdxs.size() > 0) {
        ret->mLeafBatchIndices = torch::cat(leafBatchIdxs, 0);
    }

    ret->syncMetadataToDeviceIfCUDA(false);
    ret->recomputeBatchOffsets();

    if (mListIndices.size(0) > 0) {
        TORCH_CHECK(false, "Nested lists of GridBatches are not supported yet");
    } else {
        ret->mListIndices = mListIndices;
    }
    return ret;
}

torch::Tensor
GridBatchImpl::gridToWorldMatrix(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);
    return at::linalg_inv(worldToGridMatrix(bi));
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::clone(const torch::Device &device, bool blocking) const {
    // If you're cloning an empty grid, just create a new empty grid on the right device and return
    // it
    if (batchSize() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device);
    }

    // The guide buffer is a hack to perform the correct copy (i.e. host -> device / device -> host
    // etc...) The guide carries the desired target device to the copy. The reason we do this is to
    // conform with the nanovdb which can only accept a buffer as an extra argument.
    TorchDeviceBuffer guide(0, device);

    // Make a copy of this gridHandle on the same device as the guide buffer
    nanovdb::GridHandle<TorchDeviceBuffer> clonedHdl = mGridHdl->copy<TorchDeviceBuffer>(guide);

    // Copy the voxel sizes and origins for this grid
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);

    // Build a GridBatchImpl from the cloned grid handle and voxel sizes/origins
    // FIXME: (@fwilliams) This makes an extra copy or non contiguous grids
    return GridBatchImpl::contiguous(
        c10::make_intrusive<GridBatchImpl>(std::move(clonedHdl), voxelSizes, voxelOrigins));
}

void
GridBatchImpl::syncMetadataToDeviceIfCUDA(bool blocking) {
    if (!device().is_cuda()) {
        return;
    }

    // There is something to sync and we're on a cuda device
    // Global device guards as we operate on this.
    c10::cuda::CUDAGuard deviceGuard(device());
    c10::cuda::CUDAStream wrapper = c10::cuda::getCurrentCUDAStream(device().index());

    // There are no grids in the batch so we need to free the device metadata if it exists
    if (!mHostGridMetadata && mDeviceGridMetadata) {
        freeDeviceGridMetadata(device(), mDeviceGridMetadata);
        mDeviceGridMetadata = nullptr;
        return;
    }

    if (!mDeviceGridMetadata) { // Allocate the CUDA memory if it hasn't been allocated already
        mDeviceGridMetadata = allocateDeviceGridMetadata(device(), batchSize());
    }

    // Copy host grid metadata to device grid metadata
    const size_t metadataByteSize = sizeof(GridMetadata) * batchSize();
    C10_CUDA_CHECK(cudaMemcpyAsync(mDeviceGridMetadata,
                                   mHostGridMetadata,
                                   metadataByteSize,
                                   cudaMemcpyHostToDevice,
                                   wrapper.stream()));
    // Block if you asked for it
    if (blocking) {
        C10_CUDA_CHECK(cudaStreamSynchronize(wrapper.stream()));
    }
}

namespace {

__global__ void
setGlobalPrimalTransformKernel(GridBatchImpl::GridMetadata *metadata,
                               VoxelCoordTransform transform) {
    unsigned int i               = threadIdx.x;
    metadata[i].mPrimalTransform = transform;
}

__global__ void
setGlobalDualTransformKernel(GridBatchImpl::GridMetadata *metadata, VoxelCoordTransform transform) {
    unsigned int i             = threadIdx.x;
    metadata[i].mDualTransform = transform;
}

__global__ void
setGlobalVoxelSizeKernel(GridBatchImpl::GridMetadata *metadata, nanovdb::Vec3d voxelSize) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(voxelSize, metadata[i].voxelOrigin());
}

__global__ void
setGlobalVoxelOriginKernel(GridBatchImpl::GridMetadata *metadata, nanovdb::Vec3d voxelOrigin) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(metadata[i].mVoxelSize, voxelOrigin);
}

__global__ void
setGlobalVoxelSizeAndOriginKernel(GridBatchImpl::GridMetadata *metadata,
                                  nanovdb::Vec3d voxelSize,
                                  nanovdb::Vec3d voxelOrigin) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(voxelSize, voxelOrigin);
}

__global__ void
setPrimalTransformFromDualGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *dualMetadata) {
    unsigned int i               = threadIdx.x;
    metadata[i].mDualTransform   = dualMetadata[i].mPrimalTransform;
    metadata[i].mPrimalTransform = dualMetadata[i].mDualTransform;
    metadata[i].mVoxelSize       = dualMetadata[i].mVoxelSize;
}

__hostdev__ nanovdb::Vec3d
fineVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &subdivFactor) {
    return voxelSize / subdivFactor.asVec3d();
}

__hostdev__ nanovdb::Vec3d
fineVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                const nanovdb::Vec3d &voxelOrigin,
                const nanovdb::Coord &subdivFactor) {
    return voxelOrigin - (subdivFactor.asVec3d() - nanovdb::Vec3d(1.0)) *
                             (voxelSize / subdivFactor.asVec3d()) * 0.5;
}

__hostdev__ nanovdb::Vec3d
coarseVoxelSize(const nanovdb::Vec3d &voxelSize, const nanovdb::Coord &coarseningFactor) {
    return coarseningFactor.asVec3d() * voxelSize;
}

__hostdev__ nanovdb::Vec3d
coarseVoxelOrigin(const nanovdb::Vec3d &voxelSize,
                  const nanovdb::Vec3d &voxelOrigin,
                  const nanovdb::Coord &coarseningFactor) {
    return (coarseningFactor.asVec3d() - nanovdb::Vec3d(1.0)) * voxelSize * 0.5 + voxelOrigin;
}

__global__ void
setFineTransformFromCoarseGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *coarseMetadata,
                                     nanovdb::Coord subdivisionFactor) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(fineVoxelSize(coarseMetadata[i].mVoxelSize, subdivisionFactor),
                             fineVoxelOrigin(coarseMetadata[i].mVoxelSize,
                                             coarseMetadata[i].voxelOrigin(),
                                             subdivisionFactor));
}

__global__ void
setCoarseTransformFromFineGridKernel(GridBatchImpl::GridMetadata *metadata,
                                     const GridBatchImpl::GridMetadata *fineMetadata,
                                     nanovdb::Coord coarseningFactor) {
    unsigned int i = threadIdx.x;
    metadata[i].setTransform(coarseVoxelSize(fineMetadata[i].mVoxelSize, coarseningFactor),
                             coarseVoxelOrigin(fineMetadata[i].mVoxelSize,
                                               fineMetadata[i].voxelOrigin(),
                                               coarseningFactor));
}

} // namespace

void
GridBatchImpl::setGlobalPrimalTransform(const VoxelCoordTransform &transform) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mPrimalTransform = transform;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalPrimalTransformKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                                transform);
    }
}

void
GridBatchImpl::setGlobalDualTransform(const VoxelCoordTransform &transform) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mDualTransform = transform;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalDualTransformKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                              transform);
    }
}

void
GridBatchImpl::setGlobalVoxelSize(const nanovdb::Vec3d &voxelSize) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, mHostGridMetadata[i].voxelOrigin());
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelSizeKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                          voxelSize);
    }
}

void
GridBatchImpl::setGlobalVoxelOrigin(const nanovdb::Vec3d &voxelOrigin) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel origin on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(mHostGridMetadata[i].mVoxelSize, voxelOrigin);
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelOriginKernel<<<1, batchSize(), 0, wrapper.stream()>>>(mDeviceGridMetadata,
                                                                            voxelOrigin);
    }
}

void
GridBatchImpl::setGlobalVoxelSizeAndOrigin(const nanovdb::Vec3d &voxelSize,
                                           const nanovdb::Vec3d &voxelOrigin) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, voxelOrigin);
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setGlobalVoxelSizeAndOriginKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, voxelSize, voxelOrigin);
    }
}

void
GridBatchImpl::setFineTransformFromCoarseGrid(const GridBatchImpl &coarseBatch,
                                              nanovdb::Coord subdivisionFactor) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(coarseBatch.batchSize() == batchSize(),
                "Coarse grid batch size must match fine grid batch size");
    TORCH_CHECK(subdivisionFactor[0] > 0 && subdivisionFactor[1] > 0 && subdivisionFactor[2] > 0,
                "Subdivision factor must be greater than 0");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(
            fineVoxelSize(coarseBatch.voxelSize(i), subdivisionFactor),
            fineVoxelOrigin(
                coarseBatch.voxelSize(i), coarseBatch.voxelOrigin(i), subdivisionFactor));
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        TORCH_CHECK(coarseBatch.mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setFineTransformFromCoarseGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, coarseBatch.mDeviceGridMetadata, subdivisionFactor);
    }
}

void
GridBatchImpl::setCoarseTransformFromFineGrid(const GridBatchImpl &fineBatch,
                                              nanovdb::Coord coarseningFactor) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(fineBatch.batchSize() == batchSize(),
                "Fine grid batch size must match coarse grid batch size");
    TORCH_CHECK(coarseningFactor[0] > 0 && coarseningFactor[1] > 0 && coarseningFactor[2] > 0,
                "Coarsening factor must be greater than 0");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].setTransform(
            coarseVoxelSize(fineBatch.voxelSize(i), coarseningFactor),
            coarseVoxelOrigin(fineBatch.voxelSize(i), fineBatch.voxelOrigin(i), coarseningFactor));
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        TORCH_CHECK(fineBatch.mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setCoarseTransformFromFineGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, fineBatch.mDeviceGridMetadata, coarseningFactor);
    }
}

void
GridBatchImpl::setPrimalTransformFromDualGrid(const GridBatchImpl &dualBatch) {
    TORCH_CHECK(batchSize() > 0,
                "Cannot set global voxel size and origin on an empty batch of grids");

    TORCH_CHECK(dualBatch.batchSize() == batchSize(),
                "Dual grid batch size must match primal grid batch size");

    for (int64_t i = 0; i < batchSize(); i++) {
        mHostGridMetadata[i].mDualTransform   = dualBatch.mHostGridMetadata[i].mPrimalTransform;
        mHostGridMetadata[i].mPrimalTransform = dualBatch.mHostGridMetadata[i].mDualTransform;
        mHostGridMetadata[i].mVoxelSize       = dualBatch.mHostGridMetadata[i].mVoxelSize;
    }

    if (device().is_cuda() && batchSize()) {
        TORCH_CHECK(mDeviceGridMetadata);
        c10::cuda::CUDAGuard deviceGuard(device());

        auto wrapper = at::cuda::getCurrentCUDAStream(device().index());
        setPrimalTransformFromDualGridKernel<<<1, batchSize(), 0, wrapper.stream()>>>(
            mDeviceGridMetadata, dualBatch.mDeviceGridMetadata);
    }
}

void
GridBatchImpl::setGrid(nanovdb::GridHandle<TorchDeviceBuffer> &&gridHdl,
                       const torch::Tensor listIndices,
                       const std::vector<nanovdb::Vec3d> &voxelSizes,
                       const std::vector<nanovdb::Vec3d> &voxelOrigins,
                       bool blocking) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(), "Empty grid handle");
    TORCH_CHECK(voxelSizes.size() == gridHdl.gridCount(),
                "voxelSizes array does not have the same size as the number of grids, got ",
                voxelSizes.size(),
                " expected ",
                gridHdl.gridCount());
    TORCH_CHECK(voxelOrigins.size() == gridHdl.gridCount(),
                "Voxel origins must be the same size as the number of grids");
    TORCH_CHECK((gridHdl.gridType(0) == nanovdb::GridType::OnIndex) ||
                    (gridHdl.gridType(0) == nanovdb::GridType::OnIndexMask),
                "GridBatchImpl only supports ValueOnIndex and ValueOnIndexMask grids");

    // Reallocate GridMetadata
    mBatchSize = gridHdl.gridCount();

    const torch::Device device = gridHdl.buffer().device();
    if (device.is_cpu() || device.is_cuda()) {
        freeHostGridMetadata(mHostGridMetadata);
        mHostGridMetadata = allocateHostGridMetadata(gridHdl.gridCount());

        if (device.is_cuda()) {
            freeDeviceGridMetadata(device, mDeviceGridMetadata);
            mDeviceGridMetadata = allocateDeviceGridMetadata(device, mBatchSize);
        } else {
            TORCH_CHECK(!mDeviceGridMetadata, "mDeviceGridMetadata should be null for CPU devices");
        }
    } else if (device.is_privateuseone()) {
        freeUnifiedMemoryGridMetadata(mDeviceGridMetadata);
        mDeviceGridMetadata = allocateUnifiedMemoryGridMetadata(mBatchSize);
        mHostGridMetadata   = mDeviceGridMetadata;
    } else {
        TORCH_CHECK(false, "Only CPU, CUDA, and PrivateUse1 devices are supported");
    }

    FVDB_DISPATCH_KERNEL(device, [&]() {
        // Allocate device memory for metadata
        GridBatchMetadata *deviceBatchMetadataPtr = nullptr;
        if constexpr (DeviceTag == torch::kCUDA) {
            c10::cuda::CUDAGuard deviceGuard(device);
            auto wrapper = c10::cuda::getCurrentCUDAStream(device.index());
            auto data    = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
                sizeof(GridBatchMetadata), wrapper.stream());
            deviceBatchMetadataPtr = static_cast<GridBatchMetadata *>(data);
        }

        // Populate host and/or device metadata
        ops::dispatchPopulateGridMetadata<DeviceTag>(gridHdl,
                                                     voxelSizes,
                                                     voxelOrigins,
                                                     mBatchOffsets,
                                                     mHostGridMetadata,
                                                     mDeviceGridMetadata,
                                                     &mBatchMetadata,
                                                     deviceBatchMetadataPtr);
        TORCH_CHECK(listIndices.numel() == 0 || listIndices.size(0) == (mBatchOffsets.size(0) - 1),
                    "Invalid list indices when building grid");
        mListIndices = listIndices;

        // We don't need the device copy of the global batch metadata anymore (we only carry around
        // the host version and pass it by value to device kernels), so delete it
        if constexpr (DeviceTag == torch::kCUDA) {
            c10::cuda::CUDAGuard deviceGuard(device);
            c10::cuda::CUDACachingAllocator::raw_delete(deviceBatchMetadataPtr);
        }
    });

    // FIXME: This is slow
    // Populate batch offsets for each leaf node
    {
        std::vector<torch::Tensor> leafBatchIdxs;
        leafBatchIdxs.reserve(gridHdl.gridCount());
        for (uint32_t i = 0; i < gridHdl.gridCount(); i += 1) {
            leafBatchIdxs.push_back(
                torch::full({mHostGridMetadata[i].mNumLeaves},
                            static_cast<fvdb::JIdxType>(i),
                            torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device)));
        }
        mLeafBatchIndices = torch::cat(leafBatchIdxs, 0);
    }

    // Replace the grid handle with the new one
    mGridHdl = std::make_shared<nanovdb::GridHandle<TorchDeviceBuffer>>(std::move(gridHdl));
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(int64_t bi) const {
    bi = negativeToPositiveIndexWithRangecheck(bi);

    return index(bi, bi + 1, 1);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const torch::Tensor &indices) const {
    TORCH_CHECK_INDEX(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_INDEX(!indices.is_floating_point(), "indices must be an integer tensor");

    torch::Tensor numericIndices;
    if (indices.scalar_type() == torch::kBool) {
        TORCH_CHECK_INDEX(indices.dim() == 1, "bool indices must be a 1D tensor");
        TORCH_CHECK_INDEX(
            indices.numel() == batchSize(),
            "bool indices must have the same number of entries as grids in the batch");
        numericIndices = torch::arange(
            batchSize(), torch::TensorOptions().dtype(torch::kInt64).device(indices.device()));
        numericIndices = numericIndices.masked_select(indices);
    } else {
        numericIndices = indices;
    }

    torch::Tensor indicesCpu = numericIndices.to(torch::kCPU).to(torch::kInt64);
    auto indicesAccessor     = indicesCpu.accessor<int64_t, 1>();
    return indexInternal(indicesAccessor, indicesAccessor.size(0));
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const std::vector<int64_t> &indices) const {
    return indexInternal(indices, indices.size());
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(const std::vector<bool> &indices) const {
    std::vector<int64_t> indicesInt;
    indicesInt.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); i += 1) {
        if (indices[i]) {
            indicesInt.push_back(i);
        }
    }

    return indexInternal(indicesInt, indicesInt.size());
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::index(ssize_t start, ssize_t stop, ssize_t step) const {
    struct RangeAccessor {
        ssize_t mStart;
        ssize_t mStop;
        ssize_t mStep;
        ssize_t mLen;

        RangeAccessor(ssize_t start, ssize_t stop, ssize_t step, ssize_t batchSize)
            : mStart(start), mStop(stop), mStep(step) {
            TORCH_CHECK_INDEX(step != 0, "slice step cannot be zero");
            TORCH_CHECK_INDEX(0 <= start && start <= batchSize, "slice index out of range");
            TORCH_CHECK_INDEX(-1 <= stop && stop <= batchSize, "slice index out of range");

            if (stop <= start && step > 0) {
                mLen = 0;
            } else if (stop > start && step > 0) {
                mLen = (mStop - mStart + mStep - 1) / mStep;
            } else if (stop <= start && step < 0) {
                mLen = (mStart - mStop - mStep - 1) / -mStep;
            } else {
                TORCH_CHECK_INDEX(false,
                                  "Invalid slice start=",
                                  start,
                                  ", stop=",
                                  stop,
                                  ", step=",
                                  step,
                                  " for batch size ",
                                  batchSize);
            }
        }
        size_t
        operator[](size_t i) const {
            return mStart + i * mStep;
        }
    };

    auto acc = RangeAccessor(start, stop, step, batchSize());
    return indexInternal(acc, acc.mLen);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::concatenate(const std::vector<c10::intrusive_ptr<GridBatchImpl>> &elements) {
    TORCH_CHECK_VALUE(elements.size() > 0, "Must provide at least one grid for concatenate!")

    torch::Device device = elements[0]->device();

    std::vector<std::shared_ptr<nanovdb::GridHandle<TorchDeviceBuffer>>> handles;
    std::vector<std::vector<int64_t>> byteSizes;
    std::vector<std::vector<int64_t>> readByteOffsets;
    std::vector<std::vector<int64_t>> writeByteOffsets;
    int64_t totalByteSize = 0;
    int64_t totalGrids    = 0;
    handles.reserve(elements.size());
    byteSizes.reserve(elements.size());
    readByteOffsets.reserve(elements.size());
    writeByteOffsets.reserve(elements.size());

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;

    for (size_t i = 0; i < elements.size(); i += 1) {
        TORCH_CHECK(elements[i]->device() == device,
                    "All grid batches must be on the same device!");

        // Empty grids don't contribute to the concatenation
        if (elements[i]->batchSize() == 0) {
            continue;
        }

        handles.push_back(elements[i]->mGridHdl);

        readByteOffsets.push_back(std::vector<int64_t>());
        writeByteOffsets.push_back(std::vector<int64_t>());
        byteSizes.push_back(std::vector<int64_t>());
        readByteOffsets.back().reserve(elements[i]->batchSize());
        writeByteOffsets.back().reserve(elements[i]->batchSize());
        byteSizes.back().reserve(elements[i]->batchSize());

        totalGrids += elements[i]->batchSize();

        for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
            voxelSizes.push_back(elements[i]->voxelSize(j));
            voxelOrigins.push_back(elements[i]->voxelOrigin(j));

            readByteOffsets.back().push_back(
                elements[i]->cumBytes(j)); // Where to start reading from in the current grid
            byteSizes.back().push_back(elements[i]->numBytes(j)); // How many bytes to read
            writeByteOffsets.back().push_back(
                totalByteSize); // Where to start writing to in the concatenated grid
            totalByteSize += elements[i]->numBytes(j);
        }
    }
    if (handles.size() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device);
    }

    TorchDeviceBuffer buffer(totalByteSize, device);

    int count         = 0;
    int nonEmptyCount = 0;
    if (device.is_cpu()) {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
                const uint8_t *src = elements[i]->mGridHdl->buffer().data() + readOffset;
                memcpy((void *)dst, (void *)src, numBytes);
                nanovdb::tools::updateGridCount(dst, count++, totalGrids);
            }
            nonEmptyCount += 1;
        }
    } else {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (int64_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset  = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes    = byteSizes[nonEmptyCount][j];

                c10::cuda::CUDAGuard deviceGuard(device.index());
                nanovdb::GridData *dst =
                    reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
                const uint8_t *src = elements[i]->mGridHdl->buffer().deviceData() + readOffset;
                cudaMemcpyAsync((uint8_t *)dst, src, numBytes, cudaMemcpyDeviceToDevice);

                bool dirty, *d_dirty;
                cudaMallocAsync((void **)&d_dirty, sizeof(bool), 0);
                nanovdb::cuda::updateGridCount<<<1, 1>>>(dst, count++, totalGrids, d_dirty);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost);
                if (dirty)
                    nanovdb::tools::cuda::updateChecksum(dst, nanovdb::CheckMode::Partial);
            }
            nonEmptyCount += 1;
        }
    }
    nanovdb::GridHandle<TorchDeviceBuffer> gridHdl =
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer));
    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::contiguous(c10::intrusive_ptr<GridBatchImpl> input) {
    if (input->isContiguous()) {
        return input;
    }

    const int64_t totalGrids = input->batchSize();

    int64_t totalByteSize = 0;
    for (int64_t i = 0; i < input->batchSize(); i += 1) {
        totalByteSize += input->numBytes(i);
    }

    TorchDeviceBuffer buffer(totalByteSize, input->device());

    int64_t writeOffset = 0;
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(input->batchSize());
    voxelOrigins.reserve(input->batchSize());

    if (input->device().is_cpu()) {
        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSize(i));
            voxelOrigins.push_back(input->voxelOrigin(i));

            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.data() + writeOffset);
            const uint8_t *src = input->nanoGridHandle().buffer().data() + input->cumBytes(i);
            memcpy((void *)dst, (void *)src, input->numBytes(i));
            nanovdb::tools::updateGridCount(dst, i, totalGrids);
            writeOffset += input->numBytes(i);
        }

    } else {
        for (int64_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSize(i));
            voxelOrigins.push_back(input->voxelOrigin(i));

            c10::cuda::CUDAGuard deviceGuard(input->device().index());
            nanovdb::GridData *dst =
                reinterpret_cast<nanovdb::GridData *>(buffer.deviceData() + writeOffset);
            const uint8_t *src = input->nanoGridHandle().buffer().deviceData() + input->cumBytes(i);
            cudaMemcpyAsync((uint8_t *)dst, src, input->numBytes(i), cudaMemcpyDeviceToDevice);

            bool dirty, *d_dirty;
            cudaMallocAsync((void **)&d_dirty, sizeof(bool), 0);
            nanovdb::cuda::updateGridCount<<<1, 1>>>(dst, i, totalGrids, d_dirty);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost);
            if (dirty)
                nanovdb::tools::cuda::updateChecksum(dst, nanovdb::CheckMode::Partial);
            writeOffset += input->numBytes(i);
        }
    }

    return c10::make_intrusive<GridBatchImpl>(
        nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buffer)), voxelSizes, voxelOrigins);
}

JaggedTensor
GridBatchImpl::jaggedTensor(const torch::Tensor &data) const {
    checkDevice(data);
    TORCH_CHECK(data.dim() >= 1, "Data have more than one dimensions");
    TORCH_CHECK(data.size(0) == totalVoxels(), "Data size mismatch");
    return JaggedTensor::from_data_offsets_and_list_ids(data, voxelOffsets(), jlidx());
}

torch::Tensor
GridBatchImpl::jidx() const {
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        if (batchSize() == 1 || totalVoxels() == 0) {
            return torch::empty(
                {0}, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(device()));
        }
        return ops::dispatchJIdxForGrid<DeviceTag>(*this);
    });
}

torch::Tensor
GridBatchImpl::jlidx() const {
    return mListIndices;
}

torch::Tensor
GridBatchImpl::voxelOffsets() const {
    return mBatchOffsets;
}

torch::Tensor
GridBatchImpl::serialize() const {
    return serializeV0();
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::deserialize(const torch::Tensor &serialized) {
    return deserializeV0(serialized);
}

torch::Tensor
GridBatchImpl::serializeV0() const {
    c10::intrusive_ptr<GridBatchImpl> self =
        c10::intrusive_ptr<GridBatchImpl>::reclaim_copy((GridBatchImpl *)this);
    if (!device().is_cpu()) {
        self = clone(torch::kCPU, true);
    }

    int64_t numGrids   = self->nanoGridHandle().gridCount();
    int64_t hdlBufSize = self->nanoGridHandle().buffer().size();

    struct V01Header {
        uint64_t magic   = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    } header;

    const int64_t headerSize =
        sizeof(V01Header) + numGrids * sizeof(GridMetadata) + sizeof(GridBatchMetadata);
    const int64_t totalByteSize = headerSize + hdlBufSize;

    header.totalBytes = totalByteSize;
    header.numGrids   = numGrids;

    torch::Tensor ret = torch::empty({totalByteSize}, torch::kInt8);
    int8_t *retPtr    = ret.data_ptr<int8_t>();

    memcpy(retPtr, &header, sizeof(V01Header));
    retPtr += sizeof(V01Header);

    memcpy(retPtr, &self->mBatchMetadata, sizeof(GridBatchMetadata));
    retPtr += sizeof(GridBatchMetadata);

    memcpy(retPtr, self->mHostGridMetadata, numGrids * sizeof(GridMetadata));
    retPtr += numGrids * sizeof(GridMetadata);

    memcpy(retPtr, self->nanoGridHandle().buffer().data(), hdlBufSize);
    retPtr += hdlBufSize;

    TORCH_CHECK(retPtr == (ret.data_ptr<int8_t>() + totalByteSize),
                "Something went wrong with serialization");

    return ret;
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::deserializeV0(const torch::Tensor &serialized) {
    struct V01Header {
        uint64_t magic   = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    };

    TORCH_CHECK(serialized.scalar_type() == torch::kInt8, "Serialized data must be of type int8");
    TORCH_CHECK(serialized.numel() >= sizeof(V01Header),
                "Serialized data is too small to be a valid grid handle");

    const int8_t *serializedPtr = serialized.data_ptr<int8_t>();

    const V01Header *header = reinterpret_cast<const V01Header *>(serializedPtr);
    TORCH_CHECK(header->magic == 0x0F0F0F0F0F0F0F0F,
                "Serialized data is not a valid grid handle. Bad magic.");
    TORCH_CHECK(header->version == 0, "Serialized data is not a valid grid handle. Bad version.");
    TORCH_CHECK(serialized.numel() == header->totalBytes,
                "Serialized data is not a valid grid handle. Bad total bytes.");

    const uint64_t numGrids = header->numGrids;

    const GridBatchMetadata *batchMetadata =
        reinterpret_cast<const GridBatchMetadata *>(serializedPtr + sizeof(V01Header));
    TORCH_CHECK(batchMetadata->version == 1,
                "Serialized data is not a valid grid handle. Bad batch metadata version.");

    const GridMetadata *gridMetadata = reinterpret_cast<const GridMetadata *>(
        serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata));
    for (uint64_t i = 0; i < numGrids; i += 1) {
        TORCH_CHECK(gridMetadata[i].version == 1,
                    "Serialized data is not a valid grid handle. Bad grid metadata version.");
    }
    const int8_t *gridBuffer = serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata) +
                               numGrids * sizeof(GridMetadata);

    const uint64_t sizeofMetadata =
        sizeof(V01Header) + sizeof(GridBatchMetadata) + numGrids * sizeof(GridMetadata);
    const uint64_t sizeofGrid = header->totalBytes - sizeofMetadata;

    auto buf = TorchDeviceBuffer(sizeofGrid, torch::kCPU);
    memcpy(buf.data(), gridBuffer, sizeofGrid);

    nanovdb::GridHandle gridHdl = nanovdb::GridHandle<TorchDeviceBuffer>(std::move(buf));

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(numGrids);
    voxelOrigins.reserve(numGrids);
    for (uint64_t i = 0; i < numGrids; i += 1) {
        voxelSizes.emplace_back(gridMetadata[i].mVoxelSize);
        voxelOrigins.emplace_back(gridMetadata[i].voxelOrigin());
    }

    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}

template c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const torch::TensorAccessor<int64_t, 1> &idx, int64_t size) const;
template c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::indexInternal(const std::vector<int64_t> &idx, int64_t size) const;

c10 ::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromEmpty(const torch::Device &device,
                               const nanovdb::Vec3d &voxelSize,
                               const nanovdb::Vec3d &origin) {
    return c10::make_intrusive<GridBatchImpl>(device, voxelSize, origin);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromIjk(const JaggedTensor &ijk,
                             const std::vector<nanovdb::Vec3d> &voxelSizes,
                             const std::vector<nanovdb::Vec3d> &origins) {
    detail::RAIIDeviceGuard guard(ijk.device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected coords to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK_VALUE(ijk.rdim() == 2,
                      std::string("Expected ijk to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(ijk.rsize(1) == 3,
                      "Expected 3 dimensional coords but got ijk.rshape[1] = " +
                          std::to_string(ijk.rsize(1)));
    TORCH_CHECK(ijk.num_tensors() == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(ijk.num_outer_lists() <= GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      "Cannot create a batch of grids with more than ",
                      GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      " grids in it. ",
                      "You passed in ",
                      ijk.num_outer_lists(),
                      " ijk sets.");

    const int64_t numGrids = ijk.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == ijk.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");

    auto nanovdbGridHandle = FVDB_DISPATCH_KERNEL(
        ijk.device(), [&]() { return detail::ops::dispatchCreateNanoGridFromIJK<DeviceTag>(ijk); });

    return c10::make_intrusive<GridBatchImpl>(std::move(nanovdbGridHandle), voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromPoints(const JaggedTensor &points,
                                const std::vector<nanovdb::Vec3d> &voxelSizes,
                                const std::vector<nanovdb::Vec3d> &origins) {
    detail::RAIIDeviceGuard guard(points.device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(points.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(points.rsize(1) == 3,
                      "Expected 3 dimensional points but got points.rshape[1] = " +
                          std::to_string(points.rsize(1)));
    TORCH_CHECK(
        points.num_tensors() == points.num_outer_lists(),
        "If this happens, Francis' paranoia about tensors and points was justified. File a bug");
    TORCH_CHECK_VALUE(points.num_outer_lists() <= GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      points.num_outer_lists(),
                      " points sets.");

    const int64_t numGrids = points.joffsets().size(0) - 1;
    TORCH_CHECK(
        numGrids == points.num_outer_lists(),
        "If this happens, Francis' paranoia about grids and points was justified. File a bug");

    std::vector<detail::VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(
            detail::primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }

    auto pointGridBatchHdl = FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return detail::ops::dispatchBuildGridFromPoints<DeviceTag>(points, transforms);
    });

    return c10::make_intrusive<detail::GridBatchImpl>(
        std::move(pointGridBatchHdl), voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromMesh(const JaggedTensor &meshVertices,
                              const JaggedTensor &meshFaces,
                              const std::vector<nanovdb::Vec3d> &voxelSizes,
                              const std::vector<nanovdb::Vec3d> &origins) {
    detail::RAIIDeviceGuard guard(meshVertices.device());
    TORCH_CHECK_VALUE(
        meshVertices.device() == meshFaces.device(),
        "meshVertices and meshFaces must be on the same device, but got meshVertices.device() = ",
        meshVertices.device(),
        " and meshFaces.device() = ",
        meshFaces.device());
    TORCH_CHECK_VALUE(
        meshVertices.ldim() == 1,
        "Expected meshVertices to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        meshVertices.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        meshFaces.ldim() == 1,
        "Expected meshFaces to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        meshFaces.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(meshVertices.is_floating_point(),
                     "meshVertices must have a floating point type");
    TORCH_CHECK_VALUE(
        meshVertices.rdim() == 2,
        std::string("Expected meshVertices to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(meshVertices.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(meshVertices.rsize(1) == 3,
                      "Expected 3 dimensional meshVertices but got meshVertices.rshape[1] = " +
                          std::to_string(meshVertices.rsize(1)));

    TORCH_CHECK_TYPE(!meshFaces.is_floating_point(), "meshFaces must have an integer type");
    TORCH_CHECK_VALUE(
        meshFaces.rdim() == 2,
        std::string("Expected meshFaces to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(meshFaces.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(meshFaces.rsize(1) == 3,
                      "Expected 3 dimensional meshFaces but got meshFaces.rshape[1] = " +
                          std::to_string(meshFaces.rsize(1)));

    TORCH_CHECK_VALUE(meshVertices.num_outer_lists() == meshFaces.num_outer_lists(),
                      "Expected same number of vertex and face sets got len(meshVertices) = ",
                      meshVertices.num_outer_lists(),
                      " and len(meshFaces) = ",
                      meshFaces.num_outer_lists());
    const int64_t numGrids = meshVertices.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == meshVertices.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(numGrids <= GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      numGrids,
                      " mesh sets.");

    std::vector<detail::VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(
            detail::primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }

    auto meshGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(meshVertices.device(), [&]() {
        return detail::ops::dispatchBuildGridFromMesh<DeviceTag>(
            meshVertices, meshFaces, transforms);
    });
    return c10::make_intrusive<detail::GridBatchImpl>(
        std::move(meshGridBatchHdl), voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dense(const int64_t numGrids,
                     const torch::Device &device,
                     const nanovdb::Coord &denseDims,
                     const nanovdb::Coord &ijkMin,
                     const std::vector<nanovdb::Vec3d> &voxelSizes,
                     const std::vector<nanovdb::Vec3d> &origins,
                     std::optional<torch::Tensor> mask) {
    detail::RAIIDeviceGuard guard(device);
    TORCH_CHECK_VALUE(numGrids >= 0, "numGrids must be non-negative");

    if (mask.has_value()) {
        TORCH_CHECK_VALUE(mask.value().dtype() == torch::kBool,
                          "mask must be a boolean type or None");
        TORCH_CHECK_VALUE(mask.value().dim() == 3, "mask must be 3 dimensional");
        TORCH_CHECK_VALUE(mask.value().size(0) == denseDims[0],
                          "mask must have shape (w, h, d) = denseDims");
        TORCH_CHECK_VALUE(mask.value().size(1) == denseDims[1],
                          "mask must have shape (w, h, d) = denseDims");
        TORCH_CHECK_VALUE(mask.value().size(2) == denseDims[2],
                          "mask must have shape (w, h, d) = denseDims");
    }

    TORCH_CHECK_VALUE(denseDims[0] >= 0 && denseDims[1] >= 0 && denseDims[2] >= 0,
                      "denseDims must be non-negative");

    TORCH_CHECK_VALUE(numGrids <= GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You requested ",
                      numGrids,
                      " grids.");
    TORCH_CHECK((size_t)numGrids == voxelSizes.size(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK((size_t)numGrids == origins.size(),
                "If this happens, Francis' paranoia was justified. File a bug");

    auto denseGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(device, [&]() {
        return detail::ops::dispatchCreateNanoGridFromDense<DeviceTag>(
            numGrids, ijkMin, denseDims, device, mask);
    });
    return c10::make_intrusive<detail::GridBatchImpl>(
        std::move(denseGridBatchHdl), voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::createFromNeighborVoxelsToPoints(const JaggedTensor &points,
                                                const std::vector<nanovdb::Vec3d> &voxelSizes,
                                                const std::vector<nanovdb::Vec3d> &origins) {
    detail::RAIIDeviceGuard guard(points.device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(points.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(points.rsize(1) == 3,
                      "Expected 3 dimensional points but got points.shape[1] = " +
                          std::to_string(points.rsize(1)));
    TORCH_CHECK(points.num_tensors() == points.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");
    TORCH_CHECK_VALUE(points.num_outer_lists() <= GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchImpl::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      points.num_outer_lists(),
                      " point sets.");

    const int64_t numGrids = points.joffsets().size(0) - 1;
    TORCH_CHECK(numGrids == points.num_outer_lists(),
                "If this happens, Francis' paranoia was justified. File a bug");

    std::vector<detail::VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(
            detail::primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }

    auto pointGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        return detail::ops::dispatchBuildGridFromNearestVoxelsToPoints<DeviceTag>(points,
                                                                                  transforms);
    });

    return c10::make_intrusive<detail::GridBatchImpl>(
        std::move(pointGridBatchHdl), voxelSizes, origins);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::coarsen(const nanovdb::Coord coarseningFactor) {
    detail::RAIIDeviceGuard guard(device());
    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK_VALUE(coarseningFactor[i] > 0,
                          "coarseningFactor must be strictly positive. Got [" +
                              std::to_string(coarseningFactor[0]) + ", " +
                              std::to_string(coarseningFactor[1]) + ", " +
                              std::to_string(coarseningFactor[2]) + "]");
    }

    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);

    auto coarseGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return detail::ops::dispatchBuildCoarseGridFromFine<DeviceTag>(*this, coarseningFactor);
    });
    auto ret =
        c10::make_intrusive<detail::GridBatchImpl>(std::move(coarseGridBatchHdl), voxS, voxO);
    ret->setCoarseTransformFromFineGrid(*this, coarseningFactor);
    return ret;
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::upsample(const nanovdb::Coord upsampleFactor,
                        const std::optional<JaggedTensor> mask) {
    detail::RAIIDeviceGuard guard(device());
    if (mask.has_value()) {
        TORCH_CHECK_VALUE(
            mask.value().ldim() == 1,
            "Expected mask to have 1 list dimension, i.e. be a single list of coordinate values, but got",
            mask.value().ldim(),
            "list dimensions");
        TORCH_CHECK_VALUE(
            mask.value().ldim() == 1,
            "Expected subdiv_mask to have 1 list dimension, i.e. be a single list of coordinate values, but got",
            mask.value().ldim(),
            "list dimensions");
        checkDevice(mask.value());
        TORCH_CHECK(mask.value().jdata().sizes().size() == 1,
                    "subdivision mask must have 1 dimension");
        TORCH_CHECK(mask.value().jdata().size(0) == totalVoxels(),
                    "subdivision mask must be either empty tensor or have one entry per voxel");
        TORCH_CHECK(mask.value().scalar_type() == torch::kBool,
                    "subdivision mask must be a boolean tensor");
    }

    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK_VALUE(upsampleFactor[i] > 0,
                          "subdiv_factor must be strictly positive. Got [" +
                              std::to_string(upsampleFactor[0]) + ", " +
                              std::to_string(upsampleFactor[1]) + ", " +
                              std::to_string(upsampleFactor[2]) + "]");
    }

    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }

    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);
    auto fineGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return detail::ops::dispatchBuildFineGridFromCoarse<DeviceTag>(*this, upsampleFactor, mask);
    });
    auto ret = c10::make_intrusive<detail::GridBatchImpl>(std::move(fineGridBatchHdl), voxS, voxO);
    ret->setFineTransformFromCoarseGrid(*this, upsampleFactor);
    return ret;
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dual(const bool excludeBorder) {
    detail::RAIIDeviceGuard guard(device());
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);

    auto dualGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return detail::ops::dispatchBuildPaddedGrid<DeviceTag>(*this, 0, 1, excludeBorder);
    });

    auto ret = c10::make_intrusive<detail::GridBatchImpl>(std::move(dualGridBatchHdl), voxS, voxO);
    ret->setPrimalTransformFromDualGrid(*this);
    return ret;
}

std::tuple<c10::intrusive_ptr<GridBatchImpl>, JaggedTensor>
GridBatchImpl::clipWithMask(const std::vector<nanovdb::Coord> &bboxMins,
                            const std::vector<nanovdb::Coord> &bboxMaxs) {
    detail::RAIIDeviceGuard guard(device());

    JaggedTensor activeVoxelMask = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchActiveVoxelsInBoundsMask<DeviceTag>(
            *this, bboxMins, bboxMaxs);
    });

    JaggedTensor activeVoxelCoords = FVDB_DISPATCH_KERNEL_DEVICE(
        device(), [&]() { return fvdb::detail::ops::dispatchActiveGridCoords<DeviceTag>(*this); });

    // active voxel coords masked by the voxels in bounds
    JaggedTensor activeVoxelMaskCoords = activeVoxelCoords.rmask(activeVoxelMask.jdata());

    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);
    if (batchSize() == 0) {
        return std::make_tuple(c10::make_intrusive<detail::GridBatchImpl>(device()),
                               activeVoxelMask);
    } else {
        auto clippedGridPtr = GridBatchImpl::createFromIjk(activeVoxelMaskCoords, voxS, voxO);
        return std::make_tuple(clippedGridPtr, activeVoxelMask);
    }
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::clip(const std::vector<nanovdb::Coord> &ijkMin,
                    const std::vector<nanovdb::Coord> &ijkMax) {
    auto [clippedGridPtr, activeVoxelMask] = clipWithMask(ijkMin, ijkMax);
    // We don't need the mask anymore, so we can just return the clipped grid
    return clippedGridPtr;
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::dilate(const int dilationAmt) {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(dilationAmt > 0, "dilation must be strictly positive. Got ", dilationAmt);
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);
    auto dilatedGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(
        device(), [&]() { return detail::ops::dispatchDilateGrid<DeviceTag>(*this, dilationAmt); });
    return c10::make_intrusive<detail::GridBatchImpl>(std::move(dilatedGridBatchHdl), voxS, voxO);
}

c10::intrusive_ptr<GridBatchImpl>
GridBatchImpl::convolutionOutput(const nanovdb::Coord kernelSize, const nanovdb::Coord stride) {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(nanovdb::Coord(0) < kernelSize, "kernel_size must be strictly positive.");
    TORCH_CHECK_VALUE(nanovdb::Coord(0) < stride, "stride must be strictly positive.");
    if (batchSize() == 0) {
        return c10::make_intrusive<detail::GridBatchImpl>(device());
    }
    std::vector<nanovdb::Vec3d> voxS, voxO;
    gridVoxelSizesAndOrigins(voxS, voxO);

    auto convGridBatchHdl = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return detail::ops::dispatchBuildGridForConv<DeviceTag>(*this, kernelSize, stride);
    });
    auto ret = c10::make_intrusive<detail::GridBatchImpl>(std::move(convGridBatchHdl), voxS, voxO);
    // ret->setCoarseTransformFromFineGrid(*baseGrid, stride);
    return ret;
}

} // namespace detail
} // namespace fvdb
