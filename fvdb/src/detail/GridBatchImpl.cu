#include "GridBatchImpl.h"

#include <algorithm>

#include <nanovdb/cuda/GridHandle.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>

#include "detail/ops/Ops.h"
#include "detail/build/Build.h"

namespace {

__global__ void computeBatchOffsetsFromMetadata(
    uint32_t numGrids,
    fvdb::detail::GridBatchImpl::GridMetadata* perGridMetadata,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> outBatchOffsets) {

    if (numGrids == 0) {
        return;
    }
    outBatchOffsets[0][0] = 0;
    outBatchOffsets[0][1] = perGridMetadata[0].mNumVoxels;
    for (int i = 1; i < numGrids; i += 1) {
        outBatchOffsets[i][0] = outBatchOffsets[i-1][1];
        outBatchOffsets[i][1] = outBatchOffsets[i][0] + perGridMetadata[i].mNumVoxels;
    }
}

}

namespace fvdb {
namespace detail {

GridBatchImpl::GridBatchImpl(torch::Device device, bool isMutable) {
    std::vector<nanovdb::Vec3d> dummy;
    dummy.push_back(nanovdb::Vec3d(1.0, 1.0, 1.0));
    setGrid(build::buildEmptyGrid(device, isMutable), dummy, dummy, false);
    mHostGridMetadata.clear();
    syncMetadataToDeviceIfCUDA(false);
    mBatchMetadata.mIsContiguous = true;
}

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<PytorchDeviceBuffer>&& gridHdl,
                                     const std::vector<nanovdb::Vec3d>& voxelSizes,
                                     const std::vector<nanovdb::Vec3d>& voxelOrigins) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(), "Cannot create a batched grid handle from an empty grid handle");
    for (int i = 0; i < voxelSizes.size(); i += 1) {
        TORCH_CHECK_VALUE(voxelSizes[i][0] > 0 && voxelSizes[i][1] > 0 && voxelSizes[i][2] > 0, "Voxel size must be greater than 0");
    }
    mDeviceGridMetadata = nullptr;
    setGrid(std::move(gridHdl), voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::GridBatchImpl(nanovdb::GridHandle<PytorchDeviceBuffer>&& gridHdl,
                                     const nanovdb::Vec3d& globalVoxelSize,
                                     const nanovdb::Vec3d& globalVoxelOrigin) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(), "Cannot create a batched grid handle from an empty grid handle");
    TORCH_CHECK_VALUE(globalVoxelSize[0] > 0 && globalVoxelSize[1] > 0 && globalVoxelSize[2] > 0, "Voxel size must be greater than 0");
    mDeviceGridMetadata = nullptr;
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    for(size_t i = 0; i < gridHdl.gridCount(); ++i) {
        voxelSizes.push_back(globalVoxelSize);
        voxelOrigins.push_back(globalVoxelOrigin);
    }
    setGrid(std::move(gridHdl), voxelSizes, voxelOrigins, false /* blocking */);
    mBatchMetadata.mIsContiguous = true;
};

GridBatchImpl::~GridBatchImpl() {
    mHostGridMetadata.clear();
    if (mDeviceGridMetadata != nullptr) {
        c10::cuda::CUDACachingAllocator::raw_delete(mDeviceGridMetadata);
    }
};

torch::Tensor GridBatchImpl::worldToGridMatrix(int bid) const {
    TORCH_CHECK_VALUE(bid < batchSize(), "Batch index out of range");

    torch::Tensor xformMat = torch::eye(4, torch::TensorOptions().device(device()).dtype(torch::kDouble));
    const VoxelCoordTransform& transform = primalTransform(bid);
    const nanovdb::Vec3d& scale = transform.scale<double>();
    const nanovdb::Vec3d& translate = transform.translate<double>();

    xformMat[0][0] = scale[0];
    xformMat[1][1] = scale[1];
    xformMat[2][2] = scale[2];

    xformMat[3][0] = translate[0];
    xformMat[3][1] = translate[1];
    xformMat[3][2] = translate[2];

    return xformMat;
}

void GridBatchImpl::recomputeBatchOffsets() {
    mBatchOffsets = torch::empty({batchSize(), 2}, torch::TensorOptions().dtype(torch::kInt64).device(device()));
    if (device().is_cuda()) {
        computeBatchOffsetsFromMetadata<<<1, 1>>>(batchSize(), mDeviceGridMetadata, mBatchOffsets.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto outBatchOffsets = mBatchOffsets.accessor<int64_t, 2>();
        outBatchOffsets[0][0] = 0;
        outBatchOffsets[0][1] = mHostGridMetadata[0].mNumVoxels;
        for (int i = 1; i < mHostGridMetadata.size(); i += 1) {
            outBatchOffsets[i][0] = outBatchOffsets[i-1][1];
            outBatchOffsets[i][1] = outBatchOffsets[i][0] + mHostGridMetadata[i].mNumVoxels;
        }
    }
}


torch::Tensor GridBatchImpl::gridToWorldMatrix(int bid) const {
    return torch::linalg::inv(worldToGridMatrix(bid));
}

c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::clone(torch::Device device, bool blocking) const {
    // If you're cloning an empty grid, just create a new empty grid on the right device and return it
    if (batchSize() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device, isMutable());
    }

    // The guide buffer is a hack to perform the correct copy (i.e. host -> device / device -> host etc...)
    // The guide carries the desired target device to the copy.
    // The reason we do this is to conform with the nanovdb which can only accept a buffer as an extra argument.
    PytorchDeviceBuffer guideBuffer(0, nullptr);
    guideBuffer.setDevice(device, true);

    // Make a copy of this gridHandle on the same device as the guide buffer
    nanovdb::GridHandle<PytorchDeviceBuffer> clonedHdl = mGridHdl->copy<PytorchDeviceBuffer>(guideBuffer);

    // Copy the voxel sizes and origins for this grid
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    gridVoxelSizesAndOrigins(voxelSizes, voxelOrigins);

    // Build a GridBatchImpl from the cloned grid handle and voxel sizes/origins
    // FIXME: (@fwilliams) This makes an extra copy or non contiguous grids
    return GridBatchImpl::contiguous(c10::make_intrusive<GridBatchImpl>(std::move(clonedHdl), voxelSizes, voxelOrigins));
}

void GridBatchImpl::syncMetadataToDeviceIfCUDA(bool blocking) {
    if (device().is_cuda()) { // There is something to sync and we're on a cuda device

        // We haven't allocated the cuda memory yet, so we need to do that now
        if (mDeviceGridMetadata == nullptr) {
            // We need to allocate the memory on the device
            size_t metaDataByteSize = sizeof(GridMetadata) * mHostGridMetadata.size();
            at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(device().index());
            mDeviceGridMetadata = static_cast<GridMetadata*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(metaDataByteSize, defaultStream.stream()));
        }

        // Copy host grid metadata to device buffer
        size_t metaDataByteSize = sizeof(GridMetadata) * mHostGridMetadata.size();
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(mGridHdl->buffer().device().index());
        C10_CUDA_CHECK(cudaMemcpyAsync(mDeviceGridMetadata,
                                       mHostGridMetadata.data(),
                                       metaDataByteSize,
                                       cudaMemcpyHostToDevice,
                                       defaultStream.stream()));
        // Block if you asked for it
        if (blocking) {
            C10_CUDA_CHECK(cudaStreamSynchronize(defaultStream.stream()));
        }
    }
}

void GridBatchImpl::setGlobalPrimalTransform(const VoxelCoordTransform& transform, bool syncToDevice) {
    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].mPrimalTransform = transform;
    }

    if (syncToDevice) {
        syncMetadataToDeviceIfCUDA(false);
    }
}

void GridBatchImpl::setGlobalDualTransform(const VoxelCoordTransform& transform, bool syncToDevice) {
    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].mDualTransform = transform;
    }

    if (syncToDevice) {
        syncMetadataToDeviceIfCUDA(false);
    }
}

void GridBatchImpl::setGlobalVoxelSize(const nanovdb::Vec3d& voxelSize, bool syncToDevice) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size on an empty batch of grids");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, mHostGridMetadata[i].voxelOrigin());
    }

    if (syncToDevice) {
        syncMetadataToDeviceIfCUDA(false);
    }
}

void GridBatchImpl::setGlobalVoxelOrigin(const nanovdb::Vec3d& voxelOrigin, bool syncToDevice) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel origin on an empty batch of grids");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].setTransform(mHostGridMetadata[i].mVoxelSize, voxelOrigin);
    }

    if (syncToDevice) {
        syncMetadataToDeviceIfCUDA(false);
    }
}

void GridBatchImpl::setGlobalVoxelSizeAndOrigin(const nanovdb::Vec3d& voxelSize, const nanovdb::Vec3d& voxelOrigin, bool syncToDevice) {
    TORCH_CHECK(batchSize() > 0, "Cannot set global voxel size and origin on an empty batch of grids");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].setTransform(voxelSize, voxelOrigin);
    }

    if (syncToDevice) {
        syncMetadataToDeviceIfCUDA(false);
    }
}


void GridBatchImpl::setFineTransformFromCoarseGrid(const GridBatchImpl& coarseBatch, nanovdb::Coord subdivisionFactor) {
    TORCH_CHECK(coarseBatch.batchSize() == batchSize(), "Coarse grid batch size must match fine grid batch size");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        auto sizeAndOrigin = coarseBatch.fineVoxSizeAndOrigin(i, subdivisionFactor);
        mHostGridMetadata[i].setTransform(sizeAndOrigin.first, sizeAndOrigin.second);
    }

    syncMetadataToDeviceIfCUDA(false);
}


void GridBatchImpl::setCoarseTransformFromFineGrid(const GridBatchImpl& fineBatch, nanovdb::Coord coarseningFactor) {
    TORCH_CHECK(fineBatch.batchSize() == batchSize(), "Fine grid batch size must match coarse grid batch size");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        auto sizeAndOrigin = fineBatch.coarseVoxSizeAndOrigin(i, coarseningFactor);
        mHostGridMetadata[i].setTransform(sizeAndOrigin.first, sizeAndOrigin.second);
    }

    syncMetadataToDeviceIfCUDA(false);
}


void GridBatchImpl::setPrimalTransformFromDualGrid(const GridBatchImpl& dualBatch) {
    TORCH_CHECK(dualBatch.batchSize() == batchSize(), "Dual grid batch size must match primal grid batch size");

    for (size_t i = 0; i < mHostGridMetadata.size(); i++) {
        mHostGridMetadata[i].mDualTransform = dualBatch.mHostGridMetadata[i].mPrimalTransform;
        mHostGridMetadata[i].mPrimalTransform = dualBatch.mHostGridMetadata[i].mDualTransform;
        mHostGridMetadata[i].mVoxelSize = dualBatch.mHostGridMetadata[i].mVoxelSize;
    }

    syncMetadataToDeviceIfCUDA(false);
}


void GridBatchImpl::setGrid(nanovdb::GridHandle<PytorchDeviceBuffer>&& gridHdl,
                                const std::vector<nanovdb::Vec3d>& voxelSizes,
                                const std::vector<nanovdb::Vec3d>& voxelOrigins,
                                bool blocking) {
    TORCH_CHECK(!gridHdl.buffer().isEmpty(), "Empty grid handle");
    TORCH_CHECK(voxelSizes.size() == gridHdl.gridCount(), "voxelSizes array does not have the same size as the number of grids, got ", voxelSizes.size(), " expected ", gridHdl.gridCount());
    TORCH_CHECK(voxelOrigins.size() == gridHdl.gridCount(), "Voxel origins must be the same size as the number of grids");
    TORCH_CHECK((gridHdl.gridType(0) == nanovdb::GridType::OnIndex) || (gridHdl.gridType(0) == nanovdb::GridType::OnIndexMask), "GridBatchImpl only supports ValueOnIndex and ValueOnIndexMask grids");
    const torch::Device device = gridHdl.buffer().device();

    // Clear out old grid metadata
    mHostGridMetadata.clear();
    if (mDeviceGridMetadata != nullptr) {
        c10::cuda::CUDACachingAllocator::raw_delete(mDeviceGridMetadata);
        mDeviceGridMetadata = nullptr;
    }

    // Allocate host memory for metadata
    mHostGridMetadata.resize(gridHdl.gridCount());

    FVDB_DISPATCH_KERNEL_DEVICE(device, [&]() {
        // Allocate device memory for metadata
        GridBatchMetadata* deviceBatchMetadataPtr = nullptr;
        if constexpr (DeviceTag == torch::kCUDA) {
            const size_t metaDataByteSize = sizeof(GridMetadata) * gridHdl.gridCount();
            at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(device.index());
            mDeviceGridMetadata = static_cast<GridMetadata*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(metaDataByteSize, defaultStream.stream()));
            deviceBatchMetadataPtr = static_cast<GridBatchMetadata*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(sizeof(GridBatchMetadata), defaultStream.stream()));
        }

        // Populate host and/or device metadata
        const bool isGridMutable = gridHdl.gridType(0) == nanovdb::GridType::OnIndexMask;
        ops::dispatchPopulateGridMetadata<DeviceTag>(
            gridHdl, voxelSizes, voxelOrigins, isGridMutable,
            mBatchOffsets,
            mHostGridMetadata.data(), mDeviceGridMetadata, &mBatchMetadata, deviceBatchMetadataPtr);

        // We don't need the device copy of the global batch metadata anymore (we only carry around the host version and pass it by value to device kernels), so delete it
        if constexpr (DeviceTag == torch::kCUDA) {
            c10::cuda::CUDACachingAllocator::raw_delete(deviceBatchMetadataPtr);
        }
    });


    // FIXME: This is slow
    // Populate batch offsets for each leaf node
    {
        std::vector<torch::Tensor> leafBatchIdxs;
        leafBatchIdxs.reserve(gridHdl.gridCount());
        for (size_t i = 0; i < gridHdl.gridCount(); i += 1) {
            leafBatchIdxs.push_back(
                torch::full({mHostGridMetadata[i].mNumLeaves},
                            (int16_t) i,
                            torch::TensorOptions().dtype(torch::kInt16).device(device)));
        }
        mLeafBatchIndices = torch::cat(leafBatchIdxs, 0);
    }

    // Replace the grid handle with the new one
    mGridHdl = std::make_shared<nanovdb::GridHandle<PytorchDeviceBuffer>>(std::move(gridHdl));
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::index(int32_t bid) const {
    int32_t idx = bid;
    if (bid < 0) {
        idx = batchSize() + bid;
    }
    TORCH_CHECK_INDEX(idx >= 0 && idx < batchSize(),
                      "index " + std::to_string(bid) + " is out of range for grid batch of size " +
                      std::to_string(batchSize()));

    return index(idx, idx+1, 1);
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::index(const torch::Tensor& indices) const {
    TORCH_CHECK_INDEX(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK_INDEX(!indices.is_floating_point(), "indices must be an integer tensor");

    torch::Tensor numericIndices;
    if(indices.scalar_type() == torch::kBool) {
        TORCH_CHECK_INDEX(indices.dim() == 1, "bool indices must be a 1D tensor");
        TORCH_CHECK_INDEX(indices.numel() == batchSize(), "bool indices must have the same number of entries as grids in the batch");
        numericIndices = torch::arange(batchSize(), torch::TensorOptions().dtype(torch::kInt64).device(indices.device()));
        numericIndices = numericIndices.masked_select(indices);
    } else {
        numericIndices = indices;
    }

    torch::Tensor indicesCpu = numericIndices.to(torch::kCPU).to(torch::kInt64);
    auto indicesAccessor = indicesCpu.accessor<int64_t, 1>();
    return indexInternal(indicesAccessor, indicesAccessor.size(0));
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::index(const std::vector<int64_t>& indices) const {
    return indexInternal(indices, indices.size());
}

c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::index(const std::vector<bool>& indices) const {
    std::vector<int64_t> indicesInt;
    indicesInt.reserve(indices.size());
    for (int i = 0; i < indices.size(); i += 1) {
        if (indices[i]) {
            indicesInt.push_back(i);
        }
    }

    return indexInternal(indicesInt, indicesInt.size());
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::index(ssize_t start, ssize_t stop, ssize_t step) const {
    struct RangeAccessor {
        ssize_t mStart;
        ssize_t mStop;
        ssize_t mStep;
        ssize_t mLen;

        RangeAccessor(ssize_t start, ssize_t stop, ssize_t step, ssize_t batchSize) : mStart(start), mStop(stop), mStep(step) {
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
                TORCH_CHECK_INDEX(false, "Invalid slice start=", start, ", stop=", stop, ", step=", step, " for batch size ", batchSize);
            }
        }
        size_t operator[](size_t i) const {
            return mStart + i * mStep;
        }
    };

    auto acc = RangeAccessor(start, stop, step, batchSize());
    return indexInternal(acc, acc.mLen);
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::concatenate(
        const std::vector<c10::intrusive_ptr<GridBatchImpl>>& elements) {

    TORCH_CHECK_VALUE(elements.size() > 0, "Must provide at least one grid for concatenate!")

    torch::Device device = elements[0]->device();
    bool isMutable = elements[0]->isMutable();

    std::vector<std::shared_ptr<nanovdb::GridHandle<PytorchDeviceBuffer>>> handles;
    std::vector<std::vector<int64_t>> byteSizes;
    std::vector<std::vector<int64_t>> readByteOffsets;
    std::vector<std::vector<int64_t>> writeByteOffsets;
    int64_t totalByteSize = 0;
    int64_t totalGrids = 0;
    handles.reserve(elements.size());
    byteSizes.reserve(elements.size());
    readByteOffsets.reserve(elements.size());
    writeByteOffsets.reserve(elements.size());

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;

    for (size_t i = 0; i < elements.size(); i += 1) {
        TORCH_CHECK(elements[i]->device() == device, "All grid batches must be on the same device!");
        TORCH_CHECK(elements[i]->isMutable() == isMutable, "All grid batches must have the same mutability!");

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

        for (uint64_t j = 0; j < elements[i]->batchSize(); j += 1) {
            voxelSizes.push_back(elements[i]->voxelSize(j));
            voxelOrigins.push_back(elements[i]->voxelOrigin(j));

            readByteOffsets.back().push_back(elements[i]->cumBytes(j));  // Where to start reading from in the current grid
            byteSizes.back().push_back(elements[i]->numBytes(j));        // How many bytes to read
            writeByteOffsets.back().push_back(totalByteSize);            // Where to start writing to in the concatenated grid
            totalByteSize += elements[i]->numBytes(j);
        }

    }
    if (handles.size() == 0) {
        return c10::make_intrusive<GridBatchImpl>(device, isMutable);
    }

    const bool isHost = device.is_cpu();
    PytorchDeviceBuffer buffer(totalByteSize, nullptr, isHost, device.index());

    int count = 0;
    int nonEmptyCount = 0;
    if (isHost) {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (size_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes = byteSizes[nonEmptyCount][j];

                nanovdb::GridData* dst = reinterpret_cast<nanovdb::GridData*>(buffer.data() + writeOffset);
                const uint8_t* src = elements[i]->mGridHdl->buffer().data() + readOffset;
                memcpy((void*) dst, (void*) src, numBytes);
                nanovdb::tools::updateGridCount(dst, count++, totalGrids);
            }
            nonEmptyCount += 1;
        }
    }
    else {
        for (size_t i = 0; i < elements.size(); i += 1) {
            if (elements[i]->batchSize() == 0) {
                continue;
            }

            for (size_t j = 0; j < elements[i]->batchSize(); j += 1) {
                const int64_t readOffset = readByteOffsets[nonEmptyCount][j];
                const int64_t writeOffset = writeByteOffsets[nonEmptyCount][j];
                const int64_t numBytes = byteSizes[nonEmptyCount][j];

                c10::cuda::CUDAGuard deviceGuard(device.index());
                nanovdb::GridData* dst = reinterpret_cast<nanovdb::GridData*>(buffer.deviceData() + writeOffset);
                const uint8_t* src = elements[i]->mGridHdl->buffer().deviceData() + readOffset;
                cudaMemcpyAsync((uint8_t*) dst, src, numBytes, cudaMemcpyDeviceToDevice);

                bool dirty, *d_dirty;
                cudaMallocAsync((void**)&d_dirty, sizeof(bool), 0);
                nanovdb::cuda::updateGridCount<<<1, 1>>>(dst, count++, totalGrids, d_dirty);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost);
                if (dirty) nanovdb::tools::cuda::updateChecksum(dst, nanovdb::CheckMode::Partial);
            }
            nonEmptyCount += 1;
        }
    }
    nanovdb::GridHandle<PytorchDeviceBuffer> gridHdl = nanovdb::GridHandle<PytorchDeviceBuffer>(std::move(buffer));
    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}


c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::contiguous(c10::intrusive_ptr<GridBatchImpl> input) {
    if (input->isContiguous()) {
        return input;
    }

    const int64_t totalGrids = input->batchSize();

    int64_t totalByteSize = 0;
    for (size_t i = 0; i < input->batchSize(); i += 1) {
        totalByteSize += input->numBytes(i);
    }

    const bool isHost = input->device().is_cpu();
    PytorchDeviceBuffer buffer(totalByteSize, nullptr, isHost, input->device().index());

    int64_t writeOffset = 0;
    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(input->batchSize());
    voxelOrigins.reserve(input->batchSize());

    if (isHost) {
        for (size_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSize(i));
            voxelOrigins.push_back(input->voxelOrigin(i));

            nanovdb::GridData* dst = reinterpret_cast<nanovdb::GridData*>(buffer.data() + writeOffset);
            const uint8_t* src = input->nanoGridHandle().buffer().data() + input->cumBytes(i);
            memcpy((void*) dst, (void*) src, input->numBytes(i));
            nanovdb::tools::updateGridCount(dst, i, totalGrids);
            writeOffset += input->numBytes(i);
        }

    }
    else {
        for (size_t i = 0; i < input->batchSize(); i += 1) {
            voxelSizes.push_back(input->voxelSize(i));
            voxelOrigins.push_back(input->voxelOrigin(i));

            c10::cuda::CUDAGuard deviceGuard(input->device().index());
            nanovdb::GridData* dst = reinterpret_cast<nanovdb::GridData*>(buffer.deviceData() + writeOffset);
            const uint8_t* src = input->nanoGridHandle().buffer().deviceData() + input->cumBytes(i);
            cudaMemcpyAsync((uint8_t*) dst, src, input->numBytes(i), cudaMemcpyDeviceToDevice);

            bool dirty, *d_dirty;
            cudaMallocAsync((void**)&d_dirty, sizeof(bool), 0);
            nanovdb::cuda::updateGridCount<<<1, 1>>>(dst, i, totalGrids, d_dirty);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost);
            if (dirty) nanovdb::tools::cuda::updateChecksum(dst, nanovdb::CheckMode::Partial);
            writeOffset += input->numBytes(i);
        }
    }

    return c10::make_intrusive<GridBatchImpl>(nanovdb::GridHandle<PytorchDeviceBuffer>(std::move(buffer)), voxelSizes, voxelOrigins);
}


JaggedTensor GridBatchImpl::jaggedTensor(const torch::Tensor& data, bool ignoreDisabledVoxels) const {
    checkDevice(data);
    TORCH_CHECK(data.dim() >= 1, "Data have more than one dimensions");
    if (ignoreDisabledVoxels || !isMutable()) {
        TORCH_CHECK(data.size(0) == totalVoxels(), "Data size mismatch");
    } else {
        // TODO: (@fwilliams) check data size need to call totalActiveVoxels()
    }
    return JaggedTensor::from_data_and_offsets(data, voxelOffsets(ignoreDisabledVoxels));
}


int64_t GridBatchImpl::totalEnabledVoxels(bool ignoreDisabledVoxels) const {
    if (!isMutable() || ignoreDisabledVoxels) {
        return totalVoxels();
    }
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return ops::dispatchCountEnabledVoxels<DeviceTag>(*this, -1);
    });
}


torch::Tensor GridBatchImpl::jidx(bool ignoreDisabledVoxels) const {
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        if (batchSize() == 1 || totalVoxels() == 0) {
            return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt16).device(device()));
        }
        return ops::dispatchJIdxForGrid<DeviceTag>(*this, ignoreDisabledVoxels);
    });
}


torch::Tensor GridBatchImpl::voxelOffsets(bool ignoreDisabledVoxels) const {
    if (!isMutable() || ignoreDisabledVoxels) {
        return mBatchOffsets;
    } else  {

        // FIXME: This is slow for mutable grids
        TORCH_CHECK(isMutable(), "This grid is not mutable, cannot get voxel offsets. This should never happen.");
        torch::Tensor numEnabledPerGrid = torch::empty({batchSize()}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto acc = numEnabledPerGrid.accessor<int64_t, 1>();
        for (int i = 0; i < batchSize(); i += 1) {
            acc[i] = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
                return ops::dispatchCountEnabledVoxels<DeviceTag>(*this, i);
            });
        }
        torch::Tensor cumNumEnabledVoxels = numEnabledPerGrid.cumsum(0);
        return torch::stack({cumNumEnabledVoxels - numEnabledPerGrid, cumNumEnabledVoxels}, 1).to(device());
    }
}

torch::Tensor GridBatchImpl::serialize() const {
    return serializeV0();
}

c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::deserialize(const torch::Tensor& serialized) {
    return deserializeV0(serialized);
}


torch::Tensor GridBatchImpl::serializeV0() const {
    c10::intrusive_ptr<GridBatchImpl> self = c10::intrusive_ptr<GridBatchImpl>::reclaim_copy((GridBatchImpl*) this);
    if (!device().is_cpu()) {
        self = clone(torch::kCPU, true);
    }

    int64_t numGrids = self->nanoGridHandle().gridCount();
    int64_t hdlBufSize = self->nanoGridHandle().buffer().size();

    struct V01Header {
        uint64_t magic = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    } header;

    const int64_t headerSize = sizeof(V01Header) + numGrids * sizeof(GridMetadata) + sizeof(GridBatchMetadata);
    const int64_t totalByteSize = headerSize + hdlBufSize;

    header.totalBytes = totalByteSize;
    header.numGrids = numGrids;

    torch::Tensor ret = torch::empty({totalByteSize}, torch::kInt8);
    int8_t* retPtr = ret.data_ptr<int8_t>();

    memcpy(retPtr, &header, sizeof(V01Header));
    retPtr += sizeof(V01Header);

    memcpy(retPtr, &self->mBatchMetadata, sizeof(GridBatchMetadata));
    retPtr += sizeof(GridBatchMetadata);

    memcpy(retPtr, self->mHostGridMetadata.data(), numGrids * sizeof(GridMetadata));
    retPtr += numGrids * sizeof(GridMetadata);

    memcpy(retPtr, self->nanoGridHandle().buffer().data(), hdlBufSize);
    retPtr += hdlBufSize;

    TORCH_CHECK(retPtr == (ret.data_ptr<int8_t>() + totalByteSize), "Something went wrong with serialization");

    return ret;
}

c10::intrusive_ptr<GridBatchImpl> GridBatchImpl::deserializeV0(const torch::Tensor& serialized) {
    struct V01Header {
        uint64_t magic = 0x0F0F0F0F0F0F0F0F;
        uint64_t version = 0;
        uint64_t numGrids;
        uint64_t totalBytes;
    };

    TORCH_CHECK(serialized.scalar_type() == torch::kInt8, "Serialized data must be of type int8");
    TORCH_CHECK(serialized.numel() >= sizeof(V01Header), "Serialized data is too small to be a valid grid handle");

    const int8_t* serializedPtr = serialized.data_ptr<int8_t>();

    const V01Header* header = reinterpret_cast<const V01Header*>(serializedPtr);
    TORCH_CHECK(header->magic == 0x0F0F0F0F0F0F0F0F, "Serialized data is not a valid grid handle");
    TORCH_CHECK(header->version == 0, "Serialized data is not a valid grid handle");
    TORCH_CHECK(serialized.numel() == header->totalBytes, "Serialized data is not a valid grid handle");

    const uint64_t numGrids = header->numGrids;

    const GridBatchMetadata* batchMetadata = reinterpret_cast<const GridBatchMetadata*>(serializedPtr + sizeof(V01Header));
    TORCH_CHECK(batchMetadata->version == 0, "Serialized data is not a valid grid handle");

    const GridMetadata* gridMetadata = reinterpret_cast<const GridMetadata*>(serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata));
    for (uint64_t i = 0; i < numGrids; i += 1) {
        TORCH_CHECK(gridMetadata[i].version == 0, "Serialized data is not a valid grid handle");
    }
    const int8_t* gridBuffer = serializedPtr + sizeof(V01Header) + sizeof(GridBatchMetadata) + numGrids * sizeof(GridMetadata);

    const uint64_t sizeofMetadata = sizeof(V01Header) + sizeof(GridBatchMetadata) + numGrids * sizeof(GridMetadata);
    const uint64_t sizeofGrid = header->totalBytes - sizeofMetadata;

    auto buf = PytorchDeviceBuffer(sizeofGrid, nullptr, true /* host */, -1 /* deviceIndex */);
    memcpy(buf.data(), gridBuffer, sizeofGrid);

    nanovdb::GridHandle gridHdl = nanovdb::GridHandle<PytorchDeviceBuffer>(std::move(buf));

    std::vector<nanovdb::Vec3d> voxelSizes, voxelOrigins;
    voxelSizes.reserve(numGrids);
    voxelOrigins.reserve(numGrids);
    for (uint64_t i = 0; i < numGrids; i += 1) {
        voxelSizes.push_back(gridMetadata[i].mVoxelSize);
        voxelOrigins.push_back(gridMetadata[i].voxelOrigin());
    }

    return c10::make_intrusive<GridBatchImpl>(std::move(gridHdl), voxelSizes, voxelOrigins);
}

} // namespace detail
} // namespace fvdb
