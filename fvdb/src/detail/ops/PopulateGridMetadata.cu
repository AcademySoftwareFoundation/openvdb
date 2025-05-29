// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/GridBatchImpl.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

template <template <typename T, int> typename TensorAccessorT>
__hostdev__ void
populateGridMetadataKernel(uint32_t numGrids,
                           const nanovdb::OnIndexGrid *grids,
                           const nanovdb::Vec3d *voxelSizes,
                           const nanovdb::Vec3d *voxelOrigins,
                           TensorAccessorT<fvdb::JOffsetsType, 1> gridOffsets,
                           GridBatchImpl::GridMetadata *perGridMetadata,
                           GridBatchImpl::GridBatchMetadata *batchMetadata) {
    batchMetadata->mMaxVoxels    = 0;
    batchMetadata->mMaxLeafCount = 0;

    nanovdb::Coord bbMin = nanovdb::Coord::max();
    nanovdb::Coord bbMax = nanovdb::Coord::min();

    nanovdb::OnIndexGrid *currentGrid = (nanovdb::OnIndexGrid *)&grids[0];
    uint32_t i                        = 0;
    uint64_t byteCount                = 0;

    perGridMetadata[i].mCumVoxels = 0;
    perGridMetadata[i].mCumBytes  = 0;
    perGridMetadata[i].mCumLeaves = 0;

    gridOffsets[i] = 0;
    while (i < numGrids - 1) {
        byteCount                 = currentGrid->gridSize();
        const uint32_t leafCount  = currentGrid->tree().nodeCount(0);
        const uint64_t voxelCount = currentGrid->tree().activeVoxelCount();

        GridBatchImpl::GridMetadata &metaCur  = perGridMetadata[i];
        GridBatchImpl::GridMetadata &metaNext = perGridMetadata[i + 1];

        metaCur.setTransform(voxelSizes[i], voxelOrigins[i]);
        metaCur.mNumVoxels = voxelCount;
        metaCur.mNumBytes  = byteCount;
        metaCur.mNumLeaves = leafCount;
        metaCur.mBBox      = currentGrid->tree().bbox();

        metaNext.mCumVoxels = metaCur.mCumVoxels + voxelCount;
        metaNext.mCumBytes  = metaCur.mCumBytes + byteCount;
        metaNext.mCumLeaves = metaCur.mCumLeaves + leafCount;

        gridOffsets[i + 1] = metaCur.mCumVoxels + metaCur.mNumVoxels;

        // number of voxels exceeds maximum indexable value
        assert(voxelCount <= std::numeric_limits<int64_t>::max());
        batchMetadata->mMaxVoxels =
            max(batchMetadata->mMaxVoxels, static_cast<int64_t>(voxelCount));
        batchMetadata->mMaxLeafCount = max(batchMetadata->mMaxLeafCount, leafCount);

        bbMin       = bbMin.minComponent(currentGrid->tree().bbox().min());
        bbMax       = bbMax.maxComponent(currentGrid->tree().bbox().max());
        currentGrid = (nanovdb::OnIndexGrid *)(((uint8_t *)currentGrid) + byteCount);
        i += 1;
    }

    perGridMetadata[i].setTransform(voxelSizes[i], voxelOrigins[i]);
    perGridMetadata[i].mNumVoxels = currentGrid->tree().activeVoxelCount();
    perGridMetadata[i].mNumBytes  = currentGrid->gridSize();
    perGridMetadata[i].mNumLeaves = currentGrid->tree().nodeCount(0);
    perGridMetadata[i].mBBox      = currentGrid->tree().bbox();

    gridOffsets[i + 1] = perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels;

    batchMetadata->mMaxVoxels    = max(batchMetadata->mMaxVoxels, perGridMetadata[i].mNumVoxels);
    batchMetadata->mMaxLeafCount = max(batchMetadata->mMaxLeafCount, perGridMetadata[i].mNumLeaves);

    // number of voxels exceeds maximum indexable value
    assert(perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels <=
           std::numeric_limits<int64_t>::max());
    batchMetadata->mTotalVoxels = perGridMetadata[i].mCumVoxels + perGridMetadata[i].mNumVoxels;

    // number of grid leaf nodes exceeds maximum indexable value
    assert(perGridMetadata[i].mCumLeaves + perGridMetadata[i].mNumLeaves <=
           std::numeric_limits<int64_t>::max());
    batchMetadata->mTotalLeaves = perGridMetadata[i].mCumLeaves + perGridMetadata[i].mNumLeaves;

    bbMin                     = bbMin.minComponent(currentGrid->tree().bbox().min());
    bbMax                     = bbMax.maxComponent(currentGrid->tree().bbox().max());
    batchMetadata->mTotalBBox = nanovdb::CoordBBox(bbMin, bbMax);
}

template <template <typename T, int I> typename TensorAccessorT>
__global__ void
populateGridMetadataCUDA(uint32_t numGrids,
                         const nanovdb::OnIndexGrid *grids,
                         const nanovdb::Vec3d *voxelSizes,
                         const nanovdb::Vec3d *voxelOrigins,
                         TensorAccessorT<fvdb::JOffsetsType, 1> outBatchOffsets,
                         GridBatchImpl::GridMetadata *perGridMetadata,
                         GridBatchImpl::GridBatchMetadata *batchMetadata) {
    populateGridMetadataKernel<TensorAccessorT>(
        numGrids, grids, voxelSizes, voxelOrigins, outBatchOffsets, perGridMetadata, batchMetadata);
}

template <>
void
dispatchPopulateGridMetadata<torch::kCUDA>(
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchImpl::GridMetadata *outPerGridMetadataHost,
    GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice) {
    c10::cuda::CUDAGuard deviceGuard(gridHdl.buffer().device());

    // Copy sizes and origins to device buffers
    RAIIRawDeviceBuffer<nanovdb::Vec3d> deviceVoxSizes(voxelSizes.size(),
                                                       gridHdl.buffer().device());
    deviceVoxSizes.setData((nanovdb::Vec3d *)voxelSizes.data(), true /* blocking */);
    const nanovdb::Vec3d *deviceVoxSizesPtr = deviceVoxSizes.devicePtr;

    RAIIRawDeviceBuffer<nanovdb::Vec3d> deviceVoxOrigins(voxelOrigins.size(),
                                                         gridHdl.buffer().device());
    deviceVoxOrigins.setData((nanovdb::Vec3d *)voxelOrigins.data(), true /* blocking */);
    const nanovdb::Vec3d *deviceVoxOriginsPtr = deviceVoxOrigins.devicePtr;

    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(gridHdl.buffer().device()));

    // Read metadata into device buffers
    TORCH_CHECK(gridHdl.deviceData() != nullptr, "GridHandle is empty");
    const nanovdb::OnIndexGrid *grids = (nanovdb::OnIndexGrid *)gridHdl.deviceData();
    populateGridMetadataCUDA<TorchRAcc32><<<1, 1>>>(
        gridHdl.gridCount(),
        grids,
        (const nanovdb::Vec3d *)deviceVoxSizesPtr,
        (const nanovdb::Vec3d *)deviceVoxOriginsPtr,
        outBatchOffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        outPerGridMetadataDevice,
        outBatchMetadataDevice);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const size_t metaDataByteSize = sizeof(GridBatchImpl::GridMetadata) * gridHdl.gridCount();
    cudaMemcpy(
        outPerGridMetadataHost, outPerGridMetadataDevice, metaDataByteSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outBatchMetadataHost,
               outBatchMetadataDevice,
               sizeof(GridBatchImpl::GridBatchMetadata),
               cudaMemcpyDeviceToHost);
}

template <>
void
dispatchPopulateGridMetadata<torch::kCPU>(
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchImpl::GridMetadata *outPerGridMetadataHost,
    GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice) {
    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(gridHdl.buffer().device()));
    TORCH_CHECK(gridHdl.data() != nullptr, "GridHandle is empty");
    const nanovdb::OnIndexGrid *grids = (nanovdb::OnIndexGrid *)gridHdl.data();
    populateGridMetadataKernel<TorchAcc>(gridHdl.gridCount(),
                                         grids,
                                         voxelSizes.data(),
                                         voxelOrigins.data(),
                                         outBatchOffsets.accessor<fvdb::JOffsetsType, 1>(),
                                         outPerGridMetadataHost,
                                         outBatchMetadataHost);
}

template <>
void
dispatchPopulateGridMetadata<torch::kPrivateUse1>(
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    torch::Tensor &outBatchOffsets,
    GridBatchImpl::GridMetadata *outPerGridMetadataHost,
    GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice) {
    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(gridHdl.buffer().device()));
    TORCH_CHECK(gridHdl.data() != nullptr, "GridHandle is empty");
    const nanovdb::OnIndexGrid *grids = (nanovdb::OnIndexGrid *)gridHdl.data();
    populateGridMetadataKernel<TorchAcc>(gridHdl.gridCount(),
                                         grids,
                                         voxelSizes.data(),
                                         voxelOrigins.data(),
                                         outBatchOffsets.accessor<fvdb::JOffsetsType, 1>(),
                                         outPerGridMetadataHost,
                                         outBatchMetadataHost);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
