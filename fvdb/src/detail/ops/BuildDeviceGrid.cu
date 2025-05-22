// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/GridBatchImpl.h>
#include <detail/build/Build.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>
#include <detail/utils/cuda/Utils.cuh>

#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/device_vector.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType, template <typename T, int> typename TensorAccessorT>
__hostdev__ void
populateGridMetadataKernel(uint32_t numGrids,
                           const nanovdb::NanoGrid<GridType> *grids,
                           const nanovdb::Vec3d *voxelSizes,
                           const nanovdb::Vec3d *voxelOrigins,
                           TensorAccessorT<fvdb::JOffsetsType, 1> gridOffsets,
                           GridBatchImpl::GridMetadata *perGridMetadata,
                           GridBatchImpl::GridBatchMetadata *batchMetadata) {
    batchMetadata->mMaxVoxels    = 0;
    batchMetadata->mMaxLeafCount = 0;

    batchMetadata->mIsMutable = nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value;

    nanovdb::Coord bbMin = nanovdb::Coord::max();
    nanovdb::Coord bbMax = nanovdb::Coord::min();

    nanovdb::NanoGrid<GridType> *currentGrid = (nanovdb::NanoGrid<GridType> *)&grids[0];
    uint32_t i                               = 0;
    uint64_t byteCount                       = 0;

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
        currentGrid = (nanovdb::NanoGrid<GridType> *)(((uint8_t *)currentGrid) + byteCount);
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

    batchMetadata->mIsMutable = nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value;
}

template <typename GridType, template <typename T, int I> typename TensorAccessorT>
__global__ void
populateGridMetadataCUDA(uint32_t numGrids,
                         const nanovdb::NanoGrid<GridType> *grids,
                         const nanovdb::Vec3d *voxelSizes,
                         const nanovdb::Vec3d *voxelOrigins,
                         TensorAccessorT<fvdb::JOffsetsType, 1> outBatchOffsets,
                         GridBatchImpl::GridMetadata *perGridMetadata,
                         GridBatchImpl::GridBatchMetadata *batchMetadata) {
    populateGridMetadataKernel<GridType, TensorAccessorT>(
        numGrids, grids, voxelSizes, voxelOrigins, outBatchOffsets, perGridMetadata, batchMetadata);
}

__global__ void
ijkForDense(nanovdb::Coord origin, nanovdb::Coord size, TorchRAcc32<int32_t, 2> outIJKAccessor) {
    const int32_t w = size[0], h = size[1], d = size[2];
    const uint64_t tid = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                         threadIdx.x; // = x * (h * d) + y * d + z)

    if (tid >= outIJKAccessor.size(0)) {
        return;
    }

    const int64_t xi = tid / (h * d);
    const int64_t yi = (tid - xi * (h * d)) / d;
    const int64_t zi = tid - (xi * h * d) - (yi * d);

    outIJKAccessor[tid][0] = xi + origin[0];
    outIJKAccessor[tid][1] = yi + origin[1];
    outIJKAccessor[tid][2] = zi + origin[2];
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kCUDA>(const JaggedTensor &ijk, bool isMutable) {
    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_cuda(), "device must be cuda");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32, "ijk must be int32");

    c10::cuda::CUDAGuard deviceGuard(ijk.device());

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

    nanovdb::GridHandle<TorchDeviceBuffer> ret = FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
        // function. We can't pass in a device directly but we can pass in a buffer which gets
        // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
        // passes it to the created buffer.
        TorchDeviceBuffer guide(0, ijk.device());

        // FIXME: This is slow because we have to copy this data to the host and then build the
        // grids. Ideally we want to do this in a single invocation.
        torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
        auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
        torch::Tensor ijkData          = ijk.jdata();
        TORCH_CHECK(ijkData.is_contiguous(), "ijk must be contiguous");
        TORCH_CHECK(ijkData.dim() == 2, "ijk must have shape (N, 3)");
        TORCH_CHECK(ijkData.size(1) == 3, "ijk must have shape (N, 3)");

        // Create a grid for each batch item and store the handles
        std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
        for (int i = 0; i < (ijkBOffset.size(0) - 1); i += 1) {
            const int64_t startIdx = ijkBOffset[i];
            const int64_t nVoxels  = ijkBOffset[i + 1] - startIdx;
            // torch::Tensor ijkDataSlice = ijkData.narrow(0, startIdx, nVoxels);
            const int32_t *dataPtr = ijkData.data_ptr<int32_t>() + 3 * startIdx;

            handles.push_back(nVoxels == 0 ? build::buildEmptyGrid(guide.device(), isMutable)
                                           : nanovdb::tools::cuda::voxelsToGrid<GridType,
                                                                                nanovdb::Coord *,
                                                                                TorchDeviceBuffer>(
                                                 (nanovdb::Coord *)dataPtr, nVoxels, 1.0, guide));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        if (handles.size() == 1) {
            // If there's only one handle, just return it
            return std::move(handles[0]);
        } else {
            // This copies all the handles into a single handle -- only do it if there are multie
            // grids
            return nanovdb::cuda::mergeGridHandles(handles, &guide);
        }
    });

    return ret;
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense<torch::kCUDA>(uint32_t batchSize,
                                              nanovdb::Coord origin,
                                              nanovdb::Coord size,
                                              bool isMutable,
                                              torch::Device device,
                                              const std::optional<torch::Tensor> &maybeMask) {
    TORCH_CHECK(device.is_cuda(), "device must be cuda");
    TORCH_CHECK(device.has_index(), "device must have index");

    c10::cuda::CUDAGuard deviceGuard(device);

    const int64_t gridVolume = static_cast<int64_t>(size[0]) * size[1] * size[2];

    constexpr int NUM_THREADS = 1024;
    const int64_t NUM_BLOCKS  = GET_BLOCKS(gridVolume, NUM_THREADS);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor ijkData           = torch::empty({gridVolume, 3}, opts);

    if (NUM_BLOCKS > 0) {
        ijkForDense<<<NUM_BLOCKS, NUM_THREADS>>>(
            origin, size, ijkData.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (maybeMask.has_value()) {
        torch::Tensor mask = maybeMask.value().view({-1});
        TORCH_CHECK(mask.device() == device, "mask must be on same device as ijkData");
        ijkData = ijkData.index({mask});
    }

    nanovdb::GridHandle<TorchDeviceBuffer> ret = FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
        // function. We can't pass in a device directly but we can pass in a buffer which gets
        // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
        // passes it to the created buffer.
        TorchDeviceBuffer guide(0, device);

        TORCH_CHECK(ijkData.is_contiguous(), "ijkData must be contiguous");

        // Create a grid for each batch item and store the handles
        std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
        for (int i = 0; i < batchSize; i += 1) {
            const int64_t nVoxels = ijkData.size(0);
            handles.push_back(
                nVoxels == 0 ? build::buildEmptyGrid(guide.device(), isMutable)
                             : nanovdb::tools::cuda::
                                   voxelsToGrid<GridType, nanovdb::Coord *, TorchDeviceBuffer>(
                                       (nanovdb::Coord *)ijkData.data_ptr(), nVoxels, 1.0, guide));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        if (handles.size() == 1) {
            // If there's only one handle, just return it
            return std::move(handles[0]);
        } else {
            // This copies all the handles into a single handle -- only do it if there are multie
            // grids
            return nanovdb::cuda::mergeGridHandles(handles, &guide);
        }
    });

    return ret;
}

template <>
void
dispatchPopulateGridMetadata<torch::kCUDA>(
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl,
    const std::vector<nanovdb::Vec3d> &voxelSizes,
    const std::vector<nanovdb::Vec3d> &voxelOrigins,
    const bool isMutable,
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
    FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        TORCH_CHECK(gridHdl.deviceData() != nullptr, "GridHandle is empty");
        const nanovdb::NanoGrid<GridType> *grids =
            (nanovdb::NanoGrid<GridType> *)gridHdl.deviceData();
        populateGridMetadataCUDA<GridType, TorchRAcc32><<<1, 1>>>(
            gridHdl.gridCount(),
            grids,
            (const nanovdb::Vec3d *)deviceVoxSizesPtr,
            (const nanovdb::Vec3d *)deviceVoxOriginsPtr,
            outBatchOffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
            outPerGridMetadataDevice,
            outBatchMetadataDevice);
    });
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
    const bool isMutable,
    torch::Tensor &outBatchOffsets,
    GridBatchImpl::GridMetadata *outPerGridMetadataHost,
    GridBatchImpl::GridMetadata *outPerGridMetadataDevice,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataHost,
    GridBatchImpl::GridBatchMetadata *outBatchMetadataDevice) {
    outBatchOffsets = torch::empty(
        {(fvdb::JOffsetsType)(voxelOrigins.size() + 1)},
        torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(gridHdl.buffer().device()));
    FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        TORCH_CHECK(gridHdl.data() != nullptr, "GridHandle is empty");
        const nanovdb::NanoGrid<GridType> *grids = (nanovdb::NanoGrid<GridType> *)gridHdl.data();
        populateGridMetadataKernel<GridType, TorchAcc>(
            gridHdl.gridCount(),
            grids,
            voxelSizes.data(),
            voxelOrigins.data(),
            outBatchOffsets.accessor<fvdb::JOffsetsType, 1>(),
            outPerGridMetadataHost,
            outBatchMetadataHost);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
