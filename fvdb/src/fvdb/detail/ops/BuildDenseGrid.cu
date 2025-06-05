// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

__global__ void
ijkForDense(nanovdb::Coord ijkMin, nanovdb::Coord size, TorchRAcc32<int32_t, 2> outIJKAccessor) {
    const int32_t w = size[0], h = size[1], d = size[2];
    const uint64_t tid = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                         threadIdx.x; // = x * (h * d) + y * d + z)

    if (tid >= outIJKAccessor.size(0)) {
        return;
    }

    const int64_t xi = tid / (h * d);
    const int64_t yi = (tid - xi * (h * d)) / d;
    const int64_t zi = tid - (xi * h * d) - (yi * d);

    outIJKAccessor[tid][0] = xi + ijkMin[0];
    outIJKAccessor[tid][1] = yi + ijkMin[1];
    outIJKAccessor[tid][2] = zi + ijkMin[2];
}

void
checkInputs(const torch::Device device,
            const uint32_t batchSize,
            const nanovdb::Coord &size,
            const nanovdb::Coord &ijkMin,
            const std::optional<torch::Tensor> &mask) {
    TORCH_CHECK(size[0] > 0 && size[1] > 0 && size[2] > 0,
                "Size must be greater than 0 in all dimensions");
    TORCH_CHECK((__uint128_t)size[0] * size[1] * size[2] <= std::numeric_limits<int64_t>::max(),
                "Size of dense grid exceeds the number of voxels supported by a GridBatch");
    TORCH_CHECK((__uint128_t)size[0] * size[1] * size[2] * batchSize <=
                    std::numeric_limits<int64_t>::max(),
                "Size and batch size exceed the number of voxels supported by a GridBatch");
    if (mask.has_value()) {
        TORCH_CHECK(mask.value().device() == device,
                    "Mask device must match device of dense grid to build");
        TORCH_CHECK(mask.value().dtype() == torch::kBool, "Mask must be of type bool");
        TORCH_CHECK(mask.value().dim() == 3, "Mask must be 3D");
        TORCH_CHECK(mask.value().size(0) == size[0] && mask.value().size(1) == size[1] &&
                        mask.value().size(2) == size[2],
                    "Mask must have same size as dense grid to build");
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense<torch::kCUDA>(int64_t batchSize,
                                              nanovdb::Coord ijkMin,
                                              nanovdb::Coord size,
                                              torch::Device device,
                                              const std::optional<torch::Tensor> &mask) {
    using GridType = nanovdb::ValueOnIndex;
    TORCH_CHECK(device.is_cuda(), "device must be cuda");
    TORCH_CHECK(device.has_index(), "device must have index");
    checkInputs(device, batchSize, size, ijkMin, mask);

    c10::cuda::CUDAGuard deviceGuard(device);

    const int64_t gridVolume = static_cast<int64_t>(size[0]) * size[1] * size[2];

    constexpr int NUM_THREADS = 1024;
    const int64_t NUM_BLOCKS  = GET_BLOCKS(gridVolume, NUM_THREADS);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor ijkData           = torch::empty({gridVolume, 3}, opts);

    if (NUM_BLOCKS > 0) {
        ijkForDense<<<NUM_BLOCKS, NUM_THREADS>>>(
            ijkMin, size, ijkData.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (mask.has_value()) {
        torch::Tensor maskValue = mask.value().view({-1});
        TORCH_CHECK(maskValue.device() == device, "mask must be on same device as ijkData");
        ijkData = ijkData.index({maskValue});
    }

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, device);

    TORCH_CHECK(ijkData.is_contiguous(), "ijkData must be contiguous");

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int64_t i = 0; i < batchSize; i += 1) {
        const int64_t nVoxels = ijkData.size(0);
        handles.push_back(
            nVoxels == 0
                ? createEmptyGridHandle(guide.device())
                : nanovdb::tools::cuda::voxelsToGrid<GridType, nanovdb::Coord *, TorchDeviceBuffer>(
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
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromDense<torch::kCPU>(int64_t batchSize,
                                             nanovdb::Coord ijkMin,
                                             nanovdb::Coord size,
                                             torch::Device device,
                                             const std::optional<torch::Tensor> &mask) {
    using GridType = nanovdb::ValueOnIndex;
    checkInputs(device, batchSize, size, ijkMin, mask);

    torch::TensorAccessor<bool, 3> maskAccessor(nullptr, nullptr, nullptr);
    if (mask.has_value()) {
        maskAccessor = mask.value().accessor<bool, 3>();
    }

    using ProxyGridT       = nanovdb::tools::build::Grid<float>;
    auto proxyGrid         = std::make_shared<ProxyGridT>(0.0f);
    auto proxyGridAccessor = proxyGrid->getWriteAccessor();

    for (int32_t i = 0; i < size[0]; i += 1) {
        for (int32_t j = 0; j < size[1]; j += 1) {
            for (int32_t k = 0; k < size[2]; k += 1) {
                const nanovdb::Coord ijk = ijkMin + nanovdb::Coord(i, j, k);
                if (mask.has_value()) {
                    if (maskAccessor[i][j][k] == false) {
                        continue;
                    } else {
                        proxyGridAccessor.setValue(ijk, 1.0f);
                    }
                } else {
                    proxyGridAccessor.setValue(ijk, 1.0f);
                }
            }
        }
    }

    proxyGridAccessor.merge();
    nanovdb::GridHandle<TorchDeviceBuffer> ret =
        nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
    ret.buffer().to(torch::kCPU);

    TorchDeviceBuffer guide(0, torch::kCPU);

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(batchSize);
    batchHandles.push_back(std::move(ret));
    for (uint32_t i = 1; i < batchSize; i += 1) {
        batchHandles.push_back(batchHandles[0].copy(guide));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
