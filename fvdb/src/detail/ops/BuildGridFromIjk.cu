// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/GridBatchImpl.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/CreateEmptyGrid.h>
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>

#define CCCL_INCLUSIVE_SUM_INIT_SUPPORTED (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 8)

#if CCCL_INCLUSIVE_SUM_INIT_SUPPORTED
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>
#else
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/csrc/api/include/torch/types.h>

#include <thrust/device_vector.h>

namespace fvdb {
namespace detail {
namespace ops {

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kCUDA>(const JaggedTensor &ijk) {
    using GridType = nanovdb::ValueOnIndex;

    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_cuda(), "device must be cuda");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32 || ijk.scalar_type() == torch::kInt64,
                "ijk must be int32 or int64");

    c10::cuda::CUDAGuard deviceGuard(ijk.device());

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

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
    if (ijkData.scalar_type() != torch::kInt32) {
        ijkData = ijkData.to(torch::kInt32);
    }
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

        handles.push_back(
            nVoxels == 0
                ? createEmptyGrid(guide.device())
                : nanovdb::tools::cuda::voxelsToGrid<GridType, nanovdb::Coord *, TorchDeviceBuffer>(
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
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kPrivateUse1>(const JaggedTensor &ijk) {
    using GridType = nanovdb::ValueOnIndex;

#if CCCL_INCLUSIVE_SUM_INIT_SUPPORTED
    TORCH_CHECK(ijk.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijk.device().is_privateuseone(), "device must be privateuseone");
    TORCH_CHECK(ijk.device().has_index(), "device must have index");
    TORCH_CHECK(ijk.scalar_type() == torch::kInt32, "ijk must be int32");

    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t), "nanovdb::Coord must be 3 ints");

    // This guide buffer is a hack to pass in a device with an index to the
    // cudaCreateNanoGrid function. We can't pass in a device directly but we can pass in a
    // buffer which gets passed to TorchDeviceBuffer::create. The guide buffer holds the
    // device and effectively passes it to the created buffer.
    TorchDeviceBuffer guide(0, ijk.device());

    // FIXME: This is slow because we have to copy this data to the host and then build the
    // grids. Ideally we want to do this in a single invocation.
    torch::Tensor ijkBOffsetTensor = ijk.joffsets().cpu();
    auto ijkBOffset                = ijkBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();
    torch::Tensor ijkData          = ijk.jdata();
    TORCH_CHECK(ijkData.is_contiguous(), "ijk must be contiguous");
    TORCH_CHECK(ijkData.dim() == 2, "ijk must have shape (N, 3)");
    TORCH_CHECK(ijkData.size(1) == 3, "ijk must have shape (N, 3)");

    for (const auto device_index: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(device_index).synchronize();
    }

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < (ijkBOffset.size(0) - 1); i += 1) {
        const int64_t startIdx = ijkBOffset[i];
        const int64_t nVoxels  = ijkBOffset[i + 1] - startIdx;

        if (!nVoxels) {
            auto handle = createEmptyGrid(ijk.device());
            handles.emplace_back(std::move(handle));
        } else {
            int32_t *dataPtr = ijkData.data_ptr<int32_t>() + ijkData.stride(0) * startIdx;
            auto coordPtr    = reinterpret_cast<nanovdb::Coord *>(dataPtr);

            nanovdb::cuda::DeviceMesh mesh;
            nanovdb::tools::cuda::DistributedPointsToGrid<GridType> converter(mesh);
            auto handle =
                converter.getHandle<nanovdb::Coord *, TorchDeviceBuffer>(coordPtr, nVoxels, guide);
            handles.emplace_back(std::move(handle));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    for (const auto device_index: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(device_index).synchronize();
    }

    if (handles.size() == 1) {
        // If there's only one handle, just return it
        return std::move(handles[0]);
    } else {
        // This copies all the handles into a single handle -- only do it if there are multiple
        // grids
        return nanovdb::cuda::mergeGridHandles(handles, &guide);
    }
#else
    TORCH_CHECK(false, "Distributed creation of grids requires CUDA 12.8 or later");
    return nanovdb::GridHandle<TorchDeviceBuffer>();
#endif
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchCreateNanoGridFromIJK<torch::kCPU>(const JaggedTensor &jaggedCoords) {
    using GridType = nanovdb::ValueOnIndex;

    return AT_DISPATCH_V2(
        jaggedCoords.scalar_type(),
        "buildPaddedGridFromCoords",
        AT_WRAP([&]() {
            using ScalarT = scalar_t;
            jaggedCoords.check_valid();

            static_assert(std::is_integral<ScalarT>::value,
                          "Invalid type for coords, must be integral");

            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            const torch::TensorAccessor<ScalarT, 2> &coordsAcc =
                jaggedCoords.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &coordsBOffsetsAcc =
                jaggedCoords.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(coordsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (coordsBOffsetsAcc.size(0) - 1); bi += 1) {
                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = coordsBOffsetsAcc[bi];
                const int64_t end   = coordsBOffsetsAcc[bi + 1];

                for (unsigned ci = start; ci < end; ci += 1) {
                    nanovdb::Coord ijk(coordsAcc[ci][0], coordsAcc[ci][1], coordsAcc[ci][2]);
                    proxyGridAccessor.setValue(ijk, 11);
                }

                proxyGridAccessor.merge();
                auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
                    *proxyGrid, 0u, false, false);
                ret.buffer().to(torch::kCPU);
                batchHandles.push_back(std::move(ret));
            }

            if (batchHandles.size() == 1) {
                return std::move(batchHandles[0]);
            } else {
                return nanovdb::mergeGrids(batchHandles);
            }
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));
}

} // namespace ops
} // namespace detail
} // namespace fvdb
