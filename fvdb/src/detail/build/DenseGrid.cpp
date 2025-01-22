// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Build.h"
#include <detail/ops/Ops.h>
#include <detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace fvdb {
namespace detail {
namespace build {

template <typename GridType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildDenseGridCPU(const uint32_t batchSize, const nanovdb::Coord &size,
                  const nanovdb::Coord &ijkMin, torch::optional<torch::Tensor> mask) {
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
        nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(*proxyGrid, 0u,
                                                                                false, false);
    ret.buffer().setDevice(torch::kCPU, true /* sync */);

    TorchDeviceBuffer guide(0, nullptr);
    guide.setDevice(torch::kCPU, true);

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

nanovdb::GridHandle<TorchDeviceBuffer>
buildDenseGrid(torch::Device device, bool isMutable, const uint32_t batchSize,
               const nanovdb::Coord &size, const nanovdb::Coord &ijkMin,
               const torch::optional<torch::Tensor> &mask) {
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
    if (device.is_cuda()) {
        return ops::dispatchCreateNanoGridFromDense<torch::kCUDA>(batchSize, ijkMin, size,
                                                                  isMutable, device, mask);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildDenseGridCPU<GridType>(batchSize, size, ijkMin, mask);
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
