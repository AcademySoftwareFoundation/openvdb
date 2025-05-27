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
buildDilatedGridCPU(const GridBatchImpl &gridBatch, const int dilation) {
    using IndexTree = nanovdb::NanoTree<GridType>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl = gridBatch.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(gridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < gridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *grid = gridHdl.template grid<GridType>(bidx);
        if (!grid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &tree = grid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType, -1>(tree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk = it->first;

            for (int i = -dilation; i <= dilation; i += 1) {
                for (int j = -dilation; j <= dilation; j += 1) {
                    for (int k = -dilation; k <= dilation; k += 1) {
                        const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                        proxyGridAccessor.setValue(fineIjk, 1);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        gridHandles.push_back(std::move(ret));
    }

    if (gridHandles.size() == 1) {
        return std::move(gridHandles[0]);
    } else {
        return nanovdb::mergeGrids(gridHandles);
    }
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildDilatedGridFromGrid(bool isMutable, const GridBatchImpl &gridBatch, const int dilation) {
    if (gridBatch.device().is_cuda()) {
        return ops::dispatchDilateGrid<torch::kCUDA>(gridBatch, dilation);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(
            isMutable, [&]() { return buildDilatedGridCPU<GridType>(gridBatch, dilation); });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
