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
buildCoarseGridFromFineGridCPU(const GridBatchImpl &fineBatchHdl,
                               const nanovdb::Coord branchingFactor) {
    using IndexTree = nanovdb::NanoTree<GridType>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &fineGridHdl = fineBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(fineGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < fineGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *fineGrid = fineGridHdl.template grid<GridType>(bidx);
        if (!fineGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &fineTree = fineGrid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType>(fineTree); it.isValid(); it++) {
            const nanovdb::Coord coarseIjk =
                (it->first.asVec3d() / branchingFactor.asVec3d()).floor();
            proxyGridAccessor.setValue(coarseIjk, 1.0f);
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
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildCoarseGridFromFineGrid(bool isMutable, const GridBatchImpl &fineBatchHdl,
                            const nanovdb::Coord branchingFactor) {
    if (fineBatchHdl.device().is_cuda()) {
        JaggedTensor coords =
            ops::dispatchCoarseIJKForFineGrid<torch::kCUDA>(fineBatchHdl, branchingFactor);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildCoarseGridFromFineGridCPU<GridType>(fineBatchHdl, branchingFactor);
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
