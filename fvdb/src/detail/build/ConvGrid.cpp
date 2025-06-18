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

template <typename GridType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildConvGridFromGridCPU(const GridBatchImpl &baseBatchHdl, const nanovdb::Coord &kernelSize,
                         const nanovdb::Coord &stride) {
    if (stride == nanovdb::Coord(1) || stride == kernelSize) {
        return buildCoarseGridFromFineGridCPU<GridType>(baseBatchHdl, stride);
    }

    const nanovdb::GridHandle<TorchDeviceBuffer>       &baseGridHdl = baseBatchHdl.nanoGridHandle();
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());

    int lower[3], upper[3];
    for (int i = 0; i < 3; i += 1) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0;
            upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType>(baseGrid->tree()); it.isValid(); it++) {
            const nanovdb::Coord &ijk0 = it->first;

            for (int di = lower[0]; di <= upper[0]; di += 1) {
                for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
                    for (int dk = lower[2]; dk <= upper[2]; dk += 1) {
                        const nanovdb::Coord dstIjk = ijk0 + nanovdb::Coord(dk, dj, di);
                        if (dstIjk[0] % stride[2] != 0 || dstIjk[1] % stride[1] != 0 ||
                            dstIjk[2] % stride[0] != 0)
                            continue;
                        proxyGridAccessor.setValue(nanovdb::Coord(dstIjk[0] / stride[2],
                                                                  dstIjk[1] / stride[1],
                                                                  dstIjk[2] / stride[0]),
                                                   1.0f);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildConvGridFromGrid(bool isMutable, const GridBatchImpl &baseGridHdl,
                      const nanovdb::Coord &kernelSize, const nanovdb::Coord &stride) {
    /**
     * Logic for building the conv grid is the same as torchsparse 2.0.0b.
     *  However, torchsparse has a bug that creates excessive voxels in the void space, it is fixed
     * in a customized branch - hence the additional URL for pre-built wheels.
     */

    if (baseGridHdl.device().is_cuda()) {
        JaggedTensor coords =
            ops::dispatchConvIJKForGrid<torch::kCUDA>(baseGridHdl, kernelSize, stride);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildConvGridFromGridCPU<GridType>(baseGridHdl, kernelSize, stride);
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
