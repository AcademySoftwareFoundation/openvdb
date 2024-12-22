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
buildPaddedGridFromGridWithoutBorderCPU(const GridBatchImpl &baseBatchHdl, int BMIN, int BMAX) {
    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        auto baseGridAccessor = baseGrid->getAccessor();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType>(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0   = it->first;
            bool           active = true;
            for (int di = BMIN; di <= BMAX && active; di += 1) {
                for (int dj = BMIN; dj <= BMAX && active; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX && active; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        if (ijk != ijk0) {
                            active = active && baseGridAccessor.isActive(
                                                   ijk); // if any surrounding is off, turn it off.
                        }
                    }
                }
            }
            if (active) {
                proxyGridAccessor.setValue(ijk0, 1.0f);
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().setDevice(torch::kCPU, true);
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
buildPaddedGridFromGridCPU(const GridBatchImpl &baseBatchHdl, int BMIN, int BMAX) {
    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType>(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            for (int di = BMIN; di <= BMAX; di += 1) {
                for (int dj = BMIN; dj <= BMAX; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        proxyGridAccessor.setValue(ijk, 1.0f);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().setDevice(torch::kCPU, true);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGrid(bool isMutable, const GridBatchImpl &baseBatchHdl, int bmin, int bmax,
                        bool excludeBorder) {
    if (baseBatchHdl.device().is_cuda()) {
        JaggedTensor coords;
        if (excludeBorder) {
            coords = ops::dispatchPaddedIJKForGridWithoutBorder<torch::kCUDA>(
                baseBatchHdl, nanovdb::Coord(bmin), nanovdb::Coord(bmax));
        } else {
            coords = ops::dispatchPaddedIJKForGrid<torch::kCUDA>(baseBatchHdl, nanovdb::Coord(bmin),
                                                                 nanovdb::Coord(bmax));
        }
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            if (excludeBorder) {
                return buildPaddedGridFromGridWithoutBorderCPU<GridType>(baseBatchHdl, bmin, bmax);
            } else {
                return buildPaddedGridFromGridCPU<GridType>(baseBatchHdl, bmin, bmax);
            }
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
