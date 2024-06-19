#include "Build.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "detail/utils/Utils.h"
#include "detail/ops/Ops.h"


namespace fvdb {
namespace detail {
namespace build {


template <typename GridType>
nanovdb::GridHandle<PytorchDeviceBuffer> buildFineGridFromCoarseGridCPU(const GridBatchImpl& coarseBatchHdl,
                                                                        const torch::Tensor& subdivMask,
                                                                        const nanovdb::Coord subdivisionFactor) {

    using IndexTree = nanovdb::NanoTree<GridType>;

    const nanovdb::GridHandle<PytorchDeviceBuffer>& coarseGridHdl = coarseBatchHdl.nanoGridHandle();
    const torch::TensorAccessor<bool, 1>& subdivMaskAcc = subdivMask.accessor<bool, 1>();

    std::vector<nanovdb::GridHandle<PytorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < coarseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType>* coarseGrid = coarseGridHdl.template grid<GridType>(bidx);
        if (!coarseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree& coarseTree = coarseGrid->tree();

        using ProxyGridT = nanovdb::tools::build::Grid<float>;
        auto proxyGrid = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator<GridType, -1>(coarseTree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor[0],
                                         it->first[1] * subdivisionFactor[1],
                                         it->first[2] * subdivisionFactor[2]);

            if (subdivMaskAcc.size(0) > 0 && !subdivMaskAcc[it->second]) {
                continue;
            }

            for (int i = 0; i < subdivisionFactor[0]; i += 1) {
                for (int j = 0; j < subdivisionFactor[1]; j += 1) {
                    for (int k = 0; k < subdivisionFactor[2]; k += 1) {
                        const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                        proxyGridAccessor.setValue(fineIjk, 1);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, PytorchDeviceBuffer>(*proxyGrid, 0u, false, false);
        ret.buffer().setDevice(torch::kCPU, true);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}


nanovdb::GridHandle<PytorchDeviceBuffer> buildFineGridFromCoarseGrid(bool isMutable,
                                                                     const GridBatchImpl& coarseBatchHdl,
                                                                     const torch::optional<JaggedTensor>& subdivMask,
                                                                     const nanovdb::Coord subdivisionFactor) {

    if (coarseBatchHdl.device().is_cuda()) {
        JaggedTensor coords = ops::dispatchFineIJKForCoarseGrid<torch::kCUDA>(coarseBatchHdl, subdivisionFactor, subdivMask);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        torch::Tensor subdivMaskTensor;
        if (subdivMask.has_value()) {
            subdivMaskTensor = subdivMask.value().jdata();
        } else {
            subdivMaskTensor = torch::zeros(0, torch::TensorOptions().dtype(torch::kBool));
        }
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildFineGridFromCoarseGridCPU<GridType>(coarseBatchHdl, subdivMaskTensor, subdivisionFactor);
        });
    }
}


} // namespace build
} // namespace detail
} // namespace fvdb
