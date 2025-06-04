// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/TorchDeviceBuffer.h>
#include <fvdb/detail/ops/Ops.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/util/MorphologyHelpers.h>

namespace fvdb::detail::ops {

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchMergeGrids<torch::kCUDA>(const GridBatchImpl &gridBatch1, const GridBatchImpl &gridBatch2) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch1.device());
    TORCH_CHECK_VALUE(gridBatch1.device() == gridBatch2.device(),
                      "All arguments to MergeGrids must be on the same device");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, gridBatch1.device());

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < gridBatch1.batchSize(); i += 1) {
        nanovdb::OnIndexGrid *grid1 =
            gridBatch1.nanoGridHandleMut().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid1, "First Grid is null");
        nanovdb::OnIndexGrid *grid2 =
            gridBatch2.nanoGridHandleMut().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid2, "Second Grid is null");

        nanovdb::tools::cuda::MergeGrids<nanovdb::ValueOnIndex> mergeOp(grid1, grid2);
        mergeOp.setChecksum(nanovdb::CheckMode::Default);
        mergeOp.setVerbose(0);

        auto handle = mergeOp.getHandle(guide);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        handles.push_back(std::move(handle));
    }

    if (handles.size() == 1) {
        // If there's only one handle, just return it
        return std::move(handles[0]);
    } else {
        // This copies all the handles into a single handle -- only do it if there are multiple
        // grids
        return nanovdb::cuda::mergeGridHandles(handles, &guide);
    }
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchMergeGrids<torch::kCPU>(const GridBatchImpl &gridBatch1, const GridBatchImpl &gridBatch2) {
    using GridType  = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridType>;
    TORCH_CHECK(gridBatch1.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK(gridBatch2.device().is_cpu(), "All arguments to MergeGrids must be on the CPU");
    TORCH_CHECK_VALUE(gridBatch1.batchSize() == gridBatch2.batchSize(),
                      "GridBatches to merge should have the same batch size");

    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl1 = gridBatch1.nanoGridHandle();
    const nanovdb::GridHandle<TorchDeviceBuffer> &gridHdl2 = gridBatch2.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> gridHandles;
    gridHandles.reserve(gridHdl1.gridCount());
    for (uint32_t bidx = 0; bidx < gridHdl1.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *grid1 = gridHdl1.template grid<GridType>(bidx);
        const nanovdb::OnIndexGrid *grid2 = gridHdl2.template grid<GridType>(bidx);
        TORCH_CHECK(grid1, "Failed to get pointer to nanovdb index grid (first argument to merge)");
        TORCH_CHECK(grid2,
                    "Failed to get pointer to nanovdb index grid (second argument to merge)");
        const IndexTree &tree1 = grid1->tree();
        const IndexTree &tree2 = grid2->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int64_t joffset = gridBatch1.cumVoxels(bidx);
        for (auto it = ActiveVoxelIterator<-1>(tree1); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
        }
        for (auto it = ActiveVoxelIterator<-1>(tree2); it.isValid(); it++) {
            proxyGridAccessor.setValue(it->first, 1);
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

} // namespace fvdb::detail::ops
