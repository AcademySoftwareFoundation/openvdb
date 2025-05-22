// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "detail/TorchDeviceBuffer.h"
#include "nanovdb/util/MorphologyHelpers.h"

#include <detail/ops/Ops.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/DilateGrid.cuh>

namespace fvdb::detail::ops {

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchDilateGrid<torch::kCUDA>(const GridBatchImpl &gridBatch, const int dilation) {
    c10::cuda::CUDAGuard deviceGuard(gridBatch.device());

    // This guide buffer is a hack to pass in a device with an index to the cudaCreateNanoGrid
    // function. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, gridBatch.device());

    // Create a grid for each batch item and store the handles
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
    for (int i = 0; i < gridBatch.batchSize(); i += 1) {
        nanovdb::GridHandle<TorchDeviceBuffer> handle;
        nanovdb::NanoGrid<nanovdb::ValueOnIndex> *grid =
            gridBatch.nanoGridHandleMut().deviceGrid<nanovdb::ValueOnIndex>(i);
        TORCH_CHECK(grid, "Grid is null");

        for (auto j = 0; j < dilation; j += 1) {
            nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex> dilateOp(grid);
            dilateOp.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
            dilateOp.setChecksum(nanovdb::CheckMode::Default);
            dilateOp.setVerbose(0);

            handle = dilateOp.getHandle(guide);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
        }

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

} // namespace fvdb::detail::ops
