// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H
#define FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H

#include <fvdb/detail/TorchDeviceBuffer.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace fvdb {
namespace detail {

inline nanovdb::GridHandle<TorchDeviceBuffer>
createEmptyGridHandle(torch::Device device) {
    using GridType         = nanovdb::ValueOnIndex;
    using ProxyGridT       = nanovdb::tools::build::Grid<float>;
    auto proxyGrid         = std::make_shared<ProxyGridT>(0.0f);
    auto proxyGridAccessor = proxyGrid->getWriteAccessor();

    proxyGridAccessor.merge();
    auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
        *proxyGrid, 0u, false, false);
    ret.buffer().to(device);
    return ret;
}

} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_NANOVDB_CREATEEMPTYGRIDHANDLE_H
