// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Build.h"

#include <detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace fvdb {
namespace detail {
namespace build {

nanovdb::GridHandle<TorchDeviceBuffer>
buildEmptyGrid(torch::Device device, bool isMutable) {
    return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(0.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().setDevice(device, true /* sync */);
        return ret;
    });
}

} // namespace build
} // namespace detail
} // namespace fvdb
