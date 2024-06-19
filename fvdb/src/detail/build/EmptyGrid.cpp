#include "Build.h"

#include "detail/utils/Utils.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>


namespace fvdb {
namespace detail {
namespace build {


nanovdb::GridHandle<PytorchDeviceBuffer> buildEmptyGrid(torch::Device device, bool isMutable) {
    return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
        using ProxyGridT = nanovdb::tools::build::Grid<float>;
        auto proxyGrid = std::make_shared<ProxyGridT>(0.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, PytorchDeviceBuffer>(*proxyGrid, 0u, false, false);
        ret.buffer().setDevice(device, true /* sync */);
        return ret;
    });
}


} // namespace build
} // namespace detail
} // namespace fvdb
