// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/NanoVDB.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Device-side polymorphic deviceGrid(n) — same dispatch shape as
// pyHostGrid<BufferT> in PyGridHandle.h, but returns the device pointer.
// gridType(n) is read from the host-side GridData header (the handle keeps
// a host mirror), so this works whether or not the grid has been uploaded.
// Returns None if the device-side grid is null (i.e. no deviceUpload yet)
// or the BuildT is not Python-visible.
static nb::object pyDeviceGrid(nb::handle py_handle, uint32_t n)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto& handle = nb::cast<GridHandle<BufferT>&>(py_handle);
    if (n >= handle.gridCount()) return nb::none();
    switch (handle.gridType(n)) {
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)              \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)               \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#include "../BuildTypes.def"
        default:
            return nb::none();
    }
}

void defineDeviceGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    defineGridHandle<BufferT>(m, "DeviceGridHandle")
        .def(
            "__init__",
            [](GridHandle<BufferT>&                                 handle,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cpu>  cpu_t,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cuda> cuda_t) {
                assert(cpu_t.size() == cuda_t.size());
                BufferT buffer(cpu_t.size() * sizeof(uint32_t), cpu_t.data(), cuda_t.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "cpu_t"_a.noconvert(),
            "cuda_t"_a.noconvert())
        .def("deviceGrid", &pyDeviceGrid, "n"_a = 0,
             "Return the n-th device-resident grid as a typed Grid subclass "
             "selected by gridType(n), or None if the BuildT is not bound in "
             "Python or the device copy has not been uploaded yet.")
        .def(
            "deviceUpload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceUpload(nullptr, sync); }, "sync"_a = true)
        .def(
            "deviceDownload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceDownload(nullptr, sync); }, "sync"_a = true);
    // NOTE: defineGridHandleUtilities<BufferT> intentionally NOT called for
    // DeviceBuffer. Registering nanovdb.splitGrids / nanovdb.mergeGrids as a
    // second overload taking a DeviceGridHandle list conflicts with the host
    // overload because both signatures take nb::list, and nanobind's
    // overload resolution can't disambiguate by element type — it picks one
    // and the inner cast fails with std::bad_cast. The host-only utilities
    // are what the Phase 1 plan calls for; a properly typed device variant
    // (with its own name, or with strongly-typed std::vector<HandleT> args
    // and nanobind/stl/vector.h support) can land later if it's needed.
}

} // namespace pynanovdb

#endif
