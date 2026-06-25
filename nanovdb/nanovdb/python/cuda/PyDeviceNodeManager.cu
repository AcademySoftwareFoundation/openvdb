// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyTree.h"

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/NodeManager.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/NodeManager.cuh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Device-side polymorphic mgr() — same dispatch shape as pyNodeMgr<BufferT> in
// PyTree.cc, but probes the DEVICE-resident NodeManager pointer
// (handle.deviceMgr<T>()) instead of the host one. The per-BuildT
// NanoVDB NodeManager<T> classes registered on the root module are reused
// as-is; nanobind does not distinguish a host vs device `this` pointer — the
// returned object is a NodeManager<T> whose underlying address lives in device
// memory, so its accessors (leaf(i)/lower(i)/upper(i)/...) must only be used
// from CUDA kernels, never dereferenced on the host.
static nb::object pyDeviceNodeMgr(nb::handle py_self)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    using HandleT = NodeManagerHandle<BufferT>;
    auto& handle = nb::cast<HandleT&>(py_self);
    // cuda::createNodeManager builds a DEVICE-only NodeManager: the buffer's
    // host mirror (handle.data()) stays null while the device side is
    // populated. Gate on size() (the allocated NodeManagerData byte count),
    // NOT data(), so the device handle is observable.
    if (handle.size() == 0) return nb::none();
    // The stored gridType is private; deviceMgr<BuildT>() returns NULL on a
    // type mismatch, so iterate by BuildT (first non-null wins). The X-macro
    // produces one case per bound BuildT.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
    if (auto* m = handle.template deviceMgr<T>()) {                           \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    if (auto* m = handle.template deviceMgr<T>()) {                           \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
    if (auto* m = handle.template deviceMgr<T>()) {                           \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
    if (auto* m = handle.template deviceMgr<T>()) {                           \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#include "../BuildTypes.def"
    return nb::none();
}

static void defineDeviceNodeManagerHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    using HandleT = NodeManagerHandle<BufferT>;
    // Distinct name from the host "NodeManagerHandle" because both live in the
    // same nanobind type registry; a clashing name would collide. The device
    // variant lives on the nanovdb.cuda submodule.
    //
    // No deviceUpload/deviceDownload here: cuda::createNodeManager builds the
    // NodeManager directly on the device (no host mirror), so there is nothing
    // to upload; NodeManagerHandle::deviceUpload would null-deref its host
    // NodeManagerData. The handle is created device-resident and ready for
    // kernel use.
    nb::class_<HandleT>(m, "DeviceNodeManagerHandle",
        "Owns the device memory backing a device-resident NodeManager. "
        "Move-only. Obtain via nanovdb.cuda.createDeviceNodeManager(device_grid). "
        "The NodeManager returned by mgr() is a device pointer: its node "
        "accessors must only be used from CUDA kernels, never dereferenced on "
        "the host.")
        .def("size",
             [](const HandleT& h) { return h.size(); },
             "Byte size of the device buffer backing this handle.")
        .def(
            "__bool__",
            [](const HandleT& h) { return h.size() != 0; },
            nb::is_operator(),
            "True iff this handle owns a non-empty (device-resident) buffer.")
        .def("mgr", &pyDeviceNodeMgr,
             nb::keep_alive<0, 1>(),
             "Return the typed device NodeManager for the grid this handle was "
             "built from, or None if the BuildT is not Python-visible. The "
             "returned NodeManager's `this` is a DEVICE pointer — use it only "
             "from CUDA kernels. It keeps this handle alive.");
}

// cuda::createNodeManager has one template instantiation per BuildT. We expose
// a single polymorphic createDeviceNodeManager(device_grid, stream) that picks
// the right one based on the runtime type of `device_grid` (any bound
// NanoGrid<T> whose underlying pointer is a device pointer, e.g. from
// DeviceGridHandle.deviceGrid(n)). The created handle stores a raw pointer back
// to the device grid, so the handle must keep the grid alive.
template<typename BuildT>
static nb::object tryCreateDeviceNodeManager(nb::handle py_grid, cudaStream_t stream)
{
    using GridT = NanoGrid<BuildT>;
    if (!nb::isinstance<GridT>(py_grid)) {
        return nb::object();  // sentinel: "not this BuildT, try next"
    }
    // &grid is the device pointer (the NanoGrid<T> object wraps a device this).
    auto* d_grid = &nb::cast<GridT&>(py_grid);
    NodeManagerHandle<nanovdb::cuda::DeviceBuffer> handle;
    {
        nb::gil_scoped_release release;
        handle = nanovdb::cuda::createNodeManager<BuildT, nanovdb::cuda::DeviceBuffer>(
            d_grid, nanovdb::cuda::DeviceBuffer(), stream);
    }
    return nb::cast(std::move(handle));
}

static void defineCreateDeviceNodeManager(nb::module_& m)
{
    m.def("createDeviceNodeManager",
        [](nb::handle py_grid, uintptr_t stream) -> nb::object {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // Try every bound BuildT; first matching runtime type wins.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
            if (auto obj = tryCreateDeviceNodeManager<T>(py_grid, s); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
            if (auto obj = tryCreateDeviceNodeManager<T>(py_grid, s); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
            if (auto obj = tryCreateDeviceNodeManager<T>(py_grid, s); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
            if (auto obj = tryCreateDeviceNodeManager<T>(py_grid, s); obj.is_valid()) { \
                return obj;                                                    \
            }
#include "../BuildTypes.def"
            throw nb::type_error(
                "createDeviceNodeManager: argument is not a NanoVDB device "
                "grid of any bound BuildT. Pass a device grid obtained from "
                "DeviceGridHandle.deviceGrid(n).");
        },
        "device_grid"_a, "stream"_a = 0,
        // The constructed NodeManager stores a raw pointer back to the device
        // grid; the handle must therefore keep the grid (and transitively the
        // DeviceGridHandle that owns the grid's device buffer) alive.
        nb::keep_alive<0, 1>(),
        "Build a device-resident NodeManager for the given DEVICE grid, "
        "returning a DeviceNodeManagerHandle that owns the underlying device "
        "buffer. device_grid MUST be a device grid (from "
        "DeviceGridHandle.deviceGrid(n)); passing a host grid is a usage "
        "error. stream is a raw CUDA stream handle (Python int; 0 = default "
        "stream). The handle's mgr() returns the typed device NodeManager and "
        "keeps the source grid alive for as long as it lives.");
}

void defineDeviceNodeManager(nb::module_& m)
{
    defineDeviceNodeManagerHandle(m);
    defineCreateDeviceNodeManager(m);
}

} // namespace pynanovdb

#endif
