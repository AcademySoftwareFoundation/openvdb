// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyTree.h"

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

// cuda::createNodeManager has one template instantiation per BuildT. We expose
// a single polymorphic createDeviceNodeManager(deviceGrid, stream) that picks
// the right one based on the runtime type of `deviceGrid` (any bound
// NanoGrid<T> whose underlying pointer is a device pointer, e.g. from
// DeviceGridHandle.deviceGrid(n)) and returns the typed NodeManager directly,
// mirroring the host createNodeManager in PyTree.cc.
//
// The per-BuildT NodeManager<T> classes registered on the root module are
// reused as-is; nanobind does not distinguish a host from a device `this`
// pointer, so the returned object wraps an address in device memory and its
// accessors (leaf(i)/lower(i)/upper(i)/...) must only be used from CUDA
// kernels, never dereferenced on the host.
//
// Lifetime: the C++ NodeManagerHandle that owns the device buffer is moved to
// the heap and owned by an nb::capsule; reference_internal parents the
// returned NodeManager to that capsule. The def-level keep_alive<0,1> on
// createDeviceNodeManager additionally ties the manager to the source grid,
// whose device memory the NodeManager points into.
template<typename BuildT>
static nb::object tryCreateDeviceNodeManager(nb::handle py_grid, cudaStream_t stream)
{
    using GridT = NanoGrid<BuildT>;
    using BufferT = nanovdb::cuda::DualDeviceBuffer;
    using HandleT = NodeManagerHandle<BufferT>;
    if (!nb::isinstance<GridT>(py_grid)) {
        return nb::object();  // sentinel: "not this BuildT, try next"
    }
    // &grid is the device pointer (the NanoGrid<T> object wraps a device this).
    auto* dGrid = &nb::cast<GridT&>(py_grid);
    HandleT* handle = nullptr;
    {
        nb::gil_scoped_release release;
        handle = new HandleT(
            nanovdb::cuda::createNodeManager<BuildT, BufferT>(dGrid, BufferT(), stream));
    }
    nb::capsule owner(handle, [](void* p) noexcept {
        delete static_cast<HandleT*>(p);
    });
    // Non-null by construction: the handle was just built for this BuildT.
    NodeManager<BuildT>* mgr = handle->template deviceMgr<BuildT>();
    return nb::cast(mgr, nb::rv_policy::reference_internal, owner);
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
#include "BuildTypes.def"
            throw nb::type_error(
                "createDeviceNodeManager: argument is not a NanoVDB device "
                "grid of any bound BuildT. Pass a device grid obtained from "
                "DeviceGridHandle.deviceGrid(n).");
        },
        "deviceGrid"_a, "stream"_a = 0,
        // The constructed NodeManager stores a raw pointer back to the device
        // grid; the returned manager must therefore keep the grid (and
        // transitively the DeviceGridHandle that owns its device buffer) alive.
        nb::keep_alive<0, 1>(),
        "Build and return the typed device-resident NodeManager (e.g. "
        "FloatNodeManager) for the given DEVICE grid. deviceGrid MUST be a "
        "device grid (from DeviceGridHandle.deviceGrid(n)); passing a host "
        "grid is a usage error. stream is a raw CUDA stream handle (Python "
        "int; 0 = default stream). The manager's `this` is a DEVICE pointer, "
        "so its node accessors must only be used from CUDA kernels. It owns "
        "its device buffer internally and keeps the source grid alive for as "
        "long as it lives.");
}

void defineDeviceNodeManager(nb::module_& m)
{
    defineCreateDeviceNodeManager(m);
}

} // namespace pynanovdb

#endif
