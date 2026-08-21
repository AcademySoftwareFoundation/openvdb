// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceMesh.h"

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/DeviceMesh.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

void defineDeviceMesh(nb::module_& m)
{
    using DeviceNode = nanovdb::cuda::DeviceNode;
    using DeviceMesh = nanovdb::cuda::DeviceMesh;

    nb::class_<DeviceNode>(m, "DeviceNode",
        "A device id paired with a CUDA stream created on that device.")
        .def_ro("id", &DeviceNode::id,
            "CUDA device id this node refers to (-1 if unset).")
        .def_prop_ro(
            "stream",
            [](const DeviceNode& node) {
                return reinterpret_cast<uintptr_t>(node.stream);
            },
            "Raw CUDA stream handle for this device as a Python int "
            "(0 = default stream).");

    nb::class_<DeviceMesh>(m, "DeviceMesh",
        "Multi-GPU context: enumerates every CUDA device on the host, creates "
        "a stream per device, and caches peer-to-peer connectivity. Pass it to "
        "nanovdb.tools.cuda.DistributedPointsToGrid. Move-only / not copyable.")
        .def(
            "__init__",
            [](DeviceMesh* self) {
                // The ctor touches every device (stream creation, P2P probe),
                // so release the GIL while it runs.
                nb::gil_scoped_release release;
                new (self) DeviceMesh();
            },
            "Construct a DeviceMesh spanning every CUDA device on the host. "
            "Each device must support managed memory.")
        .def(
            "deviceCount",
            [](const DeviceMesh& mesh) { return static_cast<size_t>(mesh.deviceCount()); },
            "Number of devices in this mesh.")
        .def(
            "__len__",
            [](const DeviceMesh& mesh) { return static_cast<size_t>(mesh.deviceCount()); },
            "Number of devices in this mesh (same as deviceCount()).")
        .def(
            "__getitem__",
            [](const DeviceMesh& mesh, int deviceId) -> const DeviceNode& {
                if (deviceId < 0 || static_cast<size_t>(deviceId) >= mesh.deviceCount())
                    throw nb::index_error("DeviceMesh index out of range [0, deviceCount()).");
                return mesh[deviceId];
            },
            "deviceId"_a,
            nb::rv_policy::reference_internal,
            "Return the DeviceNode (id + stream) for the given device index.")
        .def(
            "canAccessPeer",
            [](const DeviceMesh& mesh, int deviceId, int peerId) {
                return mesh.canAccessPeer(deviceId, peerId);
            },
            "deviceId"_a, "peerId"_a,
            "True iff `deviceId` can directly access memory on `peerId` "
            "(peer-to-peer support).");
}

} // namespace pynanovdb

#endif
