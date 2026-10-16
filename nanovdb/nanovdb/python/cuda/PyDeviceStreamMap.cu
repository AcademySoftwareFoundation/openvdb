// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceStreamMap.h"

#include <nanobind/stl/vector.h>

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nanovdb/cuda/DeviceStreamMap.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

void defineDeviceStreamMap(nb::module_& m)
{
    using DeviceStreamMap = nanovdb::cuda::DeviceStreamMap;

    nb::class_<DeviceStreamMap> cls(m, "DeviceStreamMap",
        "Maps each suitable CUDA device id to a freshly-created stream. The "
        "constructor filters devices by a DeviceType policy. Iterates as "
        "(device_id -> raw stream handle) pairs.");

    nb::enum_<DeviceStreamMap::DeviceType>(cls, "DeviceType",
        "Device-inclusion policy for DeviceStreamMap.")
        .value("Any", DeviceStreamMap::DeviceType::Any,
            "Include every available device (no filtering).")
        .value("PeerToPeer", DeviceStreamMap::DeviceType::PeerToPeer,
            "Include only devices that can peer-access every already-added "
            "device.")
        .value("Unified", DeviceStreamMap::DeviceType::Unified,
            "Include only devices supporting unified addressing + concurrent "
            "managed access AND peer access to the already-added devices "
            "(the default).");

    cls
        .def(
            "__init__",
            [](DeviceStreamMap* self, DeviceStreamMap::DeviceType type,
               std::vector<int> exclude, int verbose) {
                // Touches every device (attribute queries, stream creation),
                // so release the GIL.
                nb::gil_scoped_release release;
                new (self) DeviceStreamMap(type, std::move(exclude), verbose);
            },
            "type"_a = DeviceStreamMap::DeviceType::Unified,
            "exclude"_a = std::vector<int>{},
            "verbose"_a = 0,
            "Build a map of device id -> CUDA stream over the devices that "
            "satisfy `type`. `exclude` is a list of device ids to skip; "
            "`verbose` is 0 (quiet), 1 (print ignored devices) or 2 (print "
            "included devices).")
        .def("deviceCount", &DeviceStreamMap::deviceCount,
            "Number of devices in this map.")
        .def("__len__", &DeviceStreamMap::deviceCount,
            "Number of devices in this map (same as deviceCount()).")
        .def(
            "getMinPageSize",
            [](const DeviceStreamMap& map) {
                nb::gil_scoped_release release;
                return map.getMinPageSize();
            },
            "Minimum CUDA allocation granularity (in bytes) across all devices "
            "in this map.")
        .def(
            "printDevInfo",
            [](const DeviceStreamMap& map) { map.printDevInfo(); },
            "Print device information for every device in this map to stdout.")
        .def(
            "items",
            [](const DeviceStreamMap& map) {
                // Expose the underlying std::map<int, cudaStream_t> as a Python
                // dict {device_id -> raw stream handle (int)}.
                nb::dict out;
                for (const auto& kv : map)
                    out[nb::int_(kv.first)] =
                        nb::int_(reinterpret_cast<uintptr_t>(kv.second));
                return out;
            },
            "Return a dict mapping each device id to its raw CUDA stream handle "
            "(Python int).")
        .def(
            "stream",
            [](const DeviceStreamMap& map, int deviceId) {
                auto it = map.find(deviceId);
                if (it == map.end())
                    throw nb::key_error("DeviceStreamMap has no stream for that device id.");
                return reinterpret_cast<uintptr_t>(it->second);
            },
            "deviceId"_a,
            "Raw CUDA stream handle (Python int) for the given device id.")
        .def(
            "__contains__",
            [](const DeviceStreamMap& map, int deviceId) {
                return map.find(deviceId) != map.end();
            },
            "deviceId"_a,
            "True iff this map holds a stream for the given device id.")
        .def(
            "__getitem__",
            [](const DeviceStreamMap& map, int deviceId) {
                auto it = map.find(deviceId);
                if (it == map.end())
                    throw nb::key_error("DeviceStreamMap has no stream for that device id.");
                return reinterpret_cast<uintptr_t>(it->second);
            },
            "deviceId"_a,
            "Raw CUDA stream handle (Python int) for the given device id.")
        .def(
            "__iter__",
            [](const DeviceStreamMap& map) {
                nb::list keys;
                for (const auto& kv : map) keys.append(kv.first);
                return nb::iter(keys);
            },
            "Iterate over the device ids in this map.");
}

} // namespace pynanovdb

#endif
