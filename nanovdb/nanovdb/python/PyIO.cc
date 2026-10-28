// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyIO.h"

#include <nanovdb/io/IO.h>
#ifdef NANOVDB_USE_CUDA
#include "cuda/PyDeviceGridHandle.h"
#endif

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/bind_vector.h>

#include <stdexcept> // for std::runtime_error
#include <utility> // for std::forward, std::move
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

void defineFileGridMetaData(nb::module_& m)
{
    nb::class_<io::FileMetaData>(m, "FileMetaData")
        .def_ro("gridSize", &io::FileMetaData::gridSize)
        .def_ro("fileSize", &io::FileMetaData::fileSize)
        .def_ro("nameKey", &io::FileMetaData::nameKey)
        .def_ro("voxelCount", &io::FileMetaData::voxelCount)
        .def_ro("gridType", &io::FileMetaData::gridType)
        .def_ro("gridClass", &io::FileMetaData::gridClass)
        .def_ro("indexBBox", &io::FileMetaData::indexBBox)
        .def_ro("worldBBox", &io::FileMetaData::worldBBox)
        .def_ro("voxelSize", &io::FileMetaData::voxelSize)
        .def_ro("nameSize", &io::FileMetaData::nameSize)
        .def_prop_ro("nodeCount",
                     [](io::FileMetaData& metaData) {
                         return std::make_tuple(metaData.nodeCount[0], metaData.nodeCount[1], metaData.nodeCount[2], metaData.nodeCount[3]);
                     })
        .def_prop_ro("tileCount",
                     [](io::FileMetaData& metaData) { return std::make_tuple(metaData.tileCount[0], metaData.tileCount[1], metaData.tileCount[2]); })
        .def_ro("codec", &io::FileMetaData::codec)
        .def_ro("blindDataCount", &io::FileMetaData::blindDataCount)
        .def_ro("version", &io::FileMetaData::version);

    nb::bind_vector<std::vector<io::FileMetaData>>(m, "FileMetaDataVector");

    nb::class_<io::FileGridMetaData, io::FileMetaData>(m, "FileGridMetaData")
        .def_ro("gridName", &io::FileGridMetaData::gridName)
        .def("memUsage", &io::FileGridMetaData::memUsage);

    nb::bind_vector<std::vector<io::FileGridMetaData>>(m, "FileGridMetaDataVector");
}

void defineReadWriteGrid(nb::module_& m)
{
    m.def("hasGrid", nb::overload_cast<const std::string&, const std::string&>(&io::hasGrid), "fileName"_a, "gridName"_a);
    m.def("readGridMetaData", nb::overload_cast<const std::string&>(&io::readGridMetaData), "fileName"_a);
}

template<typename BufferT> nb::list readGrids(const std::string& fileName, int verbose, const BufferT& buffer)
{
    auto     handles = nanovdb::io::readGrids(fileName, verbose, buffer);
    nb::list handleList;
    for (size_t i = 0; i < handles.size(); ++i) {
        handleList.append(std::move(handles[i]));
    }
    return handleList;
}

// Calls f with the handle holding the host copy of the grids.
template<typename F> void withHostHandle(const GridHandle<HostBuffer>& handle, const char*, F&& f)
{
    f(handle);
}
#ifdef NANOVDB_USE_CUDA
template<typename F> void withHostHandle(const DeviceGridHandle& handle, const char* caller, F&& f)
{
    handle.withHost(caller, std::forward<F>(f));
}
#endif

template<typename HandleT> void writeGrids(const std::string& fileName, nb::list handles, io::Codec codec, int verbose)
{
    // check every handle before truncating the file
    std::vector<const HandleT*> hostHandles;
    for (size_t i = 0; i < handles.size(); ++i) {
        hostHandles.push_back(&nb::cast<const HandleT&>(handles[i]));
        withHostHandle(*hostHandles.back(), "writeGrids", [](const auto& h) {
            if (h.gridCount() == 0) throw std::runtime_error("writeGrids: a handle holds no grids");
        });
    }
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
    }
    for (const HandleT* handle : hostHandles) {
        withHostHandle(*handle, "writeGrids", [&](const auto& h) { nanovdb::io::writeGrid(os, h, codec); });
    }
    if (verbose) {
        std::cout << "Wrote " << handles.size() << " nanovdb::Grid(s) to file named \"" << fileName << "\"" << std::endl;
    }
}

void defineHostReadWriteGrid(nb::module_& m)
{
    using BufferT = HostBuffer;
    defineReadWriteGrid(m);

    m.def("writeGrid",
          nb::overload_cast<const std::string&, const GridHandle<BufferT>&, io::Codec, int>(&io::template writeGrid<BufferT>),
          "fileName"_a,
          "handle"_a,
          "codec"_a = io::Codec::NONE,
          "verbose"_a = 0);
    m.def("writeGrids", &writeGrids<GridHandle<BufferT>>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0);
    m.def("readGrid",
          nb::overload_cast<const std::string&, int, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "n"_a = 0,
          "verbose"_a = 0,
          "buffer"_a = BufferT());
    m.def("readGrid",
          nb::overload_cast<const std::string&, const std::string&, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "gridName"_a,
          "verbose"_a = 0,
          "buffer"_a = BufferT());
    m.def("readGrids", &readGrids<BufferT>, "fileName"_a, "verbose"_a = 0, "buffer"_a = BufferT());
}

#ifdef NANOVDB_USE_CUDA
void writeDeviceGrid(const std::string& fileName, const DeviceGridHandle& handle, io::Codec codec, int verbose)
{
    handle.withHost("writeGrid", [&](const auto& h) { io::writeGrid(fileName, h, codec, verbose); });
}

nb::list readDeviceGrids(const std::string& fileName, int verbose)
{
    auto     handles = io::readGrids<HostBuffer>(fileName, verbose);
    nb::list handleList;
    for (auto& h : handles) handleList.append(DeviceGridHandle(std::move(h)));
    return handleList;
}

// The grids are read into the handle's host side.
void defineDeviceReadWriteGrid(nb::module_& m)
{
    m.def("deviceWriteGrid", &writeDeviceGrid, "fileName"_a, "handle"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0);
    m.def("deviceWriteGrids", &writeGrids<DeviceGridHandle>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0);
    m.def(
        "deviceReadGrid",
        [](const std::string& fileName, int n, int verbose) { return DeviceGridHandle(io::readGrid<HostBuffer>(fileName, n, verbose)); },
        "fileName"_a,
        "n"_a = 0,
        "verbose"_a = 0);
    m.def(
        "deviceReadGrid",
        [](const std::string& fileName, const std::string& gridName, int verbose) {
            return DeviceGridHandle(io::readGrid<HostBuffer>(fileName, gridName, verbose));
        },
        "fileName"_a,
        "gridName"_a,
        "verbose"_a = 0);
    m.def("deviceReadGrids", &readDeviceGrids, "fileName"_a, "verbose"_a = 0);
}
#endif

} // namespace

void defineIOModule(nb::module_& m)
{
    nb::enum_<io::Codec>(m, "Codec")
        .value("NONE", io::Codec::NONE)
        .value("ZIP", io::Codec::ZIP)
        .value("BLOSC", io::Codec::BLOSC)
        .export_values();
        // .def("__repr__", [](const io::Codec& codec) {
        //     char str[strlen<io::Codec>()];
        //     toStr(str, codec);
        //     return std::string(str);
        // });

    defineFileGridMetaData(m);
    defineHostReadWriteGrid(m);
#ifdef NANOVDB_USE_CUDA
    defineDeviceReadWriteGrid(m);
#endif
}

} // namespace pynanovdb
