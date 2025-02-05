// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyIO.h"

#include <nanovdb/io/IO.h>
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/bind_vector.h>

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
        .def_ro("padding", &io::FileMetaData::padding)
        .def_ro("version", &io::FileMetaData::version);

    nb::bind_vector<std::vector<io::FileMetaData>>(m, "FileMetaDataVector");

    nb::class_<io::FileGridMetaData, io::FileMetaData>(m, "FileGridMetaData")
        .def_ro("gridName", &io::FileGridMetaData::gridName)
        .def("memUsage", &io::FileGridMetaData::memUsage);

    nb::bind_vector<std::vector<io::FileGridMetaData>>(m, "FileGridMetaDataVector");
}

template<typename BufferT> void defineReadWriteGrid(nb::module_& m)
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

template<typename BufferT> void writeGrids(const std::string& fileName, nb::list handles, io::Codec codec, int verbose)
{
    for (size_t i = 0; i < handles.size(); ++i) {
        nanovdb::io::writeGrid(fileName, nb::cast<const GridHandle<BufferT>&>(handles[i]), codec, verbose);
    }
}

void defineHostReadWriteGrid(nb::module_& m)
{
    using BufferT = HostBuffer;
    defineReadWriteGrid<BufferT>(m);

    m.def("writeGrid",
          nb::overload_cast<const std::string&, const GridHandle<BufferT>&, io::Codec, int>(&io::template writeGrid<BufferT>),
          "fileName"_a,
          "handle"_a,
          "codec"_a = io::Codec::NONE,
          "verbose"_a = 0);
    m.def("writeGrids", &writeGrids<BufferT>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0);
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
void defineDeviceReadWriteGrid(nb::module_& m)
{
    using BufferT = cuda::DeviceBuffer;
    defineReadWriteGrid<BufferT>(m);

    m.def("deviceWriteGrid",
          nb::overload_cast<const std::string&, const GridHandle<BufferT>&, io::Codec, int>(&io::template writeGrid<BufferT>),
          "fileName"_a,
          "handle"_a,
          "codec"_a = io::Codec::NONE,
          "verbose"_a = 0);
    m.def("deviceWriteGrids", &writeGrids<BufferT>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0);
    m.def("deviceReadGrid",
          nb::overload_cast<const std::string&, int, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "n"_a = 0,
          "verbose"_a = 0,
          "buffer"_a = BufferT());
    m.def("deviceReadGrid",
          nb::overload_cast<const std::string&, const std::string&, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "gridName"_a,
          "verbose"_a = 0,
          "buffer"_a = BufferT());
    m.def("deviceReadGrids", &readGrids<BufferT>, "fileName"_a, "verbose"_a = 0, "buffer"_a = BufferT());
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
