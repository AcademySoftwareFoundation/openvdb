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
    nb::class_<io::FileMetaData>(m, "FileMetaData",
        "Per-grid header read from the .nvdb file index. Mirrors the C++ "
        "io::FileMetaData layout; subclassed by FileGridMetaData which adds "
        "the grid name string.")
        .def_ro("gridSize", &io::FileMetaData::gridSize,
                "Uncompressed grid size in bytes.")
        .def_ro("fileSize", &io::FileMetaData::fileSize,
                "On-disk byte size of this grid (post-codec).")
        .def_ro("nameKey", &io::FileMetaData::nameKey,
                "Hash of the grid name used as a fast lookup key.")
        .def_ro("voxelCount", &io::FileMetaData::voxelCount,
                "Number of active voxels in this grid.")
        .def_ro("gridType", &io::FileMetaData::gridType,
                "GridType enumerator naming the BuildT of this grid.")
        .def_ro("gridClass", &io::FileMetaData::gridClass,
                "GridClass enumerator (LevelSet, FogVolume, ...).")
        .def_ro("indexBBox", &io::FileMetaData::indexBBox,
                "Axis-aligned bounding box of active voxels in index space.")
        .def_ro("worldBBox", &io::FileMetaData::worldBBox,
                "Axis-aligned bounding box of active voxels in world space.")
        .def_ro("voxelSize", &io::FileMetaData::voxelSize,
                "World-space size of a single voxel.")
        .def_ro("nameSize", &io::FileMetaData::nameSize,
                "Length of the grid name string including the null terminator.")
        .def_prop_ro("nodeCount",
                     [](io::FileMetaData& metaData) {
                         return std::make_tuple(metaData.nodeCount[0], metaData.nodeCount[1], metaData.nodeCount[2], metaData.nodeCount[3]);
                     },
                     "Tuple (leaf, lower, upper, root) of node counts in this grid.")
        .def_prop_ro("tileCount",
                     [](io::FileMetaData& metaData) { return std::make_tuple(metaData.tileCount[0], metaData.tileCount[1], metaData.tileCount[2]); },
                     "Tuple (lower-tile, upper-tile, root-tile) of active-tile counts.")
        .def_ro("codec", &io::FileMetaData::codec,
                "Codec used to compress this grid on disk.")
        .def_ro("blindDataCount", &io::FileMetaData::blindDataCount,
                "Number of blind-data channels attached to this grid.")
        .def_ro("version", &io::FileMetaData::version,
                "NanoVDB version stored in the file when this grid was written.");

    nb::bind_vector<std::vector<io::FileMetaData>>(m, "FileMetaDataVector",
        "List of FileMetaData entries, one per grid in a .nvdb file.");

    nb::class_<io::FileGridMetaData, io::FileMetaData>(m, "FileGridMetaData",
        "FileMetaData extended with the grid name. Returned by "
        "readGridMetaData() so callers can identify grids by name without "
        "materializing them.")
        .def_ro("gridName", &io::FileGridMetaData::gridName,
                "Grid name as a Python string.")
        .def("memUsage", &io::FileGridMetaData::memUsage,
             "Byte size of this metadata record in memory.");

    nb::bind_vector<std::vector<io::FileGridMetaData>>(m, "FileGridMetaDataVector",
        "List of FileGridMetaData entries, one per grid in a .nvdb file.");
}

template<typename BufferT> void defineReadWriteGrid(nb::module_& m)
{
    m.def("hasGrid", nb::overload_cast<const std::string&, const std::string&>(&io::hasGrid), "fileName"_a, "gridName"_a,
          "Return True iff the .nvdb file at fileName contains a grid named gridName.");
    m.def("readGridMetaData", nb::overload_cast<const std::string&>(&io::readGridMetaData), "fileName"_a,
          "Return a FileGridMetaDataVector describing every grid stored in the .nvdb file.");
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
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
    }
    for (size_t i = 0; i < handles.size(); ++i) {
        nanovdb::io::writeGrid(os, nb::cast<const GridHandle<BufferT>&>(handles[i]), codec);
    }
    if (verbose) {
        std::cout << "Wrote " << handles.size() << " nanovdb::Grid(s) to file named \"" << fileName << "\"" << std::endl;
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
          "verbose"_a = 0,
          "Write the grids in handle to the .nvdb file at fileName using the given codec.");
    m.def("writeGrids", &writeGrids<BufferT>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0,
          "Write every GridHandle in the handles list to the .nvdb file at fileName.");
    m.def("readGrid",
          nb::overload_cast<const std::string&, int, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "n"_a = 0,
          "verbose"_a = 0,
          "buffer"_a = BufferT(),
          "Read the n-th grid from the .nvdb file at fileName into a fresh GridHandle.");
    m.def("readGrid",
          nb::overload_cast<const std::string&, const std::string&, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "gridName"_a,
          "verbose"_a = 0,
          "buffer"_a = BufferT(),
          "Read the grid named gridName from the .nvdb file at fileName into a fresh GridHandle.");
    m.def("readGrids", &readGrids<BufferT>, "fileName"_a, "verbose"_a = 0, "buffer"_a = BufferT(),
          "Read every grid from the .nvdb file at fileName, returning a list of GridHandles.");
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
          "verbose"_a = 0,
          "Write the grids in a device-backed handle to the .nvdb file at fileName.");
    m.def("deviceWriteGrids", &writeGrids<BufferT>, "fileName"_a, "handles"_a, "codec"_a = io::Codec::NONE, "verbose"_a = 0,
          "Write every device-backed GridHandle in handles to the .nvdb file at fileName.");
    m.def("deviceReadGrid",
          nb::overload_cast<const std::string&, int, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "n"_a = 0,
          "verbose"_a = 0,
          "buffer"_a = BufferT(),
          "Read the n-th grid from the .nvdb file at fileName into a fresh DeviceGridHandle.");
    m.def("deviceReadGrid",
          nb::overload_cast<const std::string&, const std::string&, int, const BufferT&>(&io::template readGrid<BufferT>),
          "fileName"_a,
          "gridName"_a,
          "verbose"_a = 0,
          "buffer"_a = BufferT(),
          "Read the grid named gridName from the .nvdb file at fileName into a fresh DeviceGridHandle.");
    m.def("deviceReadGrids", &readGrids<BufferT>, "fileName"_a, "verbose"_a = 0, "buffer"_a = BufferT(),
          "Read every grid from the .nvdb file at fileName into device-backed handles.");
}
#endif

} // namespace

void defineIOModule(nb::module_& m)
{
    nb::enum_<io::Codec>(m, "Codec",
        "Compression codec selector used when writing a .nvdb file. NONE "
        "writes raw bytes; ZIP uses zlib; BLOSC uses the blosc codec when "
        "compiled in.")
        .value("NONE", io::Codec::NONE)
        .value("ZIP", io::Codec::ZIP)
        .value("BLOSC", io::Codec::BLOSC)
        .export_values()
        .def("__repr__", [](const io::Codec& codec) {
            char str[strlen<io::Codec>()];
            toStr(str, codec);
            return std::string(str);
        });

    defineFileGridMetaData(m);
    defineHostReadWriteGrid(m);
#ifdef NANOVDB_USE_CUDA
    defineDeviceReadWriteGrid(m);
#endif
}

} // namespace pynanovdb
