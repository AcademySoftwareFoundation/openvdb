// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file IO.h

    \author Ken Museth

    \date May 1, 2020

    \brief Implements I/O for NanoVDB grids. Features optional BLOSC and ZIP
           file compression, support for multiple grids per file as well as
           multiple grid types.

    \note  This file does NOT depend on OpenVDB, but optionally on ZIP and BLOSC
*/

#ifndef NANOVDB_IO_H_HAS_BEEN_INCLUDED
#define NANOVDB_IO_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include "GridHandle.h"
#include "GridChecksum.h"

#include <fstream> // for std::ifstream
#include <iostream> // for std::cerr/cout
#include <string> // for std::string
#include <sstream> // for std::stringstream
#include <cstring> // for std::strcmp
#include <memory> // for std::unique_ptr
#include <vector> // for std::vector
#ifdef NANOVDB_USE_ZIP
#include <zlib.h> // for ZIP compression
#endif
#ifdef NANOVDB_USE_BLOSC
#include <blosc.h> // for BLOSC compression
#endif

// Due to a bug in older versions of gcc, including fstream might
// define "major" and "minor" which are used as member data below.
// See https://bugzilla.redhat.com/show_bug.cgi?id=130601
#if defined(major) || defined(minor)
#undef major
#undef minor
#endif

namespace nanovdb {

namespace io {

/// We fix a specific size for counting bytes in files so that they
/// are saved the same regardless of machine precision.  (Note there are
/// still little/bigendian issues, however)
using fileSize_t = uint64_t;

/// @brief Optional compression codecs
///
/// @note NONE is the default, ZIP is slow but compact and BLOSC offers a great balance.
///
/// @warning NanoVDB optionally supports ZIP and BLOSC compression and will throw an exception
///          if it support is required but missing.
enum class Codec : uint16_t { NONE = 0,
                              ZIP = 1,
                              BLOSC = 2,
                              END = 3 };

inline __hostdev__ const char* toStr(Codec codec)
{
    static const char * LUT[] = { "NONE", "ZIP", "BLOSC" , "END" };
    return LUT[static_cast<int>(codec)];
}

/// @brief Internal functions for compressed read/write of a NanoVDB GridHandle into a stream
///
/// @warning These functions should never be called directly by client code
namespace Internal {
static constexpr fileSize_t MAX_SIZE = 1UL << 30; // size is 1 GB

template<typename BufferT>
static fileSize_t write(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec, uint32_t n);

template<typename BufferT>
static void read(std::istream& is, BufferT& buffer, Codec codec);

static void read(std::istream& is, char* data, fileSize_t size, Codec codec);
} // namespace Internal

/// @brief Standard hash function to use on strings; std::hash may vary by
///        platform/implementation and is know to produce frequent collisions.
uint64_t stringHash(const char* cstr);

/// @brief Return a uint64_t hash key of a std::string
inline uint64_t stringHash(const std::string& str)
{
    return stringHash(str.c_str());
}

/// @brief Return a uint64_t with its bytes reversed so we can check for endianness
inline uint64_t reverseEndianness(uint64_t val)
{
    return (((val) >> 56) & 0x00000000000000FF) | (((val) >> 40) & 0x000000000000FF00) |
           (((val) >> 24) & 0x0000000000FF0000) | (((val) >>  8) & 0x00000000FF000000) |
           (((val) <<  8) & 0x000000FF00000000) | (((val) << 24) & 0x0000FF0000000000) |
           (((val) << 40) & 0x00FF000000000000) | (((val) << 56) & 0xFF00000000000000);
}

/// @brief Data encoded at the head of each segment of a file or stream.
///
/// @note A file or stream is composed of one or more segments that each contain
//        one or more grids.
// Magic number of NanoVDB files   (uint64_t) |
// Version numbers of this file    (uint32_t) | one header for each segment
// Number of grids in this segment (uint16_t) |
// Compression mode                (uint16_t) |
struct Header
{// 16 bytes
    uint64_t magic; // 8 bytes
    Version  version;// 4 bytes version numbers
    uint16_t gridCount; // 2 bytes
    Codec    codec; // 2 bytes
    Header(Codec c = Codec::NONE)
        : magic(NANOVDB_MAGIC_NUMBER) // Magic number: "NanoVDB" in hex
        , version()// major, minor and patch version numbers
        , gridCount(0)
        , codec(c)
    {
    }
}; // Header ( 16 bytes = 2 words )

/// @brief Data encoded for each of the grids associated with a segment.
// Grid size in memory             (uint64_t)   |
// Grid size on disk               (uint64_t)   |
// Grid name hash key              (uint64_t)   |
// Numer of active voxels          (uint64_t)   |
// Grid type                       (uint32_t)   |
// Grid class                      (uint32_t)   |
// Characters in grid name         (uint32_t)   |
// AABB in world space             (2*3*double) | one per grid in file
// AABB in index space             (2*3*int)    |
// Size of a voxel in world units  (3*double)   |
// Byte size of the grid name      (uint32_t)   |
// Number of nodes per level       (4*uint32_t) |
// Numer of active tiles per level (3*uint32_t) |
// Codec for file compression      (uint16_t)   |
// Padding due to 8B alignment     (uint16_t)   |
// Version number                  (uint32_t)   |
struct MetaData
{// 176 bytes
    uint64_t    gridSize, fileSize, nameKey, voxelCount; // 4 * 8 = 32B.
    GridType    gridType;  // 4B.
    GridClass   gridClass; // 4B.
    BBox<Vec3d> worldBBox; // 2 * 3 * 8 = 48B.
    CoordBBox   indexBBox; // 2 * 3 * 4 = 24B.
    Vec3d       voxelSize; // 24B.
    uint32_t    nameSize;  // 4B.
    uint32_t    nodeCount[4]; //4 x 4 = 16B
    uint32_t    tileCount[3];// 3 x 4 = 12B
    Codec       codec;  // 2B
    uint16_t    padding;// 2B, due to 8B alignment from uint64_t
    Version     version;// 4B
}; // MetaData

struct GridMetaData : public MetaData
{
    static_assert(sizeof(MetaData) == 176, "Unexpected sizeof(MetaData)");
    std::string gridName;
    void        read(std::istream& is);
    void        write(std::ostream& os) const;
    GridMetaData() {}
    template<typename ValueT>
    GridMetaData(uint64_t size, Codec c, const NanoGrid<ValueT>& grid);
    uint64_t memUsage() const { return sizeof(MetaData) + nameSize; }
}; // GridMetaData

struct Segment
{
    // Check assumptions made during read and write of Header and MetaData
    static_assert(sizeof(Header) == 16u, "Unexpected sizeof(Header)");
    Header                    header;
    std::vector<GridMetaData> meta;
    Segment(Codec c = Codec::NONE)
        : header(c)
        , meta()
    {
    }
    template<typename BufferT>
    void     add(const GridHandle<BufferT>& h);
    bool     read(std::istream& is);
    void     write(std::ostream& os) const;
    uint64_t memUsage() const;
}; // Segment

/// @brief Write a single grid to file (over-writing existing content of the file)
template<typename BufferT>
void writeGrid(const std::string& fileName, const GridHandle<BufferT>& handle, Codec codec = Codec::NONE, int verbose = 0);

/// @brief Write a single grid to stream (starting at the current position)
///
/// @note This method can be used to append grid to an existing stream
template<typename BufferT>
void writeGrid(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec = Codec::NONE);

/// @brief Write multiple grids to file (over-writing existing content of the file)
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
void writeGrids(const std::string& fileName, const VecT<GridHandle<BufferT>>& handles, Codec codec = Codec::NONE, int verbose = 0);

/// @brief Writes multiple grids to stream (starting at its current position)
///
/// @note This method can be used to append multiple grids to an existing stream
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
void writeGrids(std::ostream& os, const VecT<GridHandle<BufferT>>& handles, Codec codec = Codec::NONE);

/// @brief Read the n'th grid from file (defaults to first grid)
///
/// @throw If n exceeds the number of grids in the file
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(const std::string& fileName, uint64_t n = 0, int verbose = 0, const BufferT& buffer = BufferT());

/// @brief Read the n'th grid from stream (defaults to first grid)
///
/// @throw If n exceeds the number of grids in the stream
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(std::istream& is, uint64_t n = 0, const BufferT& buffer = BufferT());

/// @brief Read the first grid with a specific name
///
/// @warning If not grid exists with the specified name the resulting GridHandle is empty
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(const std::string& fileName, const std::string& gridName, int verbose = 0, const BufferT& buffer = BufferT());

/// @brief Read the first grid with a specific name
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(std::istream& is, const std::string& gridName, const BufferT& buffer = BufferT());

/// @brief Read all the grids in the file
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
VecT<GridHandle<BufferT>> readGrids(const std::string& fileName, int verbose = 0, const BufferT& buffer = BufferT());

/// @brief Real all grids at the current position of the input stream
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
VecT<GridHandle<BufferT>> readGrids(std::istream& is, const BufferT& buffer = BufferT());

/// @brief Return true if the file contains a grid with the specified name
bool hasGrid(const std::string& fileName, const std::string& gridName);

/// @brief Return true if the stream contains a grid with the specified name
bool hasGrid(std::istream& is, const std::string& gridName);

/// @brief Reads and returns a vector of meta data for all the grids found in the specified file
std::vector<GridMetaData> readGridMetaData(const std::string& fileName);

/// @brief Reads and returns a vector of meta data for all the grids found in the specified stream
std::vector<GridMetaData> readGridMetaData(std::istream& is);

// --------------------------> Implementations for Internal <------------------------------------

template<typename BufferT>
fileSize_t Internal::write(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec, unsigned int n)
{
    const char* data = reinterpret_cast<const char*>(handle.gridData(n));
    fileSize_t  total = 0, residual = handle.gridSize(n);

    switch (codec) {
    case Codec::ZIP: {
#ifdef NANOVDB_USE_ZIP
        uLongf                   size = compressBound(residual); // Get an upper bound on the size of the compressed data.
        std::unique_ptr<Bytef[]> tmp(new Bytef[size]);
        const int                status = compress(tmp.get(), &size, reinterpret_cast<const Bytef*>(data), residual);
        if (status != Z_OK)
            std::runtime_error("Internal write error in ZIP");
        if (size > residual)
            std::cerr << "\nWarning: Unexpected ZIP compression from " << residual << " to " << size << " bytes\n";
        const fileSize_t outBytes = size;
        os.write(reinterpret_cast<const char*>(&outBytes), sizeof(fileSize_t));
        os.write(reinterpret_cast<const char*>(tmp.get()), outBytes);
        total += sizeof(fileSize_t) + outBytes;
#else
        throw std::runtime_error("ZIP compression codec was disabled during build");
#endif
        break;
    }
    case Codec::BLOSC: {
#ifdef NANOVDB_USE_BLOSC
        do {
            fileSize_t              chunk = residual < MAX_SIZE ? residual : MAX_SIZE, size = chunk + BLOSC_MAX_OVERHEAD;
            std::unique_ptr<char[]> tmp(new char[size]);
            const int               count = blosc_compress_ctx(9, 1, sizeof(float), chunk, data, tmp.get(), size, BLOSC_LZ4_COMPNAME, 1 << 18, 1);
            if (count <= 0)
                std::runtime_error("Internal write error in BLOSC");
            const fileSize_t outBytes = count;
            os.write(reinterpret_cast<const char*>(&outBytes), sizeof(fileSize_t));
            os.write(reinterpret_cast<const char*>(tmp.get()), outBytes);
            total += sizeof(fileSize_t) + outBytes;
            data += chunk;
            residual -= chunk;
        } while (residual > 0);
#else
        throw std::runtime_error("BLOSC compression codec was disabled during build");
#endif
        break;
    }
    default:
        os.write(data, residual);
        total += residual;
    }
    if (!os) throw std::runtime_error("Failed to write Tree to file");
    return total;
} // Internal::write

template<typename BufferT>
void Internal::read(std::istream& is, BufferT& buffer, Codec codec)
{
    Internal::read(is, reinterpret_cast<char*>(buffer.data()), buffer.size(), codec);
} // Internal::read

void Internal::read(std::istream& is, char* data, fileSize_t residual, Codec codec)
{
    // read tree using optional compression
    switch (codec) {
    case Codec::ZIP: {
#ifdef NANOVDB_USE_ZIP
        fileSize_t size;
        is.read(reinterpret_cast<char*>(&size), sizeof(fileSize_t));
        std::unique_ptr<Bytef[]> tmp(new Bytef[size]);
        is.read(reinterpret_cast<char*>(tmp.get()), size);
        uLongf numBytes = residual;
        int    status = uncompress(reinterpret_cast<Bytef*>(data), &numBytes, tmp.get(), static_cast<uLongf>(size));
        if (status != Z_OK)
            std::runtime_error("Internal read error in ZIP");
        if (fileSize_t(numBytes) != residual)
            throw std::runtime_error("UNZIP failed on byte size");
#else
        throw std::runtime_error("ZIP compression codec was disabled during build");
#endif
        break;
    }
    case Codec::BLOSC: {
#ifdef NANOVDB_USE_BLOSC
        do {
            fileSize_t size;
            is.read(reinterpret_cast<char*>(&size), sizeof(fileSize_t));
            std::unique_ptr<char[]> tmp(new char[size]);
            is.read(reinterpret_cast<char*>(tmp.get()), size);
            const fileSize_t chunk = residual < MAX_SIZE ? residual : MAX_SIZE;
            const int        count = blosc_decompress_ctx(tmp.get(), data, size_t(chunk), 1); //fails with more threads :(
            if (count < 1)
                std::runtime_error("Internal read error in BLOSC");
            if (count != int(chunk))
                throw std::runtime_error("BLOSC failed on byte size");
            data += size_t(chunk);
            residual -= chunk;
        } while (residual > 0);
#else
        throw std::runtime_error("BLOSC compression codec was disabled during build");
#endif
        break;
    }
    default:
        is.read(data, residual);
    }
    if (!is) throw std::runtime_error("Failed to read Tree from file");
} // Internal::read

// --------------------------> Implementations for GridMetaData <------------------------------------

template<typename ValueT>
inline GridMetaData::GridMetaData(uint64_t size, Codec c, const NanoGrid<ValueT>& grid)
    : MetaData{size, // gridSize
               0, // fileSize
               0, // nameKey
               grid.activeVoxelCount(), // voxelCount
               grid.gridType(), // gridType
               grid.gridClass(), // gridClass
               grid.worldBBox(), // worldBBox
               grid.tree().bbox(), // indexBBox
               grid.voxelSize(), // voxelSize
               0, // nameSize
               {0, 0, 0, 1}, // nodeCount[4]
               {0, 0, 0}, // tileCount[3]
               c, // codec
               0, // padding
               Version()}// version
    , gridName(grid.gridName())
{
    nameKey = stringHash(gridName);
    nameSize = static_cast<uint32_t>(gridName.size() + 1); // include '\0'
    const uint32_t* ptr = reinterpret_cast<const TreeData*>(&grid.tree())->mNodeCount;
    for (int i = 0; i < 3; ++i) MetaData::nodeCount[i] = *ptr++;
    for (int i = 0; i < 3; ++i) MetaData::tileCount[i] = *ptr++;
}// GridMetaData::GridMetaData

inline void GridMetaData::write(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(this), sizeof(MetaData));
    os.write(gridName.c_str(), nameSize);
    if (!os) throw std::runtime_error("Failed writing GridMetaData");
}// GridMetaData::write

inline void GridMetaData::read(std::istream& is)
{
    is.read(reinterpret_cast<char*>(this), sizeof(MetaData));
    std::unique_ptr<char[]> tmp(new char[nameSize]);
    is.read(reinterpret_cast<char*>(tmp.get()), nameSize);
    gridName.assign(tmp.get());
    if (!is) throw std::runtime_error("Failed reading GridMetaData");
}// GridMetaData::read

// --------------------------> Implementations for Segment <------------------------------------

inline uint64_t Segment::memUsage() const
{
    uint64_t sum = sizeof(Header);
    for (auto& m : meta) sum += m.memUsage();
    return sum;
}// Segment::memUsage

template<typename BufferT>
inline void Segment::add(const GridHandle<BufferT>& h)
{
    for (uint32_t i = 0; i < h.gridCount(); ++i) {
        if (auto* grid = h.template grid<float>(i)) { // most common
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Vec3f>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<double>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<int32_t>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<uint32_t>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<int64_t>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<int16_t>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Vec3d>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<ValueMask>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<ValueIndex>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<ValueIndexMask>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<ValueOnIndex>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<ValueOnIndexMask>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<bool>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Rgba8>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Fp4>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Fp8>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Fp16>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<FpN>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Vec4f>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else if (auto* grid = h.template grid<Vec4d>(i)) {
            meta.emplace_back(h.gridSize(i), header.codec, *grid);
        } else {
            std::stringstream ss;
            ss << "nanovdb::io::Segment::add: Cannot write grid of unknown type \""<<toStr(h.gridType(i));
            throw std::runtime_error(ss.str() + "\" to file");
        }
    }
    header.gridCount += h.gridCount();
}// Segment::add

inline void Segment::write(std::ostream& os) const
{
    if (header.gridCount == 0) {
        throw std::runtime_error("Segment contains no grids");
    } else if (!os.write(reinterpret_cast<const char*>(&header), sizeof(Header))) {
        throw std::runtime_error("Failed to write Header of Segment");
    }
    for (auto& m : meta) m.write(os);
}// Segment::write

inline bool Segment::read(std::istream& is)
{
    is.read(reinterpret_cast<char*>(&header), sizeof(Header));
    if (is.eof()) {
        return false;
    }
    if (!is || header.magic != NANOVDB_MAGIC_NUMBER) {
        // first check for byte-swapped header magic.
        if (header.magic == reverseEndianness(NANOVDB_MAGIC_NUMBER))
            throw std::runtime_error("This nvdb file has reversed endianness");
        throw std::runtime_error("Magic number error: This is not a valid nvdb file");
    } else if ( header.version.getMajor() != NANOVDB_MAJOR_VERSION_NUMBER) {
        std::stringstream ss;
        if (header.version.getMajor() < NANOVDB_MAJOR_VERSION_NUMBER) {
            ss << "The file contains an older version of NanoVDB: " << std::string(header.version.c_str()) << "!\n\t"
               << "Recommendation: Re-generate this NanoVDB file with this version: " << NANOVDB_MAJOR_VERSION_NUMBER << ".X of NanoVDB";
        } else {
            ss << "This tool was compiled against an older version of NanoVDB: " << NANOVDB_MAJOR_VERSION_NUMBER << ".X!\n\t"
               << "Recommendation: Re-compile this tool against the newer version: " << header.version.getMajor() << ".X of NanoVDB";
        }
        throw std::runtime_error("An unrecoverable error in nanovdb::Segment::read:\n\tIncompatible file format: " + ss.str());
    }
    meta.resize(header.gridCount);
    for (auto& m : meta) {
        m.read(is);
        m.version = header.version;
    }
    return true;
}// Segment::read

// --------------------------> Implementations for read/write <------------------------------------

template<typename BufferT>
void writeGrid(const std::string& fileName, const GridHandle<BufferT>& handle, Codec codec, int verbose)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for output");
    }
    writeGrid<BufferT>(os, handle, codec);
    if (verbose) {
        std::cout << "Wrote nanovdb::Grid to file named \"" << fileName << "\"" << std::endl;
    }
}// writeGrid

template<typename BufferT>
void writeGrid(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec)
{
    Segment seg(codec);
    seg.add(handle);
    const uint64_t headerSize = seg.memUsage();
    std::streamoff seek = headerSize;
    seg.write(os); // write header without the correct fileSize
    for (uint32_t i = 0; i < handle.gridCount(); ++i) {
        seg.meta[i].fileSize = Internal::write(os, handle, codec, i);
        seek += seg.meta[i].fileSize;
    }
    os.seekp(-seek, std::ios_base::cur); // rewind to start of stream
    seg.write(os); // rewrite header with the correct fileSize
    os.seekp(seek - headerSize, std::ios_base::cur); // skip to end
}// writeGrid

template<typename BufferT, template<typename...> class VecT>
void writeGrids(const std::string& fileName, const VecT<GridHandle<BufferT>>& handles, Codec codec, int verbose)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for output");
    }
    writeGrids<BufferT, VecT>(os, handles, codec);
    if (verbose) {
        std::cout << "Wrote " << handles.size() << " nanovdb::Grid(s) to file named \"" << fileName << "\"" << std::endl;
    }
}// writeGrids

template<typename BufferT, template<typename...> class VecT>
void writeGrids(std::ostream& os, const VecT<GridHandle<BufferT>>& handles, Codec codec)
{
    for (auto& h : handles) writeGrid(os, h, codec);
}// writeGrids

/// @brief Read the n'th grid
template<typename BufferT>
GridHandle<BufferT> readGrid(const std::string& fileName, uint64_t n, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for input");
    }
    auto handle = readGrid<BufferT>(is, n, buffer);
    if (verbose) {
        std::cout << "Read NanoGrid # " << n << " from the file named \"" << fileName << "\"" << std::endl;
    }
    return handle; // is converted to r-value and return value is move constructed.
}// readGrid

template<typename BufferT>
GridHandle<BufferT> readGrid(std::istream& is, uint64_t n, const BufferT& pool)
{
    Segment seg;
    uint64_t counter = 0;
    while (seg.read(is)) {
        std::streamoff seek = 0;
        for (auto& m : seg.meta) {
            if (counter == n) {
                auto buffer = BufferT::create(m.gridSize, &pool);
                is.seekg(seek, std::ios_base::cur); // skip forward from the current position
                Internal::read(is, buffer, seg.header.codec);
                return GridHandle<BufferT>(std::move(buffer));
            } else {
                seek += m.fileSize;
            }
            ++counter;
        }
        is.seekg(seek, std::ios_base::cur); // skip forward from the current position
    }
    throw std::runtime_error("Grid index " + std::to_string(n) + "exceeds grid count (" + std::to_string(counter) + ") in file");
}// readGrid

/// @brief Read the first grid with a specific name
template<typename BufferT>
GridHandle<BufferT> readGrid(const std::string& fileName, const std::string& gridName, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for input");
    }
    auto handle = readGrid<BufferT>(is, gridName, buffer);
    if (verbose) {
        if (handle) {
            std::cout << "Read NanoGrid named \"" << gridName << "\" from the file named \"" << fileName << "\"" << std::endl;
        } else {
            std::cout << "File named \"" << fileName << "\" does not contain a grid named \"" + gridName + "\"" << std::endl;
        }
    }
    return handle; // is converted to r-value and return value is move constructed.
}// readGrid

template<typename BufferT>
GridHandle<BufferT> readGrid(std::istream& is, const std::string& gridName, const BufferT& pool)
{
    const auto key = stringHash(gridName);
    Segment seg;
    while (seg.read(is)) {
        std::streamoff seek = 0;
        for (auto& m : seg.meta) {
            if (m.nameKey == key && m.gridName == gridName) { // check for hash key collision
                auto buffer = BufferT::create(m.gridSize, &pool);
                is.seekg(seek, std::ios_base::cur); // rewind
                Internal::read(is, buffer, seg.header.codec);
                return GridHandle<BufferT>(std::move(buffer));
            } else {
                seek += m.fileSize;
            }
        }
        is.seekg(seek, std::ios_base::cur); // skip forward from the current position
    }
    throw std::runtime_error("Grid name '" + gridName + "' not found in file");
}// readGrid

/// @brief Read all the grids
template<typename BufferT, template<typename...> class VecT>
VecT<GridHandle<BufferT>> readGrids(const std::string& fileName, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for input");
    }
    auto handles = readGrids<BufferT, VecT>(is, buffer);
    if (verbose) {
        std::cout << "Read " << handles.size() << " NanoGrid(s) from the file named \"" << fileName << "\"" << std::endl;
    }
    return handles; // is converted to r-value and return value is move constructed.
}// readGrids

template<typename BufferT, template<typename...> class VecT>
VecT<GridHandle<BufferT>> readGrids(std::istream& is, const BufferT& pool)
{
    VecT<GridHandle<BufferT>> handles;
    Segment seg;
    while (seg.read(is)) {
        uint64_t bufferSize = 0;
        for (auto& m : seg.meta) bufferSize += m.gridSize;
        auto buffer = BufferT::create(bufferSize, &pool);
        uint64_t bufferOffset = 0;
        for (uint16_t i = 0; i < seg.header.gridCount; ++i) {
            Internal::read(is, reinterpret_cast<char*>(buffer.data()) + bufferOffset, seg.meta[i].gridSize, seg.header.codec);

            // The following three lines provide backwards compatibility with older files
            // that were written using writeGrids.
            auto *data = reinterpret_cast<GridData*>(buffer.data() + bufferOffset);
            data->mGridIndex = static_cast<uint32_t>(i);
            data->mGridCount = static_cast<uint32_t>(seg.header.gridCount);

            bufferOffset += seg.meta[i].gridSize;
        }
        handles.emplace_back(std::move(buffer)); // force move copy assignment
    }

    // The following two lines provide backwards compatibility with older files
    // that were written using writeGrids. Since we (force) updated the mGridIndex
    // and mGridCount above, we need to recompute the checksum as well.
    for (auto& handle : handles)
        updateChecksum(handle);

    return handles; // is converted to r-value and return value is move constructed.
}// readGrids

inline std::vector<GridMetaData> readGridMetaData(const std::string& fileName)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for input");
    }
    return readGridMetaData(is); // is converted to r-value and return value is move constructed.
}// readGridMetaData

inline std::vector<GridMetaData> readGridMetaData(std::istream& is)
{
    std::vector<GridMetaData> meta;
    Segment seg;
    while (seg.read(is)) {
        std::streamoff seek = 0;
        for (auto& m : seg.meta) {
            meta.push_back(m);
            seek += m.fileSize;
        }
        is.seekg(seek, std::ios_base::cur);
    }
    return meta; // is converted to r-value and return value is move constructed.
}// readGridMetaData

inline bool hasGrid(const std::string& fileName, const std::string& gridName)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for input");
    }
    return hasGrid(is, gridName);
}// hasGrid

inline bool hasGrid(std::istream& is, const std::string& gridName)
{
    const auto key = stringHash(gridName);
    Segment seg;
    while (seg.read(is)) {
        std::streamoff seek = 0;
        for (auto& m : seg.meta) {
            if (m.nameKey == key && m.gridName == gridName) return true; // check for hash key collision
            seek += m.fileSize;
        }
        is.seekg(seek, std::ios_base::cur);
    }
    return false;
}// hasGrid

inline uint64_t stringHash(const char* cstr)
{
    uint64_t hash = 0;
    if (cstr) {
        for (auto* str = reinterpret_cast<const unsigned char*>(cstr); *str; ++str) {
            uint64_t overflow = hash >> (64 - 8);
            hash *= 67; // Next-ish prime after 26 + 26 + 10
            hash += *str + overflow;
        }
    }
    return hash;
}// stringHash

} // namespace io

inline std::ostream&
operator<<(std::ostream& os, const CoordBBox& b)
{
    os << "(" << b[0][0] << "," << b[0][1] << "," << b[0][2] << ") -> "
       << "(" << b[1][0] << "," << b[1][1] << "," << b[1][2] << ")";
    return os;
}

inline std::ostream&
operator<<(std::ostream& os, const Coord& ijk)
{
    os << "(" << ijk[0] << "," << ijk[1] << "," << ijk[2] << ")";
    return os;
}

template<typename T>
inline std::ostream&
operator<<(std::ostream& os, const Vec3<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return os;
}

} // namespace nanovdb

#endif // NANOVDB_IO_H_HAS_BEEN_INCLUDED
