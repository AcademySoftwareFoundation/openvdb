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

    \details NanoVDB files take on of two formats:
             1) multiple segments each with multiple grids (segments have easy to access metadata about its grids)
             2) starting with verion 32.6.0 nanovdb files also support a raw buffer with one or more grids (just a
             dump of a raw grid buffer, so no new metadata).

    // 1: Segment:  FileHeader, MetaData0, gridName0...MetaDataN, gridNameN, compressed Grid0, ... compressed GridN
    // 2: Raw: Grid0, ... GridN
*/

#ifndef NANOVDB_IO_H_HAS_BEEN_INCLUDED
#define NANOVDB_IO_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/GridChecksum.h>// for updateGridCount

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

namespace nanovdb {// ==========================================================

namespace io {// ===============================================================

// --------------------------> writeGrid(s) <------------------------------------

/// @brief Write a single grid to file (over-writing existing content of the file)
template<typename BufferT>
void writeGrid(const std::string& fileName, const GridHandle<BufferT>& handle, io::Codec codec = io::Codec::NONE, int verbose = 0);

/// @brief Write multiple grids to file (over-writing existing content of the file)
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
void writeGrids(const std::string& fileName, const VecT<GridHandle<BufferT>>& handles, Codec codec = Codec::NONE, int verbose = 0);

// --------------------------> readGrid(s) <------------------------------------

/// @brief Read and return one or all grids from a file into a single GridHandle
/// @tparam BufferT Type of buffer used memory allocation
/// @param fileName string name of file to be read from
/// @param n zero-based signed index of the grid to be read.
///          The default value of 0 means read only first grid.
///          A negative value of n means read all grids in the file.
/// @param verbose specify verbosity level. Default value of zero means quiet.
/// @param buffer optional buffer used for memory allocation
/// @return return a single GridHandle with one or all grids found in the file
/// @throw will throw a std::runtime_error if the file does not contain a grid with index n
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(const std::string& fileName, int n = 0, int verbose = 0, const BufferT& buffer = BufferT());

/// @brief Read and return the first grid with a specific name from a file
/// @tparam BufferT Type of buffer used memory allocation
/// @param fileName string name of file to be read from
/// @param gridName string name of the grid to be read
/// @param verbose specify verbosity level. Default value of zero means quiet.
/// @param buffer  optional buffer used for memory allocation
/// @return return a single GridHandle containing the grid with the specific name
/// @throw will throw a std::runtime_error if the file does not contain a grid with the specific name
template<typename BufferT = HostBuffer>
GridHandle<BufferT> readGrid(const std::string& fileName, const std::string& gridName, int verbose = 0, const BufferT& buffer = BufferT());

/// @brief Read all the grids in the file and return them as a vector of multiple GridHandles, each containing
///        all grids encoded in the same segment of the file (i.e. they where written together)
/// @tparam BufferT Type of buffer used memory allocation
/// @param fileName string name of file to be read from
/// @param verbose specify verbosity level. Default value of zero means quiet.
/// @param buffer  optional buffer used for memory allocation
/// @return Return a vector of GridHandles each containing all grids encoded
///         in the same segment of the file (i.e. they where written together).
template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
VecT<GridHandle<BufferT>> readGrids(const std::string& fileName, int verbose = 0, const BufferT& buffer = BufferT());

// -----------------------------------------------------------------------

/// We fix a specific size for counting bytes in files so that they
/// are saved the same regardless of machine precision.  (Note there are
/// still little/bigendian issues, however)
using fileSize_t = uint64_t;

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
inline uint64_t stringHash(const std::string& str){return stringHash(str.c_str());}

/// @brief Return a uint64_t with its bytes reversed so we can check for endianness
inline uint64_t reverseEndianness(uint64_t val)
{
    return (((val) >> 56) & 0x00000000000000FF) | (((val) >> 40) & 0x000000000000FF00) |
           (((val) >> 24) & 0x0000000000FF0000) | (((val) >>  8) & 0x00000000FF000000) |
           (((val) <<  8) & 0x000000FF00000000) | (((val) << 24) & 0x0000FF0000000000) |
           (((val) << 40) & 0x00FF000000000000) | (((val) << 56) & 0xFF00000000000000);
}

/// @brief This class defines the meta data stored for each grid in a segment
///
/// @details A segment consists of a FileHeader followed by a list of FileGridMetaData
///          each followed by grid names and then finally the grids themselves.
///
/// @note This class should not be confused with nanovdb::GridMetaData defined in NanoVDB.h
///       Also, io::FileMetaData is defined in NanoVDB.h.
struct FileGridMetaData : public FileMetaData
{
    static_assert(sizeof(FileMetaData) == 176, "Unexpected sizeof(FileMetaData)");
    std::string gridName;
    void        read(std::istream& is);
    void        write(std::ostream& os) const;
    FileGridMetaData() {}
    FileGridMetaData(uint64_t size, Codec c, const GridData &gridData);
    uint64_t memUsage() const { return sizeof(FileMetaData) + nameSize; }
}; // FileGridMetaData

/// @brief This class defines all the data stored in segment of a file
///
/// @details A segment consists of a FileHeader followed by a list of FileGridMetaData
///          each followed by grid names and then finally the grids themselves.
struct Segment
{
    // Check assumptions made during read and write of FileHeader and FileMetaData
    static_assert(sizeof(FileHeader) == 16u, "Unexpected sizeof(FileHeader)");
    FileHeader header;// defined in NanoVDB.h
    std::vector<FileGridMetaData> meta;// defined in NanoVDB.h
    Segment(Codec c = Codec::NONE)
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        : header{NANOVDB_MAGIC_FILE, Version(), 0u, c}
#else
        : header{NANOVDB_MAGIC_NUMB, Version(), 0u, c}
#endif
        , meta()
    {
    }
    template<typename BufferT>
    void     add(const GridHandle<BufferT>& h);
    bool     read(std::istream& is);
    void     write(std::ostream& os) const;
    uint64_t memUsage() const;
}; // Segment

/// @brief Return true if the file contains a grid with the specified name
bool hasGrid(const std::string& fileName, const std::string& gridName);

/// @brief Return true if the stream contains a grid with the specified name
bool hasGrid(std::istream& is, const std::string& gridName);

/// @brief Reads and returns a vector of meta data for all the grids found in the specified file
std::vector<FileGridMetaData> readGridMetaData(const std::string& fileName);

/// @brief Reads and returns a vector of meta data for all the grids found in the specified stream
std::vector<FileGridMetaData> readGridMetaData(std::istream& is);

// --------------------------> Implementations for Internal <------------------------------------

template<typename BufferT>
fileSize_t Internal::write(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec, unsigned int n)
{
    const char* data = reinterpret_cast<const char*>(handle.gridData(n));
    fileSize_t  total = 0, residual = handle.gridSize(n);

    switch (codec) {
    case Codec::ZIP: {
#ifdef NANOVDB_USE_ZIP
        uLongf                   size = compressBound(static_cast<uLongf>(residual)); // Get an upper bound on the size of the compressed data.
        std::unique_ptr<Bytef[]> tmp(new Bytef[size]);
        const int                status = compress(tmp.get(), &size, reinterpret_cast<const Bytef*>(data), static_cast<uLongf>(residual));
        if (status != Z_OK) std::runtime_error("Internal write error in ZIP");
        if (size > residual) std::cerr << "\nWarning: Unexpected ZIP compression from " << residual << " to " << size << " bytes\n";
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
            if (count <= 0) std::runtime_error("Internal write error in BLOSC");
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

/// @brief read compressed grid from stream
/// @param is input stream to read from
/// @param data data buffer to write into. Must be of size @c residual or larger.
/// @param residual expected byte size of uncompressed data.
/// @param codec mode of compression
void Internal::read(std::istream& is, char* data, fileSize_t residual, Codec codec)
{
    // read tree using optional compression
    switch (codec) {
    case Codec::ZIP: {
#ifdef NANOVDB_USE_ZIP
        fileSize_t size;
        is.read(reinterpret_cast<char*>(&size), sizeof(fileSize_t));
        std::unique_ptr<Bytef[]> tmp(new Bytef[size]);// temp buffer for compressed data
        is.read(reinterpret_cast<char*>(tmp.get()), size);
        uLongf numBytes = static_cast<uLongf>(residual);
        int status = uncompress(reinterpret_cast<Bytef*>(data), &numBytes, tmp.get(), static_cast<uLongf>(size));
        if (status != Z_OK) std::runtime_error("Internal read error in ZIP");
        if (fileSize_t(numBytes) != residual) throw std::runtime_error("UNZIP failed on byte size");
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
            std::unique_ptr<char[]> tmp(new char[size]);// temp buffer for compressed data
            is.read(reinterpret_cast<char*>(tmp.get()), size);
            const fileSize_t chunk = residual < MAX_SIZE ? residual : MAX_SIZE;
            const int        count = blosc_decompress_ctx(tmp.get(), data, size_t(chunk), 1); //fails with more threads :(
            if (count < 1) std::runtime_error("Internal read error in BLOSC");
            if (count != int(chunk)) throw std::runtime_error("BLOSC failed on byte size");
            data += size_t(chunk);
            residual -= chunk;
        } while (residual > 0);
#else
        throw std::runtime_error("BLOSC compression codec was disabled during build");
#endif
        break;
    }
    default:
        is.read(data, residual);// read uncompressed data
    }
    if (!is) throw std::runtime_error("Failed to read Tree from file");
} // Internal::read

// --------------------------> Implementations for FileGridMetaData <------------------------------------

inline FileGridMetaData::FileGridMetaData(uint64_t size, Codec c, const GridData &gridData)
    : FileMetaData{size, // gridSize
                   size, // fileSize (will typically be redefined)
                   0u, // nameKey
                   0u, // voxelCount
                   gridData.mGridType, // gridType
                   gridData.mGridClass, // gridClass
                   gridData.mWorldBBox, // worldBBox
                   gridData.indexBBox(), // indexBBox
                   gridData.mVoxelSize, // voxelSize
                   0, // nameSize
                   {0, 0, 0, 1}, // nodeCount[4]
                   {0, 0, 0}, // tileCount[3]
                   c, // codec
                   0, // padding
                   Version()}// version
    , gridName(gridData.gridName())
{
    auto &treeData = *reinterpret_cast<const TreeData*>(gridData.treePtr());
    nameKey = stringHash(gridName);
    voxelCount = treeData.mVoxelCount;
    nameSize = static_cast<uint32_t>(gridName.size() + 1); // include '\0'
    for (int i = 0; i < 3; ++i) {
        FileMetaData::nodeCount[i] = treeData.mNodeCount[i];
        FileMetaData::tileCount[i] = treeData.mTileCount[i];
    }
}// FileGridMetaData::FileGridMetaData

inline void FileGridMetaData::write(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(this), sizeof(FileMetaData));
    os.write(gridName.c_str(), nameSize);
    if (!os) throw std::runtime_error("Failed writing FileGridMetaData");
}// FileGridMetaData::write

inline void FileGridMetaData::read(std::istream& is)
{
    is.read(reinterpret_cast<char*>(this), sizeof(FileMetaData));
    std::unique_ptr<char[]> tmp(new char[nameSize]);
    is.read(reinterpret_cast<char*>(tmp.get()), nameSize);
    gridName.assign(tmp.get());
    if (!is) throw std::runtime_error("Failed reading FileGridMetaData");
}// FileGridMetaData::read

// --------------------------> Implementations for Segment <------------------------------------

inline uint64_t Segment::memUsage() const
{
    uint64_t sum = sizeof(FileHeader);
    for (auto& m : meta) sum += m.memUsage();// includes FileMetaData + grid name
    return sum;
}// Segment::memUsage

template<typename BufferT>
inline void Segment::add(const GridHandle<BufferT>& h)
{
    for (uint32_t i = 0; i < h.gridCount(); ++i) {
        const GridData *gridData = h.gridData(i);
        if (!gridData) throw std::runtime_error("Segment::add: GridHandle does not contain grid #" + std::to_string(i));
        meta.emplace_back(h.gridSize(i), header.codec, *gridData);
    }
    header.gridCount += h.gridCount();
}// Segment::add

inline void Segment::write(std::ostream& os) const
{
    if (header.gridCount == 0) {
        throw std::runtime_error("Segment contains no grids");
    } else if (!os.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader))) {
        throw std::runtime_error("Failed to write FileHeader of Segment");
    }
    for (auto& m : meta) m.write(os);
}// Segment::write

inline bool Segment::read(std::istream& is)
{
    is.read(reinterpret_cast<char*>(&header), sizeof(FileHeader));
    if (is.eof()) {// The EOF flag is only set once a read tries to read past the end of the file
        is.clear(std::ios_base::eofbit);// clear eof flag so we can rewind and read again
        return false;
    }
    const MagicType magic = toMagic(header.magic);
    if (magic != MagicType::NanoVDB && magic != MagicType::NanoFile) {
        // first check for byte-swapped header magic.
        if (header.magic == reverseEndianness(NANOVDB_MAGIC_NUMB) ||
            header.magic == reverseEndianness(NANOVDB_MAGIC_FILE)) {
            throw std::runtime_error("This nvdb file has reversed endianness");
        } else {
            if (magic == MagicType::OpenVDB) {
                throw std::runtime_error("Expected a NanoVDB file, but read an OpenVDB file!");
            } else if (magic == MagicType::NanoGrid) {
                throw std::runtime_error("Expected a NanoVDB file, but read a raw NanoVDB grid!");
            } else {
                throw std::runtime_error("Expected a NanoVDB file, but read a file of unknown type!");
            }
        }
    } else if ( !header.version.isCompatible()) {
        std::stringstream ss;
        Version v;
        is.read(reinterpret_cast<char*>(&v), sizeof(Version));// read GridData::mVersion located at byte 16=sizeof(FileHeader) is stream
        if ( v.getMajor() == NANOVDB_MAJOR_VERSION_NUMBER) {
            ss << "This file looks like it contains a raw grid buffer and not a standard file with meta data";
        } else if ( header.version.getMajor() < NANOVDB_MAJOR_VERSION_NUMBER) {
            char str[30];
            ss << "The file contains an older version of NanoVDB: " << std::string(toStr(str, header.version)) << "!\n\t"
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

// --------------------------> writeGrid <------------------------------------

template<typename BufferT>
void writeGrid(std::ostream& os, const GridHandle<BufferT>& handle, Codec codec)
{
    Segment seg(codec);
    seg.add(handle);
    const auto start = os.tellp();
    seg.write(os); // write header without the correct fileSize (so it's allocated)
    for (uint32_t i = 0; i < handle.gridCount(); ++i) {
        seg.meta[i].fileSize = Internal::write(os, handle, codec, i);
    }
    os.seekp(start);
    seg.write(os);// re-write header with the correct fileSize
    os.seekp(0, std::ios_base::end);// skip to end
}// writeGrid

template<typename BufferT>
void writeGrid(const std::string& fileName, const GridHandle<BufferT>& handle, Codec codec, int verbose)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
    }
    writeGrid<BufferT>(os, handle, codec);
    if (verbose) {
        std::cout << "Wrote nanovdb::Grid to file named \"" << fileName << "\"" << std::endl;
    }
}// writeGrid

// --------------------------> writeGrids <------------------------------------

template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
void writeGrids(std::ostream& os, const VecT<GridHandle<BufferT>>& handles, Codec codec = Codec::NONE)
{
    for (auto& h : handles) writeGrid(os, h, codec);
}// writeGrids

template<typename BufferT, template<typename...> class VecT>
void writeGrids(const std::string& fileName, const VecT<GridHandle<BufferT>>& handles, Codec codec, int verbose)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for output");
    writeGrids<BufferT, VecT>(os, handles, codec);
    if (verbose) std::cout << "Wrote " << handles.size() << " nanovdb::Grid(s) to file named \"" << fileName << "\"" << std::endl;
}// writeGrids

// --------------------------> readGrid <------------------------------------

template<typename BufferT>
GridHandle<BufferT> readGrid(std::istream& is, int n, const BufferT& pool)
{
    GridHandle<BufferT> handle;
    if (n<0) {// read all grids into the same buffer
        try {//first try to read a raw grid buffer
            handle.read(is, pool);
        } catch(const std::logic_error&) {
            Segment seg;
            uint64_t bufferSize = 0u;
            uint32_t gridCount = 0u, gridIndex = 0u;
            const auto start = is.tellg();
            while (seg.read(is)) {
                std::streamoff skipSize = 0;
                for (auto& m : seg.meta) {
                    ++gridCount;
                    bufferSize += m.gridSize;
                    skipSize   += m.fileSize;
                }// loop over grids in segment
                is.seekg(skipSize, std::ios_base::cur); // skip forward from the current position
            }// loop over segments
            auto buffer = BufferT::create(bufferSize, &pool);
            char *ptr = (char*)buffer.data();
            is.seekg(start);// rewind
            while (seg.read(is)) {
                for (auto& m : seg.meta) {
                    Internal::read(is, ptr, m.gridSize, seg.header.codec);
                    tools::updateGridCount((GridData*)ptr, gridIndex++, gridCount);
                    ptr += m.gridSize;
                }// loop over grids in segment
            }// loop over segments
            return GridHandle<BufferT>(std::move(buffer));
        }
    } else {// read a specific grid
        try {//first try to read a raw grid buffer
            handle.read(is, uint32_t(n), pool);
            tools::updateGridCount((GridData*)handle.data(), 0u, 1u);
        } catch(const std::logic_error&) {
            Segment seg;
            int counter = -1;
            while (seg.read(is)) {
                std::streamoff seek = 0;
                for (auto& m : seg.meta) {
                    if (++counter == n) {
                        auto buffer = BufferT::create(m.gridSize, &pool);
                        Internal::read(is, buffer, seg.header.codec);
                        tools::updateGridCount((GridData*)buffer.data(), 0u, 1u);
                        return GridHandle<BufferT>(std::move(buffer));
                    } else {
                        seek += m.fileSize;
                    }
                }// loop over grids in segment
                is.seekg(seek, std::ios_base::cur); // skip forward from the current position
            }// loop over segments
            if (n != counter) throw std::runtime_error("stream does not contain a #" + std::to_string(n) + " grid");
        }
    }
    return handle;
}// readGrid

/// @brief Read the n'th grid
template<typename BufferT>
GridHandle<BufferT> readGrid(const std::string& fileName, int n, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
    auto handle = readGrid<BufferT>(is, n, buffer);
    if (verbose) {
        if (n<0) {
            std::cout << "Read all NanoGrids from the file named \"" << fileName << "\"" << std::endl;
        } else {
            std::cout << "Read NanoGrid # " << n << " from the file named \"" << fileName << "\"" << std::endl;
        }
    }
    return handle; // is converted to r-value and return value is move constructed.
}// readGrid

/// @brief Read a specific grid from an input stream given the name of the grid
/// @tparam BufferT Buffer type used for allocation
/// @param is input stream from which to read the grid
/// @param gridName string name of the (first) grid to be returned
/// @param pool optional memory pool from which to allocate the grid buffer
/// @return Return the first grid in the input stream with a specific name
/// @throw std::runtime_error with no grid exists with the specified name
template<typename BufferT>
GridHandle<BufferT> readGrid(std::istream& is, const std::string& gridName, const BufferT& pool)
{
    try {
        GridHandle<BufferT> handle;
        handle.read(is, gridName, pool);
        return handle;
    } catch(const std::logic_error&) {
        const auto key = stringHash(gridName);
        Segment seg;
        while (seg.read(is)) {// loop over all segments in stream
            std::streamoff seek = 0;
            for (auto& m : seg.meta) {// loop over all grids in segment
                if ((m.nameKey == 0u || m.nameKey == key) && m.gridName == gridName) { // check for hash key collision
                    auto buffer = BufferT::create(m.gridSize, &pool);
                    is.seekg(seek, std::ios_base::cur); // rewind
                    Internal::read(is, buffer, seg.header.codec);
                    tools::updateGridCount((GridData*)buffer.data(), 0u, 1u);
                    return GridHandle<BufferT>(std::move(buffer));
                } else {
                    seek += m.fileSize;
                }
            }
            is.seekg(seek, std::ios_base::cur); // skip forward from the current position
        }
    }
    throw std::runtime_error("Grid name '" + gridName + "' not found in file");
}// readGrid

/// @brief Read the first grid with a specific name
template<typename BufferT>
GridHandle<BufferT> readGrid(const std::string& fileName, const std::string& gridName, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
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

// --------------------------> readGrids <------------------------------------

template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>
VecT<GridHandle<BufferT>> readGrids(std::istream& is, const BufferT& pool = BufferT())
{
    VecT<GridHandle<BufferT>> handles;
    Segment seg;
    while (seg.read(is)) {
        uint64_t bufferSize = 0;
        for (auto& m : seg.meta) bufferSize += m.gridSize;
        auto buffer = BufferT::create(bufferSize, &pool);
        uint64_t bufferOffset = 0;
        for (uint16_t i = 0; i < seg.header.gridCount; ++i) {
            auto *data = util::PtrAdd<GridData>(buffer.data(), bufferOffset);
            Internal::read(is, (char*)data, seg.meta[i].gridSize, seg.header.codec);
            tools::updateGridCount(data, uint32_t(i), uint32_t(seg.header.gridCount));
            bufferOffset += seg.meta[i].gridSize;
        }// loop over grids in segment
        handles.emplace_back(std::move(buffer)); // force move copy assignment
    }// loop over segments
    return handles; // is converted to r-value and return value is move constructed.
}// readGrids

/// @brief Read all the grids
template<typename BufferT, template<typename...> class VecT>
VecT<GridHandle<BufferT>> readGrids(const std::string& fileName, int verbose, const BufferT& buffer)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
    auto handles = readGrids<BufferT, VecT>(is, buffer);
    if (verbose) std::cout << "Read " << handles.size() << " NanoGrid(s) from the file named \"" << fileName << "\"" << std::endl;
    return handles; // is converted to r-value and return value is move constructed.
}// readGrids

// --------------------------> readGridMetaData <------------------------------------

inline std::vector<FileGridMetaData> readGridMetaData(const std::string& fileName)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
    return readGridMetaData(is); // is converted to r-value and return value is move constructed.
}// readGridMetaData

inline std::vector<FileGridMetaData> readGridMetaData(std::istream& is)
{
    Segment seg;
    std::vector<FileGridMetaData> meta;
    try {
        GridHandle<> handle;// if stream contains a raw grid buffer we unfortunately have to load everything
        handle.read(is);
        seg.add(handle);
        meta = std::move(seg.meta);
    } catch(const std::logic_error&) {
        while (seg.read(is)) {
            std::streamoff skip = 0;
            for (auto& m : seg.meta) {
                meta.push_back(m);
                skip += m.fileSize;
            }// loop over grid meta data in segment
            is.seekg(skip, std::ios_base::cur);
        }// loop over segments
    }
    return meta; // is converted to r-value and return value is move constructed.
}// readGridMetaData

// --------------------------> hasGrid <------------------------------------

inline bool hasGrid(const std::string& fileName, const std::string& gridName)
{
    std::ifstream is(fileName, std::ios::in | std::ios::binary);
    if (!is.is_open()) throw std::ios_base::failure("Unable to open file named \"" + fileName + "\" for input");
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
        }// loop over grid meta data in segment
        is.seekg(seek, std::ios_base::cur);
    }// loop over segments
    return false;
}// hasGrid

// --------------------------> stringHash <------------------------------------

inline uint64_t stringHash(const char* c_str)
{
    uint64_t hash = 0;// zero is returned when cstr = nullptr or "\0"
    if (c_str) {
        for (auto* str = reinterpret_cast<const unsigned char*>(c_str); *str; ++str) {
            uint64_t overflow = hash >> (64 - 8);
            hash *= 67; // Next-ish prime after 26 + 26 + 10
            hash += *str + overflow;
        }
    }
    return hash;
}// stringHash

} // namespace io ======================================================================

template<typename T>
inline std::ostream&
operator<<(std::ostream& os, const math::BBox<math::Vec3<T>>& b)
{
    os << "(" << b[0][0] << "," << b[0][1] << "," << b[0][2] << ") -> "
       << "(" << b[1][0] << "," << b[1][1] << "," << b[1][2] << ")";
    return os;
}

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
operator<<(std::ostream& os, const math::Vec3<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return os;
}

template<typename T>
inline std::ostream&
operator<<(std::ostream& os, const math::Vec4<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << ")";
    return os;
}

} // namespace nanovdb ===================================================================

#endif // NANOVDB_IO_H_HAS_BEEN_INCLUDED
