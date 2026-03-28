// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "Stream.h"

#include "File.h" ///< @todo refactor
#include "GridDescriptor.h"
#include <openvdb/Exceptions.h>
#include <cstdint>

#include <cstdio> // for remove()
#include <functional> // for std::bind()
#include <iostream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {


Stream::Stream(std::istream& is)
{
    if (!is) return;

    // Delayed loading has been removed - always read directly from the stream
    readHeader(is);

    // Tag the input stream with the library and file format version numbers
    // and the compression options specified in the header.
    StreamMetadata::Ptr streamMetadata(new StreamMetadata);
    io::setStreamMetadataPtr(is, streamMetadata, /*transfer=*/false);
    io::setVersion(is, libraryVersion(), fileVersion());
    io::setDataCompression(is, compression());

    // Read in the VDB metadata.
    mMeta.reset(new MetaMap);
    mMeta->readMeta(is);

    // Read in the number of grids.
    const int32_t gridCount = readGridCount(is);

    // Read in all grids and insert them into mGrids.
    mGrids.reset(new GridPtrVec);
    std::vector<GridDescriptor> descriptors;
    descriptors.reserve(gridCount);
    Archive::NamedGridMap namedGrids;
    for (int32_t i = 0; i < gridCount; ++i) {
        GridDescriptor gd;
        gd.readHeader(is);
        gd.readStreamPos(is);
        descriptors.push_back(gd);
        GridBase::Ptr grid = Archive::readGrid(gd, is, io::ReadOptions{});
        mGrids->push_back(grid);
        namedGrids[gd.uniqueName()] = grid;
    }

    // Connect instances (grids that share trees with other grids).
    for (size_t i = 0, N = descriptors.size(); i < N; ++i) {
        Archive::connectInstance(descriptors[i], namedGrids);
    }
}


Stream::Stream(std::ostream& os)
    : Archive()
    , mOutputStream(&os)
{
}


Stream::Stream(const Stream& other)
    : Archive(other)
    , mMeta(other.mMeta)
    , mGrids(other.mGrids)
    , mOutputStream(other.mOutputStream)
{
}


Stream&
Stream::operator=(const Stream& other)
{
    if (&other != this) {
        mMeta = other.mMeta;
        mGrids = other.mGrids;
        mOutputStream = other.mOutputStream;
    }
    return *this;
}


SharedPtr<Archive>
Stream::copy() const
{
    return SharedPtr<Archive>(new Stream(*this));
}


////////////////////////////////////////


void
Stream::write(const GridCPtrVec& grids, const MetaMap& metadata,
    const io::WriteOptions& writeOptions) const
{
    if (mOutputStream == nullptr) {
        OPENVDB_THROW(ValueError, "no output stream was specified");
    }
    this->writeGrids(*mOutputStream, grids, metadata, writeOptions);
}


void
Stream::writeGrids(std::ostream& os, const GridCPtrVec& grids, const MetaMap& metadata,
    const io::WriteOptions& writeOptions) const
{
    Archive::write(os, grids, /*seekable=*/false, metadata, writeOptions);
}


////////////////////////////////////////


MetaMap::Ptr
Stream::getMetadata() const
{
    MetaMap::Ptr result;
    if (mMeta) {
        // Return a deep copy of the file-level metadata
        // that was read when this object was constructed.
        result.reset(new MetaMap(*mMeta));
    }
    return result;
}


GridPtrVecPtr
Stream::getGrids()
{
    return mGrids;
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
