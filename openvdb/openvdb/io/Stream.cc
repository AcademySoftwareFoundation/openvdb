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

struct Stream::Impl
{
    Impl(): mOutputStream{nullptr} {}
    Impl(const Impl& other) { *this = other; }
    Impl& operator=(const Impl& other)
    {
        if (&other != this) {
            mMeta = other.mMeta; ///< @todo deep copy?
            mGrids = other.mGrids; ///< @todo deep copy?
            mOutputStream = other.mOutputStream;
            mFile.reset();
        }
        return *this;
    }

    MetaMap::Ptr mMeta;
    GridPtrVecPtr mGrids;
    std::ostream* mOutputStream;
    std::unique_ptr<File> mFile;
};


////////////////////////////////////////


Stream::Stream(std::istream& is): mImpl(new Impl)
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
    mImpl->mMeta.reset(new MetaMap);
    mImpl->mMeta->readMeta(is);

    // Read in the number of grids.
    const int32_t gridCount = readGridCount(is);

    // Read in all grids and insert them into mGrids.
    mImpl->mGrids.reset(new GridPtrVec);
    std::vector<GridDescriptor> descriptors;
    descriptors.reserve(gridCount);
    Archive::NamedGridMap namedGrids;
    for (int32_t i = 0; i < gridCount; ++i) {
        GridDescriptor gd;
        gd.readHeader(is);
        gd.readStreamPos(is);
        descriptors.push_back(gd);
        GridBase::Ptr grid = readGrid(gd, is);
        mImpl->mGrids->push_back(grid);
        namedGrids[gd.uniqueName()] = grid;
    }

    // Connect instances (grids that share trees with other grids).
    for (size_t i = 0, N = descriptors.size(); i < N; ++i) {
        Archive::connectInstance(descriptors[i], namedGrids);
    }
}


Stream::Stream(): mImpl(new Impl)
{
}


Stream::Stream(std::ostream& os): mImpl(new Impl)
{
    mImpl->mOutputStream = &os;
}


Stream::~Stream()
{
}


Stream::Stream(const Stream& other): Archive(other), mImpl(new Impl(*other.mImpl))
{
}


Stream&
Stream::operator=(const Stream& other)
{
    if (&other != this) {
        mImpl.reset(new Impl(*other.mImpl));
    }
    return *this;
}


SharedPtr<Archive>
Stream::copy() const
{
    return SharedPtr<Archive>(new Stream(*this));
}


////////////////////////////////////////


GridBase::Ptr
Stream::readGrid(const GridDescriptor& gd, std::istream& is) const
{
    GridBase::Ptr grid;

    if (!GridBase::isRegistered(gd.gridType())) {
        OPENVDB_THROW(TypeError, "can't read grid \""
            << GridDescriptor::nameAsString(gd.uniqueName()) <<
            "\" from input stream because grid type " << gd.gridType() << " is unknown");
    } else {
        grid = GridBase::createGrid(gd.gridType());
        if (grid) grid->setSaveFloatAsHalf(gd.saveFloatAsHalf());

        Archive::readGrid(grid, gd, is);
    }
    return grid;
}


void
Stream::write(const GridCPtrVec& grids, const MetaMap& metadata) const
{
    if (mImpl->mOutputStream == nullptr) {
        OPENVDB_THROW(ValueError, "no output stream was specified");
    }
    this->writeGrids(*mImpl->mOutputStream, grids, metadata);
}


void
Stream::writeGrids(std::ostream& os, const GridCPtrVec& grids, const MetaMap& metadata) const
{
    Archive::write(os, grids, /*seekable=*/false, metadata);
}


////////////////////////////////////////


MetaMap::Ptr
Stream::getMetadata() const
{
    MetaMap::Ptr result;
    if (mImpl->mMeta) {
        // Return a deep copy of the file-level metadata
        // that was read when this object was constructed.
        result.reset(new MetaMap(*mImpl->mMeta));
    }
    return result;
}


GridPtrVecPtr
Stream::getGrids()
{
    return mImpl->mGrids;
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
