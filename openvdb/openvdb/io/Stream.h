// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED
#define OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED

#include "Archive.h"
#include <iosfwd>
#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

class GridDescriptor;


/// Grid archive associated with arbitrary input and output streams (not necessarily files)
class OPENVDB_API Stream: public Archive
{
public:
    /// @brief Read grids from an input stream.
    /// @param is The input stream to read from
    explicit Stream(std::istream&);

    OPENVDB_DEPRECATED_MESSAGE("Use Stream(std::istream&) instead. This method is deprecated and will be removed. Delayed loading is no longer supported.")
    Stream(std::istream& is, bool /*delayLoad*/) : Stream(is) { }

    /// Construct an archive for stream output.
    Stream() = default;
    /// Construct an archive for output to the given stream.
    explicit Stream(std::ostream&);

    Stream(const Stream&);
    Stream& operator=(const Stream&);

    ~Stream() override { }

    /// @brief Return a copy of this archive.
    Archive::Ptr copy() const override;

    /// Return the file-level metadata in a newly created MetaMap.
    MetaMap::Ptr getMetadata() const;

    /// Return pointers to the grids that were read from the input stream.
    GridPtrVecPtr getGrids();

    /// @brief Write the grids in the given container to this archive's output stream.
    /// @throw ValueError if this archive was constructed without specifying an output stream.
    void write(const GridCPtrVec&, const MetaMap& = MetaMap(),
        const io::WriteOptions& = io::WriteOptions{}) const override;

    /// @brief Write the grids in the given container to this archive's output stream.
    /// @throw ValueError if this archive was constructed without specifying an output stream.
    template<typename GridPtrContainerT>
    void write(const GridPtrContainerT&, const MetaMap& = MetaMap(),
        const io::WriteOptions& = io::WriteOptions{}) const;

private:
    void writeGrids(std::ostream&, const GridCPtrVec&, const MetaMap&,
        const io::WriteOptions&) const;

    MetaMap::Ptr mMeta;
    GridPtrVecPtr mGrids;
    std::ostream* mOutputStream = nullptr;
};


////////////////////////////////////////


template<typename GridPtrContainerT>
inline void
Stream::write(const GridPtrContainerT& container, const MetaMap& metadata,
    const io::WriteOptions& writeOptions) const
{
    GridCPtrVec grids;
    std::copy(container.begin(), container.end(), std::back_inserter(grids));
    this->write(grids, metadata, writeOptions);
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED
