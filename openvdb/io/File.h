///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file File.h

#ifndef OPENVDB_IO_FILE_HAS_BEEN_INCLUDED
#define OPENVDB_IO_FILE_HAS_BEEN_INCLUDED

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include "Archive.h"
#include "GridDescriptor.h"


class TestFile;
class TestStream;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

/// Grid archive associated with a file on disk
class OPENVDB_API File: public Archive
{
public:
    typedef std::multimap<Name, GridDescriptor> NameMap;
    typedef NameMap::const_iterator NameMapCIter;

    explicit File(const std::string& filename);
    ~File();

    const std::string& filename() const { return mFilename; }

    /// Open the file, read the file header and the file-level metadata, and
    /// populate the grid descriptors, but do not load any grids into memory.
    /// @throw IoError if the file is not a valid VDB file.
    /// @return @c true if the file's UUID has changed since it was last read.
    bool open();

    /// Return @c true if the file has been opened for reading, false otherwise.
    bool isOpen() const { return mIsOpen; }

    /// Close the file once we are done reading from it.
    void close();

    /// Return @c true if a grid of the given name exists in this file.
    bool hasGrid(const Name&) const;

    /// Return (in a newly created MetaMap) the file-level metadata.
    MetaMap::Ptr getMetadata() const;

    /// Read the entire contents of the file and return a list of grid pointers.
    GridPtrVecPtr getGrids() const;

    /// @brief Read just the grid metadata and transforms from the file and return a list
    /// of pointers to grids that are empty except for their metadata and transforms.
    /// @throw IoError if this file is not open for reading.
    GridPtrVecPtr readAllGridMetadata();

    /// @brief Read a grid's metadata and transform only.
    /// @return A pointer to a grid that is empty except for its metadata and transform.
    /// @throw IoError if this file is not open for reading.
    /// @throw KeyError if no grid with the given name exists in this file.
    GridBase::Ptr readGridMetadata(const Name&);

    /// @brief Read a grid's metadata, topology, transform, etc., but not
    /// any of its leaf node data blocks.
    /// @return the grid pointer to the partially loaded grid.
    /// @note This returns a @c const pointer, so that the grid can't be
    /// changed before its data blocks have been loaded.  A non-<tt>const</tt>
    /// pointer is only returned when readGrid() is called.
    GridBase::ConstPtr readGridPartial(const Name&);

    /// Read an entire grid, including all of its data blocks.
    GridBase::Ptr readGrid(const Name&);

    /// @todo GridPtrVec readAllGridsPartial(const Name&)
    /// @todo GridPtrVec readAllGrids(const Name&)

    /// @brief Write the grids in the given container to the file whose name
    /// was given in the constructor.
    template<typename GridPtrContainerT>
    void write(const GridPtrContainerT&, const MetaMap& = MetaMap()) const;

    /// A const iterator that iterates over all names in the file. This is only
    /// valid once the file has been opened.
    class NameIterator
    {
    public:
        NameIterator(const NameMapCIter& iter): mIter(iter) {}
        ~NameIterator() {}

        NameIterator& operator++() { mIter++; return *this; }

        bool operator==(const NameIterator& iter) const { return mIter == iter.mIter; }
        bool operator!=(const NameIterator& iter) const { return mIter != iter.mIter; }

        Name operator*() const { return this->gridName(); }

        Name gridName() const { return GridDescriptor::nameAsString(mIter->second.uniqueName()); }

    private:
        NameMapCIter mIter;
    };

    /// @return a NameIterator to iterate over all grid names in the file.
    NameIterator beginName() const;

    /// @return the ending iterator for all grid names in the file.
    NameIterator endName() const;

private:
    /// Resets the input stream to the beginning.
    void resetInStream() const { mInStream.seekg(0, std::ios::beg); }

    /// Read in all grid descriptors that are stored in the given stream.
    void readGridDescriptors(std::istream&);

    /// @brief Return an iterator to the descriptor for the grid with the given name.
    /// If the name is non-unique, return an iterator to the first matching descriptor.
    NameMapCIter findDescriptor(const Name&) const;

    /// Return a newly created, empty grid of the type specified by the given grid descriptor.
    GridBase::Ptr createGrid(const GridDescriptor&) const;

    /// Read in and return the partially-populated grid specified by the given grid descriptor.
    GridBase::ConstPtr readGridPartial(const GridDescriptor&, bool readTopology) const;

    /// Read in and return the grid specified by the given grid descriptor.
    GridBase::Ptr readGrid(const GridDescriptor&) const;

    /// Partially populate the given grid by reading its metadata and transform and,
    /// if the grid is not an instance, its tree structure, but not the tree's leaf nodes.
    void readGridPartial(GridBase::Ptr, std::istream&, bool isInstance, bool readTopology) const;

    void writeGrids(const GridCPtrVec&, const MetaMap&) const;

    // Disallow copying of instances of this class.
    File(const File& other);
    File& operator=(const File& other);

    friend class ::TestFile;
    friend class ::TestStream;


    std::string mFilename;
    /// The file-level metadata
    MetaMap::Ptr mMeta;
    /// The file stream that is open for reading
    mutable std::ifstream mInStream;
    /// Flag indicating if we have read in the global information (header,
    /// metadata, and grid descriptors) for this VDB file
    bool mIsOpen;
    /// Grid descriptors for all grids stored in the file, indexed by grid name
    NameMap mGridDescriptors;
    /// All grids, indexed by unique name (used only when mHasGridOffsets is false)
    Archive::NamedGridMap mNamedGrids;
    /// All grids stored in the file (used only when mHasGridOffsets is false)
    GridPtrVecPtr mGrids;
};


////////////////////////////////////////


template<typename GridPtrContainerT>
inline void
File::write(const GridPtrContainerT& container, const MetaMap& metadata) const
{
    GridCPtrVec grids;
    std::copy(container.begin(), container.end(), std::back_inserter(grids));
    this->writeGrids(grids, metadata);
}

template<>
inline void
File::write<GridCPtrVec>(const GridCPtrVec& grids, const MetaMap& metadata) const
{
    this->writeGrids(grids, metadata);
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_FILE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
