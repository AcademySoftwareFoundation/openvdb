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
/// @file File.cc

#include "File.h"

#include <cassert>
#include <sstream>
#include <boost/cstdint.hpp>
#include <openvdb/Exceptions.h>
#include <openvdb/util/logging.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

File::File(const std::string& filename):
    mFilename(filename),
    mIsOpen(false)
{
    setInputHasGridOffsets(true);
}


File::~File()
{
}


////////////////////////////////////////


bool
File::open()
{
#if defined(_MSC_VER)
    // The original C++ standard library specified that open() only _sets_
    // the fail bit upon an error. It does not clear any bits upon success.
    // This was later addressed by the Library Working Group (LWG) for DR #409
    // and implemented by gcc 4.0. Visual Studio 2008 however is one of those
    // which has not caught up.
    // See: http://gcc.gnu.org/onlinedocs/libstdc++/ext/lwg-defects.html#22
    mInStream.clear();
#endif

    // Open the file.
    mInStream.open(mFilename.c_str(), std::ios_base::in | std::ios_base::binary);

    if (mInStream.fail()) {
        OPENVDB_THROW(IoError, "could not open file " << mFilename);
    }

    // Read in the file header.
    bool newFile = false;
    try {
        newFile = Archive::readHeader(mInStream);
    } catch (IoError& e) {
        mInStream.close();
        if (e.what() && std::string("not a VDB file") == e.what()) {
            // Rethrow, adding the filename.
            OPENVDB_THROW(IoError, mFilename << " is not a VDB file");
        }
        throw;
    }

    // Tag the input stream with the file format and library version numbers.
    Archive::setFormatVersion(mInStream);
    Archive::setLibraryVersion(mInStream);
    Archive::setDataCompression(mInStream);

    // Read in the VDB metadata.
    mMeta = MetaMap::Ptr(new MetaMap);
    mMeta->readMeta(mInStream);

    if (!inputHasGridOffsets()) {
        OPENVDB_LOG_WARN("file " << mFilename << " does not support partial reading");

        mGrids.reset(new GridPtrVec);
        mNamedGrids.clear();

        // Stream in the entire contents of the file and append all grids to mGrids.
        const boost::int32_t gridCount = readGridCount(mInStream);
        for (boost::int32_t i = 0; i < gridCount; ++i) {
            GridDescriptor gd;
            gd.read(mInStream);

            GridBase::Ptr grid = createGrid(gd);
            Archive::readGrid(grid, gd, mInStream);

            mGridDescriptors.insert(std::make_pair(gd.gridName(), gd));
            mGrids->push_back(grid);
            mNamedGrids[gd.uniqueName()] = grid;
        }
        // Connect instances (grids that share trees with other grids).
        for (NameMapCIter it = mGridDescriptors.begin(); it != mGridDescriptors.end(); ++it) {
            Archive::connectInstance(it->second, mNamedGrids);
        }
    } else {
        // Read in just the grid descriptors.
        readGridDescriptors(mInStream);
    }

    mIsOpen = true;
    return newFile; // true if file is not identical to opened file
}


void
File::close()
{
    // Close the stream.
    if (mInStream.is_open()) {
        mInStream.close();
    }

    // Reset all data.
    mMeta.reset();
    mGridDescriptors.clear();
    mGrids.reset();
    mNamedGrids.clear();

    mIsOpen = false;
    setInputHasGridOffsets(true);
}


////////////////////////////////////////


bool
File::hasGrid(const Name& name) const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    return (findDescriptor(name) != mGridDescriptors.end());
}


MetaMap::Ptr
File::getMetadata() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    // Return a deep copy of the file-level metadata, which was read
    // when the file was opened.
    return MetaMap::Ptr(new MetaMap(*mMeta));
}


GridPtrVecPtr
File::getGrids() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }

    GridPtrVecPtr ret;
    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        ret = mGrids;
    } else {
        ret.reset(new GridPtrVec);

        Archive::NamedGridMap namedGrids;

        // Read all grids represented by the GridDescriptors.
        for (NameMapCIter i = mGridDescriptors.begin(), e = mGridDescriptors.end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            GridBase::Ptr grid = readGrid(gd);
            ret->push_back(grid);
            namedGrids[gd.uniqueName()] = grid;
        }

        // Connect instances (grids that share trees with other grids).
        for (NameMapCIter i = mGridDescriptors.begin(), e = mGridDescriptors.end(); i != e; ++i) {
            Archive::connectInstance(i->second, namedGrids);
        }
    }
    return ret;
}


////////////////////////////////////////


GridPtrVecPtr
File::readAllGridMetadata()
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }

    GridPtrVecPtr ret(new GridPtrVec);

    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        for (size_t i = 0, N = mGrids->size(); i < N; ++i) {
            // Return copies of the grids, but with empty trees.
            ret->push_back((*mGrids)[i]->copyGrid(/*treePolicy=*/CP_NEW));
        }
    } else {
        // Read just the metadata and transforms for all grids.
        for (NameMapCIter i = mGridDescriptors.begin(), e = mGridDescriptors.end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            GridBase::ConstPtr grid = readGridPartial(gd, /*readTopology=*/false);
            // Return copies of the grids, but with empty trees.
            // (As of 0.98.0, at least, it would suffice to just const cast
            // the grid pointers returned by readGridPartial(), but shallow
            // copying the grids helps to ensure future compatibility.)
            ret->push_back(grid->copyGrid(/*treePolicy=*/CP_NEW));
        }
    }
    return ret;
}


GridBase::Ptr
File::readGridMetadata(const Name& name)
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    GridBase::ConstPtr ret;
    if (!inputHasGridOffsets()) {
        // Retrieve the grid from mGrids, which should already contain
        // the entire contents of the file.
        ret = readGrid(name);
    } else {
        NameMapCIter it = findDescriptor(name);
        if (it == mGridDescriptors.end()) {
            OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
        }

        // Seek to and read in the grid from the file.
        const GridDescriptor& gd = it->second;
        ret = readGridPartial(gd, /*readTopology=*/false);
    }
    return ret->copyGrid(/*treePolicy=*/CP_NEW);
}


////////////////////////////////////////


GridBase::ConstPtr
File::readGridPartial(const Name& name)
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    GridBase::ConstPtr ret;
    if (!inputHasGridOffsets()) {
        // Retrieve the grid from mGrids, which should already contain
        // the entire contents of the file.
        if (GridBase::Ptr grid = readGrid(name)) {
            ret = boost::const_pointer_cast<const GridBase>(grid);
        }
    } else {
        NameMapCIter it = findDescriptor(name);
        if (it == mGridDescriptors.end()) {
            OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
        }

        // Seek to and read in the grid from the file.
        const GridDescriptor& gd = it->second;
        ret = readGridPartial(gd, /*readTopology=*/true);

        if (gd.isInstance()) {
            NameMapCIter parentIt =
                findDescriptor(GridDescriptor::nameAsString(gd.instanceParentName()));
            if (parentIt == mGridDescriptors.end()) {
                OPENVDB_THROW(KeyError, "missing instance parent \""
                    << GridDescriptor::nameAsString(gd.instanceParentName())
                    << "\" for grid " << GridDescriptor::nameAsString(gd.uniqueName())
                    << " in file " << mFilename);
            }
            if (GridBase::ConstPtr parent =
                readGridPartial(parentIt->second, /*readTopology=*/true))
            {
                if (Archive::isInstancingEnabled()) {
                    // Share the instance parent's tree.
                    boost::const_pointer_cast<GridBase>(ret)->setTree(
                        boost::const_pointer_cast<GridBase>(parent)->baseTreePtr());
                } else {
                    // Copy the instance parent's tree.
                    boost::const_pointer_cast<GridBase>(ret)->setTree(
                        parent->baseTree().copy());
                }
            }
        }
    }
    return ret;
}


GridBase::Ptr
File::readGrid(const Name& name)
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    GridBase::Ptr ret;
    if (!inputHasGridOffsets()) {
        // Retrieve the grid from mNamedGrids, which should already contain
        // the entire contents of the file.

        // Search by unique name.
        Archive::NamedGridMap::const_iterator it =
            mNamedGrids.find(GridDescriptor::stringAsUniqueName(name));
        // If not found, search by grid name.
        if (it == mNamedGrids.end()) it = mNamedGrids.find(name);
        if (it == mNamedGrids.end()) {
            OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
        }
        ret = it->second;
    } else {
        NameMapCIter it = findDescriptor(name);
        if (it == mGridDescriptors.end()) {
            OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
        }

        // Seek to and read in the grid from the file.
        const GridDescriptor& gd = it->second;
        ret = readGrid(gd);

        if (gd.isInstance()) {
            NameMapCIter parentIt =
                findDescriptor(GridDescriptor::nameAsString(gd.instanceParentName()));
            if (parentIt == mGridDescriptors.end()) {
                OPENVDB_THROW(KeyError, "missing instance parent \""
                    << GridDescriptor::nameAsString(gd.instanceParentName())
                    << "\" for grid " << GridDescriptor::nameAsString(gd.uniqueName())
                    << " in file " << mFilename);
            }
            if (GridBase::Ptr parent = readGrid(parentIt->second)) {
                if (Archive::isInstancingEnabled()) {
                    // Share the instance parent's tree.
                    ret->setTree(parent->baseTreePtr());
                } else {
                    // Copy the instance parent's tree.
                    ret->setTree(parent->baseTree().copy());
                }
            }
        }
    }
    return ret;
}


////////////////////////////////////////


void
File::writeGrids(const GridCPtrVec& grids, const MetaMap& metadata) const
{
    if (isOpen()) {
        OPENVDB_THROW(IoError,
            mFilename << " cannot be written because it is open for reading");
    }

    // Create a file stream and write it out.
    std::ofstream file;
    file.open(mFilename.c_str(),
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    if (file.fail()) {
        OPENVDB_THROW(IoError, "could not open " << mFilename << " for writing");
    }

    // Write out the vdb.
    Archive::write(file, grids, /*seekable=*/true, metadata);

    file.close();
}


////////////////////////////////////////


void
File::readGridDescriptors(std::istream& is)
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    mGridDescriptors.clear();

    for (boost::int32_t i = 0, N = readGridCount(is); i < N; ++i) {
        // Read the grid descriptor.
        GridDescriptor gd;
        gd.read(is);

        // Add the descriptor to the dictionary.
        mGridDescriptors.insert(std::make_pair(gd.gridName(), gd));

        // Skip forward to the next descriptor.
        gd.seekToEnd(is);
    }
}


////////////////////////////////////////


File::NameMapCIter
File::findDescriptor(const Name& name) const
{
    const Name uniqueName = GridDescriptor::stringAsUniqueName(name);

    // Find all descriptors with the given grid name.
    std::pair<NameMapCIter, NameMapCIter> range = mGridDescriptors.equal_range(name);

    if (range.first == range.second) {
        // If no descriptors were found with the given grid name, the name might have
        // a suffix ("name[N]").  In that case, remove the "[N]" suffix and search again.
        range = mGridDescriptors.equal_range(GridDescriptor::stripSuffix(uniqueName));
    }

    const size_t count = size_t(std::distance(range.first, range.second));
    if (count > 1 && name == uniqueName) {
        OPENVDB_LOG_WARN(mFilename << " has more than one grid named \"" << name << "\"");
    }

    NameMapCIter ret = mGridDescriptors.end();

    if (count > 0) {
        if (name == uniqueName) {
            // If the given grid name is unique or if no "[N]" index was given,
            // use the first matching descriptor.
            ret = range.first;
        } else {
            // If the given grid name has a "[N]" index, find the descriptor
            // with a matching unique name.
            for (NameMapCIter it = range.first; it != range.second; ++it) {
                const Name candidateName = it->second.uniqueName();
                if (candidateName == uniqueName || candidateName == name) {
                    ret = it;
                    break;
                }
            }
        }
    }
    return ret;
}


////////////////////////////////////////


GridBase::Ptr
File::createGrid(const GridDescriptor& gd) const
{
    // Create the grid.
    if (!GridBase::isRegistered(gd.gridType())) {
        OPENVDB_THROW(KeyError, "Cannot read grid "
            << GridDescriptor::nameAsString(gd.uniqueName())
            << " from " << mFilename << ": grid type "
            << gd.gridType() << " is not registered");
    }

    GridBase::Ptr grid = GridBase::createGrid(gd.gridType());
    if (grid) grid->setSaveFloatAsHalf(gd.saveFloatAsHalf());

    return grid;
}


GridBase::ConstPtr
File::readGridPartial(const GridDescriptor& gd, bool readTopology) const
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    GridBase::Ptr grid = createGrid(gd);

    // Seek to grid.
    gd.seekToGrid(mInStream);

    // Read the grid partially.
    readGridPartial(grid, mInStream, gd.isInstance(), readTopology);

    // Promote to a const grid.
    GridBase::ConstPtr constGrid = grid;

    return constGrid;
}


GridBase::Ptr
File::readGrid(const GridDescriptor& gd) const
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    GridBase::Ptr grid = createGrid(gd);

    // Seek to where the grid is.
    gd.seekToGrid(mInStream);

    // Read in the grid.
    Archive::readGrid(grid, gd, mInStream);

    return grid;
}


void
File::readGridPartial(GridBase::Ptr grid, std::istream& is,
    bool isInstance, bool readTopology) const
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    // This code needs to stay in sync with io::Archive::readGrid(), in terms of
    // the order of operations.
    readGridCompression(is);
    grid->readMeta(is);
    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_GRID_INSTANCING) {
        grid->readTransform(is);
        if (!isInstance && readTopology) {
            grid->readTopology(is);
        }
    } else {
        if (readTopology) {
            grid->readTopology(is);
            grid->readTransform(is);
        }
    }
}


////////////////////////////////////////


File::NameIterator
File::beginName() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    return File::NameIterator(mGridDescriptors.begin());
}


File::NameIterator
File::endName() const
{
    return File::NameIterator(mGridDescriptors.end());
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
