// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file io/File.cc

#include "File.h"

#include <openvdb/Exceptions.h>
#include <openvdb/util/logging.h>
#include <openvdb/util/Assert.h>
#include <cstdint>

#include <sys/stat.h> // stat()

#include <cstdlib> // for getenv(), strtoul()
#include <cstring> // for strerror_r()
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {


File::File(const std::string& filename)
    : Archive()
    , mFilename(filename)
{
    setInputHasGridOffsets(true);
}


File::File(const File& other)
    : Archive(other)
    , mFilename(other.mFilename)
    , mMeta(other.mMeta)
    , mIsOpen(false)
    , mGridDescriptors(other.mGridDescriptors)
    , mNamedGrids(other.mNamedGrids)
    , mGrids(other.mGrids)
{
}


File&
File::operator=(const File& other)
{
    if (&other != this) {
        Archive::operator=(other);
        mFilename = other.mFilename;
        mMeta = other.mMeta;
        mIsOpen = false; // don't want two file objects reading from the same stream
        mGridDescriptors = other.mGridDescriptors;
        mNamedGrids = other.mNamedGrids;
        mGrids = other.mGrids;
    }
    return *this;
}


SharedPtr<Archive>
File::copy() const
{
    return SharedPtr<Archive>{new File{*this}};
}


////////////////////////////////////////


const std::string&
File::filename() const
{
    return mFilename;
}


MetaMap::Ptr
File::fileMetadata()
{
    return mMeta;
}

MetaMap::ConstPtr
File::fileMetadata() const
{
    return mMeta;
}


const File::NameMap&
File::gridDescriptors() const
{
    return mGridDescriptors;
}

File::NameMap&
File::gridDescriptors()
{
    return mGridDescriptors;
}


std::istream&
File::inputStream() const
{
    if (!mInStream) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    return *mInStream;
}


////////////////////////////////////////


Index64
File::getSize() const
{
    /// @internal boost::filesystem::file_size() would be a more portable alternative,
    /// but as of 9/2014, Houdini ships without the Boost.Filesystem library,
    /// which makes it much less convenient to use that library.

    Index64 result = std::numeric_limits<Index64>::max();

    std::string mesg = "could not get size of file " + mFilename;

#ifdef _WIN32
    // Get the file size by seeking to the end of the file.
    std::ifstream fstrm(mFilename);
    if (fstrm) {
        fstrm.seekg(0, fstrm.end);
        result = static_cast<Index64>(fstrm.tellg());
    } else {
        OPENVDB_THROW(IoError, mesg);
    }
#else
    // Get the file size using the stat() system call.
    struct stat info;
    if (0 != ::stat(mFilename.c_str(), &info)) {
        std::string s = getErrorString();
        if (!s.empty()) mesg += " (" + s + ")";
        OPENVDB_THROW(IoError, mesg);
    }
    if (!S_ISREG(info.st_mode)) {
        mesg += " (not a regular file)";
        OPENVDB_THROW(IoError, mesg);
    }
    result = static_cast<Index64>(info.st_size);
#endif

    return result;
}


////////////////////////////////////////


bool
File::isOpen() const
{
    return mIsOpen;
}


bool
File::open()
{
    if (mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is already open");
    }
    mInStream.reset();

    // Open the file using standard I/O (delayed loading has been removed)
    std::unique_ptr<std::istream> newStream;
    newStream.reset(new std::ifstream(
        mFilename.c_str(), std::ios_base::in | std::ios_base::binary));

    if (newStream->fail()) {
        OPENVDB_THROW(IoError, "could not open file " << mFilename);
    }

    // Read in the file header.
    bool newFile = false;
    try {
        newFile = Archive::readHeader(*newStream);
    } catch (IoError& e) {
        if (e.what() && std::string("not a VDB file") == e.what()) {
            // Rethrow, adding the filename.
            OPENVDB_THROW(IoError, mFilename << " is not a VDB file");
        }
        throw;
    }

    mInStream.swap(newStream);

    // Tag the input stream with the file format and library version numbers
    // and other metadata.
    mStreamMetadata.reset(new StreamMetadata);
    mStreamMetadata->setSeekable(true);
    io::setStreamMetadataPtr(inputStream(), mStreamMetadata, /*transfer=*/false);
    Archive::setFormatVersion(inputStream());
    Archive::setLibraryVersion(inputStream());
    Archive::setDataCompression(inputStream());

    // Read in the VDB metadata.
    mMeta = MetaMap::Ptr(new MetaMap);
    mMeta->readMeta(inputStream());

    if (!inputHasGridOffsets()) {
        OPENVDB_LOG_DEBUG_RUNTIME("file " << mFilename << " does not support partial reading");

        mGrids.reset(new GridPtrVec);
        mNamedGrids.clear();

        // Stream in the entire contents of the file and append all grids to mGrids.
        const int32_t gridCount = readGridCount(inputStream());
        for (int32_t i = 0; i < gridCount; ++i) {
            GridDescriptor gd;
            gd.readHeader(inputStream());
            gd.readStreamPos(inputStream());

            GridBase::Ptr grid = createGrid(gd);
            Archive::readGrid(grid, gd, inputStream(), BBoxd());

            mGridDescriptors.insert(std::make_pair(gd.gridName(), gd));
            mGrids->push_back(grid);
            mNamedGrids[gd.uniqueName()] = grid;
        }
        // Connect instances (grids that share trees with other grids).
        for (NameMapCIter it = mGridDescriptors.begin(); it != mGridDescriptors.end(); ++it) {
            Archive::connectInstance(it->second, mNamedGrids);
        }
    } else {
        mGridDescriptors.clear();

        for (int32_t i = 0, N = readGridCount(inputStream()); i < N; ++i) {
            // Read the grid descriptor.
            GridDescriptor gd;
            gd.readHeader(inputStream());
            gd.readStreamPos(inputStream());

            // Add the descriptor to the dictionary.
            mGridDescriptors.insert(std::make_pair(gd.gridName(), gd));

            // Skip forward to the next descriptor.
            gd.seekToEnd(inputStream());
        }
    }

    mIsOpen = true;
    return newFile; // true if file is not identical to opened file
}


void
File::close()
{
    // Reset all data.
    mMeta.reset();
    mGridDescriptors.clear();
    mGrids.reset();
    mNamedGrids.clear();
    mInStream.reset();
    mStreamMetadata.reset();

    mIsOpen = false;
    setInputHasGridOffsets(true);
}


////////////////////////////////////////


bool
File::hasGrid(const Name& name) const
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    return (findDescriptor(name) != mGridDescriptors.end());
}


MetaMap::Ptr
File::getMetadata() const
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }
    // Return a deep copy of the file-level metadata, which was read
    // when the file was opened.
    return MetaMap::Ptr(new MetaMap(*mMeta));
}


GridPtrVecPtr
File::getGrids() const
{
    if (!mIsOpen) {
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
            GridBase::Ptr grid = readGrid(gd, BBoxd());
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


GridBase::Ptr
File::retrieveCachedGrid(const Name& name) const
{
    // If the file has grid offsets, grids are read on demand
    // and not cached in mNamedGrids.
    if (inputHasGridOffsets()) return GridBase::Ptr();

    // If the file does not have grid offsets, mNamedGrids should already
    // contain the entire contents of the file.

    // Search by unique name.
    Archive::NamedGridMap::const_iterator it =
        mNamedGrids.find(GridDescriptor::stringAsUniqueName(name));
    // If not found, search by grid name.
    if (it == mNamedGrids.end()) it = mNamedGrids.find(name);
    if (it == mNamedGrids.end()) {
        OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
    }
    return it->second;
}


////////////////////////////////////////


GridPtrVecPtr
File::readAllGridMetadata()
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }

    if (fileVersion() < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        OPENVDB_THROW(IoError,
            "VDB file version < 222 (NODE_MASK_COMPRESSION) is no longer supported.");
    }

    GridPtrVecPtr ret(new GridPtrVec);

    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        for (size_t i = 0, N = mGrids->size(); i < N; ++i) {
            // Return copies of the grids, but with empty trees.
            ret->push_back((*mGrids)[i]->copyGridWithNewTree());
        }
    } else {
        // Read just the metadata and transforms for all grids.
        for (NameMapCIter i = mGridDescriptors.begin(), e = mGridDescriptors.end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            GridBase::ConstPtr grid = readGridPartial(gd);
            // Return copies of the grids, but with empty trees.
            // (As of 0.98.0, at least, it would suffice to just const cast
            // the grid pointers returned by readGridPartial(), but shallow
            // copying the grids helps to ensure future compatibility.)
            ret->push_back(grid->copyGridWithNewTree());
        }
    }
    return ret;
}


GridBase::Ptr
File::readGridMetadata(const Name& name)
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    if (fileVersion() < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        OPENVDB_THROW(IoError,
            "VDB file version < 222 (NODE_MASK_COMPRESSION) is no longer supported.");
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
        ret = readGridPartial(gd);
    }
    return ret->copyGridWithNewTree();
}


////////////////////////////////////////


GridBase::Ptr
File::readGrid(const Name& name)
{
    return readGrid(name, BBoxd());
}


GridBase::Ptr
File::readGrid(const Name& name, const BBoxd& bbox)
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    const bool clip = bbox.isSorted();

    // If a grid with the given name was already read and cached
    // (along with the entire contents of the file, because the file
    // doesn't support random access), retrieve and return it.
    GridBase::Ptr grid = retrieveCachedGrid(name);
    if (grid) {
        if (clip) {
            grid = grid->deepCopyGrid();
            grid->clipGrid(bbox);
        }
        return grid;
    }

    NameMapCIter it = findDescriptor(name);
    if (it == mGridDescriptors.end()) {
        OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
    }

    // Seek to and read in the grid from the file.
    const GridDescriptor& gd = it->second;
    grid = readGrid(gd, bbox);

    if (gd.isInstance()) {
        /// @todo Refactor to share code with Archive::connectInstance()?
        NameMapCIter parentIt =
            findDescriptor(GridDescriptor::nameAsString(gd.instanceParentName()));
        if (parentIt == mGridDescriptors.end()) {
            OPENVDB_THROW(KeyError, "missing instance parent \""
                << GridDescriptor::nameAsString(gd.instanceParentName())
                << "\" for grid " << GridDescriptor::nameAsString(gd.uniqueName())
                << " in file " << mFilename);
        }

        GridBase::Ptr parent;
        parent = readGrid(parentIt->second, bbox);
        if (parent) grid->setTree(parent->baseTreePtr());
    }
    return grid;
}


////////////////////////////////////////


void
File::writeGrids(const GridCPtrVec& grids, const MetaMap& meta) const
{
    if (mIsOpen) {
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
    Archive::write(file, grids, /*seekable=*/true, meta);

    file.close();
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
File::readGridPartial(const GridDescriptor& gd) const
{
    // This method should not be called for files that don't contain grid offsets.
    OPENVDB_ASSERT(inputHasGridOffsets());

    std::istream& is = inputStream();

    GridBase::Ptr grid = createGrid(gd);

    // Seek to grid.
    gd.seekToGrid(is);

    // This code needs to stay in sync with io::Archive::readGrid(), in terms of
    // the order of operations.
    readGridCompression(is);
    grid->readMeta(is);

    // Delayed loading is no longer supported - always remove metadata related to delayed loading if it exists
    if ((*grid)[GridBase::META_FILE_DELAYED_LOAD]) {
        grid->removeMeta(GridBase::META_FILE_DELAYED_LOAD);
    }

    // Read the transform.
    grid->readTransform(is);

    // Promote to a const grid.
    GridBase::ConstPtr constGrid = grid;

    return constGrid;
}


GridBase::Ptr
File::readGrid(const GridDescriptor& gd, const BBoxd& bbox) const
{
    // This method should not be called for files that don't contain grid offsets.
    OPENVDB_ASSERT(inputHasGridOffsets());

    GridBase::Ptr grid = createGrid(gd);
    gd.seekToGrid(inputStream());
    Archive::readGrid(grid, gd, inputStream(), bbox);
    return grid;
}


////////////////////////////////////////


File::NameIterator
File::beginName() const
{
    if (!mIsOpen) {
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
