// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file io/File.cc

#include "File.h"

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h> // for GridTypes
#include <openvdb/util/logging.h>
#include <openvdb/util/Assert.h>
#include <cstdint>

#include <sys/stat.h> // stat()

#include <cstdlib> // for getenv(), strtoul()
#include <cstring> // for strerror_r()
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

namespace {

/// @brief Convert @a source to the grid type that @a readOptions would have
///   produced had it been read through @a codec (looked up by the caller via
///   the protected @c Archive::findCodec(), since this is a free function).
///   Returns null if no conversion is needed or possible, in which case the
///   caller keeps the original grid.  A conversion that was requested but cannot
///   be done is reported to @a diagnostics and logged against @a filename.
GridBase::Ptr convertGridForReadMode(const GridBase& source, const ReadOptions& readOptions,
    Codec* codec, ReadDiagnostics& diagnostics, const std::string& filename);

/// @brief Return the name used in diagnostics and log messages for @a mode.
std::string readModeName(ReadMode mode);

} // anonymous namespace


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

            GridBase::Ptr grid = Archive::readGrid(gd, inputStream(), io::ReadOptions{});

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
File::getGrids(const io::ReadOptions& readOptions) const
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading");
    }

    GridPtrVecPtr ret;
    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        const auto& bbox = readOptions.clipBBox;
        const bool clip = bbox.isSorted();

        if (readOptions.readMode == io::ReadMode::Original && !clip) {
            // Nothing to convert or clip: preserve pointer identity with mGrids.
            ret = mGrids;
        } else {
            ret.reset(new GridPtrVec);

            // Instances (grids sharing a source tree) share the converted tree
            // too, unless instancing is disabled. Under a clip, share only
            // when transforms agree, since a clip depends on each grid's own
            // transform.
            const bool shareConvertedTrees = isInstancingEnabled() &&
                readOptions.readMode != io::ReadMode::MetadataOnly;
            struct Resolved { GridBase::Ptr grid; math::Transform::ConstPtr transform; };
            std::map<const TreeBase*, Resolved> resolvedBySourceTree;

            for (const auto& cachedGrid : *mGrids) {
                const TreeBase* sourceTree = &cachedGrid->constBaseTree();
                GridBase::Ptr grid;

                if (shareConvertedTrees) {
                    auto it = resolvedBySourceTree.find(sourceTree);
                    if (it != resolvedBySourceTree.end() &&
                        (!clip || *it->second.transform == cachedGrid->transform()))
                    {
                        const GridBase::Ptr& resolved = it->second.grid;
                        grid = resolved->copyGridWithNewTree();
                        grid->clearMetadata();
                        grid->insertMeta(*cachedGrid);
                        grid->setTransform(cachedGrid->transformPtr());
                        grid->setTree(resolved->baseTreePtr());
                        ret->push_back(grid);
                        continue;
                    }
                }

                grid = resolveCachedGrid(cachedGrid, readOptions, mReadDiagnostics);

                if (shareConvertedTrees) {
                    // Keep the first-seen entry as canonical, so a later
                    // mismatched transform under clip doesn't overwrite it.
                    resolvedBySourceTree.try_emplace(
                        sourceTree, Resolved{grid, cachedGrid->transformPtr()});
                }
                ret->push_back(grid);
            }
        }
    } else {
        ret.reset(new GridPtrVec);

        Archive::NamedGridMap namedGrids;

        // Read all grids represented by the GridDescriptors.
        for (NameMapCIter i = mGridDescriptors.begin(), e = mGridDescriptors.end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            // Seek to the grid in the file.
            gd.seekToGrid(inputStream());
            GridBase::Ptr grid = Archive::readGrid(gd, inputStream(), readOptions, mReadDiagnostics);
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

    if (fileVersion() < OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX) {
        OPENVDB_THROW(IoError,
            "VDB file version < 221 (FLOAT_FRUSTUM_BBOX) is no longer supported.");
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
            // Seek to the grid in the file.
            gd.seekToGrid(inputStream());
            io::ReadOptions readOptions;
            readOptions.readMode = io::ReadMode::MetadataOnly;
            GridBase::ConstPtr grid = Archive::readGrid(gd, inputStream(), readOptions);
            // Return copies of the grids, but with empty trees.
            // (As of 0.98.0, at least, it would suffice to just const cast
            // the grid pointers returned by readGrid(partial=true), but shallow
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

    if (fileVersion() < OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX) {
        OPENVDB_THROW(IoError,
            "VDB file version < 221 (FLOAT_FRUSTUM_BBOX) is no longer supported.");
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
        gd.seekToGrid(inputStream());
        io::ReadOptions readOptions;
        readOptions.readMode = io::ReadMode::MetadataOnly;
        ret = Archive::readGrid(gd, inputStream(), readOptions);
    }
    return ret->copyGridWithNewTree();
}


////////////////////////////////////////


GridBase::Ptr
File::readGrid(const Name& name, const BBoxd& bbox)
{
    io::ReadOptions readOptions;
    readOptions.clipBBox = bbox;
    return readGrid(name, readOptions);
}


GridBase::Ptr
File::readGrid(const Name& name, const io::ReadOptions& readOptions)
{
    if (!mIsOpen) {
        OPENVDB_THROW(IoError, mFilename << " is not open for reading.");
    }

    // If a grid with the given name was already read and cached
    // (along with the entire contents of the file, because the file
    // doesn't support random access), retrieve and return it.
    GridBase::Ptr cachedGrid = retrieveCachedGrid(name);
    GridBase::Ptr grid;
    if (cachedGrid) {
        return resolveCachedGrid(cachedGrid, readOptions, mReadDiagnostics);
    }

    NameMapCIter it = findDescriptor(name);
    if (it == mGridDescriptors.end()) {
        OPENVDB_THROW(KeyError, mFilename << " has no grid named \"" << name << "\"");
    }

    // Seek to and read in the grid from the file.
    const GridDescriptor& gd = it->second;
    // This method should not be called for files that don't contain grid offsets.
    OPENVDB_ASSERT(inputHasGridOffsets());
    // Seek to the grid in the file.
    gd.seekToGrid(inputStream());
    grid = Archive::readGrid(gd, inputStream(), readOptions, mReadDiagnostics);

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

        // Read the parent without clipping. Archive::readGrid() converts the
        // world-space clip region into index space using the grid's own
        // transform, but an instance has its own transform that may differ
        // from the parent's. Instead, read the full parent tree and clip the
        // assembled instance below using the instance's transform, so that the
        // retained region matches the requested world-space bbox.
        io::ReadOptions parentOptions = readOptions;
        parentOptions.clipBBox = BBoxd();

        GridBase::Ptr parent;
        OPENVDB_ASSERT(inputHasGridOffsets());
        parentIt->second.seekToGrid(inputStream());
        parent = Archive::readGrid(parentIt->second, inputStream(), parentOptions, mReadDiagnostics);
        if (parent) {
            grid->setTree(parent->baseTreePtr());
            const auto& clipBBox = readOptions.clipBBox;
            if (clipBBox.isSorted()) {
                grid->clipGrid(clipBBox);
            }
        }
    }
    return grid;
}


////////////////////////////////////////


void
File::writeGrids(const GridCPtrVec& grids, const MetaMap& meta, const io::WriteOptions& writeOptions) const
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
    Archive::write(file, grids, /*seekable=*/true, meta, writeOptions);

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


////////////////////////////////////////


namespace {

namespace convert_grid_internal {

/// @brief Convert @a source to the grid type @c ValueConverter<TargetBuildT>::Type,
///   provided the registry-reported @a targetType agrees and the value conversion
///   is legal.  Sets @a alreadyTargetType when @a source has that build type
///   already.  Returns null in both cases, so the caller keeps the original grid.
template<typename TargetBuildT>
inline GridBase::Ptr
convertToTargetType(const GridBase& source, const std::string& targetType,
    bool& alreadyTargetType)
{
    // A MaskGrid source is only visited when the target is itself a mask, where
    // the build types match and the branch below attempts no conversion.  No
    // LeafNode conversion constructor exists from ValueMask to another build type.
    using SourceGridTypes = std::conditional_t<std::is_same_v<TargetBuildT, ValueMask>,
        GridTypes, GridTypes::Remove<MaskGrid>>;

    GridBase::Ptr result;
    source.apply<SourceGridTypes>([&](const auto& typedSource) {
        using SourceGridT = std::decay_t<decltype(typedSource)>;
        using TargetGridT =
            typename SourceGridT::template ValueConverter<TargetBuildT>::Type;
        if constexpr (std::is_same_v<typename SourceGridT::BuildType, TargetBuildT>) {
            alreadyTargetType = true;
        } else if constexpr (CanConvertType<typename SourceGridT::BuildType, TargetBuildT>::value) {
            if (TargetGridT::gridType() != targetType)   return;
            if constexpr (std::is_same_v<TargetBuildT, ValueMask>) {
                // A mask records active state, not values, so copy the topology
                // instead of casting values.  Casting would let a non-zero
                // background or an inactive non-zero tile become true, which the
                // codec path does not do.  create() takes the GridBase overload to
                // copy the metadata and transform without converting the values.
                auto target = TargetGridT::create(static_cast<const GridBase&>(typedSource));
                target->setTree(typename TargetGridT::TreeType::Ptr(
                    new typename TargetGridT::TreeType(typedSource.constTree(),
                        /*inactiveValue=*/false, /*activeValue=*/true, TopologyCopy())));
                result = target;
            } else {
                result = typename TargetGridT::Ptr(new TargetGridT(typedSource));
            }
        }
    });
    return result;
}

} // namespace convert_grid_internal

/// @brief Return the name used in diagnostics and log messages for @a mode.
std::string
readModeName(ReadMode mode)
{
    switch (mode) {
        case ReadMode::Half: return "Half";
        case ReadMode::Bool: return "Bool";
        case ReadMode::Mask: return "Mask";
        case ReadMode::TopologyOnly: return "TopologyOnly";
        case ReadMode::MetadataOnly: return "MetadataOnly";
        default: return "Original";
    }
}

GridBase::Ptr
convertGridForReadMode(const GridBase& source, const ReadOptions& readOptions,
    Codec* codec, ReadDiagnostics& diagnostics, const std::string& filename)
{
    if (readOptions.readMode != ReadMode::Half &&
        readOptions.readMode != ReadMode::Bool &&
        readOptions.readMode != ReadMode::Mask)
    {
        return GridBase::Ptr();
    }

    // Ask the codec that would have read this grid which type it produces.
    CodecData::Ptr codecData = codec ? codec->createData() : CodecData::Ptr();
    const std::string targetType =
        (codecData && codecData->grid) ? codecData->grid->type() : std::string();

    GridBase::Ptr result;
    bool alreadyTargetType = false;
    if (readOptions.readMode == ReadMode::Half) {
        result = convert_grid_internal::convertToTargetType<Half>(
            source, targetType, alreadyTargetType);
    } else if (readOptions.readMode == ReadMode::Bool) {
        result = convert_grid_internal::convertToTargetType<bool>(
            source, targetType, alreadyTargetType);
    } else {
        result = convert_grid_internal::convertToTargetType<ValueMask>(
            source, targetType, alreadyTargetType);
    }

    // Either no conversion codec is registered for this grid type (targetType is
    // empty, or names the plain gridType codec), or the registry named a target
    // type this dispatch cannot produce, such as a grid type outside GridTypes or
    // a pair that CanConvertType rejects.  A grid that already has the requested
    // build type is not a failure, so it is not reported.
    if (!result && !alreadyTargetType) {
        const std::string modeStr = readModeName(readOptions.readMode);
        diagnostics.addWarning(source.getName(),
            "ReadMode::" + modeStr + " conversion is not supported for grid type '"
            + source.type() + "'; reading as original type");
        OPENVDB_LOG_WARN(filename << ": grid \"" << source.getName()
            << "\" requested ReadMode::" << modeStr << ", but no conversion is "
            "available for grid type \"" << source.type()
            << "\"; returning the original type");
    }

    return result;
}

} // anonymous namespace


GridBase::Ptr
File::resolveCachedGrid(const GridBase::Ptr& cachedGrid, const io::ReadOptions& readOptions,
    ReadDiagnostics& diagnostics) const
{
    if (readOptions.readMode == ReadMode::MetadataOnly) {
        return cachedGrid->copyGridWithNewTree();
    }

    GridBase::Ptr grid = cachedGrid;

    if (readOptions.readMode == ReadMode::TopologyOnly) {
        diagnostics.addWarning(grid->getName(),
            "ReadMode::TopologyOnly is not supported for grids cached from a file "
            "without grid offsets; returning the original grid with values intact");
        OPENVDB_LOG_WARN(mFilename << ": grid \"" << grid->getName()
            << "\" requested ReadMode::TopologyOnly, but this file has no grid offsets "
            "and the grid is already fully cached; returning the original grid "
            "with values intact");
    } else {
        io::Codec* codec = Archive::findCodec(grid->type(), readOptions);
        if (GridBase::Ptr converted =
            convertGridForReadMode(*grid, readOptions, codec, diagnostics, mFilename))
        {
            grid = converted;
        }
    }

    const auto& bbox = readOptions.clipBBox;
    if (bbox.isSorted()) {
        if (grid == cachedGrid) {
            // Don't mutate the cached grid in place, it stays owned by the caller.
            grid = grid->deepCopyGrid();
        }
        grid->clipGrid(bbox);
        diagnostics.addWarning(cachedGrid->getName(),
            "bounding box clipping was applied as a post-process because the grid "
            "was cached from a file without grid offsets");
    }

    return grid;
}


} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
