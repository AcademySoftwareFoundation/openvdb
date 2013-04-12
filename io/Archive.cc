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

#include "Archive.h"

#include <algorithm> // for std::find_if()
#include <cstring> // for std::memcpy()
#include <iostream>
#include <map>
#include <sstream>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>
#include <openvdb/util/logging.h>
#include "GridDescriptor.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

/// Allocate storage for per-stream file format and library version numbers
/// and other values of use to readers and writers.
const int
    Archive::sFormatVersionIndex = std::ios_base::xalloc(),
    Archive::sLibraryMajorVersionIndex = std::ios_base::xalloc(),
    Archive::sLibraryMinorVersionIndex = std::ios_base::xalloc(),
    Archive::sDataCompressionIndex = std::ios_base::xalloc(),
    Archive::sWriteGridStatsMetadataIndex = std::ios_base::xalloc(),
    Archive::sGridBackgroundIndex = std::ios_base::xalloc(),
    Archive::sGridClassIndex = std::ios_base::xalloc();

const uint32_t Archive::DEFAULT_COMPRESSION_FLAGS = (COMPRESS_ZIP | COMPRESS_ACTIVE_MASK);


Archive::Archive():
    mFileVersion(OPENVDB_FILE_VERSION),
    mUuid(boost::uuids::nil_uuid()),
    mInputHasGridOffsets(false),
    mEnableInstancing(true),
    mCompression(DEFAULT_COMPRESSION_FLAGS),
    mEnableGridStats(true)
{
    mLibraryVersion.first = OPENVDB_LIBRARY_MAJOR_VERSION;
    mLibraryVersion.second = OPENVDB_LIBRARY_MINOR_VERSION;
}


Archive::~Archive()
{
}


////////////////////////////////////////


std::string
Archive::getUniqueTag() const
{
    /// @todo Once versions of Boost < 1.44.0 are no longer in use,
    /// this can be replaced with "return boost::uuids::to_string(mUuid);".
    std::ostringstream ostr;
    ostr << mUuid;
    return ostr.str();
}


bool
Archive::isIdentical(const std::string& uuidStr) const
{
    return uuidStr == getUniqueTag();
}


////////////////////////////////////////


uint32_t
getFormatVersion(std::istream& is)
{
    return static_cast<uint32_t>(is.iword(Archive::sFormatVersionIndex));
}


void
Archive::setFormatVersion(std::istream& is)
{
    is.iword(sFormatVersionIndex) = mFileVersion;
}


VersionId
getLibraryVersion(std::istream& is)
{
    VersionId version;
    version.first = static_cast<uint32_t>(is.iword(Archive::sLibraryMajorVersionIndex));
    version.second = static_cast<uint32_t>(is.iword(Archive::sLibraryMinorVersionIndex));
    return version;
}


void
Archive::setLibraryVersion(std::istream& is)
{
    is.iword(sLibraryMajorVersionIndex) = mLibraryVersion.first;
    is.iword(sLibraryMinorVersionIndex) = mLibraryVersion.second;
}


std::string
getVersion(std::istream& is)
{
    VersionId version = getLibraryVersion(is);
    std::ostringstream ostr;
    ostr << version.first << "." << version.second << "/" << getFormatVersion(is);
    return ostr.str();
}


void
setCurrentVersion(std::istream& is)
{
    is.iword(Archive::sFormatVersionIndex) = OPENVDB_FILE_VERSION;
    is.iword(Archive::sLibraryMajorVersionIndex) = OPENVDB_LIBRARY_MAJOR_VERSION;
    is.iword(Archive::sLibraryMinorVersionIndex) = OPENVDB_LIBRARY_MINOR_VERSION;
}


std::string
Archive::version() const
{
    std::ostringstream ostr;
    ostr << mLibraryVersion.first << "." << mLibraryVersion.second << "/" << mFileVersion;
    return ostr.str();
}


////////////////////////////////////////


uint32_t
getDataCompression(std::ios_base& strm)
{
    return uint32_t(strm.iword(Archive::sDataCompressionIndex));
}


void
Archive::setDataCompression(std::istream& is)
{
    is.iword(Archive::sDataCompressionIndex) = mCompression;
}


bool
Archive::isCompressionEnabled() const
{
    return (mCompression & COMPRESS_ZIP);
}


void
Archive::setCompressionEnabled(bool b)
{
    if (b) mCompression |= COMPRESS_ZIP;
    else mCompression &= ~COMPRESS_ZIP;
}


void
Archive::setGridCompression(std::ostream& os, const GridBase& grid) const
{
    // Start with the options that are enabled globally for this archive.
    uint32_t compression = compressionFlags();

    // Disable options that are inappropriate for the given grid.
    switch (grid.getGridClass()) {
        case GRID_LEVEL_SET:
        case GRID_FOG_VOLUME:
            // Zip compression is not used on level sets or fog volumes.
            compression = compression & ~COMPRESS_ZIP;
            break;
        default:
            break;
    }
    os.iword(Archive::sDataCompressionIndex) = compression;

    os.write(reinterpret_cast<const char*>(&compression), sizeof(uint32_t));
}


void
Archive::readGridCompression(std::istream& is)
{
    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        uint32_t compression = COMPRESS_NONE;
        is.read(reinterpret_cast<char*>(&compression), sizeof(uint32_t));
        is.iword(Archive::sDataCompressionIndex) = compression;
    }
}


////////////////////////////////////////


bool
getWriteGridStatsMetadata(std::ostream& os)
{
    return os.iword(Archive::sWriteGridStatsMetadataIndex) != 0;
}


void
Archive::setWriteGridStatsMetadata(std::ostream& os)
{
    os.iword(Archive::sWriteGridStatsMetadataIndex) = mEnableGridStats;
}


////////////////////////////////////////


uint32_t
getGridClass(std::ios_base& strm)
{
    const uint32_t val = strm.iword(Archive::sGridClassIndex);
    if (val >= NUM_GRID_CLASSES) return GRID_UNKNOWN;
    return val;
}


const void*
getGridBackgroundValuePtr(std::ios_base& strm)
{
    return strm.pword(Archive::sGridBackgroundIndex);
}


void
setGridBackgroundValuePtr(std::ios_base& strm, const void* background)
{
    strm.pword(Archive::sGridBackgroundIndex) = const_cast<void*>(background);
}


////////////////////////////////////////


bool
Archive::readHeader(std::istream& is)
{
    // 1) Read the magic number for VDB.
    int64_t magic;
    is.read(reinterpret_cast<char*>(&magic), sizeof(int64_t));

    if (magic != OPENVDB_MAGIC) {
        OPENVDB_THROW(IoError, "not a VDB file");
    }

    // 2) Read the file format version number.
    is.read(reinterpret_cast<char*>(&mFileVersion), sizeof(uint32_t));
    if (mFileVersion > OPENVDB_FILE_VERSION) {
        OPENVDB_LOG_WARN("unsupported VDB file format (expected version "
            << OPENVDB_FILE_VERSION << " or earlier, got version " << mFileVersion << ")");
    } else if (mFileVersion < 211) {
        // Versions prior to 211 stored separate major, minor and patch numbers.
        uint32_t version;
        is.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        mFileVersion = 100 * mFileVersion + 10 * version;
        is.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        mFileVersion += version;
    }

    // 3) Read the library version numbers (not stored prior to file format version 211).
    mLibraryVersion.first = mLibraryVersion.second = 0;
    if (mFileVersion >= 211) {
        uint32_t version;
        is.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        mLibraryVersion.first = version; // major version
        is.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        mLibraryVersion.second = version; // minor version
    }

    // 4) Read the flag indicating whether the stream supports partial reading.
    //    (Versions prior to 212 have no flag because they always supported partial reading.)
    mInputHasGridOffsets = true;
    if (mFileVersion >= 212) {
        char hasGridOffsets;
        is.read(&hasGridOffsets, sizeof(char));
        mInputHasGridOffsets = hasGridOffsets;
    }

    // 5) Read the flag that indicates whether data is compressed.
    //    (From version 222 on, compression information is stored per grid.)
    mCompression = DEFAULT_COMPRESSION_FLAGS;
    if (mFileVersion >= OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION &&
        mFileVersion < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION)
    {
        char isCompressed;
        is.read(&isCompressed, sizeof(char));
        mCompression = (isCompressed != 0 ? COMPRESS_ZIP : COMPRESS_NONE);
    }

    // 6) Read the 16-byte (128-bit) uuid.
    boost::uuids::uuid oldUuid = mUuid;
    if (mFileVersion >= OPENVDB_FILE_VERSION_BOOST_UUID) {
        // UUID is stored as an ASCII string.
        is >> mUuid;
    } else {
        // Older versions stored the UUID as a byte string.
        char uuidBytes[16];
        is.read(uuidBytes, 16);
        std::memcpy(&mUuid.data[0], uuidBytes, std::min<size_t>(16, mUuid.size()));
    }
    return oldUuid != mUuid; // true if UUID in input stream differs from old UUID
}


void
Archive::writeHeader(std::ostream& os, bool seekable) const
{
    using boost::uint32_t;
    using boost::int64_t;

    // 1) Write the magic number for VDB.
    int64_t magic = OPENVDB_MAGIC;
    os.write(reinterpret_cast<char*>(&magic), sizeof(int64_t));

    // 2) Write the file format version number.
    uint32_t version = OPENVDB_FILE_VERSION;
    os.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    // 3) Write the library version numbers.
    version = OPENVDB_LIBRARY_MAJOR_VERSION;
    os.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    version = OPENVDB_LIBRARY_MINOR_VERSION;
    os.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    // 4) Write a flag indicating that this stream contains no grid offsets.
    char hasGridOffsets = seekable;
    os.write(&hasGridOffsets, sizeof(char));

    // 5) Write a flag indicating that this stream contains compressed leaf data.
    //    (Omitted as of version 222)

    // 6) Generate a new random 16-byte (128-bit) uuid and write it to the stream.
    boost::mt19937 ran;
    ran.seed(time(NULL));
    boost::uuids::basic_random_generator<boost::mt19937> gen(&ran);
    mUuid = gen(); // mUuid is mutable
    os << mUuid;
}


////////////////////////////////////////


int32_t
Archive::readGridCount(std::istream& is)
{
    int32_t gridCount = 0;
    is.read(reinterpret_cast<char*>(&gridCount), sizeof(int32_t));
    return gridCount;
}


////////////////////////////////////////


void
Archive::connectInstance(const GridDescriptor& gd, const NamedGridMap& grids) const
{
    if (!gd.isInstance() || grids.empty()) return;

    NamedGridMap::const_iterator it = grids.find(gd.uniqueName());
    if (it == grids.end()) return;
    GridBase::Ptr grid = it->second;
    if (!grid) return;

    it = grids.find(gd.instanceParentName());
    if (it != grids.end()) {
        GridBase::Ptr parent = it->second;
        if (mEnableInstancing) {
            // Share the instance parent's tree.
            grid->setTree(parent->baseTreePtr());
        } else {
            // Copy the instance parent's tree.
            grid->setTree(parent->baseTree().copy());
        }
    } else {
        OPENVDB_THROW(KeyError, "missing instance parent \""
            << GridDescriptor::nameAsString(gd.instanceParentName())
            << "\" for grid " << GridDescriptor::nameAsString(gd.uniqueName()));
    }
}


////////////////////////////////////////


void
Archive::readGrid(GridBase::Ptr grid, const GridDescriptor& gd, std::istream& is)
{
    // Read the compression settings for this grid and tag the stream with them
    // so that downstream functions can reference them.
    readGridCompression(is);

    is.iword(Archive::sGridClassIndex) = GRID_UNKNOWN;
    is.pword(Archive::sGridBackgroundIndex) = NULL;

    grid->readMeta(is);

    // Add a description of the compression settings to the grid as metadata.
    /// @todo Would this be useful?
    //const uint32_t compression = getDataCompression(is);
    //grid->insertMeta(GridBase::META_FILE_COMPRESSION,
    //    StringMetadata(compressionToString(compression)));

    const GridClass gridClass = grid->getGridClass();
    is.iword(Archive::sGridClassIndex) = gridClass;

    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_GRID_INSTANCING) {
        grid->readTransform(is);
        if (!gd.isInstance()) {
            grid->readTopology(is);
            grid->readBuffers(is);
        }
    } else {
        grid->readTopology(is);
        grid->readTransform(is);
        grid->readBuffers(is);
    }
    if (getFormatVersion(is) < OPENVDB_FILE_VERSION_NO_GRIDMAP) {
        // Older versions of the library didn't store grid names as metadata,
        // so when reading older files, copy the grid name from the descriptor
        // to the grid's metadata.
        if (grid->getName().empty()) {
            grid->setName(gd.gridName());
        }
    }
}


void
Archive::write(std::ostream& os, const GridPtrVec& grids, bool seekable,
    const MetaMap& metadata) const
{
    this->write(os, GridCPtrVec(grids.begin(), grids.end()), seekable, metadata);
}


void
Archive::write(std::ostream& os, const GridCPtrVec& grids, bool seekable,
    const MetaMap& metadata) const
{
    // Set stream flags so that downstream functions can reference them.
    os.iword(Archive::sDataCompressionIndex) = compressionFlags();
    os.iword(Archive::sWriteGridStatsMetadataIndex) = isGridStatsMetadataEnabled();

    this->writeHeader(os, seekable);

    metadata.writeMeta(os);

    // Write the number of non-null grids.
    int32_t gridCount = 0;
    for (GridCPtrVecCIter i = grids.begin(), e = grids.end(); i != e; ++i) {
        if (*i) ++gridCount;
    }
    os.write(reinterpret_cast<char*>(&gridCount), sizeof(int32_t));

    typedef std::map<const TreeBase*, GridDescriptor> TreeMap;
    typedef TreeMap::iterator TreeMapIter;
    TreeMap treeMap;

    std::set<std::string> uniqueNames;

    // Write out the non-null grids.
    for (GridCPtrVecCIter i = grids.begin(), e = grids.end(); i != e; ++i) {
        if (const GridBase::ConstPtr& grid = *i) {

            // Ensure that the grid's descriptor has a unique grid name.
            std::string name = grid->getName();
            for (int n = 1; uniqueNames.find(name) != uniqueNames.end(); ++n) {
                name = GridDescriptor::addSuffix(grid->getName(), n);
            }
            uniqueNames.insert(name);

            // Create a grid descriptor.
            GridDescriptor gd(name, grid->type(), grid->saveFloatAsHalf());

            // Check if this grid's tree is shared with a grid that has already been written.
            const TreeBase* treePtr = &(grid->baseTree());
            TreeMapIter mapIter = treeMap.find(treePtr);

            bool isInstance = ((mapIter != treeMap.end())
                && (mapIter->second.saveFloatAsHalf() == gd.saveFloatAsHalf()));

            if (mEnableInstancing && isInstance) {
                // This grid's tree is shared with another grid that has already been written.
                // Get the name of the other grid.
                gd.setInstanceParentName(mapIter->second.uniqueName());
                // Write out this grid's descriptor and metadata, but not its tree.
                writeGridInstance(gd, grid, os, seekable);

                OPENVDB_LOG_DEBUG_RUNTIME("io::Archive::write(): "
                    << GridDescriptor::nameAsString(gd.uniqueName())
                    << " (" << std::hex << treePtr << std::dec << ")"
                    << " is an instance of "
                    << GridDescriptor::nameAsString(gd.instanceParentName()));
            } else {
                // Write out the grid descriptor and its associated grid.
                writeGrid(gd, grid, os, seekable);
                // Record the grid's tree pointer so that the tree doesn't get written
                // more than once.
                treeMap[treePtr] = gd;
            }
        }

        // Some compression options (e.g., mask compression) are set per grid.
        // Restore the original settings before writing the next grid.
        os.iword(Archive::sDataCompressionIndex) = compressionFlags();
    }
}


void
Archive::writeGrid(GridDescriptor& gd, GridBase::ConstPtr grid,
    std::ostream& os, bool seekable) const
{
    // Write out the Descriptor's header information (grid name and type)
    gd.writeHeader(os);

    // Save the curent stream position as postion to where the offsets for
    // this GridDescriptor will be written to.
    int64_t offsetPos = (seekable ? int64_t(os.tellp()) : 0);

    // Write out the offset information. At this point it will be incorrect.
    // But we need to write it out to move the stream head forward.
    gd.writeStreamPos(os);

    // Now we know the starting grid storage position.
    if (seekable) gd.setGridPos(os.tellp());

    // Save the compression settings for this grid.
    setGridCompression(os, *grid);

    // Save the grid's metadata and transform.
    if (!getWriteGridStatsMetadata(os)) {
        grid->writeMeta(os);
    } else {
        // Compute and add grid statistics metadata.
        GridBase::Ptr copyOfGrid = grid->copyGrid(); // shallow copy
        copyOfGrid->addStatsMetadata();
        copyOfGrid->writeMeta(os);
    }
    grid->writeTransform(os);

    // Save the grid's structure.
    grid->writeTopology(os);

    // Now we know the grid block storage position.
    if (seekable) gd.setBlockPos(os.tellp());

    // Save out the data blocks of the grid.
    grid->writeBuffers(os);

    // Now we know the end position of this grid.
    if (seekable) gd.setEndPos(os.tellp());

    if (seekable) {
        // Now, go back to where the Descriptor's offset information is written
        // and write the offsets again.
        os.seekp(offsetPos, std::ios_base::beg);
        gd.writeStreamPos(os);

        // Now seek back to the end.
        gd.seekToEnd(os);
    }
}


void
Archive::writeGridInstance(GridDescriptor& gd, GridBase::ConstPtr grid,
    std::ostream& os, bool seekable) const
{
    // Write out the Descriptor's header information (grid name, type
    // and instance parent name).
    gd.writeHeader(os);

    // Save the curent stream position as postion to where the offsets for
    // this GridDescriptor will be written to.
    int64_t offsetPos = (seekable ? int64_t(os.tellp()) : 0);

    // Write out the offset information. At this point it will be incorrect.
    // But we need to write it out to move the stream head forward.
    gd.writeStreamPos(os);

    // Now we know the starting grid storage position.
    if (seekable) gd.setGridPos(os.tellp());

    // Save the compression settings for this grid.
    setGridCompression(os, *grid);

    // Save the grid's metadata and transform.
    grid->writeMeta(os);
    grid->writeTransform(os);

    // Now we know the end position of this grid.
    if (seekable) gd.setEndPos(os.tellp());

    if (seekable) {
        // Now, go back to where the Descriptor's offset information is written
        // and write the offsets again.
        os.seekp(offsetPos, std::ios_base::beg);
        gd.writeStreamPos(os);

        // Now seek back to the end.
        gd.seekToEnd(os);
    }
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
