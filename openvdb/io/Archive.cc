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

// Indices into a stream's internal extensible array of values used by readers and writers
struct StreamState
{
    static const long MAGIC_NUMBER;

    StreamState();
    ~StreamState();

    int magicNumber;
    int fileVersion;
    int libraryMajorVersion;
    int libraryMinorVersion;
    int dataCompression;
    int writeGridStatsMetadata;
    int gridBackground;
    int gridClass;
}
sStreamState;

const long StreamState::MAGIC_NUMBER =
    long((uint64_t(OPENVDB_MAGIC) << 32) | (uint64_t(OPENVDB_MAGIC)));

const uint32_t Archive::DEFAULT_COMPRESSION_FLAGS = (COMPRESS_ZIP | COMPRESS_ACTIVE_MASK);


////////////////////////////////////////


StreamState::StreamState(): magicNumber(std::ios_base::xalloc())
{
    // Having reserved an entry (the one at index magicNumber) in the extensible array
    // associated with every stream, store a magic number at that location in the
    // array belonging to the cout stream.
    std::cout.iword(magicNumber) = MAGIC_NUMBER;
    std::cout.pword(magicNumber) = this;

    // Search for a lower-numbered entry in cout's array that already contains the magic number.
    /// @todo This assumes that the indices returned by xalloc() increase monotonically.
    int existingArray = -1;
    for (int i = 0; i < magicNumber; ++i) {
        if (std::cout.iword(i) == MAGIC_NUMBER) {
            existingArray = i;
            break;
        }
    }

    if (existingArray >= 0 && std::cout.pword(existingArray) != NULL) {
        // If a lower-numbered entry was found to contain the magic number,
        // a coexisting version of this library must have registered it.
        // In that case, the corresponding pointer should point to an existing
        // StreamState struct.  Copy the other array indices from that StreamState
        // into this one, so as to share state with the other library.
        const StreamState& other =
            *static_cast<const StreamState*>(std::cout.pword(existingArray));
        fileVersion =            other.fileVersion;
        libraryMajorVersion =    other.libraryMajorVersion;
        libraryMinorVersion =    other.libraryMinorVersion;
        dataCompression =        other.dataCompression;
        writeGridStatsMetadata = other.writeGridStatsMetadata;
        gridBackground =         other.gridBackground;
        gridClass =              other.gridClass;
    } else {
        // Reserve storage for per-stream file format and library version numbers
        // and other values of use to readers and writers.  Each of the following
        // values is an index into the extensible arrays associated with all streams.
        // The indices are common to all streams, but the values stored at those indices
        // are unique to each stream.
        fileVersion =            std::ios_base::xalloc();
        libraryMajorVersion =    std::ios_base::xalloc();
        libraryMinorVersion =    std::ios_base::xalloc();
        dataCompression =        std::ios_base::xalloc();
        writeGridStatsMetadata = std::ios_base::xalloc();
        gridBackground =         std::ios_base::xalloc();
        gridClass =              std::ios_base::xalloc();
    }
}


StreamState::~StreamState()
{
    // Ensure that this StreamState struct can no longer be accessed.
    std::cout.iword(magicNumber) = 0;
    std::cout.pword(magicNumber) = NULL;
}


////////////////////////////////////////


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


boost::shared_ptr<Archive>
Archive::copy() const
{
    return boost::shared_ptr<Archive>(new Archive(*this));
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
    return static_cast<uint32_t>(is.iword(sStreamState.fileVersion));
}


void
Archive::setFormatVersion(std::istream& is)
{
    is.iword(sStreamState.fileVersion) = mFileVersion;
}


VersionId
getLibraryVersion(std::istream& is)
{
    VersionId version;
    version.first = static_cast<uint32_t>(is.iword(sStreamState.libraryMajorVersion));
    version.second = static_cast<uint32_t>(is.iword(sStreamState.libraryMinorVersion));
    return version;
}


void
Archive::setLibraryVersion(std::istream& is)
{
    is.iword(sStreamState.libraryMajorVersion) = mLibraryVersion.first;
    is.iword(sStreamState.libraryMinorVersion) = mLibraryVersion.second;
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
    is.iword(sStreamState.fileVersion) = OPENVDB_FILE_VERSION;
    is.iword(sStreamState.libraryMajorVersion) = OPENVDB_LIBRARY_MAJOR_VERSION;
    is.iword(sStreamState.libraryMinorVersion) = OPENVDB_LIBRARY_MINOR_VERSION;
}


void
setVersion(std::ios_base& strm, const VersionId& libraryVersion, uint32_t fileVersion)
{
    strm.iword(sStreamState.fileVersion) = fileVersion;
    strm.iword(sStreamState.libraryMajorVersion) = libraryVersion.first;
    strm.iword(sStreamState.libraryMinorVersion) = libraryVersion.second;
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
    return uint32_t(strm.iword(sStreamState.dataCompression));
}


void
setDataCompression(std::ios_base& strm, uint32_t compression)
{
    strm.iword(sStreamState.dataCompression) = compression;
}


void
Archive::setDataCompression(std::istream& is)
{
    io::setDataCompression(is, mCompression);
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
    io::setDataCompression(os, compression);

    os.write(reinterpret_cast<const char*>(&compression), sizeof(uint32_t));
}


void
Archive::readGridCompression(std::istream& is)
{
    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        uint32_t compression = COMPRESS_NONE;
        is.read(reinterpret_cast<char*>(&compression), sizeof(uint32_t));
        io::setDataCompression(is, compression);
    }
}


////////////////////////////////////////


bool
getWriteGridStatsMetadata(std::ostream& os)
{
    return os.iword(sStreamState.writeGridStatsMetadata) != 0;
}


void
Archive::setWriteGridStatsMetadata(std::ostream& os)
{
    os.iword(sStreamState.writeGridStatsMetadata) = mEnableGridStats;
}


////////////////////////////////////////


uint32_t
getGridClass(std::ios_base& strm)
{
    const uint32_t val = strm.iword(sStreamState.gridClass);
    if (val >= NUM_GRID_CLASSES) return GRID_UNKNOWN;
    return val;
}


void
setGridClass(std::ios_base& strm, uint32_t cls)
{
    strm.iword(sStreamState.gridClass) = long(cls);
}


const void*
getGridBackgroundValuePtr(std::ios_base& strm)
{
    return strm.pword(sStreamState.gridBackground);
}


void
setGridBackgroundValuePtr(std::ios_base& strm, const void* background)
{
    strm.pword(sStreamState.gridBackground) = const_cast<void*>(background);
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

    io::setGridClass(is, GRID_UNKNOWN);
    io::setGridBackgroundValuePtr(is, NULL);

    grid->readMeta(is);

    // Add a description of the compression settings to the grid as metadata.
    /// @todo Would this be useful?
    //const uint32_t compression = getDataCompression(is);
    //grid->insertMeta(GridBase::META_FILE_COMPRESSION,
    //    StringMetadata(compressionToString(compression)));

    const GridClass gridClass = grid->getGridClass();
    io::setGridClass(is, gridClass);

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
    io::setDataCompression(os, compressionFlags());
    os.iword(sStreamState.writeGridStatsMetadata) = isGridStatsMetadataEnabled();

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

            // Ensure that the grid's descriptor has a unique grid name, by appending
            // a number to it if a grid with the same name was already written.
            // Always add a number if the grid name is empty, so that the grid can be
            // properly identified as an instance parent, if necessary.
            std::string name = grid->getName();
            if (name.empty()) name = GridDescriptor::addSuffix(name, 0);
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
        io::setDataCompression(os, compressionFlags());
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
