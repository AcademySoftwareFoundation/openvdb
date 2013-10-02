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

#ifndef OPENVDB_VERSION_HAS_BEEN_INCLUDED
#define OPENVDB_VERSION_HAS_BEEN_INCLUDED

#include "Platform.h"
#include <iosfwd> // for std::istream
#include <string>


/// The version namespace name for this library version
///
/// Fully-namespace-qualified symbols are named as follows:
/// vdb::vX_Y_Z::Vec3i, vdb::vX_Y_Z::io::File, vdb::vX_Y_Z::tree::Tree, etc.,
/// where X, Y and Z are OPENVDB_LIBRARY_MAJOR_VERSION, OPENVDB_LIBRARY_MINOR_VERSION
/// and OPENVDB_LIBRARY_PATCH_VERSION, respectively (defined below).
#define OPENVDB_VERSION_NAME v2_0_0

/// If OPENVDB_REQUIRE_VERSION_NAME is undefined, symbols from the version
/// namespace are promoted to the top-level namespace (e.g., vdb::v1_0_0::io::File
/// can be referred to simply as vdb::io::File).  Otherwise, symbols must be fully
/// namespace-qualified.
#ifdef OPENVDB_REQUIRE_VERSION_NAME
#define OPENVDB_USE_VERSION_NAMESPACE
#else
/// @note The empty namespace clause below ensures that
/// OPENVDB_VERSION_NAME is recognized as a namespace name.
#define OPENVDB_USE_VERSION_NAMESPACE \
    namespace OPENVDB_VERSION_NAME {} \
    using namespace OPENVDB_VERSION_NAME;
#endif


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief The magic number is stored in the first four bytes of every VDB file.
/// @details This can be used to quickly test whether we have a valid file or not.
const int32_t OPENVDB_MAGIC = 0x56444220;

const uint32_t
    OPENVDB_LIBRARY_MAJOR_VERSION = 2,
    OPENVDB_LIBRARY_MINOR_VERSION = 0,
    OPENVDB_LIBRARY_PATCH_VERSION = 0;

/// @brief The current version number of the VDB file format
/// @details  This can be used to enable various backwards compatability switches
/// or to reject files that cannot be read.
const uint32_t OPENVDB_FILE_VERSION = 222;

/// Notable file format version numbers
enum {
    OPENVDB_FILE_VERSION_ROOTNODE_MAP = 213,
    OPENVDB_FILE_VERSION_INTERNALNODE_COMPRESSION = 214,
    OPENVDB_FILE_VERSION_SIMPLIFIED_GRID_TYPENAME = 215,
    OPENVDB_FILE_VERSION_GRID_INSTANCING = 216,
    OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION = 217,
    OPENVDB_FILE_VERSION_BOOST_UUID = 218,
    OPENVDB_FILE_VERSION_NO_GRIDMAP = 219,
    OPENVDB_FILE_VERSION_NEW_TRANSFORM = 219,
    OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION = 220,
    OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX = 221,
    OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION = 222
};


struct VersionId { uint32_t first, second; VersionId(): first(0), second(0) {} };

namespace io {
/// @brief Return the file format version number associated with the given input stream.
OPENVDB_API uint32_t getFormatVersion(std::istream&);
/// @brief Return the (major, minor) library version number associated with the given input stream.
OPENVDB_API VersionId getLibraryVersion(std::istream&);
/// @brief Return a string of the form "<major>.<minor>/<format>", giving the library
/// and file format version numbers associated with the given input stream.
OPENVDB_API std::string getVersion(std::istream&);
// Associate the current file format and library version numbers with the given input stream.
OPENVDB_API void setCurrentVersion(std::istream&);
// Associate specific file format and library version numbers with the given stream.
OPENVDB_API void setVersion(std::ios_base&, const VersionId& libraryVersion, uint32_t fileVersion);
// Return a bitwise OR of compression option flags (COMPRESS_ZIP, COMPRESS_ACTIVE_MASK, etc.)
// specifying whether and how input data is compressed or output data should be compressed.
OPENVDB_API uint32_t getDataCompression(std::ios_base&);
// Associate with the given stream a bitwise OR of compression option flags (COMPRESS_ZIP,
// COMPRESS_ACTIVE_MASK, etc.) specifying whether and how input data is compressed
// or output data should be compressed.
OPENVDB_API void setDataCompression(std::ios_base&, uint32_t compressionFlags);
// Return the class (GRID_LEVEL_SET, GRID_UNKNOWN, etc.) of the grid
// currently being read from or written to the given stream.
OPENVDB_API uint32_t getGridClass(std::ios_base&);
// brief Associate with the given stream the class (GRID_LEVEL_SET, GRID_UNKNOWN, etc.)
// of the grid currently being read or written.
OPENVDB_API void setGridClass(std::ios_base&, uint32_t);
// Return a pointer to the background value of the grid currently being
// read from or written to the given stream.
OPENVDB_API const void* getGridBackgroundValuePtr(std::ios_base&);
// Specify (a pointer to) the background value of the grid currently being
// read from or written to the given stream.
// The pointer must remain valid until the entire grid has been read or written.
OPENVDB_API void setGridBackgroundValuePtr(std::ios_base&, const void* background);
} // namespace io

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_VERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
