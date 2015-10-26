///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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


/// The version namespace name for this library version
///
/// Fully-namespace-qualified symbols are named as follows:
/// openvdb::vX_Y_Z::Vec3i, openvdb::vX_Y_Z::io::File, openvdb::vX_Y_Z::tree::Tree, etc.,
/// where X, Y and Z are OPENVDB_LIBRARY_MAJOR_VERSION, OPENVDB_LIBRARY_MINOR_VERSION
/// and OPENVDB_LIBRARY_PATCH_VERSION, respectively (defined below).
#define OPENVDB_VERSION_NAME v3_1_0

// Library major, minor and patch version numbers
#define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER 3
#define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER 1
#define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER 0

/// @brief Library version number string of the form "<major>.<minor>.<patch>"
/// @details This is a macro rather than a static constant because we typically
/// want the compile-time version number, not the runtime version number
/// (although the two are usually the same).
#define OPENVDB_LIBRARY_VERSION_STRING "3.1.0"

/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
#define OPENVDB_LIBRARY_VERSION_NUMBER \
    ((OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER << 24) | \
    ((OPENVDB_LIBRARY_MINOR_VERSION_NUMBER & 0xFF) << 16) | \
    (OPENVDB_LIBRARY_PATCH_VERSION_NUMBER & 0xFFFF))

/// If OPENVDB_REQUIRE_VERSION_NAME is undefined, symbols from the version
/// namespace are promoted to the top-level namespace (e.g., openvdb::v1_0_0::io::File
/// can be referred to simply as openvdb::io::File).  Otherwise, symbols must be fully
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

// Library major, minor and patch version numbers
const uint32_t
    OPENVDB_LIBRARY_MAJOR_VERSION = OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER,
    OPENVDB_LIBRARY_MINOR_VERSION = OPENVDB_LIBRARY_MINOR_VERSION_NUMBER,
    OPENVDB_LIBRARY_PATCH_VERSION = OPENVDB_LIBRARY_PATCH_VERSION_NUMBER;
/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
const uint32_t OPENVDB_LIBRARY_VERSION = OPENVDB_LIBRARY_VERSION_NUMBER;

/// @brief The current version number of the VDB file format
/// @details  This can be used to enable various backwards compatability switches
/// or to reject files that cannot be read.
const uint32_t OPENVDB_FILE_VERSION = 223;

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
    OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION = 222,
    OPENVDB_FILE_VERSION_BLOSC_COMPRESSION = 223,
    OPENVDB_FILE_VERSION_POINT_INDEX_GRID = 223
};


/// Return a library version number string of the form "<major>.<minor>.<patch>".
inline const char* getLibraryVersionString() { return OPENVDB_LIBRARY_VERSION_STRING; }


struct VersionId {
    uint32_t first, second;
    VersionId(): first(0), second(0) {}
    VersionId(uint32_t major, uint32_t minor): first(major), second(minor) {}
};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_VERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
