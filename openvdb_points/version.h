///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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

#ifndef OPENVDB_POINTS_VERSION_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_VERSION_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

/// Always disable Houdini warnings
#ifndef IGNORE_HDK_DEPRECATIONS
#define IGNORE_HDK_DEPRECATIONS
#endif

// Library major, minor and patch version numbers
#define OPENVDB_POINTS_LIBRARY_MAJOR_VERSION_NUMBER 0
#define OPENVDB_POINTS_LIBRARY_MINOR_VERSION_NUMBER 2
#define OPENVDB_POINTS_LIBRARY_PATCH_VERSION_NUMBER 0

/// @brief Library version number string of the form "<major>.<minor>.<patch>"
/// @details This is a macro rather than a static constant because we typically
/// want the compile-time version number, not the runtime version number
/// (although the two are usually the same).
#define OPENVDB_POINTS_LIBRARY_VERSION_STRING "0.2.0"

/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
#define OPENVDB_POINTS_LIBRARY_VERSION_NUMBER \
    ((OPENVDB_POINTS_LIBRARY_MAJOR_VERSION_NUMBER << 24) | \
    ((OPENVDB_POINTS_LIBRARY_MINOR_VERSION_NUMBER & 0xFF) << 16) | \
    (OPENVDB_POINTS_LIBRARY_PATCH_VERSION_NUMBER & 0xFFFF))

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace points {

// Library major, minor and patch version numbers
const uint32_t
    OPENVDB_POINTS_LIBRARY_MAJOR_VERSION = OPENVDB_POINTS_LIBRARY_MAJOR_VERSION_NUMBER,
    OPENVDB_POINTS_LIBRARY_MINOR_VERSION = OPENVDB_POINTS_LIBRARY_MINOR_VERSION_NUMBER,
    OPENVDB_POINTS_LIBRARY_PATCH_VERSION = OPENVDB_POINTS_LIBRARY_PATCH_VERSION_NUMBER;
/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
const uint32_t OPENVDB_POINTS_LIBRARY_VERSION = OPENVDB_POINTS_LIBRARY_VERSION_NUMBER;


/// Return a library version number string of the form "<major>.<minor>.<patch>".
inline const char* getLibraryVersionString() { return OPENVDB_POINTS_LIBRARY_VERSION_STRING; }

} // namespace points

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_VERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
