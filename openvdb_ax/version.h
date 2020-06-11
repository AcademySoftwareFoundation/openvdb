///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
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

/// @file version.h
///
/// @brief Library and file format version numbers
///
/// @details  Based off of the versioning system within openvdb. There is
///   currently no AX namespace versioning which instead leverages openvdb's
///   version namespace.
///
///   The library minor version number gets incremented whenever a change is made
///   to any aspect of the public API (not just the grid API) that necessitates
///   changes to client code.  Changes to APIs in private or internal namespaces
///   do not trigger a minor version number increment; such APIs should not be
///   used in client code.
///
///    A patch version number increment indicates a change, usually a new feature
///    or a bug fix, that does not necessitate changes to client code but rather
///    only recompilation of that code (because the library namespace
///    incorporates the version number).
///

#ifndef OPENVDB_AX_VERSION_HAS_BEEN_INCLUDED
#define OPENVDB_AX_VERSION_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

// Library major, minor and patch version numbers
#define OPENVDB_AX_LIBRARY_MAJOR_VERSION_NUMBER 0
#define OPENVDB_AX_LIBRARY_MINOR_VERSION_NUMBER 2
#define OPENVDB_AX_LIBRARY_PATCH_VERSION_NUMBER 0

#define OPENVDB_AX_VERSION_NAME                                          \
    OPENVDB_PREPROC_CONCAT(v,                                            \
    OPENVDB_PREPROC_CONCAT(OPENVDB_AX_LIBRARY_MAJOR_VERSION_NUMBER,      \
    OPENVDB_PREPROC_CONCAT(_, OPENVDB_AX_LIBRARY_MINOR_VERSION_NUMBER)))

/// @brief Library version number string of the form "<major>.<minor>.<patch>"
/// @details This is a macro rather than a static constant because we typically
/// want the compile-time version number, not the runtime version number
/// (although the two are usually the same).
/// @hideinitializer
#define OPENVDB_AX_LIBRARY_VERSION_STRING \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_AX_LIBRARY_MAJOR_VERSION_NUMBER) "." \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_AX_LIBRARY_MINOR_VERSION_NUMBER) "." \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_AX_LIBRARY_PATCH_VERSION_NUMBER)

/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
#define OPENVDB_AX_LIBRARY_VERSION_NUMBER \
    ((OPENVDB_AX_LIBRARY_MAJOR_VERSION_NUMBER << 24) | \
    ((OPENVDB_AX_LIBRARY_MINOR_VERSION_NUMBER & 0xFF) << 16) | \
    (OPENVDB_AX_LIBRARY_PATCH_VERSION_NUMBER & 0xFFFF))

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

// Library major, minor and patch version numbers
const uint32_t
    OPENVDB_AX_LIBRARY_MAJOR_VERSION = OPENVDB_AX_LIBRARY_MAJOR_VERSION_NUMBER,
    OPENVDB_AX_LIBRARY_MINOR_VERSION = OPENVDB_AX_LIBRARY_MINOR_VERSION_NUMBER,
    OPENVDB_AX_LIBRARY_PATCH_VERSION = OPENVDB_AX_LIBRARY_PATCH_VERSION_NUMBER;
/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
const uint32_t OPENVDB_AX_LIBRARY_VERSION = OPENVDB_AX_LIBRARY_VERSION_NUMBER;

/// Return a library version number string of the form "<major>.<minor>.<patch>".
inline constexpr const char* getLibraryVersionString() { return OPENVDB_AX_LIBRARY_VERSION_STRING; }

/// @brief  Some extract compiler macros for ignoring specific warnings
/// @note   The attributes ignore will likely go away with the function framework
///         restructuring, but should be migrated to VDB is necessary

#if defined __INTEL_COMPILER
    #define OPENVDB_NO_ATTRIBUTES_WARNING_BEGIN
    #define OPENVDB_NO_ATTRIBUTES_WARNING_END
#elif defined __clang__
    #define OPENVDB_NO_ATTRIBUTES_WARNING_BEGIN \
        _Pragma("clang diagnostic push") \
        _Pragma("clang diagnostic ignored \"-Wignored-attributes\"")
    #define OPENVDB_NO_ATTRIBUTES_WARNING_END \
        _Pragma("clang diagnostic pop")
#elif defined __GNUC__ && OPENVDB_CHECK_GCC(6, 1)
    // Wignored-attributes only introduced in version 6.1
    #define OPENVDB_NO_ATTRIBUTES_WARNING_BEGIN \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wignored-attributes\"")
    #define OPENVDB_NO_ATTRIBUTES_WARNING_END \
        _Pragma("GCC diagnostic pop")
#else
    #define OPENVDB_NO_ATTRIBUTES_WARNING_BEGIN
    #define OPENVDB_NO_ATTRIBUTES_WARNING_END
#endif


} // namespace ax
} // namspace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_VERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
