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

#include "openvdb.h"
#include <tbb/mutex.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

typedef tbb::mutex Mutex;
typedef Mutex::scoped_lock Lock;

// Declare this at file scope to ensure thread-safe initialization.
Mutex sInitMutex;
bool sIsInitialized = false;

void
initialize()
{
    Lock lock(sInitMutex);
    if (sIsInitialized) return;

    // Register metadata.
    Metadata::clearRegistry();
    BoolMetadata::registerType();
    DoubleMetadata::registerType();
    FloatMetadata::registerType();
    Int32Metadata::registerType();
    UInt32Metadata::registerType();
    Int64Metadata::registerType();
    StringMetadata::registerType();
    Vec2IMetadata::registerType();
    Vec2SMetadata::registerType();
    Vec2DMetadata::registerType();
    Vec3IMetadata::registerType();
    Vec3SMetadata::registerType();
    Vec3DMetadata::registerType();
    Mat4SMetadata::registerType();
    Mat4DMetadata::registerType();

    // Register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::UnitaryMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Register common grid types.
    GridBase::clearRegistry();
    BoolGrid::registerGrid();
    FloatGrid::registerGrid();
    DoubleGrid::registerGrid();
    Int32Grid::registerGrid();
    UInt32Grid::registerGrid();
    Int64Grid::registerGrid();
    HermiteGrid::registerGrid();
    StringGrid::registerGrid();
    Vec3IGrid::registerGrid();
    Vec3SGrid::registerGrid();
    Vec3DGrid::registerGrid();

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
// This assignment is mutex-protected and therefore thread-safe.
__pragma(warning(disable:1711))
#endif

    sIsInitialized = true;

#ifdef __ICC
__pragma(warning(default:1711))
#endif
}


void
uninitialize()
{
    Lock lock(sInitMutex);

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
// This assignment is mutex-protected and therefore thread-safe.
__pragma(warning(disable:1711))
#endif

    sIsInitialized = false;

#ifdef __ICC
__pragma(warning(default:1711))
#endif

    Metadata::clearRegistry();
    GridBase::clearRegistry();
    math::MapRegistry::clear();
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
