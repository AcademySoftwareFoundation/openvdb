///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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

#include "openvdb.h"

#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb_points/tools/AttributeGroup.h>
#include <openvdb_points/tools/PointDataGrid.h>

#include <tbb/mutex.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

using namespace openvdb::tools;
using namespace openvdb::math;

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

    // no compression

    TypedAttributeArray<bool>::registerType();
    TypedAttributeArray<int16_t>::registerType();
    TypedAttributeArray<int32_t>::registerType();
    TypedAttributeArray<int64_t>::registerType();
    TypedAttributeArray<half>::registerType();
    TypedAttributeArray<float>::registerType();
    TypedAttributeArray<double>::registerType();
    TypedAttributeArray<Vec2<half> >::registerType();
    TypedAttributeArray<Vec2<float> >::registerType();
    TypedAttributeArray<Vec2<double> >::registerType();
    TypedAttributeArray<Vec3<half> >::registerType();
    TypedAttributeArray<Vec3<float> >::registerType();
    TypedAttributeArray<Vec3<double> >::registerType();

    // group attribute

    GroupAttributeArray::registerType();

    // truncate compression

    TypedAttributeArray<float, NullAttributeCodec<half> >::registerType();
    TypedAttributeArray<Vec3<float>, NullAttributeCodec<Vec3<half> > >::registerType();

    // fixed point compression

    TypedAttributeArray<Vec3<float>, FixedPointAttributeCodec<Vec3<uint8_t> > >::registerType();
    TypedAttributeArray<Vec3<float>, FixedPointAttributeCodec<Vec3<uint16_t> > >::registerType();

    // unit vector compression

    TypedAttributeArray<Vec3<float>, UnitVecAttributeCodec>::registerType();

    // Register types associated with point data grids.
    Metadata::registerType(typeNameAsString<PointDataIndex32>(), Int32Metadata::createMetadata);
    Metadata::registerType(typeNameAsString<PointDataIndex64>(), Int64Metadata::createMetadata);
    tools::PointDataGrid::registerGrid();

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.<
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
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
