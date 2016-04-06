///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#ifndef OPENVDB_OPENVDB_HAS_BEEN_INCLUDED
#define OPENVDB_OPENVDB_HAS_BEEN_INCLUDED

#include "Platform.h"
#include "Types.h"
#include "Metadata.h"
#include "math/Maps.h"
#include "math/Transform.h"
#include "Grid.h"
#include "tree/Tree.h"
#include "io/File.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// Common tree types
typedef tree::Tree4<ValueMask,   5, 4, 3>::Type  MaskTree;
typedef tree::Tree4<bool,        5, 4, 3>::Type  BoolTree;
typedef tree::Tree4<float,       5, 4, 3>::Type  FloatTree;
typedef tree::Tree4<double,      5, 4, 3>::Type  DoubleTree;
typedef tree::Tree4<int32_t,     5, 4, 3>::Type  Int32Tree;
typedef tree::Tree4<uint32_t,    5, 4, 3>::Type  UInt32Tree;
typedef tree::Tree4<int64_t,     5, 4, 3>::Type  Int64Tree;
typedef tree::Tree4<Vec2i,       5, 4, 3>::Type  Vec2ITree;
typedef tree::Tree4<Vec2s,       5, 4, 3>::Type  Vec2STree;
typedef tree::Tree4<Vec2d,       5, 4, 3>::Type  Vec2DTree;
typedef tree::Tree4<Vec3i,       5, 4, 3>::Type  Vec3ITree;
typedef tree::Tree4<Vec3f,       5, 4, 3>::Type  Vec3STree;
typedef tree::Tree4<Vec3d,       5, 4, 3>::Type  Vec3DTree;
typedef tree::Tree4<std::string, 5, 4, 3>::Type  StringTree;
typedef MaskTree  TopologyTree;    
typedef Vec3STree Vec3fTree;
typedef Vec3DTree Vec3dTree;
typedef FloatTree ScalarTree;
typedef Vec3fTree VectorTree;

/// Common grid types
typedef Grid<MaskTree>      MaskGrid;
typedef Grid<BoolTree>      BoolGrid;
typedef Grid<FloatTree>     FloatGrid;
typedef Grid<DoubleTree>    DoubleGrid;
typedef Grid<Int32Tree>     Int32Grid;
typedef Grid<Int64Tree>     Int64Grid;
typedef Grid<Vec3ITree>     Vec3IGrid;
typedef Grid<Vec3STree>     Vec3SGrid;
typedef Grid<Vec3DTree>     Vec3DGrid;
typedef Grid<StringTree>    StringGrid;
typedef Vec3SGrid           Vec3fGrid;
typedef Vec3DGrid           Vec3dGrid;
typedef FloatGrid           ScalarGrid;
typedef Vec3fGrid           VectorGrid;
typedef MaskGrid            TopologyGrid;

/// Global registration of basic types
OPENVDB_API void initialize();

/// Global deregistration of basic types
OPENVDB_API void uninitialize();

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_OPENVDB_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
