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

/// @file test/integration/CompareGrids.h
///
/// @authors Francisco Gochez, Nick Avramoussis
///
/// @brief  Functions for comparing entire VDB grids and generating
///   reports on their differences
///

#ifndef OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED
#define OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Prune.h>

#ifdef OPENVDB_AX_NO_MATRIX
namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
#define MATRIX_OPS(TYPE) \
inline TYPE operator+(const TYPE&, const float&) { throw std::runtime_error("Invalid Matrix op+ called."); } \
inline bool operator<(const TYPE&, const TYPE&) { throw std::runtime_error("Invalid Matrix op< called."); } \
inline bool operator>(const TYPE&, const TYPE&) { throw std::runtime_error("Invalid Matrix op> called."); } \
inline TYPE Abs(const TYPE&) { throw std::runtime_error("Invalid Matrix op abs called."); }

MATRIX_OPS(Mat3<double>)
MATRIX_OPS(Mat3<float>)
MATRIX_OPS(Mat4<double>)
MATRIX_OPS(Mat4<float>)
#undef MATRIX_OPS
}
}
}
#endif // OPENVDB_AX_NO_MATRIX

namespace unittest_util
{


struct ComparisonSettings
{
    bool mCheckTransforms = true;         // Check grid transforms
    bool mCheckTopologyStructure = true;  // Checks node (voxel/leaf/tile) layout
    bool mCheckActiveStates = true;       // Checks voxel active states match
    bool mCheckBufferValues = true;       // Checks voxel buffer values match

    bool mCheckDescriptors = true;        // Check points leaf descriptors
    bool mCheckArrayValues = true;        // Checks attribute array sizes and values
    bool mCheckArrayFlags = true;         // Checks attribute array flags
};

/// @brief The results collected from compareGrids()
///
struct ComparisonResult
{
    ComparisonResult(std::ostream& os = std::cout)
        : mOs(os)
        , mDifferingTopology(openvdb::MaskGrid::create())
        , mDifferingValues(openvdb::MaskGrid::create()) {}

    std::ostream& mOs;
    openvdb::MaskGrid::Ptr mDifferingTopology; // Always empty if mCheckActiveStates is false
    openvdb::MaskGrid::Ptr mDifferingValues;   // Always empty if mCheckBufferValues is false
                                               // or if mCheckBufferValues and mCheckArrayValues
                                               // is false for point data grids
};

template <typename GridType>
bool compareGrids(ComparisonResult& resultData,
                  const GridType& firstGrid,
                  const GridType& secondGrid,
                  const ComparisonSettings& settings,
                  const openvdb::MaskGrid::ConstPtr maskGrid,
                  const typename GridType::ValueType tolerance =
                    openvdb::zeroVal<typename GridType::ValueType>());

bool compareUntypedGrids(ComparisonResult& resultData,
                         const openvdb::GridBase& firstGrid,
                         const openvdb::GridBase& secondGrid,
                         const ComparisonSettings& settings,
                         const openvdb::MaskGrid::ConstPtr maskGrid);

} // namespace unittest_util

#endif // OPENVDB_POINTS_UNITTEST_COMPARE_GRIDS_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
