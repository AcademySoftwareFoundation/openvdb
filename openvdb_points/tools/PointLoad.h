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
//
/// @author Dan Bailey
///
/// @file PointLoad.h
///
/// @brief  Various point loading methods using a VDB Point Grid.
///


#ifndef OPENVDB_TOOLS_POINT_LOAD_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_LOAD_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/PointDataGrid.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Loads all leaf node voxel data in the given grid.
///
/// @param grid  the Grid to be loaded.
/// @note This method wraps readNonresidentBuffers().
template <typename PointDataGridT>
void loadPoints(PointDataGridT& grid);


/// @brief Loads all leaf node voxel data in the given grid that
/// overlap with mask grid leaf nodes.
///
/// @param grid  the Grid to be loaded.
/// @param mask  the mask to denote region of points to load
template <typename PointDataGridT, typename MaskGridT>
void loadPoints(PointDataGridT& grid, const MaskGridT& mask);


/// @brief Load the leaf node voxel data in the given grid that
/// overlap with a world-space bounding box.
///
/// @param grid  the Grid to be loaded.
/// @param bbox  the bbox to denote region of points to load
///
/// @note Does not clip to the bounding box, leaf nodes with any
/// overlap will be loaded.
template <typename PointDataGridT>
void loadPoints(PointDataGridT& grid, const BBoxd& bbox);


////////////////////////////////////////


#ifndef OPENVDB_2_ABI_COMPATIBLE
template <typename PointDataGridT>
void loadPoints(PointDataGridT& grid)
{
    grid.constTree().readNonresidentBuffers();
}
#else
template <typename PointDataGridT>
void loadPoints(PointDataGridT&)
{
    // out-of-core not supported with ABI 2
}
#endif


template <typename PointDataGridT, typename MaskGridT>
void loadPoints(PointDataGridT& grid, const MaskGridT& mask)
{
    using PointDataTreeT = typename PointDataGridT::TreeType;

    tree::ValueAccessor<const PointDataTreeT> pointsAcc(grid.constTree());

    auto leafIter = mask.constTree().cbeginLeaf();

    for (; leafIter; ++leafIter) {
        const Coord& ijk = leafIter->origin();
        const typename PointDataTreeT::LeafNodeType* leaf = pointsAcc.probeConstLeaf(ijk);

        if (!leaf)  continue;

        // load out of core leaf nodes
        if (leaf->buffer().isOutOfCore())    leaf->buffer().data();
    }
}


template <typename PointDataGridT>
void loadPoints(PointDataGridT& grid, const BBoxd& bbox)
{
    using BoolGridT = typename PointDataGridT::template ValueConverter<bool>::Type;

    // Transform the world-space bounding box into the source grid's index space.
    Vec3d idxMin, idxMax;
    math::calculateBounds(grid.constTransform(), bbox.min(), bbox.max(), idxMin, idxMax);
    CoordBBox region(Coord::floor(idxMin), Coord::floor(idxMax));

    // Construct a boolean mask grid that is true inside the index-space bounding box
    // and false everywhere else.
    BoolGridT clipMask(/*background=*/false);
    clipMask.fill(region, /*value=*/true, /*active=*/true);

    // MaskGrid introduced in OpenVDB 3.2
    using MaskType = BoolGrid;

    // Convert the input grid to a mask grid (with the same tree configuration).
    auto pointsMask = MaskType::create(/*background=*/false);
    pointsMask->topologyUnion(grid);
    pointsMask->topologyIntersection(clipMask);

    loadPoints(grid, *pointsMask);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_LOAD_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
