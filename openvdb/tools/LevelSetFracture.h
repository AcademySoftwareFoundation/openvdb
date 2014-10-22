///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
//
/// @file tools/LevelSetFracture.h
///
/// @brief Divide volumes represented by level set grids into multiple,
/// disjoint pieces by intersecting them with one or more "cutter" volumes,
/// also represented by level sets.

#ifndef OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Quat.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/util/NullInterrupter.h>
#include "Composite.h" // for csgIntersection() and csgDifference()
#include "GridTransformer.h" // for resampleToMatch()
#include "LevelSetUtil.h" // for MinMaxVoxel()
#include <list>
#include <deque>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Level set fracturing
template<class GridType, class InterruptType = util::NullInterrupter>
class LevelSetFracture
{
public:
    typedef std::vector<Vec3s> Vec3sList;
    typedef std::vector<math::Quats> QuatsList;
    typedef std::list<typename GridType::Ptr> GridPtrList;
    typedef typename GridPtrList::iterator GridPtrListIter;


    /// @brief Default constructor
    ///
    /// @param interrupter  optional interrupter object
    explicit LevelSetFracture(InterruptType* interrupter = NULL);

    /// @brief Divide volumes represented by level set grids into multiple,
    /// disjoint pieces by intersecting them with one or more "cutter" volumes,
    /// also represented by level sets.
    /// @details If desired, the process can be applied iteratively, so that
    /// fragments created with one cutter are subdivided by other cutters.
    ///
    /// @note  The incoming @a grids and the @a cutter are required to have matching
    ///        transforms and narrow band widths!
    ///
    /// @param grids          list of grids to fracture. The residuals of the
    ///                       fractured grids will remain in this list
    /// @param cutter         a level set grid to use as the cutter object
    /// @param segment        toggle to split disjoint fragments into their own grids
    /// @param points         optional list of world space points at which to instance the
    ///                       cutter object (if null, use the cutter's current position only)
    /// @param rotations      optional list of custom rotations for each cutter instance
    /// @param cutterOverlap  toggle to allow consecutive cutter instances to fracture
    ///                       previously generated fragments
    void fracture(GridPtrList& grids, const GridType& cutter, bool segment = false,
        const Vec3sList* points = NULL, const QuatsList* rotations = NULL,
        bool cutterOverlap = true);

    /// Return a list of new fragments, not including the residuals from the input grids.
    GridPtrList& fragments() { return mFragments; }

    /// Remove all elements from the fragment list.
    void clear() { mFragments.clear(); }

private:
    // disallow copy by assignment
    void operator=(const LevelSetFracture&) {}

    bool wasInterrupted(int percent = -1) const {
        return mInterrupter && mInterrupter->wasInterrupted(percent);
    }

    bool isValidFragment(GridType&) const;
    void segmentFragments(GridPtrList&) const;
    void process(GridPtrList&, const GridType& cutter);

    InterruptType* mInterrupter;
    GridPtrList mFragments;
};


////////////////////////////////////////


// Internal utility objects and implementation details

namespace internal {

/// @brief Segmentation scheme, splits disjoint fragments into separate grids.
/// @note This is a temporary solution and it will be replaced soon.
template<typename GridType, typename InterruptType>
inline std::vector<typename GridType::Ptr>
segment(GridType& grid, InterruptType* interrupter = NULL)
{
    typedef typename GridType::Ptr GridPtr;
    typedef typename GridType::TreeType TreeType;
    typedef typename TreeType::Ptr TreePtr;
    typedef typename TreeType::ValueType ValueType;

    std::vector<GridPtr> segments;

    while (grid.activeVoxelCount() > 0) {

        if (interrupter && interrupter->wasInterrupted()) break;

        // Deep copy the grid's metadata (tree and transform are shared)
        GridPtr segment(new GridType(grid, ShallowCopy()));
        // Make the transform unique and insert an empty tree
        segment->setTransform(grid.transform().copy());
        TreePtr tree(new TreeType(grid.background()));
        segment->setTree(tree);

        std::deque<Coord> coordList;
        coordList.push_back(grid.tree().beginLeaf()->beginValueOn().getCoord());

        Coord ijk, n_ijk;
        ValueType value;

        typename tree::ValueAccessor<TreeType> sourceAcc(grid.tree());
        typename tree::ValueAccessor<TreeType> targetAcc(segment->tree());

        while (!coordList.empty()) {

            if (interrupter && interrupter->wasInterrupted()) break;

            ijk = coordList.back();
            coordList.pop_back();

            if (!sourceAcc.probeValue(ijk, value)) continue;
            if (targetAcc.isValueOn(ijk)) continue;

            targetAcc.setValue(ijk, value);
            sourceAcc.setValueOff(ijk);

            for (int n = 0; n < 6; n++) {
                n_ijk = ijk + util::COORD_OFFSETS[n];
                if (!targetAcc.isValueOn(n_ijk) && sourceAcc.isValueOn(n_ijk)) {
                    coordList.push_back(n_ijk);
                }
            }
        }
        
        tools::pruneInactive(grid.tree());
        tools::signedFloodFill(segment->tree());
        segments.push_back(segment);
    }
    return segments;
}

} // namespace internal


////////////////////////////////////////


template<class GridType, class InterruptType>
LevelSetFracture<GridType, InterruptType>::LevelSetFracture(InterruptType* interrupter)
    : mInterrupter(interrupter)
    , mFragments()
{
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::fracture(GridPtrList& grids, const GridType& cutter,
    bool segmentation, const Vec3sList* points, const QuatsList* rotations, bool cutterOverlap)
{
    // We can process all incoming grids with the same cutter instance,
    // this optimization is enabled by the requirement of having matching
    // transforms between all incoming grids and the cutter object.
    if (points && points->size() != 0) {

        math::Transform::Ptr originalCutterTransform = cutter.transform().copy();
        GridType cutterGrid(cutter, ShallowCopy());

        const bool hasInstanceRotations =
            points && rotations && points->size() == rotations->size();

        // for each instance point..
        for (size_t p = 0, P = points->size(); p < P; ++p) {

            int percent = int((float(p) / float(P)) * 100.0);
            if (wasInterrupted(percent)) break;

            GridType instCutterGrid;
            instCutterGrid.setTransform(originalCutterTransform->copy());
            math::Transform::Ptr xform = originalCutterTransform->copy();

            if (hasInstanceRotations) {
                const Vec3s& rot = (*rotations)[p].eulerAngles(math::XYZ_ROTATION);
                xform->preRotate(rot[0], math::X_AXIS);
                xform->preRotate(rot[1], math::Y_AXIS);
                xform->preRotate(rot[2], math::Z_AXIS);
                xform->postTranslate((*points)[p]);
            } else {
                xform->postTranslate((*points)[p]);
            }

            cutterGrid.setTransform(xform);

            if (wasInterrupted()) break;

            // Since there is no scaling, use the generic resampler instead of
            // the more expensive level set rebuild tool.
            if (mInterrupter != NULL) {
                doResampleToMatch<BoxSampler>(cutterGrid, instCutterGrid, *mInterrupter);
            } else {
                util::NullInterrupter interrupter;
                doResampleToMatch<BoxSampler>(cutterGrid, instCutterGrid, interrupter);
            }

            if (cutterOverlap && !mFragments.empty()) process(mFragments, instCutterGrid);
            process(grids, instCutterGrid);
        }

    } else {
        // use cutter in place
        if (cutterOverlap && !mFragments.empty()) process(mFragments, cutter);
        process(grids, cutter);
    }

    if (segmentation) {
        segmentFragments(mFragments);
        segmentFragments(grids);
    }
}


template<class GridType, class InterruptType>
bool
LevelSetFracture<GridType, InterruptType>::isValidFragment(GridType& grid) const
{
    typedef typename GridType::TreeType TreeType;
    if (grid.activeVoxelCount() < 27) return false;

    // Check if valid level-set
    {
        tree::LeafManager<TreeType> leafs(grid.tree());
        MinMaxVoxel<TreeType> minmax(leafs);
        minmax.runParallel();

        if ((minmax.minVoxel() < 0) == (minmax.maxVoxel() < 0)) return false;
    }

    return true;
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::segmentFragments(GridPtrList& grids) const
{
    GridPtrList newFragments;

    for (GridPtrListIter it = grids.begin(); it != grids.end(); ++it) {

        if (wasInterrupted()) break;

        std::vector<typename GridType::Ptr> segments = internal::segment(*(*it), mInterrupter);
        for (size_t n = 0, N = segments.size(); n < N; ++n) {

            if (wasInterrupted()) break;

            if (isValidFragment(*segments[n])) {
                newFragments.push_back(segments[n]);
            }
        }
    }

    grids.swap(newFragments);
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::process(
    GridPtrList& grids, const GridType& cutter)
{
    typedef typename GridType::Ptr GridPtr;

    GridPtrList newFragments;

    for (GridPtrListIter it = grids.begin(); it != grids.end(); ++it) {

        if (wasInterrupted()) break;

        GridPtr grid = *it;

        // gen new fragment
        GridPtr fragment = grid->deepCopy();
        csgIntersection(*fragment, *cutter.deepCopy());

        if (wasInterrupted()) break;

        if (!isValidFragment(*fragment)) continue;

        // update residual
        GridPtr residual = grid->deepCopy();
        csgDifference(*residual, *cutter.deepCopy());

        if (wasInterrupted()) break;

        if (!isValidFragment(*residual)) continue;

        newFragments.push_back(fragment);

        grid->tree().clear();
        grid->tree().merge(residual->tree());
    }

    if (!newFragments.empty()) {
        mFragments.splice(mFragments.end(), newFragments);
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
