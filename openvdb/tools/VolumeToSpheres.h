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

#ifndef OPENVDB_TOOLS_VOLUME_TO_SPHERES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_SPHERES_HAS_BEEN_INCLUDED

#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Morphology.h> // for erodeVoxels()

#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>
#include <limits> // std::numeric_limits

//////////


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief  Threaded method to fill a closed level set or fog volume
///         with adaptively sized spheres.
///
/// @param grid             a scalar gird to fill with spheres.
///
/// @param spheres          a @c Vec4 array representing the spheres that returned by this
///                         method. The first three components specify the sphere center
///                         and the fourth is the radius. The spheres in this array are
///                         ordered by radius, biggest to smallest.
///
/// @param maxSphereCount   no more than this number of spheres are generated.
///
/// @param overlapping      toggle to allow spheres to overlap/intersect
///
/// @param minRadius        determines the smallest sphere size in voxel units.
///
/// @param maxRadius        determines the largest sphere size in voxel units.
///
/// @param isovalue         the crossing point of the volume values that is considered
///                         the surface. The zero default value works for signed distance
///                         fields while fog volumes require a larger positive value,
///                         0.5 is a good initial guess.
///
/// @param instanceCount    how many interior points to consider for the sphere placement,
///                         increasing this count increases the chances of finding optimal
///                         sphere sizes.
///
/// @param interrupter      a pointer adhering to the util::NullInterrupter interface
///
template<typename GridT, typename InterrupterT>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    int maxSphereCount,
    bool overlapping = false,
    float minRadius = 1.0,
    float maxRadius = std::numeric_limits<float>::max(),
    float isovalue = 0.0,
    int instanceCount = 10000,
    InterrupterT* interrupter = NULL);


/// @brief  @c fillWithSpheres method variant that automatically infers
///         the util::NullInterrupter.
template<typename GridT>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    int maxSphereCount,
    bool overlapping = false,
    float minRadius = 1.0,
    float maxRadius = std::numeric_limits<float>::max(),
    float isovalue = 0.0,
    int instanceCount = 10000)
{
    fillWithSpheres<GridT, util::NullInterrupter>(grid, spheres,
        maxSphereCount, overlapping, minRadius, maxRadius, isovalue, instanceCount);
}


////////////////////////////////////////


/// @brief  Accelerated closest surface point queries for narrow band level sets.
///         Supports queries that originate at arbitrary world-space locations, is
///         not confined to the narrow band region of the input volume geometry.
template<typename GridT>
class ClosestSurfacePoint
{
public:
    typedef typename GridT::TreeType TreeT;
    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename TreeT::template ValueConverter<Int16>::Type Int16TreeT;


    ClosestSurfacePoint();


    /// @brief  Extracts the surface points and constructs a spatial acceleration structure.
    ///
    /// @param grid             a scalar gird, level set or fog volume.
    ///
    /// @param isovalue         the crossing point of the volume values that is considered
    ///                         the surface. The zero default value works for signed distance
    ///                         fields while fog volumes require a larger positive value,
    ///                         0.5 is a good initial guess.
    ///
    /// @param interrupter      a pointer adhering to the util::NullInterrupter interface.
    ///
    template<typename InterrupterT>
    void initialize(const GridT& grid, float isovalue = 0.0, InterrupterT* interrupter = NULL);


    /// @brief  @c initialize method variant that automatically infers
    ///         the util::NullInterrupter.
    void initialize(const GridT& grid, float isovalue = 0.0);



    /// @brief Computes distance to closest surface.
    ///
    /// @param points       search locations in world space.
    ///
    /// @param distances    list of closest surface point distances, populated by this method.
    ///
    bool search(const std::vector<Vec3R>& points, std::vector<float>& distances);


    /// @brief Performs closest point searches.
    ///
    /// @param points       search locations in world space to be replaced by their closest
    ///                     surface point.
    ///
    /// @param distances    list of closest surface point distances, populated by this method.
    ///
    bool searchAndReplace(std::vector<Vec3R>& points, std::vector<float>& distances);


    /// @{
    /// @brief Tree accessors
    const IntTreeT& indexTree() const { return *mIdxTreePt; }
    const Int16TreeT& signTree() const { return *mSignTreePt; }
    /// @}

private:
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef std::pair<size_t, size_t> IndexRange;

    bool mIsInitialized;
    std::vector<Vec4R> mLeafBoundingSpheres, mNodeBoundingSpheres;
    std::vector<IndexRange> mLeafRanges;
    std::vector<const IntLeafT*> mLeafNodes;
    PointList mSurfacePointList;
    size_t mPointListSize, mMaxNodeLeafs;
    float mMaxRadiusSqr;
    typename IntTreeT::Ptr mIdxTreePt;
    typename Int16TreeT::Ptr mSignTreePt;

    bool search(std::vector<Vec3R>&, std::vector<float>&, bool transformPoints);
};


////////////////////////////////////////




// Internal utility methods


namespace internal {

struct PointAccessor
{
    PointAccessor(std::vector<Vec3R>& points)
        : mPoints(points)
    {
    }

    void add(const Vec3R &pos)
    {
        mPoints.push_back(pos);
    }
private:
    std::vector<Vec3R>& mPoints;
};


template<typename IntLeafT>
class LeafBS
{
public:

    LeafBS(std::vector<Vec4R>& leafBoundingSpheres,
        const std::vector<const IntLeafT*>& leafNodes,
        const math::Transform& transform,
        const PointList& surfacePointList);

    void run(bool threaded = true);


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    std::vector<Vec4R>& mLeafBoundingSpheres;
    const std::vector<const IntLeafT*>& mLeafNodes;
    const math::Transform& mTransform;
    const PointList& mSurfacePointList;
};

template<typename IntLeafT>
LeafBS<IntLeafT>::LeafBS(
    std::vector<Vec4R>& leafBoundingSpheres,
    const std::vector<const IntLeafT*>& leafNodes,
    const math::Transform& transform,
    const PointList& surfacePointList)
    : mLeafBoundingSpheres(leafBoundingSpheres)
    , mLeafNodes(leafNodes)
    , mTransform(transform)
    , mSurfacePointList(surfacePointList)
{
}

template<typename IntLeafT>
void
LeafBS<IntLeafT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mLeafNodes.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mLeafNodes.size()));
    }
}

template<typename IntLeafT>
void
LeafBS<IntLeafT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typename IntLeafT::ValueOnCIter iter;
    Vec3s avg;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = 0.0;

        int count = 0;
        for (iter = mLeafNodes[n]->cbeginValueOn(); iter; ++iter) {
            avg += mSurfacePointList[iter.getValue()];
            ++count;
        }

        if (count > 1) avg *= float(1.0 / double(count));

        float maxDist = 0.0;

        for (iter = mLeafNodes[n]->cbeginValueOn(); iter; ++iter) {
            float tmpDist = (mSurfacePointList[iter.getValue()] - avg).lengthSqr();
            if (tmpDist > maxDist) maxDist = tmpDist;
        }

        Vec4R& sphere = mLeafBoundingSpheres[n];

        sphere[0] = avg[0];
        sphere[1] = avg[1];
        sphere[2] = avg[2];
        sphere[3] = maxDist * 2.0; // padded radius
    }
}


class NodeBS
{
public:
    typedef std::pair<size_t, size_t> IndexRange;

    NodeBS(std::vector<Vec4R>& nodeBoundingSpheres,
        const std::vector<IndexRange>& leafRanges,
        const std::vector<Vec4R>& leafBoundingSpheres);

    inline void run(bool threaded = true);

    inline void operator()(const tbb::blocked_range<size_t>&) const;

private:
    std::vector<Vec4R>& mNodeBoundingSpheres;
    const std::vector<IndexRange>& mLeafRanges;
    const std::vector<Vec4R>& mLeafBoundingSpheres;
};

inline
NodeBS::NodeBS(std::vector<Vec4R>& nodeBoundingSpheres,
    const std::vector<IndexRange>& leafRanges,
    const std::vector<Vec4R>& leafBoundingSpheres)
    : mNodeBoundingSpheres(nodeBoundingSpheres)
    , mLeafRanges(leafRanges)
    , mLeafBoundingSpheres(leafBoundingSpheres)
{
}

inline void
NodeBS::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mLeafRanges.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mLeafRanges.size()));
    }
}

inline void
NodeBS::operator()(const tbb::blocked_range<size_t>& range) const
{
    Vec3d avg, pos;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = 0.0;

        int count = int(mLeafRanges[n].second) - int(mLeafRanges[n].first);

        for (size_t i = mLeafRanges[n].first; i < mLeafRanges[n].second; ++i) {
            avg[0] += mLeafBoundingSpheres[i][0];
            avg[1] += mLeafBoundingSpheres[i][1];
            avg[2] += mLeafBoundingSpheres[i][2];
        }

        if (count > 1) avg *= float(1.0 / double(count));


        double maxDist = 0.0;

        for (size_t i = mLeafRanges[n].first; i < mLeafRanges[n].second; ++i) {
            pos[0] = mLeafBoundingSpheres[i][0];
            pos[1] = mLeafBoundingSpheres[i][1];
            pos[2] = mLeafBoundingSpheres[i][2];

            double tmpDist = (pos - avg).lengthSqr() + mLeafBoundingSpheres[i][3];
            if (tmpDist > maxDist) maxDist = tmpDist;
        }

        Vec4R& sphere = mNodeBoundingSpheres[n];

        sphere[0] = avg[0];
        sphere[1] = avg[1];
        sphere[2] = avg[2];
        sphere[3] = maxDist * 2.0; // padded radius
    }
}



////////////////////////////////////////


template<typename IntLeafT>
class ClosestPointDist
{
public:
    typedef std::pair<size_t, size_t> IndexRange;

    ClosestPointDist(
        std::vector<Vec3R>& instancePoints,
        std::vector<float>& instanceDistances,
        const PointList& surfacePointList,
        const std::vector<const IntLeafT*>& leafNodes,
        const std::vector<IndexRange>& leafRanges,
        const std::vector<Vec4R>& leafBoundingSpheres,
        const std::vector<Vec4R>& nodeBoundingSpheres,
        size_t maxNodeLeafs,
        bool transformPoints = false);


    void run(bool threaded = true);


    void operator()(const tbb::blocked_range<size_t>&) const;

private:

    void evalLeaf(size_t index, const IntLeafT& leaf) const;
    void evalNode(size_t pointIndex, size_t nodeIndex) const;


    std::vector<Vec3R>& mInstancePoints;
    std::vector<float>& mInstanceDistances;

    const PointList& mSurfacePointList;

    const std::vector<const IntLeafT*>& mLeafNodes;
    const std::vector<IndexRange>& mLeafRanges;
    const std::vector<Vec4R>& mLeafBoundingSpheres;
    const std::vector<Vec4R>& mNodeBoundingSpheres;

    std::vector<float> mLeafDistances, mNodeDistances;

    const bool mTransformPoints;
    size_t mClosestPointIndex;
};


template<typename IntLeafT>
ClosestPointDist<IntLeafT>::ClosestPointDist(
    std::vector<Vec3R>& instancePoints,
    std::vector<float>& instanceDistances,
    const PointList& surfacePointList,
    const std::vector<const IntLeafT*>& leafNodes,
    const std::vector<IndexRange>& leafRanges,
    const std::vector<Vec4R>& leafBoundingSpheres,
    const std::vector<Vec4R>& nodeBoundingSpheres,
    size_t maxNodeLeafs,
    bool transformPoints)
    : mInstancePoints(instancePoints)
    , mInstanceDistances(instanceDistances)
    , mSurfacePointList(surfacePointList)
    , mLeafNodes(leafNodes)
    , mLeafRanges(leafRanges)
    , mLeafBoundingSpheres(leafBoundingSpheres)
    , mNodeBoundingSpheres(nodeBoundingSpheres)
    , mLeafDistances(maxNodeLeafs, 0.0)
    , mNodeDistances(leafRanges.size(), 0.0)
    , mTransformPoints(transformPoints)
    , mClosestPointIndex(0)
{
}


template<typename IntLeafT>
void
ClosestPointDist<IntLeafT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mInstancePoints.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mInstancePoints.size()));
    }
}

template<typename IntLeafT>
void
ClosestPointDist<IntLeafT>::evalLeaf(size_t index, const IntLeafT& leaf) const
{
    typename IntLeafT::ValueOnCIter iter;
    const Vec3s center = mInstancePoints[index];
    size_t& closestPointIndex = const_cast<size_t&>(mClosestPointIndex);

    for (iter = leaf.cbeginValueOn(); iter; ++iter) {

        const Vec3s& point = mSurfacePointList[iter.getValue()];
        float tmpDist = (point - center).lengthSqr();

        if (tmpDist < mInstanceDistances[index]) {
            mInstanceDistances[index] = tmpDist;
            closestPointIndex = iter.getValue();
        }
    }
}


template<typename IntLeafT>
void
ClosestPointDist<IntLeafT>::evalNode(size_t pointIndex, size_t nodeIndex) const
{
    const Vec3R& pos = mInstancePoints[pointIndex];
    float minDist = mInstanceDistances[pointIndex];
    size_t minDistIdx = 0;
    Vec3R center;
    bool updatedDist = false;

    for (size_t i = mLeafRanges[nodeIndex].first, n = 0;
        i < mLeafRanges[nodeIndex].second; ++i, ++n)
    {
        float& distToLeaf = const_cast<float&>(mLeafDistances[n]);

        center[0] = mLeafBoundingSpheres[i][0];
        center[1] = mLeafBoundingSpheres[i][1];
        center[2] = mLeafBoundingSpheres[i][2];

        distToLeaf = float((pos - center).lengthSqr() - mLeafBoundingSpheres[i][3]);

        if (distToLeaf < minDist) {
            minDist = distToLeaf;
            minDistIdx = i;
            updatedDist = true;
        }
    }

    if (!updatedDist) return;

    evalLeaf(pointIndex, *mLeafNodes[minDistIdx]);

    for (size_t i = mLeafRanges[nodeIndex].first, n = 0;
        i < mLeafRanges[nodeIndex].second; ++i, ++n)
    {
        if (mLeafDistances[n] < mInstanceDistances[pointIndex] && i != minDistIdx) {
            evalLeaf(pointIndex, *mLeafNodes[i]);
        }
    }
}


template<typename IntLeafT>
void
ClosestPointDist<IntLeafT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    Vec3R center;
    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Vec3R& pos = mInstancePoints[n];
        float minDist = mInstanceDistances[n];
        size_t minDistIdx = 0;

        for (size_t i = 0, I = mNodeDistances.size(); i < I; ++i) {
            float& distToNode = const_cast<float&>(mNodeDistances[i]);

            center[0] = mNodeBoundingSpheres[i][0];
            center[1] = mNodeBoundingSpheres[i][1];
            center[2] = mNodeBoundingSpheres[i][2];

            distToNode = float((pos - center).lengthSqr() - mNodeBoundingSpheres[i][3]);

            if (distToNode < minDist) {
                minDist = distToNode;
                minDistIdx = i;
            }
        }

        evalNode(n, minDistIdx);

        for (size_t i = 0, I = mNodeDistances.size(); i < I; ++i) {
            if (mNodeDistances[i] < mInstanceDistances[n] && i != minDistIdx) {
                evalNode(n, i);
            }
        }

        mInstanceDistances[n] = std::sqrt(mInstanceDistances[n]);

        if (mTransformPoints) mInstancePoints[n] = mSurfacePointList[mClosestPointIndex];
    }
}


class UpdatePoints
{
public:
    UpdatePoints(
        const Vec4s& sphere,
        const std::vector<Vec3R>& points,
        std::vector<float>& distances,
        std::vector<unsigned char>& mask,
        bool overlapping);

    float radius() const { return mRadius; }
    int index() const { return mIndex; }

    inline void run(bool threaded = true);


    UpdatePoints(UpdatePoints&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>& range);
    void join(const UpdatePoints& rhs)
    {
        if (rhs.mRadius > mRadius) {
            mRadius = rhs.mRadius;
            mIndex = rhs.mIndex;
        }
    }

private:

    const Vec4s& mSphere;
    const std::vector<Vec3R>& mPoints;

    std::vector<float>& mDistances;
    std::vector<unsigned char>& mMask;

    bool mOverlapping;
    float mRadius;
    int mIndex;
};

inline
UpdatePoints::UpdatePoints(
    const Vec4s& sphere,
    const std::vector<Vec3R>& points,
    std::vector<float>& distances,
    std::vector<unsigned char>& mask,
    bool overlapping)
    : mSphere(sphere)
    , mPoints(points)
    , mDistances(distances)
    , mMask(mask)
    , mOverlapping(overlapping)
    , mRadius(0.0)
    , mIndex(0)
{
}

inline
UpdatePoints::UpdatePoints(UpdatePoints& rhs, tbb::split)
    : mSphere(rhs.mSphere)
    , mPoints(rhs.mPoints)
    , mDistances(rhs.mDistances)
    , mMask(rhs.mMask)
    , mOverlapping(rhs.mOverlapping)
    , mRadius(rhs.mRadius)
    , mIndex(rhs.mIndex)
{
}

inline void
UpdatePoints::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPoints.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mPoints.size()));
    }
}

inline void
UpdatePoints::operator()(const tbb::blocked_range<size_t>& range)
{
    Vec3s pos;
    for (size_t n = range.begin(); n != range.end(); ++n) {
        if (mMask[n]) continue;

        pos.x() = float(mPoints[n].x()) - mSphere[0];
        pos.y() = float(mPoints[n].y()) - mSphere[1];
        pos.z() = float(mPoints[n].z()) - mSphere[2];

        float dist = pos.length();

        if (dist < mSphere[3]) {
            mMask[n] = 1;
            continue;
        }

        if (!mOverlapping) {
            mDistances[n] = std::min(mDistances[n], (dist - mSphere[3]));
        }

        if (mDistances[n] > mRadius) {
            mRadius = mDistances[n];
            mIndex = int(n);
        }
    }
}


} // namespace internal


////////////////////////////////////////


template<typename GridT, typename InterrupterT>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    int maxSphereCount,
    bool overlapping,
    float minRadius,
    float maxRadius,
    float isovalue,
    int instanceCount,
    InterrupterT* interrupter)
{
    spheres.clear();
    spheres.reserve(maxSphereCount);

    const bool addNBPoints = grid.activeVoxelCount() < 10000;
    int instances = std::max(instanceCount, maxSphereCount);

    typedef typename GridT::TreeType TreeT;
    typedef typename GridT::ValueType ValueT;

    typedef typename TreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename TreeT::template ValueConverter<Int16>::Type Int16TreeT;

    typedef boost::mt11213b RandGen;
    RandGen mtRand(/*seed=*/0);

    const TreeT& tree = grid.tree();
    const math::Transform& transform = grid.transform();

    std::vector<Vec3R> instancePoints;

    { // Scatter candidate sphere centroids (instancePoints)
        typename Grid<BoolTreeT>::Ptr interiorMaskPtr;

        if (grid.getGridClass() == GRID_LEVEL_SET) {
            interiorMaskPtr = sdfInteriorMask(grid, ValueT(isovalue));
        } else {
            interiorMaskPtr = typename Grid<BoolTreeT>::Ptr(Grid<BoolTreeT>::create(false));
            interiorMaskPtr->setTransform(transform.copy());
            interiorMaskPtr->tree().topologyUnion(tree);
        }

        if (interrupter && interrupter->wasInterrupted()) return;

        erodeVoxels(interiorMaskPtr->tree(), 1);

        instancePoints.reserve(instances);
        internal::PointAccessor ptnAcc(instancePoints);

        UniformPointScatter<internal::PointAccessor, RandGen, InterrupterT> scatter(
            ptnAcc, Index64(addNBPoints ? (instances / 2) : instances), mtRand, interrupter);

        scatter(*interiorMaskPtr);
    }

    if (interrupter && interrupter->wasInterrupted()) return;

    std::vector<float> instanceRadius;

    ClosestSurfacePoint<GridT> csp;
    csp.initialize(grid, isovalue, interrupter);

    // add extra instance points in the interior narrow band.
    if (instancePoints.size() < size_t(instances)) {
        const Int16TreeT& signTree = csp.signTree();
        typename Int16TreeT::LeafNodeType::ValueOnCIter it;
        typename Int16TreeT::LeafCIter leafIt = signTree.cbeginLeaf();

        for (; leafIt; ++leafIt) {
            for (it = leafIt->cbeginValueOn(); it; ++it) {
                const int flags = it.getValue();
                if (!(0xE00 & flags) && (flags & 0x100)) {
                    instancePoints.push_back(transform.indexToWorld(it.getCoord()));
                }

                if (instancePoints.size() == size_t(instances)) break;
            }
            if (instancePoints.size() == size_t(instances)) break;
        }
    }


    if (interrupter && interrupter->wasInterrupted()) return;

    if (!csp.search(instancePoints, instanceRadius)) return;

    std::vector<unsigned char> instanceMask(instancePoints.size(), 0);
    float largestRadius = 0.0;
    int largestRadiusIdx = 0;

    for (size_t n = 0, N = instancePoints.size(); n < N; ++n) {
        if (instanceRadius[n] > largestRadius) {
            largestRadius = instanceRadius[n];
            largestRadiusIdx = int(n);
        }
    }

    Vec3s pos;
    Vec4s sphere;
    minRadius = float(minRadius * transform.voxelSize()[0]);
    maxRadius = float(maxRadius * transform.voxelSize()[0]);

    for (size_t s = 0, S = std::min(size_t(maxSphereCount), instancePoints.size()); s < S; ++s) {

        if (interrupter && interrupter->wasInterrupted()) return;

        largestRadius = std::min(maxRadius, largestRadius);

        if (s != 0 && largestRadius < minRadius) break;

        sphere[0] = float(instancePoints[largestRadiusIdx].x());
        sphere[1] = float(instancePoints[largestRadiusIdx].y());
        sphere[2] = float(instancePoints[largestRadiusIdx].z());
        sphere[3] = largestRadius;

        spheres.push_back(sphere);
        instanceMask[largestRadiusIdx] = 1;

        internal::UpdatePoints op(
            sphere, instancePoints, instanceRadius, instanceMask, overlapping);
        op.run();

        largestRadius = op.radius();
        largestRadiusIdx = op.index();
    }
}

////////////////////////////////////////


template<typename GridT>
ClosestSurfacePoint<GridT>::ClosestSurfacePoint()
    : mIsInitialized(false)
    , mLeafBoundingSpheres(0)
    , mNodeBoundingSpheres(0)
    , mLeafRanges(0)
    , mLeafNodes(0)
    , mSurfacePointList()
    , mPointListSize(0)
    , mMaxNodeLeafs(0)
    , mMaxRadiusSqr(0.0)
    , mIdxTreePt()
{
}

template<typename GridT>
void
ClosestSurfacePoint<GridT>::initialize(const GridT& grid, float isovalue)
{
    initialize<GridT, util::NullInterrupter>(grid, isovalue, NULL);
}


template<typename GridT>
template<typename InterrupterT>
void
ClosestSurfacePoint<GridT>::initialize(
    const GridT& grid, float isovalue, InterrupterT* interrupter)
{
    mIsInitialized = false;
    typedef tree::LeafManager<const TreeT> LeafManagerT;
    typedef tree::LeafManager<IntTreeT>    IntLeafManagerT;
    typedef tree::LeafManager<Int16TreeT>  Int16LeafManagerT;
    typedef typename GridT::ValueType ValueT;

    const TreeT& tree = grid.tree();
    const math::Transform& transform = grid.transform();

    { // Extract surface point cloud

        {
            LeafManagerT leafs(tree);
            internal::SignData<TreeT, LeafManagerT>
                signDataOp(tree, leafs, ValueT(isovalue));

            signDataOp.run();

            mSignTreePt = signDataOp.signTree();
            mIdxTreePt = signDataOp.idxTree();
        }

        if (interrupter && interrupter->wasInterrupted()) return;

        Int16LeafManagerT signLeafs(*mSignTreePt);

        std::vector<size_t> regions(signLeafs.leafCount(), 0);
        signLeafs.foreach(internal::CountPoints(regions));

        mPointListSize = 0;
        for (size_t tmp = 0, n = 0, N = regions.size(); n < N; ++n) {
            tmp = regions[n];
            regions[n] = mPointListSize;
            mPointListSize += tmp;
        }

        if (mPointListSize == 0) return;

        mSurfacePointList.reset(new Vec3s[mPointListSize]);

        internal::GenPoints<TreeT, Int16LeafManagerT>
            pointOp(signLeafs, tree, *mIdxTreePt, mSurfacePointList, regions, transform, isovalue);

        pointOp.run();

        mIdxTreePt->topologyUnion(*mSignTreePt);
    }

    if (interrupter && interrupter->wasInterrupted()) return;

    // estimate max sphere radius (sqr dist)
    CoordBBox bbox =  grid.evalActiveVoxelBoundingBox();

    Vec3s dim = transform.indexToWorld(bbox.min()) -
        transform.indexToWorld(bbox.max());

    dim[0] = std::abs(dim[0]);
    dim[1] = std::abs(dim[1]);
    dim[2] = std::abs(dim[2]);

    mMaxRadiusSqr = std::min(std::min(dim[0], dim[1]), dim[2]);
    mMaxRadiusSqr *= 0.51f;
    mMaxRadiusSqr *= mMaxRadiusSqr;


    IntLeafManagerT idxLeafs(*mIdxTreePt);


    typedef typename IntTreeT::RootNodeType IntRootNodeT;
    typedef typename IntRootNodeT::NodeChainType IntNodeChainT;
    BOOST_STATIC_ASSERT(boost::mpl::size<IntNodeChainT>::value > 1);
    typedef typename boost::mpl::at<IntNodeChainT, boost::mpl::int_<1> >::type IntInternalNodeT;


    typename IntTreeT::NodeCIter nIt = mIdxTreePt->cbeginNode();
    nIt.setMinDepth(IntTreeT::NodeCIter::LEAF_DEPTH - 1);
    nIt.setMaxDepth(IntTreeT::NodeCIter::LEAF_DEPTH - 1);

    std::vector<const IntInternalNodeT*> internalNodes;

    const IntInternalNodeT* node = NULL;
    for (; nIt; ++nIt) {
        nIt.getNode(node);
        if (node) internalNodes.push_back(node);
    }

    std::vector<IndexRange>().swap(mLeafRanges);
    mLeafRanges.resize(internalNodes.size());

    std::vector<const IntLeafT*>().swap(mLeafNodes);
    mLeafNodes.reserve(idxLeafs.leafCount());

    typename IntInternalNodeT::ChildOnCIter leafIt;
    mMaxNodeLeafs = 0;
    for (size_t n = 0, N = internalNodes.size(); n < N; ++n) {

        mLeafRanges[n].first = mLeafNodes.size();

        size_t leafCount = 0;
        for (leafIt = internalNodes[n]->cbeginChildOn(); leafIt; ++leafIt) {
            mLeafNodes.push_back(&(*leafIt));
            ++leafCount;
        }

        mMaxNodeLeafs = std::max(leafCount, mMaxNodeLeafs);

        mLeafRanges[n].second = mLeafNodes.size();
    }

    std::vector<Vec4R>().swap(mLeafBoundingSpheres);
    mLeafBoundingSpheres.resize(mLeafNodes.size());

    internal::LeafBS<IntLeafT> leafBS(
        mLeafBoundingSpheres, mLeafNodes, transform, mSurfacePointList);
    leafBS.run();


    std::vector<Vec4R>().swap(mNodeBoundingSpheres);
    mNodeBoundingSpheres.resize(internalNodes.size());

    internal::NodeBS nodeBS(mNodeBoundingSpheres, mLeafRanges, mLeafBoundingSpheres);
    nodeBS.run();
    mIsInitialized = true;
}


template<typename GridT>
bool
ClosestSurfacePoint<GridT>::search(std::vector<Vec3R>& points,
    std::vector<float>& distances, bool transformPoints)
{
    if (!mIsInitialized) return false;

    distances.clear();
    distances.resize(points.size(), mMaxRadiusSqr);

    internal::ClosestPointDist<IntLeafT> cpd(points, distances, mSurfacePointList,
        mLeafNodes, mLeafRanges, mLeafBoundingSpheres, mNodeBoundingSpheres,
        mMaxNodeLeafs, transformPoints);

    cpd.run();

    return true;
}


template<typename GridT>
bool
ClosestSurfacePoint<GridT>::search(const std::vector<Vec3R>& points, std::vector<float>& distances)
{
    return search(const_cast<std::vector<Vec3R>& >(points), distances, false);
}


template<typename GridT>
bool
ClosestSurfacePoint<GridT>::searchAndReplace(std::vector<Vec3R>& points,
    std::vector<float>& distances)
{
    return search(points, distances, true);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
