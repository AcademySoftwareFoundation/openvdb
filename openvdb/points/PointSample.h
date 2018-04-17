///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

/// @author Nick Avramoussis, Francisco Gochez
///
/// @file points/PointSample.h
///
/// @brief Sample a VDB Grid onto a VDB Points attribute

#ifndef OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED

#include <type_traits> // enable_if

#include "PointDataGrid.h"
#include "PointGroup.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Samples values from an OpenVDB grid onto VDB Points
///
/// @details Performs sampling of an OpenVDB grid onto a collection of points and stores
///          the results into a target attribute. If the attribute does not yet exist on the
///          points, it will be added.  If it already exists it can be of a different type from the
///          grid provided it can be cast with default constructors.
///          The "Order" template parameter determines the form of sampling  which is performed.
///          0 will use closest point, 1 will use tri-linear interpolation, and 2 will use
///          tri-quadratic (see also the Sampler struct in tools/Interpolation.h) interpolation.
/// @note    Samplers of order higher than 2 are not supported.
///
///  @param points           the PointDataGrid whose points will be sampled on to
///  @param sourceGrid       VDB grid which will be sampled
///  @param targetAttribute  a target attribute on the points which will hold samples. This
///                          attribute will be created with the source grid type if it does
///                          not exist
///  @param includeGroups    names of groups to include
///  @param excludeGroups    names of groups to exclude
template <typename SourceGridT, typename PointDataGridT, size_t Order = 1>
inline void sampleGrid(PointDataGridT& points,
                       const SourceGridT& sourceGrid,
                       const Name& targetAttribute,
                       const std::vector<Name>& includeGroups = std::vector<Name>(),
                       const std::vector<Name>& excludeGroups = std::vector<Name>());


///////////////////////////////////////////////////


namespace point_sample_internal {

template<typename SamplerT,
         typename AccessorT,
         typename TargetValueT,
         typename PositionT = Vec3f,
         typename PointFilterT = points::NullFilter,
         typename CodecT = points::UnknownCodec,
         typename PointDataTreeT = points::PointDataTree,
         bool Transform = false>
class PointSampleOp
{
public:
    using LeafManagerT = tree::LeafManager<PointDataTreeT>;
    using PointDataLeafNodeT = typename PointDataTreeT::LeafNodeType;
    using PointIteratorT = IndexIter<points::ValueVoxelCIter, PointFilterT>;

    using PositionAttributeHandleT = AttributeHandle<PositionT, CodecT>;
    using TargetAttributeHandleT = AttributeWriteHandle<TargetValueT>;

    PointSampleOp(const AccessorT& sourceAccessor,
                  const size_t positionAttributeIndex,
                  const size_t targetAttributeIndex,
                  const PointFilterT& filter = PointFilterT(),
                  const math::Transform* const pointDataGridTransform = nullptr,
                  const math::Transform* const sourceGridTransform = nullptr)
    : mSourceAccessor(sourceAccessor)
    , mPositionAttributeIndex(positionAttributeIndex)
    , mTargetAttributeIndex(targetAttributeIndex)
    , mFilter(filter)
    , mPointDataGridTransform(pointDataGridTransform)
    , mSourceGridTransform(sourceGridTransform)
    {
        if (Transform) {
            assert(mSourceGridTransform);
            assert(mPointDataGridTransform);
        }
    }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        PointFilterT filterCopy(mFilter);

        for (auto leaf = range.begin(); leaf; ++leaf) {
            // get the relevant point data attribute arrays for the voxel in question
            typename PositionAttributeHandleT::Ptr positionHandle =
                PositionAttributeHandleT::create(
                    leaf->constAttributeArray(mPositionAttributeIndex));

            typename TargetAttributeHandleT::Ptr targetHandle =
                TargetAttributeHandleT::create(
                    leaf->attributeArray(mTargetAttributeIndex));

            filterCopy.reset(*leaf);

            for (auto voxel = leaf->cbeginValueOn(); voxel; ++voxel) {
                const Coord& coord = voxel.getCoord();

                PointIteratorT iter(leaf->beginValueVoxel(coord), filterCopy);
                if (!iter) continue;

                const Vec3d coordVec3d = coord.asVec3d();

                for (; iter; ++iter) {
                    const Index index = *iter;

                    // get the position in index space. Do this in double precision to make sure
                    // we doesn't move points which are close to the -0.5/0.5 boundary into other
                    // voxels when doing a staggered offset by 0.5

                    Vec3d position = Vec3d(positionHandle->get(index)) + coordVec3d;

                    // transform it to source index space if transforms differ

                    if (Transform) {
                        Vec3d worldPosition = mPointDataGridTransform->indexToWorld(position);
                        position = mSourceGridTransform->worldToIndex(worldPosition);
                    }

                    // Sample the grid at samplePosition
                    typename AccessorT::ValueType sample =
                        SamplerT::sample(mSourceAccessor, position);

                    targetHandle->set(index, static_cast<TargetValueT>(sample));
                }
            }
        }
    }

protected:
    const AccessorT mSourceAccessor;
    const size_t mPositionAttributeIndex;
    const size_t mTargetAttributeIndex;
    const PointFilterT& mFilter;
    const math::Transform* const mPointDataGridTransform;
    const math::Transform* const mSourceGridTransform;
};

template <typename SourceGridT,
          typename PointDataGridT,
          typename PointFilterT = points::NullFilter,
          size_t Order = 1>
class PointDataSampler
{
public:
    using SourceGridValueT = typename SourceGridT::ValueType;
    using SourceGridTraits = VecTraits<SourceGridValueT>;

    PointDataSampler(PointDataGridT& points,
                     const SourceGridT& grid,
                     const PointFilterT& filter = PointFilterT())
        : mPoints(points)
        , mSourceGrid(grid)
        , mFilter(filter) {}

    /// @brief  Sample from the contained source VDB onto an attribute at position
    ///         targetIndex. Throws if the attribute and source VDB are incompatible
    ///         types.
    inline void
    sample(const size_t targetIndex) const
    {
        const auto leafIter = mPoints.constTree().cbeginLeaf();
        if (!leafIter) return;

        const AttributeSet::Descriptor& descriptor =
            leafIter->attributeSet().descriptor();
        const Name& targetType = descriptor.type(targetIndex).first;
        sample<SourceGridTraits::IsVec>(targetType, targetIndex);
    }

    /// @brief  Sample from the contained source VDB onto an attribute at position
    ///         targetIndex of type TargetValueT.
    template <typename TargetValueT>
    inline void
    sample(const size_t targetIndex) const
    {
        using PointDataTreeT = typename PointDataGridT::TreeType;
        using AccessorT = typename SourceGridT::ConstAccessor;

        const auto leafIter = mPoints.constTree().cbeginLeaf();
        if (!leafIter) return;

        const size_t positionIndex = leafIter->attributeSet().find("P");
        if (positionIndex == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(RuntimeError, "Failed to find position attribute");
        }

        const math::Transform& pointTransform = mPoints.constTransform();
        const math::Transform& sourceTransform = mSourceGrid.constTransform();
        const AccessorT sourceGridAccessor = mSourceGrid.getConstAccessor();

        const bool staggered(SourceGridTraits::IsVec &&
            mSourceGrid.getGridClass() == GRID_STAGGERED);

        tree::LeafManager<PointDataTree> leafManager(mPoints.tree());

        if (staggered) {

            // we can only use a staggered sampler if the source grid is a vector grid
            using SamplerT = tools::Sampler<Order, SourceGridTraits::IsVec>;

            if (pointTransform != sourceTransform) {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, Vec3f, PointFilterT,
                    UnknownCodec, PointDataTreeT, true>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                       &pointTransform, &sourceTransform);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
            else {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, Vec3f, PointFilterT,
                    UnknownCodec, PointDataTreeT>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
        }
        else {
            using SamplerT = tools::Sampler<Order, false>;

            if (pointTransform != sourceTransform) {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, Vec3f, PointFilterT,
                    UnknownCodec, PointDataTreeT, true>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                        &pointTransform, &sourceTransform);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
            else {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, Vec3f, PointFilterT,
                    UnknownCodec, PointDataTreeT>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
        }
    }

private:

    template<bool IsVectorGrid>
    typename std::enable_if<IsVectorGrid, void>::type
    sample(const std::string& type,
           const size_t targetIndex) const
    {
        if (type == typeNameAsString<Vec3f>())      sample<Vec3f>(targetIndex);
        else if (type == typeNameAsString<Vec3d>()) sample<Vec3d>(targetIndex);
        else if (type == typeNameAsString<Vec3i>()) sample<Vec3i>(targetIndex);
        else {
            const std::string gridType = typeNameAsString<SourceGridValueT>();
            OPENVDB_THROW(TypeError,
                "Unsupported point sampling from a Source VDB of type \""
                + gridType + "\" to a point attribute of type \"" + type + "\".");
        }
    }

    template<bool IsVectorGrid>
    typename std::enable_if<!IsVectorGrid, void>::type
    sample(const std::string& type,
           const size_t targetIndex) const
    {
        if (type == typeNameAsString<int16_t>())      sample<int16_t>(targetIndex);
        else if (type == typeNameAsString<int32_t>()) sample<int32_t>(targetIndex);
        else if (type == typeNameAsString<int64_t>()) sample<int64_t>(targetIndex);
        else if (type == typeNameAsString<float>())   sample<float>(targetIndex);
        else if (type == typeNameAsString<double>())  sample<double>(targetIndex);
        else if (type == typeNameAsString<bool>())    sample<bool>(targetIndex);
        else {
            const std::string gridType = typeNameAsString<SourceGridValueT>();
            OPENVDB_THROW(TypeError,
                "Unsupported point sampling from a Source VDB of type \""
                + gridType + "\" to a point attribute of type \"" + type + "\".");
        }
    }

private:

    PointDataGridT& mPoints;
    const SourceGridT& mSourceGrid;
    const PointFilterT mFilter;
};

} // namespace point_sample_internal


template <typename SourceGridT, typename PointDataGridT, size_t Order = 1>
inline void sampleGrid(PointDataGridT& points,
                       const SourceGridT& sourceGrid,
                       const Name& targetAttribute,
                       const std::vector<Name>& includeGroups,
                       const std::vector<Name>& excludeGroups)
{
    // we do not allow sampling onto the "P" attribute
    if (targetAttribute == "P") {
        OPENVDB_THROW(RuntimeError, "Cannot sample onto the \"P\" attribute");
    }

    const auto leafIter = points.tree().cbeginLeaf();
    if (!leafIter) return;

    const AttributeSet::Descriptor& descriptor =
        leafIter->attributeSet().descriptor();
    size_t targetIndex = descriptor.find(targetAttribute);

    if (targetIndex == AttributeSet::INVALID_POS) {
        // if the attribute is missing, append one based on the source grid's value type

        appendAttribute<typename SourceGridT::ValueType>(points.tree(), targetAttribute);
        targetIndex = leafIter->attributeSet().descriptor().find(targetAttribute);
        assert(targetIndex != AttributeSet::INVALID_POS);
    }

    const bool filterByGroups = !includeGroups.empty() || !excludeGroups.empty();

    if (filterByGroups) {
        const points::MultiGroupFilter multiGroupfilter(includeGroups, excludeGroups,
            leafIter->attributeSet());
        point_sample_internal::PointDataSampler
            <SourceGridT, PointDataGridT, points::MultiGroupFilter, Order>
                pointDataSampler(points, sourceGrid, multiGroupfilter);

        pointDataSampler.sample(targetIndex);
    }
    else {
        point_sample_internal::PointDataSampler
            <SourceGridT, PointDataGridT, points::NullFilter, Order>
                pointDataSampler(points, sourceGrid);

        pointDataSampler.sample(targetIndex);
    }
}

////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
