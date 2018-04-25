///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
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

#include <openvdb/util/NullInterrupter.h>

#include "PointDataGrid.h"
#include "PointGroup.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Performs closest point sampling from an OpenVDB grid onto VDB Points
///
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
///
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param includeGroups    names of groups to include
/// @param excludeGroups    names of groups to exclude
/// @param interrupter      an optional interrupter
template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT = util::NullInterrupter>
inline void
pointSample(PointDataGridT& points,
            const SourceGridT& sourceGrid,
            const Name& targetAttribute = "",
            const std::vector<Name>& includeGroups = std::vector<Name>(),
            const std::vector<Name>& excludeGroups = std::vector<Name>(),
            InterrupterT* const interrupter = nullptr);

/// @brief Performs tri-linear sampling from an OpenVDB grid onto VDB Points
///
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
///
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param includeGroups    names of groups to include
/// @param excludeGroups    names of groups to exclude
/// @param interrupter      an optional interrupter
template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT = util::NullInterrupter>
inline void
boxSample(PointDataGridT& points,
          const SourceGridT& sourceGrid,
          const Name& targetAttribute = "",
          const std::vector<Name>& includeGroups = std::vector<Name>(),
          const std::vector<Name>& excludeGroups = std::vector<Name>(),
          InterrupterT* const interrupter = nullptr);

/// @brief Performs tri-quadratic samples values from an OpenVDB grid onto VDB Points
///
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
///
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param includeGroups    names of groups to include
/// @param excludeGroups    names of groups to exclude
/// @param interrupter      an optional interrupter
template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT = util::NullInterrupter>
inline void
quadraticSample(PointDataGridT& points,
                const SourceGridT& sourceGrid,
                const Name& targetAttribute = "",
                const std::vector<Name>& includeGroups = std::vector<Name>(),
                const std::vector<Name>& excludeGroups = std::vector<Name>(),
                InterrupterT* const interrupter = nullptr);


///////////////////////////////////////////////////


namespace point_sample_internal {

template<typename SamplerT,
         typename AccessorT,
         typename TargetValueT,
         typename PointFilterT = points::NullFilter,
         typename PointDataTreeT = points::PointDataTree,
         typename InterrupterT = util::NullInterrupter,
         bool Transform = false>
class PointSampleOp
{
public:
    using LeafManagerT = tree::LeafManager<PointDataTreeT>;
    using PositionHandleT = AttributeHandle<Vec3f>;
    using TargetHandleT = AttributeWriteHandle<TargetValueT>;

    PointSampleOp(const AccessorT& sourceAccessor,
                  const size_t positionAttributeIndex,
                  const size_t targetAttributeIndex,
                  const PointFilterT& filter = PointFilterT(),
                  const math::Transform* const pointDataGridTransform = nullptr,
                  const math::Transform* const sourceGridTransform = nullptr,
                  InterrupterT* const interrupter = nullptr)
    : mSourceAccessor(sourceAccessor)
    , mPositionAttributeIndex(positionAttributeIndex)
    , mTargetAttributeIndex(targetAttributeIndex)
    , mFilter(filter)
    , mPointDataGridTransform(pointDataGridTransform)
    , mSourceGridTransform(sourceGridTransform)
    , mInterrupter(interrupter)
    {
        if (Transform) {
            assert(mSourceGridTransform);
            assert(mPointDataGridTransform);
        }
    }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        if (util::wasInterrupted(mInterrupter)) {
            tbb::task::self().cancel_group_execution();
            return;
        }

        for (auto leaf = range.begin(); leaf; ++leaf) {
            PositionHandleT::Ptr positionHandle =
                PositionHandleT::create(leaf->constAttributeArray(mPositionAttributeIndex));
            typename TargetHandleT::Ptr targetHandle =
                TargetHandleT::create(leaf->attributeArray(mTargetAttributeIndex));

            for (auto iter = leaf->beginIndexOn(mFilter); iter; ++iter) {

                const Index index = *iter;
                const Vec3d coord = iter.getCoord().asVec3d();

                // get the position in index space. Do this in double precision to make sure
                // we doesn't move points which are close to the -0.5/0.5 boundary into other
                // voxels when doing a staggered offset by 0.5

                Vec3d position = Vec3d(positionHandle->get(index)) + coord;

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

protected:
    const AccessorT mSourceAccessor;
    const size_t mPositionAttributeIndex;
    const size_t mTargetAttributeIndex;
    const PointFilterT& mFilter;
    const math::Transform* const mPointDataGridTransform;
    const math::Transform* const mSourceGridTransform;
    InterrupterT* mInterrupter;
};

template <typename SourceGridT,
          typename PointDataGridT,
          typename PointFilterT = points::NullFilter,
          typename InterrupterT = util::NullInterrupter,
          size_t Order = 1>
class PointDataSampler
{
public:
    using SourceGridValueT = typename SourceGridT::ValueType;
    using SourceGridTraits = VecTraits<SourceGridValueT>;

    PointDataSampler(PointDataGridT& points,
                     const SourceGridT& grid,
                     const PointFilterT& filter = PointFilterT(),
                     InterrupterT* const interrupter = nullptr)
        : mPoints(points)
        , mSourceGrid(grid)
        , mFilter(filter)
        , mInterrupter(interrupter) {}

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
                PointSampleOp<SamplerT, AccessorT, TargetValueT, PointFilterT,
                    PointDataTreeT, InterrupterT, true>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                       &pointTransform, &sourceTransform, mInterrupter);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
            else {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, PointFilterT,
                    PointDataTreeT, InterrupterT>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                       nullptr, nullptr, mInterrupter);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
        }
        else {
            using SamplerT = tools::Sampler<Order, false>;

            if (pointTransform != sourceTransform) {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, PointFilterT,
                    PointDataTreeT, InterrupterT, true>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                        &pointTransform, &sourceTransform, mInterrupter);
                tbb::parallel_for(leafManager.leafRange(), op);
            }
            else {
                PointSampleOp<SamplerT, AccessorT, TargetValueT, PointFilterT,
                    PointDataTreeT, InterrupterT>
                    op(sourceGridAccessor, positionIndex, targetIndex, mFilter,
                       nullptr, nullptr, mInterrupter);
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
    InterrupterT* mInterrupter;
};

template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT,
    size_t Order = 1>
inline void
sampleGrid(PointDataGridT& points,
           const SourceGridT& sourceGrid,
           const Name& targetAttribute,
           const std::vector<Name>& includeGroups,
           const std::vector<Name>& excludeGroups,
           InterrupterT* const interrupter)
{
    Name attribute(targetAttribute);
    if (targetAttribute.empty()) {
        attribute = sourceGrid.getName();
    }

    // we do not allow sampling onto the "P" attribute
    if (attribute == "P") {
        OPENVDB_THROW(RuntimeError, "Cannot sample onto the \"P\" attribute");
    }

    const auto leafIter = points.tree().cbeginLeaf();
    if (!leafIter) return;

    const AttributeSet::Descriptor& descriptor =
        leafIter->attributeSet().descriptor();
    size_t targetIndex = descriptor.find(attribute);

    if (targetIndex == AttributeSet::INVALID_POS) {
        // if the attribute is missing, append one based on the source grid's value type

        appendAttribute<typename SourceGridT::ValueType>(points.tree(), attribute);
        targetIndex = leafIter->attributeSet().descriptor().find(attribute);
        assert(targetIndex != AttributeSet::INVALID_POS);
    }

    const bool filterByGroups = !includeGroups.empty() || !excludeGroups.empty();

    if (filterByGroups) {
        const points::MultiGroupFilter multiGroupfilter(includeGroups, excludeGroups,
            leafIter->attributeSet());
        point_sample_internal::PointDataSampler
            <SourceGridT, PointDataGridT, points::MultiGroupFilter, InterrupterT, Order>
                pointDataSampler(points, sourceGrid, multiGroupfilter);

        pointDataSampler.sample(targetIndex);
    }
    else {
        point_sample_internal::PointDataSampler
            <SourceGridT, PointDataGridT, points::NullFilter, InterrupterT, Order>
                pointDataSampler(points, sourceGrid);

        pointDataSampler.sample(targetIndex);
    }
}


} // namespace point_sample_internal


template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT>
inline void
pointSample(PointDataGridT& points,
            const SourceGridT& sourceGrid,
            const Name& targetAttribute,
            const std::vector<Name>& includeGroups,
            const std::vector<Name>& excludeGroups,
            InterrupterT* const interrupter)
{
    using namespace point_sample_internal;
    sampleGrid<SourceGridT, PointDataGridT, InterrupterT, 0>
        (points, sourceGrid, targetAttribute, includeGroups, excludeGroups, interrupter);
}

template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT>
inline void
boxSample(PointDataGridT& points,
          const SourceGridT& sourceGrid,
          const Name& targetAttribute,
          const std::vector<Name>& includeGroups,
          const std::vector<Name>& excludeGroups,
          InterrupterT* const interrupter)
{
    using namespace point_sample_internal;
    sampleGrid<SourceGridT, PointDataGridT, InterrupterT, 1>
        (points, sourceGrid, targetAttribute, includeGroups, excludeGroups, interrupter);
}

template<
    typename SourceGridT,
    typename PointDataGridT,
    typename InterrupterT>
inline void
quadraticSample(PointDataGridT& points,
                const SourceGridT& sourceGrid,
                const Name& targetAttribute,
                const std::vector<Name>& includeGroups,
                const std::vector<Name>& excludeGroups,
                InterrupterT* const interrupter)
{
    using namespace point_sample_internal;
    sampleGrid<SourceGridT, PointDataGridT, InterrupterT, 2>
        (points, sourceGrid, targetAttribute, includeGroups, excludeGroups, interrupter);
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
