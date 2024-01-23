// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey, Nick Avramoussis
///
/// @file points/PointConversion.h
///
/// @brief  Convert points and attributes to and from VDB Point Data grids.

#ifndef OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED

#include <openvdb/math/Transform.h>

#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/tools/PointsToMask.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Assert.h>

#include "AttributeArrayString.h"
#include "AttributeSet.h"
#include "IndexFilter.h"
#include "PointAttribute.h"
#include "PointDataGrid.h"
#include "PointGroup.h"

#include <tbb/parallel_reduce.h>

#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

////////////////////////////////////////


/// @brief Point-partitioner compatible STL vector attribute wrapper for convenience
template<typename ValueType>
class PointAttributeVector {
public:
    using PosType = ValueType;
    using value_type= ValueType;

    PointAttributeVector(const std::vector<value_type>& data,
                         const Index stride = 1)
        : mData(data)
        , mStride(stride) { }

    size_t size() const { return mData.size(); }
    void getPos(size_t n, ValueType& xyz) const { xyz = mData[n]; }
    void get(ValueType& value, size_t n) const { value = mData[n]; }
    void get(ValueType& value, size_t n, openvdb::Index m) const { value = mData[n * mStride + m]; }

private:
    const std::vector<value_type>& mData;
    const Index mStride;
}; // PointAttributeVector


////////////////////////////////////////

/// @brief  Localises points with position into a @c PointDataGrid into two stages:
///         allocation of the leaf attribute data and population of the positions.
///
/// @param  pointIndexGrid  a PointIndexGrid into the points.
/// @param  positions       list of world space point positions.
/// @param  xform           world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   The position data must be supplied in a Point-Partitioner compatible
///         data structure. A convenience PointAttributeVector class is offered.
///
/// @note   The position data is populated separately to perform world space to
///         voxel space conversion and apply quantisation.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. Typically this is built implicitly by the PointDataGrid constructor.

template<
    typename CompressionT,
    typename PointDataGridT,
    typename PositionArrayT,
    typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid,
                    const PositionArrayT& positions,
                    const math::Transform& xform,
                    const Metadata* positionDefaultValue = nullptr);


/// @brief  Convenience method to create a @c PointDataGrid from a std::vector of
///         point positions.
///
/// @param  positions     list of world space point positions.
/// @param  xform         world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   This method implicitly wraps the std::vector for a Point-Partitioner compatible
///         data structure and creates the required @c PointIndexGrid to the points.

template <typename CompressionT, typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const math::Transform& xform,
                    const Metadata* positionDefaultValue = nullptr);


/// @brief  Stores point attribute data in an existing @c PointDataGrid attribute.
///
/// @param  tree            the PointDataGrid to be populated.
/// @param  pointIndexTree  a PointIndexTree into the points.
/// @param  attributeName   the name of the VDB Points attribute to be populated.
/// @param  data            a wrapper to the attribute data.
/// @param  stride          the stride of the attribute
/// @param  insertMetadata  true if strings are to be automatically inserted as metadata.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. This is required to ensure the same point index ordering.
template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT>
inline void
populateAttribute(  PointDataTreeT& tree,
                    const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName,
                    const PointArrayT& data,
                    const Index stride = 1,
                    const bool insertMetadata = true);

/// @brief Convert the position attribute from a Point Data Grid
///
/// @param positionAttribute    the position attribute to be populated.
/// @param grid                 the PointDataGrid to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename PositionAttribute, typename PointDataGridT, typename FilterT = NullFilter>
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const FilterT& filter = NullFilter(),
                                const bool inCoreOnly = false);


/// @brief Convert the attribute from a PointDataGrid
///
/// @param attribute            the attribute to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf.
/// @param startOffset          a value to shift all the point offsets by
/// @param arrayIndex           the index in the Descriptor of the array to be converted.
/// @param stride               the stride of the attribute
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
template <typename TypedAttribute, typename PointDataTreeT, typename FilterT = NullFilter>
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride = 1,
                                const FilterT& filter = NullFilter(),
                                const bool inCoreOnly = false);


/// @brief Convert the group from a PointDataGrid
///
/// @param group                the group to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param index                the group index to be converted.
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename Group, typename PointDataTreeT, typename FilterT = NullFilter>
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false);

// for internal use only - this traits class extracts T::value_type if defined,
// otherwise falls back to using Vec3R
namespace internal {
template <typename...> using void_t = void;
template <typename T, typename = void>
struct ValueTypeTraits { using Type = Vec3R; /* default type if T::value_type is not defined*/ };
template <typename T>
struct ValueTypeTraits <T, void_t<typename T::value_type>> { using Type = typename T::value_type; };
} // namespace internal

/// @ brief Given a container of world space positions and a target points per voxel,
/// compute a uniform voxel size that would best represent the storage of the points in a grid.
/// This voxel size is typically used for conversion of the points into a PointDataGrid.
///
/// @param positions        array of world space positions
/// @param pointsPerVoxel   the target number of points per voxel, must be positive and non-zero
/// @param transform        voxel size will be computed using this optional transform if provided
/// @param decimalPlaces    for readability, truncate voxel size to this number of decimals
/// @param interrupter      an optional interrupter
///
/// @note VecT will be PositionWrapper::value_type or Vec3R (if there is no value_type defined)
///
/// @note if none or one point provided in positions, the default voxel size of 0.1 will be returned
///
template<   typename PositionWrapper,
            typename InterrupterT = openvdb::util::NullInterrupter,
            typename VecT = typename internal::ValueTypeTraits<PositionWrapper>::Type>
inline float
computeVoxelSize(  const PositionWrapper& positions,
                   const uint32_t pointsPerVoxel,
                   const math::Mat4d transform = math::Mat4d::identity(),
                   const Index decimalPlaces = 5,
                   InterrupterT* const interrupter = nullptr);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointConversionImpl.h"

#endif // OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED
