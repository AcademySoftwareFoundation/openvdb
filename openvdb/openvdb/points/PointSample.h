// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis, Francisco Gochez, Dan Bailey
///
/// @file points/PointSample.h
///
/// @brief Sample a VDB Grid onto a VDB Points attribute

#ifndef OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED

#include <openvdb/util/NullInterrupter.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/tools/Interpolation.h>

#include "PointDataGrid.h"
#include "PointAttribute.h"

#include <sstream>
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Performs closest point sampling from a VDB grid onto a VDB Points attribute
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param filter           an optional index filter
/// @param interrupter      an optional interrupter
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
template<typename PointDataGridT, typename SourceGridT,
    typename FilterT = NullFilter, typename InterrupterT = util::NullInterrupter>
inline void pointSample(PointDataGridT& points,
                        const SourceGridT& sourceGrid,
                        const Name& targetAttribute = "",
                        const FilterT& filter = NullFilter(),
                        InterrupterT* const interrupter = nullptr);

/// @brief Performs tri-linear sampling from a VDB grid onto a VDB Points attribute
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param filter           an optional index filter
/// @param interrupter      an optional interrupter
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
template<typename PointDataGridT, typename SourceGridT,
    typename FilterT = NullFilter, typename InterrupterT = util::NullInterrupter>
inline void boxSample(  PointDataGridT& points,
                        const SourceGridT& sourceGrid,
                        const Name& targetAttribute = "",
                        const FilterT& filter = NullFilter(),
                        InterrupterT* const interrupter = nullptr);

/// @brief Performs tri-quadratic sampling from a VDB grid onto a VDB Points attribute
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param filter           an optional index filter
/// @param interrupter      an optional interrupter
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
template<typename PointDataGridT, typename SourceGridT,
    typename FilterT = NullFilter, typename InterrupterT = util::NullInterrupter>
inline void quadraticSample(PointDataGridT& points,
                            const SourceGridT& sourceGrid,
                            const Name& targetAttribute = "",
                            const FilterT& filter = NullFilter(),
                            InterrupterT* const interrupter = nullptr);


// This struct samples the source grid accessor using the world-space position supplied,
// with SamplerT providing the sampling scheme. In the case where ValueT does not match
// the value type of the source grid, the sample() method will also convert the sampled
// value into a ValueT value, using round-to-nearest for float-to-integer conversion.
struct SampleWithRounding
{
    template<typename ValueT, typename SamplerT, typename AccessorT>
    inline ValueT sample(const AccessorT& accessor, const Vec3d& position) const;
};

// A dummy struct that is used to mean that the sampled attribute should either match the type
// of the existing attribute or the type of the source grid (if the attribute doesn't exist yet)
struct DummySampleType { };

/// @brief Performs sampling and conversion from a VDB grid onto a VDB Points attribute
/// @param order            the sampling order - 0 = closest-point, 1 = trilinear, 2 = triquadratic
/// @param points           the PointDataGrid whose points will be sampled on to
/// @param sourceGrid       VDB grid which will be sampled
/// @param targetAttribute  a target attribute on the points which will hold samples. This
///                         attribute will be created with the source grid type if it does
///                         not exist, and with the source grid name if the name is empty
/// @param filter           an optional index filter
/// @param sampler          handles sampling and conversion into the target attribute type,
///                         which by default this uses the SampleWithRounding struct.
/// @param interrupter      an optional interrupter
/// @param threaded         enable or disable threading  (threading is enabled by default)
/// @note  The target attribute may exist provided it can be cast to the SourceGridT ValueType
template<typename PointDataGridT, typename SourceGridT, typename TargetValueT = DummySampleType,
    typename SamplerT = SampleWithRounding, typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
inline void sampleGrid( size_t order,
                        PointDataGridT& points,
                        const SourceGridT& sourceGrid,
                        const Name& targetAttribute,
                        const FilterT& filter = NullFilter(),
                        const SamplerT& sampler = SampleWithRounding(),
                        InterrupterT* const interrupter = nullptr,
                        const bool threaded = true);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointSampleImpl.h"

#endif // OPENVDB_POINTS_POINT_SAMPLE_HAS_BEEN_INCLUDED
