// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointStatistics.h
///
/// @brief Functions to perform multi threaded reductions and analysis of
///   arbitrary point attribute types. Each function imposes various
///   requirements on the point ValueType (such as expected operators) and
///   supports arbitrary point filters.
///

#ifndef OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/tree/LeafManager.h>

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Evaluates the minimum and maximum values of a point attribute.
/// @details Performs parallel reduction by comparing values using their less
///   than and greater than operators. If the PointDataGrid is empty or the
///   filter evalutes to empty, zeroVal<ValueT>() is returned for both values.
/// @note The ValueT of the attribute must be copy constructible. This method
///   will throw if the templated ValueT does not match the given attribute.
///   For vectors and matrices, this results in per component comparisons.
///   See evalExtents for magnitudes or more custom control.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return min,max value pair
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
std::pair<ValueT, ValueT>
evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the average value of a point attribute.
/// @details Performs parallel reduction by cumulative moving average. The
///   reduction arithmetic and return value precision evaluates to:
///      ConvertElementType<ValueT, double>::Type
///   which, for POD and VDB math types, is ValueT at double precision. If the
///   PointDataGrid is empty or the filter evalutes to empty, zeroVal<ValueT>()
///   is returned.
/// @note The ConvertElementType of the attribute must be copy constructible,
///   support the same type + - * operators and * / operators from a double.
///   This method will throw if ValueT does not match the given attribute. The
///   function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return the average value
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
typename ConvertElementType<ValueT, double>::Type
evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the total value of a point attribute.
/// @details Performs parallel reduction by summing all values. The reduction
///   arithmetic and return value precision evaluates to:
///      PromoteType<ValueT>::Highest
///   which, for POD and VDB math types, is ValueT at its highest bit precision.
///   If the PointDataGrid is empty or the filter evalutes to empty,
///   zeroVal<ValueT>() is returned.
/// @note The PromoteType of the attribute must be copy constructible, support
///   the same type + operator. This method will throw if ValueT does not match
///   the given attribute. The function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return the total value
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
typename PromoteType<ValueT>::Highest
accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the minimum and maximum values of a point attribute and
///   returns whether the values are valid. Optionally constructs localised
///   min and max value trees.
/// @details Performs parallel reduction by comparing values using their less
///   than and greater than operators. This method will return true if min and
///   max have been set, false otherwise (when no points existed or a filter
///   evaluated to empty).
/// @note The ValueT of the attribute must also be copy constructible. This
///   method will throw if the templated ValueT does not match the given
///   attribute. For vectors and matrices, this results in per component
///   comparisons. See evalExtents for magnitudes or more custom control.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param min        the computed min value
/// @param max        the computed max value
/// @param filter     a filter to apply to points
/// @param minTree    if provided, builds a tiled tree of localised min results
/// @param maxTree    if provided, builds a tiled tree of localised max results
/// @return true if min and max have been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
bool evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    ValueT& min,
    ValueT& max,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* minTree = nullptr,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* maxTree = nullptr);

/// @brief Evaluates the average value of a point attribute and returns whether
///   the value is valid. Optionally constructs localised average value trees.
/// @details Performs parallel reduction by cumulative moving average. The
///   reduction arithmetic and return value precision evaluates to:
///      ConvertElementType<ValueT, double>::Type
///   which, for POD and VDB math types, is ValueT at double precision. This
///   method will return true average has been set, false otherwise (when no
///   points existed or a filter evaluated to empty).
/// @note The ConvertElementType of the attribute must be copy constructible,
///   support the same type + - * operators and * / operators from a double.
///   This method will throw if ValueT does not match the given attribute. The
///   function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param average    the computed averaged value at double precision
/// @param filter     a filter to apply to points
/// @param averageTree  if provided, builds a tiled tree of localised avg results.
/// @return true if average has been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
/// @par Example:
/// @code
///    using namespace openvdb;
///    using namespace openvdb::points
///
///    // average and store per leaf values in a new tree
///    ConvertElementType<uint8_t, double>::Type avg;  // evaluates to double
///    PointDataTree::ValueConverter<decltype(avg)>::Type avgTree; // double tree of averages
///    bool success = evalAverage<uint8_t>(tree, "attrib", avg, NullFilter(), &avgTree);
/// @endcode
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT,
    typename ResultTreeT = typename ConvertElementType<ValueT, double>::Type>
bool evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    typename ConvertElementType<ValueT, double>::Type& average,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* averageTree = nullptr);

/// @brief Evaluates the total value of a point attribute and returns whether
///   the value is valid. Optionally constructs localised total value trees.
/// @details Performs parallel reduction by summing all values. The reduction
///   arithmetic and return value precision evaluates to:
///      PromoteType<ValueT>::Highest
///   which, for POD and VDB math types, is ValueT at its highest bit precision.
///   This method will return true total has been set, false otherwise (when no
///   points existed or a filter evaluated to empty).
/// @note The PromoteType of the attribute must be copy constructible, support
///   the same type + operator. This method will throw if ValueT does not match
///   the given attribute. The function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param total      the computed total value
/// @param filter     a filter to apply to points
/// @param totalTree  if provided, builds a tiled tree of localised total results.
/// @return true if total has been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
/// @par Example:
/// @code
///    using namespace openvdb;
///    using namespace openvdb::points;
///
///    // accumulate and store per leaf values in a new tree
///    PromoteType<uint8_t>::Highest total;  // evaluates to uint64_t
///    PointDataTree::ValueConverter<decltype(total)>::Type totalTree; // uint64_t tree of totals
///    bool success = accumulate<uint8_t>(tree, "attrib", total, NullFilter(), &totalTree);
/// @endcode
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT,
    typename ResultTreeT = typename PromoteType<ValueT>::Highest>
bool accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    typename PromoteType<ValueT>::Highest& total,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* totalTree = nullptr);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointStatisticsImpl.h"

#endif // OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED
