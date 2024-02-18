// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file   VolumeToElements.h
///
/// @brief  Extract tetrahedral and hexahedral finite elements from scalar volumes.
///
/// @author Megidd Git

#ifndef OPENVDB_TOOLS_VOLUME_TO_ELEMENTS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_ELEMENTS_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/Assert.h>
#include <openvdb/openvdb.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


// Wrapper functions for the VolumeToElements converter


/// @brief Uniformly process any scalar grid that has a continuous isosurface.
///
/// @param grid     a scalar grid to process
/// @param points   output list of world space points
/// @param tets    output tet index list
/// @param hexs    output hex index list
/// @param isovalue determines which isosurface to process
///
/// @throw TypeError if @a grid does not have a scalar value type
template<typename GridType>
void
volumeToElements(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& tets,
    std::vector<Vec8I>& hexs,
    double isovalue = 0.0);


////////////////////////////////////////


/// @brief Process any scalar grid that has a continuous isosurface.
struct VolumeToElements
{

    /// @param isovalue                   Determines which isosurface to process.
    VolumeToElements(double isovalue = 0);

    //////////

    /// @brief Main call
    /// @note Call with scalar typed grid.
    template<typename InputGridType>
    void operator()(const InputGridType&);


private:
    // Disallow copying
    VolumeToElements(const VolumeToElements&);
    VolumeToElements& operator=(const VolumeToElements&);

    PointList mPoints;
    ElementPoolList mElements;

    double mIsovalue;

}; // struct VolumeToElements


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// Internal utility objects and implementation details

/// @cond OPENVDB_DOCS_INTERNAL

namespace volume_to_elements_internal {


template<typename GridType>
void
doVolumeToElements(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& tets,
    std::vector<Vec8I>& hexs,
    double isovalue)
{
    static_assert(std::is_scalar<typename GridType::ValueType>::value,
        "volume to elements conversion is supported only for scalar grids");

    VolumeToElements elementer(isovalue);
    elementer(grid);

    // TODO: copy points, tets, and hexs.
}


} // volume_to_elements_internal namespace

/// @endcond

////////////////////////////////////////


inline
VolumeToElements::VolumeToElements(double isovalue)
    : mPoints(nullptr)
    , mElements()
    , mIsovalue(isovalue)
{
}


template<typename InputGridType>
inline void
VolumeToElements::operator()(const InputGridType& inputGrid)
{
}


////////////////////////////////////////


template<typename GridType>
void volumeToElements(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& tets,
    std::vector<Vec8I>& hexs,
    double isovalue)
{
    volume_to_elements_internal::doVolumeToElements(grid, points, tets, hexs, isovalue);
}

template<typename GridType>
void volumeToElements(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec8I>& hexs,
    double isovalue)
{
    std::vector<Vec4I> tets;
    volumeToElements(grid, points, tets, hexs, isovalue);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_VOLUMETOELEMENTS
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    void volumeToElements(const Grid<TreeT>&, std::vector<Vec3s>&, std::vector<Vec8I>&, double)
OPENVDB_NUMERIC_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void volumeToElements(const Grid<TreeT>&, std::vector<Vec3s>&, std::vector<Vec4I>&, std::vector<Vec8I>&, double)
OPENVDB_NUMERIC_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_ELEMENTS_HAS_BEEN_INCLUDED
