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


/// @brief Element flags, used for reference based meshing.
enum { ELEMFLAG_EXTERIOR = 0x1, ELEMFLAG_FRACTURE_SEAM = 0x2,  ELEMFLAG_SUBDIVIDED = 0x4 };


/// @brief Collection of hexs and tets
class ElementPool
{
public:

    inline ElementPool();
    inline ElementPool(const size_t numHexs, const size_t numTets);

    inline void copy(const ElementPool& rhs);

    inline void resetHexs(size_t size);
    inline void clearHexs();

    inline void resetTets(size_t size);
    inline void clearTets();


    // element accessor methods

    const size_t& numHexs() const                      { return mNumHexs; }

    openvdb::Vec8I& hex(size_t n)                      { return mHexs[n]; }
    const openvdb::Vec8I& hex(size_t n) const          { return mHexs[n]; }


    const size_t& numTets() const                  { return mNumTets; }

    openvdb::Vec4I& tet(size_t n)                  { return mTets[n]; }
    const openvdb::Vec4I& tet(size_t n) const      { return mTets[n]; }


    // element flags accessor methods

    char& hexFlags(size_t n)                           { return mHexFlags[n]; }
    const char& hexFlags(size_t n) const               { return mHexFlags[n]; }

    char& tetFlags(size_t n)                       { return mTetFlags[n]; }
    const char& tetFlags(size_t n) const           { return mTetFlags[n]; }


    // reduce the element containers, n has to
    // be smaller than the current container size.

    inline bool trimHexs(const size_t n, bool reallocate = false);
    inline bool trimTrinagles(const size_t n, bool reallocate = false);

private:
    // disallow copy by assignment
    void operator=(const ElementPool&) {}

    size_t mNumHexs, mNumTets;
    std::unique_ptr<openvdb::Vec8I[]> mHexs;
    std::unique_ptr<openvdb::Vec4I[]> mTets;
    std::unique_ptr<char[]> mHexFlags, mTetFlags;
};


/// @{
/// @brief Point and primitive list types.
using PointList = std::unique_ptr<openvdb::Vec3s[]>;
using ElementPoolList = std::unique_ptr<ElementPool[]>;
/// @}


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
