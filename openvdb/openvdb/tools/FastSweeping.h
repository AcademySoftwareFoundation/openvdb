// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file    FastSweeping.h
///
/// @author  Ken Museth
///
/// @brief   Defined the six functions {fog,sdf}To{Sdf,Ext,SdfAndExt} in
///          addition to the two functions maskSdf and dilateSdf. Sdf denotes
///          a signed-distance field (i.e. negative values are inside), fog
///          is a scalar fog volume (i.e. higher values are inside), and Ext is
///          a field (of arbitrary type) that is extended off the iso-surface.
///          All these functions are implemented with the methods in the class
///          named FastSweeping.
///
/// @note    Solves the (simplified) Eikonal Eq: @f$|\nabla \phi|^2 = 1@f$ and
///          performs velocity extension,  @f$\nabla f\nabla \phi = 0@f$, both
///          by means of the fast sweeping algorithm detailed in:
///          "A Fast Sweeping Method For Eikonal Equations"
///          by H. Zhao, Mathematics of Computation, Vol 74(230), pp 603-627, 2004
///
/// @details The algorithm used below for parallel fast sweeping was first published in:
///          "New Algorithm for Sparse and Parallel Fast Sweeping: Efficient
///          Computation of Sparse Distance Fields" by K. Museth, ACM SIGGRAPH Talk,
///          2017, http://www.museth.org/Ken/Publications_files/Museth_SIG17.pdf

#ifndef OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED

//#define BENCHMARK_FAST_SWEEPING

#include <openvdb/openvdb.h>
#include <openvdb/Platform.h>
#include <openvdb/math/Math.h> // for Abs() and isExactlyEqual()
#include <openvdb/math/Stencils.h> // for GradStencil
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h> // for PruneMinMaxFltKernel

#include "LevelSetUtil.h"
#include "Morphology.h"

#include "Statistics.h"
#ifdef BENCHMARK_FAST_SWEEPING
#include <openvdb/util/CpuTimer.h>
#endif

#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_group.h>

#include <type_traits>// for static_assert
#include <cmath>
#include <limits>
#include <deque>
#include <unordered_map>
#include <utility>// for std::make_pair

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Fast Sweeping update mode. This is useful to determine
///        narrow-band extension or field extension in one side
///        of a signed distance field.
enum class FastSweepingDomain {
    /// Update all voxels affected by the sweeping algorithm
    SWEEP_ALL,
    // Update voxels corresponding to an sdf/fog values that are greater than a given isovalue
    SWEEP_GREATER_THAN_ISOVALUE,
    // Update voxels corresponding to an sdf/fog values that are less than a given isovalue
    SWEEP_LESS_THAN_ISOVALUE
};

/// @brief Converts a scalar fog volume into a signed distance function. Active input voxels
///        with scalar values above the given isoValue will have NEGATIVE distance
///        values on output, i.e. they are assumed to be INSIDE the iso-surface.
///
/// @return A shared pointer to a signed-distance field defined on the active values
///         of the input fog volume.
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbritary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a fogGrid then the returned grid has all its active values set to
///          plus or minus infinity, depending on if the input values are larger or
///          smaller than @a isoValue.
template<typename GridT>
typename GridT::Ptr
fogToSdf(const GridT &fogGrid,
         typename GridT::ValueType isoValue,
         int nIter = 1);

/// @brief Given an existing approximate SDF it solves the Eikonal equation for all its
///        active voxels. Active input voxels with a signed distance value above the
///        given isoValue will have POSITIVE distance values on output, i.e. they are
///        assumed to be OUTSIDE the iso-surface.
///
/// @return A shared pointer to a signed-distance field defined on the active values
///         of the input sdf volume.
///
/// @param sdfGrid  An approximate signed distance field to the specified iso-surface.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @note The only difference between this method and fogToSdf, defined above, is the
///       convention of the sign of the output distance field.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a sdfGrid then the returned grid has all its active values set to
///          plus or minus infinity, depending on if the input values are larger or
///          smaller than @a isoValue.
template<typename GridT>
typename GridT::Ptr
sdfToSdf(const GridT &sdfGrid,
         typename GridT::ValueType isoValue = 0,
         int nIter = 1);

/// @brief Computes the extension of a field (scalar, vector, or int are supported), defined
///        by the specified functor, off an iso-surface from an input FOG volume.
///
/// @return A shared pointer to the extension field defined from the active values in
///         the input fog volume.
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->ExtValueT that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param background Background value of return grid with the extension field.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @param mode     Determines the mode of updating the extension field. SWEEP_ALL
///                 will update all voxels of the extension field affected by the
///                 fast sweeping algorithm. SWEEP_GREATER_THAN_ISOVALUE will update
///                 all voxels corresponding to fog values that are greater than a given
///                 isovalue. SWEEP_LESS_THAN_ISOVALUE will update all voxels corresponding
///                 to fog values that are less than a given isovalue. If a mode other
///                 than SWEEP_ALL is chosen, a user needs to supply @a extGrid.
///
/// @param extGrid  Optional parameter required to supply a default value for the extension
///                 field when SWEEP_GREATER_THAN_ISOVALUE or SWEEP_LESS_THAN_ISOVALUE
///                 mode is picked for @a mode. When SWEEP_GREATER_THAN_ISOVALUE is supplied
///                 as an argument for @a mode, the extension field voxel will default
///                 to the value of the @a extGrid in that position if it corresponds to a fog
///                 value that is less than the isovalue. Otherwise, the extension
///                 field voxel value will be computed by the Fast Sweeping algorithm.
///                 The opposite convention is implemented when SWEEP_LESS_THAN_ISOVALUE
///                 is supplied as an argument for @a mode.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbritary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a fogGrid then the returned grid has all its active values set to
///          @a background.
template<typename FogGridT, typename ExtOpT, typename ExtValueT>
typename FogGridT::template ValueConverter<ExtValueT>::Type::Ptr
fogToExt(const FogGridT &fogGrid,
         const ExtOpT &op,
         const ExtValueT& background,
         typename FogGridT::ValueType isoValue,
         int nIter = 1,
         FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL,
         const typename FogGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid = nullptr);

/// @brief Computes the extension of a field (scalar, vector, or int are supported), defined
///        by the specified functor, off an iso-surface from an input SDF volume.
///
/// @return A shared pointer to the extension field defined on the active values in the
///         input signed distance field.
///
/// @param sdfGrid  An approximate signed distance field to the specified iso-surface.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->ExtValueT that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param background Background value of return grid with the extension field.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @param mode     Determines the mode of updating the extension field. SWEEP_ALL
///                 will update all voxels of the extension field affected by the
///                 fast sweeping algorithm. SWEEP_GREATER_THAN_ISOVALUE will update
///                 all voxels corresponding to level set values that are greater than a given
///                 isovalue. SWEEP_LESS_THAN_ISOVALUE will update all voxels corresponding
///                 to level set values that are less than a given isovalue. If a mode other
///                 than SWEEP_ALL is chosen, a user needs to supply @a extGrid.
///
/// @param extGrid  Optional parameter required to supply a default value for the extension
///                 field when SWEEP_GREATER_THAN_ISOVALUE or SWEEP_LESS_THAN_ISOVALUE
///                 mode is picked for @a mode. When SWEEP_GREATER_THAN_ISOVALUE is supplied
///                 as an argument for @a mode, the extension field voxel will default
///                 to the value of the @a extGrid in that position if it corresponds to a level-set
///                 value that is less than the isovalue. Otherwise, the extension
///                 field voxel value will be computed by the Fast Sweeping algorithm.
///                 The opposite convention is implemented when SWEEP_LESS_THAN_ISOVALUE
///                 is supplied as an argument for @a mode.
///
/// @note The only difference between this method and fogToExt, defined above, is the
///       convention of the sign of the signed distance field.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a sdfGrid then the returned grid has all its active values set to
///          @a background.
template<typename SdfGridT, typename ExtOpT, typename ExtValueT>
typename SdfGridT::template ValueConverter<ExtValueT>::Type::Ptr
sdfToExt(const SdfGridT &sdfGrid,
         const ExtOpT &op,
         const ExtValueT &background,
         typename SdfGridT::ValueType isoValue = 0,
         int nIter = 1,
         FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL,
         const typename SdfGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid = nullptr);

/// @brief Computes the signed distance field and the extension of a field (scalar, vector, or
///        int are supported), defined by the specified functor, off an iso-surface from an input
///        FOG volume.
///
/// @return An pair of two shared pointers to respectively the SDF and extension field
///
/// @param fogGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->ExtValueT that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param background Background value of return grid with the extension field.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a fogGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @param mode     Determines the mode of updating the extension field. SWEEP_ALL
///                 will update all voxels of the extension field affected by the
///                 fast sweeping algorithm. SWEEP_GREATER_THAN_ISOVALUE will update
///                 all voxels corresponding to fog values that are greater than a given
///                 isovalue. SWEEP_LESS_THAN_ISOVALUE will update all voxels corresponding
///                 to fog values that are less than a given isovalue. If a mode other
///                 than SWEEP_ALL is chosen, a user needs to supply @a extGrid.
///
/// @param extGrid  Optional parameter required to supply a default value for the extension
///                 field when SWEEP_GREATER_THAN_ISOVALUE or SWEEP_LESS_THAN_ISOVALUE
///                 mode is picked for @a mode. When SWEEP_GREATER_THAN_ISOVALUE is supplied
///                 as an argument for @a mode, the extension field voxel will default
///                 to the value of the @a extGrid in that position if it corresponds to a fog
///                 value that is less than the isovalue. Otherwise, the extension
///                 field voxel value will be computed by the Fast Sweeping algorithm.
///                 The opposite convention is implemented when SWEEP_LESS_THAN_ISOVALUE
///                 is supplied as an argument for @a mode.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbritary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grids are identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grids!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a fogGrid then a pair of the following grids is returned: The first
///          is a signed distance grid with its active values set to plus or minus
///          infinity depending of whether its input values are above or below @a isoValue.
///          The second grid, which represents the extension field, has all its active
///          values set to @a background.
template<typename FogGridT, typename ExtOpT, typename ExtValueT>
std::pair<typename FogGridT::Ptr, typename FogGridT::template ValueConverter<ExtValueT>::Type::Ptr>
fogToSdfAndExt(const FogGridT &fogGrid,
               const ExtOpT &op,
               const ExtValueT &background,
               typename FogGridT::ValueType isoValue,
               int nIter = 1,
               FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL,
               const typename FogGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid = nullptr);

/// @brief Computes the signed distance field and the extension of a field (scalar, vector, or
///        int are supported), defined by the specified functor, off an iso-surface from an input
///        SDF volume.
///
/// @return A pair of two shared pointers to respectively the SDF and extension field
///
/// @param sdfGrid  Scalar (floating-point) volume from which an
///                 iso-surface can be defined.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->ExtValueT that
///                 defines the Dirichlet boundary condition, on the iso-surface,
///                 of the field to be extended.
///
/// @param background Background value of return grid with the extension field.
///
/// @param isoValue A value which defines a smooth iso-surface that
///                 intersects active voxels in @a sdfGrid.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @param mode     Determines the mode of updating the extension field. SWEEP_ALL
///                 will update all voxels of the extension field affected by the
///                 fast sweeping algorithm. SWEEP_GREATER_THAN_ISOVALUE will update
///                 all voxels corresponding to level set values that are greater than a given
///                 isovalue. SWEEP_LESS_THAN_ISOVALUE will update all voxels corresponding
///                 to level set values that are less than a given isovalue. If a mode other
///                 than SWEEP_ALL is chosen, a user needs to supply @a extGrid.
///
/// @param extGrid  Optional parameter required to supply a default value for the extension
///                 field when SWEEP_GREATER_THAN_ISOVALUE or SWEEP_LESS_THAN_ISOVALUE
///                 mode is picked for @a mode. When SWEEP_GREATER_THAN_ISOVALUE is supplied
///                 as an argument for @a mode, the extension field voxel will default
///                 to the value of the @a extGrid in that position if it corresponds to a level-set
///                 value that is less than the isovalue. Otherwise, the extension
///                 field voxel value will be computed by the Fast Sweeping algorithm.
///                 The opposite convention is implemented when SWEEP_LESS_THAN_ISOVALUE
///                 is supplied as an argument for @a mode.
///
/// @note Strictly speaking a fog volume is normalized to the range [0,1] but this
///       method accepts a scalar volume with an arbritary range, as long as the it
///       includes the @a isoValue.
///
/// @details Topology of output grids are identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grids!
///
/// @warning If @a isoValue does not intersect any active values in
///          @a sdfGrid then a pair of the following grids is returned: The first
///          is a signed distance grid with its active values set to plus or minus
///          infinity depending of whether its input values are above or below @a isoValue.
///          The second grid, which represents the extension field, has all its active
///          values set to @a background.
template<typename SdfGridT, typename ExtOpT, typename ExtValueT>
std::pair<typename SdfGridT::Ptr, typename SdfGridT::template ValueConverter<ExtValueT>::Type::Ptr>
sdfToSdfAndExt(const SdfGridT &sdfGrid,
               const ExtOpT &op,
               const ExtValueT &background,
               typename SdfGridT::ValueType isoValue = 0,
               int nIter = 1,
               FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL,
               const typename SdfGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid = nullptr);

/// @brief Dilates the narrow band of an existing signed distance field by
///        a specified number of voxels (like adding "onion-rings").
///
/// @note This operation is not to be confused with morphological dilation
///       of a level set, which is implemented in LevelSetFilter::offset,
///       and involves actual interface tracking of the narrow band.
///
/// @return A shared pointer to the dilated signed distance field.
///
/// @param sdfGrid  Input signed distance field to be dilated.
///
/// @param dilation Numer of voxels that the narrow band of the input SDF will be dilated.
///
/// @param nn       Stencil-pattern used for dilation
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @param mode     Determines the direction of the dilation. SWEEP_ALL
///                 will dilate in both sides of the signed distance function,
///                 SWEEP_GREATER_THAN_ISOVALUE will dilate in the positive
///                 side of the iso-surface, SWEEP_LESS_THAN_ISOVALUE will dilate
///                 in the negative side of the iso-surface.
///
/// @details Topology will change as a result of this dilation. E.g. if
///          sdfGrid has a width of 3 and @a dilation = 6 then the grid
///          returned by this method is a narrow band signed distance field
///          with a total width of 9 units.
template<typename GridT>
typename GridT::Ptr
dilateSdf(const GridT &sdfGrid,
          int dilation,
          NearestNeighbors nn = NN_FACE,
          int nIter = 1,
          FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL);

/// @brief Fills mask by extending an existing signed distance field into
///        the active values of this input ree of arbitrary value type.
///
/// @return A shared pointer to the masked signed distance field.
///
/// @param sdfGrid  Input signed distance field to be extended into the mask.
///
/// @param mask     Mask used to identify the topology of the output SDF.
///                 Note this mask is assume to overlap with the sdfGrid.
///
/// @param ignoreActiveTiles If false, active tiles in the mask are treated
///                 as active voxels. Else they are ignored.
///
/// @param nIter    Number of iterations of the fast sweeping algorithm.
///                 Each iteration performs 2^3 = 8 individual sweeps.
///
/// @details Topology of the output SDF is determined by the union of the active
///          voxels (or optionally values) in @a sdfGrid and @a mask.
template<typename GridT, typename MaskTreeT>
typename GridT::Ptr
maskSdf(const GridT &sdfGrid,
        const Grid<MaskTreeT> &mask,
        bool ignoreActiveTiles = false,
        int nIter = 1);

////////////////////////////////////////////////////////////////////////////////
/// @brief Computes signed distance values from an initial iso-surface and
///        optionally performs velocity extension at the same time. This is
///        done by means of a novel sparse and parallel fast sweeping
///        algorithm based on a first order Godunov's scheme.
///
///        Solves: @f$|\nabla \phi|^2 = 1 @f$
///
/// @warning Note, it is important to call one of the initialization methods before
///          called the sweep function. Failure to do so will throw a RuntimeError.
///          Consider instead call one of the many higher-level free-standing functions
///          defined above!
template<typename SdfGridT, typename ExtValueT = typename SdfGridT::ValueType>
class FastSweeping
{
    static_assert(std::is_floating_point<typename SdfGridT::ValueType>::value,
                  "FastSweeping requires SdfGridT to have floating-point values");
    // Defined types related to the signed distance (or fog) grid
    using SdfValueT = typename SdfGridT::ValueType;
    using SdfTreeT = typename SdfGridT::TreeType;
    using SdfAccT  = tree::ValueAccessor<SdfTreeT, false>;//don't register accessors
    using SdfConstAccT  = typename tree::ValueAccessor<const SdfTreeT, false>;//don't register accessors

    // define types related to the extension field
    using ExtGridT = typename SdfGridT::template ValueConverter<ExtValueT>::Type;
    using ExtTreeT = typename ExtGridT::TreeType;
    using ExtAccT  = tree::ValueAccessor<ExtTreeT, false>;

    // define types related to the tree that masks out the active voxels to be solved for
    using SweepMaskTreeT = typename SdfTreeT::template ValueConverter<ValueMask>::Type;
    using SweepMaskAccT = tree::ValueAccessor<SweepMaskTreeT, false>;//don't register accessors

public:

    /// @brief Constructor
    FastSweeping();

     /// @brief Destructor.
    ~FastSweeping() { this->clear(); }

    /// @brief Disallow copy construction.
    FastSweeping(const FastSweeping&) = delete;

    /// @brief Disallow copy assignment.
    FastSweeping& operator=(const FastSweeping&) = delete;

    /// @brief Returns a shared pointer to the signed distance field computed
    ///        by this class.
    ///
    /// @warning This shared pointer might point to NULL if the grid has not been
    ///          initialize (by one of the init methods) or computed (by the sweep
    ///          method).
    typename SdfGridT::Ptr sdfGrid() { return mSdfGrid; }

    /// @brief Returns a shared pointer to the extension field computed
    ///        by this class.
    ///
    /// @warning This shared pointer might point to NULL if the grid has not been
    ///          initialize (by one of the init methods) or computed (by the sweep
    ///          method).
    typename ExtGridT::Ptr extGrid() { return mExtGrid; }

    /// @brief Returns a shared pointer to the extension grid input. This is non-NULL
    ///        if this class is used to extend a field with a non-default sweep direction.
    ///
    /// @warning This shared pointer might point to NULL. This is non-NULL
    ///          if this class is used to extend a field with a non-default sweep direction,
    ///          i.e. SWEEP_LESS_THAN_ISOVALUE or SWEEP_GREATER_THAN_ISOVALUE.
    typename ExtGridT::Ptr extGridInput() { return mExtGridInput; }

    /// @brief Initializer for input grids that are either a signed distance
    ///        field or a scalar fog volume.
    ///
    /// @return True if the initialization succeeded.
    ///
    /// @param sdfGrid Input scalar grid that represents an existing signed distance
    ///                field or a fog volume (signified by @a isInputSdf).
    ///
    /// @param isoValue Iso-value to be used to define the Dirichlet boundary condition
    ///                 of the fast sweeping algorithm (typically 0 for sdfs and a
    ///                 positive value for fog volumes).
    ///
    /// @param isInputSdf Used to determine if @a sdfGrid is a sigend distance field (true)
    ///                   or a scalar fog volume (false).
    ///
    /// @details This, or any of ther other initialization methods, should be called
    ///          before any call to sweep(). Failure to do so will throw a RuntimeError.
    ///
    /// @warning Note, if this method fails, i.e. returns false, a subsequent call
    ///          to sweep will trow a RuntimeError. Instead call clear and try again.
    bool initSdf(const SdfGridT &sdfGrid, SdfValueT isoValue, bool isInputSdf);

    /// @brief Initializer used whenever velocity extension is performed in addition
    ///        to the computation of signed distance fields.
    ///
    /// @return True if the initialization succeeded.
    ///
    ///
    /// @param sdfGrid Input scalar grid that represents an existing signed distance
    ///                field or a fog volume (signified by @a isInputSdf).
    ///
    /// @param op Functor with signature [](const Vec3R &xyz)->ExtValueT that
    ///           defines the Dirichlet boundary condition, on the iso-surface,
    ///           of the field to be extended. Strictly the return type of this functor
    ///           is only required to be convertible to ExtValueT!
    ///
    /// @param background Background value of return grid with the extension field.
    ///
    /// @param isoValue Iso-value to be used for the boundary condition of the fast
    ///                 sweeping algorithm (typically 0 for sdfs and a positive value
    ///                 for fog volumes).
    ///
    /// @param isInputSdf Used to determine if @a sdfGrid is a sigend distance field (true)
    ///                   or a scalar fog volume (false).
    ///
    /// @param mode     Determines the mode of updating the extension field. SWEEP_ALL
    ///                 will update all voxels of the extension field affected by the
    ///                 fast sweeping algorithm. SWEEP_GREATER_THAN_ISOVALUE will update
    ///                 all voxels corresponding to fog values that are greater than a given
    ///                 isovalue. SWEEP_LESS_THAN_ISOVALUE will update all voxels corresponding
    ///                 to fog values that are less than a given isovalue. If a mode other
    ///                 than SWEEP_ALL is chosen, a user needs to supply @a extGrid.
    ///
    /// @param extGrid  Optional parameter required to supply a default value for the extension
    ///                 field when SWEEP_GREATER_THAN_ISOVALUE or SWEEP_LESS_THAN_ISOVALUE
    ///                 mode is picked for @a mode. When SWEEP_GREATER_THAN_ISOVALUE is supplied
    ///                 as an argument for @a mode, the extension field voxel will default
    ///                 to the value of the @a extGrid in that position if it corresponds to a level-set
    ///                 value that is less than the isovalue. Otherwise, the extension
    ///                 field voxel value will be computed by the Fast Sweeping algorithm.
    ///                 The opposite convention is implemented when SWEEP_LESS_THAN_ISOVALUE
    ///                 is supplied as an argument for @a mode.
    ///
    /// @details This, or any of ther other initialization methods, should be called
    ///          before any call to sweep(). Failure to do so will throw a RuntimeError.
    ///
    /// @warning Note, if this method fails, i.e. returns false, a subsequent call
    ///          to sweep will trow a RuntimeError. Instead call clear and try again.
    template <typename ExtOpT>
    bool initExt(const SdfGridT &sdfGrid,
                 const ExtOpT &op,
                 const ExtValueT &background,
                 SdfValueT isoValue,
                 bool isInputSdf,
                 FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL,
                 const typename ExtGridT::ConstPtr extGrid = nullptr);

    /// @brief Initializer used when dilating an existing signed distance field.
    ///
    /// @return True if the initialization succeeded.
    ///
    /// @param sdfGrid  Input signed distance field to to be dilated.
    ///
    /// @param dilation Numer of voxels that the input SDF will be dilated.
    ///
    /// @param nn       Stencil-pattern used for dilation
    ///
    /// @param mode     Determines the direction of the dilation. SWEEP_ALL
    ///                 will dilate in both sides of the signed distance function,
    ///                 SWEEP_GREATER_THAN_ISOVALUE will dilate in the positive
    ///                 side of the iso-surface, SWEEP_LESS_THAN_ISOVALUE will dilate
    ///                 in the negative side of the iso-surface.
    ///
    /// @details This, or any of ther other initialization methods, should be called
    ///          before any call to sweep(). Failure to do so will throw a RuntimeError.
    ///
    /// @warning Note, if this method fails, i.e. returns false, a subsequent call
    ///          to sweep will trow a RuntimeError. Instead call clear and try again.
    bool initDilate(const SdfGridT &sdfGrid,
                    int dilation,
                    NearestNeighbors nn = NN_FACE,
                    FastSweepingDomain mode = FastSweepingDomain::SWEEP_ALL);

    /// @brief Initializer used for the extension of an existing signed distance field
    ///        into the active values of an input mask of arbitrary value type.
    ///
    /// @return True if the initialization succeeded.
    ///
    /// @param sdfGrid  Input signed distance field to be extended into the mask.
    ///
    /// @param mask     Mask used to identify the topology of the output SDF.
    ///                 Note this mask is assume to overlap with the sdfGrid.
    ///
    /// @param ignoreActiveTiles If false, active tiles in the mask are treated
    ///                 as active voxels. Else they are ignored.
    ///
    /// @details This, or any of ther other initialization methods, should be called
    ///          before any call to sweep(). Failure to do so will throw a RuntimeError.
    ///
    /// @warning Note, if this method fails, i.e. returns false, a subsequent call
    ///          to sweep will trow a RuntimeError. Instead call clear and try again.
    template<typename MaskTreeT>
    bool initMask(const SdfGridT &sdfGrid, const Grid<MaskTreeT> &mask, bool ignoreActiveTiles = false);

    /// @brief Perform @a nIter iterations of the fast sweeping algorithm.
    ///
    /// @param nIter Number of iterations of the fast sweeping algorithm.
    ///              Each iteration performs 2^3 = 8 individual sweeps.
    ///
    /// @param finalize If true the (possibly asymmetric) inside and outside values of the
    ///                 resulting signed distance field are properly set. Unless you're
    ///                 an expert this should remain true!
    ///
    /// @throw RuntimeError if sweepingVoxelCount() or boundaryVoxelCount() return zero.
    ///        This might happen if none of the initialization methods above were called
    ///        or if that initialization failed.
    void sweep(int nIter = 1,
               bool finalize = true);

    /// @brief Clears all the grids and counters so initialization can be called again.
    void clear();

    /// @brief Return the number of voxels that will be solved for.
    size_t sweepingVoxelCount() const { return mSweepingVoxelCount; }

    /// @brief Return the number of voxels that defined the boundary condition.
    size_t boundaryVoxelCount() const { return mBoundaryVoxelCount; }

    /// @brief Return true if there are voxels and boundaries to solve for
    bool isValid() const { return mSweepingVoxelCount > 0 && mBoundaryVoxelCount > 0; }

    /// @brief Return whether the sweep update is in all direction (SWEEP_ALL),
    ///        greater than isovalue (SWEEP_GREATER_THAN_ISOVALUE), or less than isovalue
    ///        (SWEEP_LESS_THAN_ISOVALUE).
    ///
    /// @note SWEEP_GREATER_THAN_ISOVALUE and SWEEP_LESS_THAN_ISOVALUE modes are used
    ///       in dilating the narrow-band of a levelset or in extending a field.
    FastSweepingDomain sweepDirection() const { return mSweepDirection; }

    /// @brief Return whether the fast-sweeping input grid a signed distance function or not (fog).
    bool isInputSdf() { return mIsInputSdf; }

private:

    /// @brief Private method to prune the sweep mask and cache leaf origins.
    void computeSweepMaskLeafOrigins();

    // Private utility classes
    template<typename>
    struct MaskKernel;//   initialization to extend a SDF into a mask
    template<typename>
    struct InitExt;
    struct InitSdf;
    struct DilateKernel;// initialization to dilate a SDF
    struct MinMaxKernel;
    struct PruneMinMaxFltKernel;// prune sdf if it has min float or max float
    struct SweepingKernel;// performs the actual concurrent sparse fast sweeping

    // Define the topology (i.e. stencil) of the neighboring grid points
    static const Coord mOffset[6];// = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

    // Private member data of FastSweeping
    typename SdfGridT::Ptr mSdfGrid;
    typename ExtGridT::Ptr mExtGrid;
    typename ExtGridT::Ptr mExtGridInput; // optional: only used in extending a field in one direction
    SweepMaskTreeT mSweepMask; // mask tree containing all non-boundary active voxels, in the case of dilation, does not include active voxel
    std::vector<Coord> mSweepMaskLeafOrigins; // cache of leaf node origins for mask tree
    size_t mSweepingVoxelCount, mBoundaryVoxelCount;
    FastSweepingDomain mSweepDirection; // only used in dilate and extending a field
    bool mIsInputSdf;
};// FastSweeping

////////////////////////////////////////////////////////////////////////////////

// Static member data initialization
template <typename SdfGridT, typename ExtValueT>
const Coord FastSweeping<SdfGridT, ExtValueT>::mOffset[6] = {{-1,0,0},{1,0,0},
                                                             {0,-1,0},{0,1,0},
                                                             {0,0,-1},{0,0,1}};

template <typename SdfGridT, typename ExtValueT>
FastSweeping<SdfGridT, ExtValueT>::FastSweeping()
    : mSdfGrid(nullptr), mExtGrid(nullptr), mSweepingVoxelCount(0), mBoundaryVoxelCount(0), mSweepDirection(FastSweepingDomain::SWEEP_ALL), mIsInputSdf(true)
{
}

template <typename SdfGridT, typename ExtValueT>
void FastSweeping<SdfGridT, ExtValueT>::clear()
{
    mSdfGrid.reset();
    mExtGrid.reset();
    mSweepMask.clear();
    if (mExtGridInput) mExtGridInput.reset();
    mSweepingVoxelCount = mBoundaryVoxelCount = 0;
    mSweepDirection = FastSweepingDomain::SWEEP_ALL;
    mIsInputSdf = true;
}

template <typename SdfGridT, typename ExtValueT>
void FastSweeping<SdfGridT, ExtValueT>::computeSweepMaskLeafOrigins()
{
    // replace any inactive leaf nodes with tiles and voxelize any active tiles

    pruneInactive(mSweepMask);
    mSweepMask.voxelizeActiveTiles();

    using LeafManagerT = tree::LeafManager<SweepMaskTreeT>;
    using LeafT = typename SweepMaskTreeT::LeafNodeType;
    LeafManagerT leafManager(mSweepMask);

    mSweepMaskLeafOrigins.resize(leafManager.leafCount());
    std::atomic<size_t> sweepingVoxelCount{0};
    auto kernel = [&](const LeafT& leaf, size_t leafIdx) {
        mSweepMaskLeafOrigins[leafIdx] = leaf.origin();
        sweepingVoxelCount += leaf.onVoxelCount();
    };
    leafManager.foreach(kernel, /*threaded=*/true, /*grainsize=*/1024);

    mBoundaryVoxelCount = 0;
    mSweepingVoxelCount = sweepingVoxelCount;
    if (mSdfGrid) {
        const size_t totalCount = mSdfGrid->constTree().activeVoxelCount();
        assert( totalCount >= mSweepingVoxelCount );
        mBoundaryVoxelCount = totalCount - mSweepingVoxelCount;
    }
}// FastSweeping::computeSweepMaskLeafOrigins

template <typename SdfGridT, typename ExtValueT>
bool FastSweeping<SdfGridT, ExtValueT>::initSdf(const SdfGridT &fogGrid, SdfValueT isoValue, bool isInputSdf)
{
    this->clear();
    mSdfGrid = fogGrid.deepCopy();//very fast
    mIsInputSdf = isInputSdf;
    InitSdf kernel(*this);
    kernel.run(isoValue);
    return this->isValid();
}

template <typename SdfGridT, typename ExtValueT>
template <typename OpT>
bool FastSweeping<SdfGridT, ExtValueT>::initExt(const SdfGridT &fogGrid, const OpT &op, const ExtValueT &background, SdfValueT isoValue, bool isInputSdf, FastSweepingDomain mode, const typename ExtGridT::ConstPtr extGrid)
{
    if (mode != FastSweepingDomain::SWEEP_ALL) {
        if (!extGrid)
            OPENVDB_THROW(RuntimeError, "FastSweeping::initExt Calling initExt with mode != SWEEP_ALL requires an extension grid!");
        if (extGrid->transform() != fogGrid.transform())
            OPENVDB_THROW(RuntimeError, "FastSweeping::initExt extension grid input should have the same transform as Fog/SDF grid!");
    }

    this->clear();
    mSdfGrid = fogGrid.deepCopy();//very fast
    mExtGrid = createGrid<ExtGridT>( background );
    mSweepDirection = mode;
    mIsInputSdf = isInputSdf;
    if (mSweepDirection != FastSweepingDomain::SWEEP_ALL) {
        mExtGridInput = extGrid->deepCopy();
    }
    mExtGrid->topologyUnion( *mSdfGrid );//very fast
    InitExt<OpT> kernel(*this);
    kernel.run(isoValue, op);
    return this->isValid();
}


template <typename SdfGridT, typename ExtValueT>
bool FastSweeping<SdfGridT, ExtValueT>::initDilate(const SdfGridT &sdfGrid, int dilate, NearestNeighbors nn, FastSweepingDomain mode)
{
    this->clear();
    mSdfGrid = sdfGrid.deepCopy();//very fast
    mSweepDirection = mode;
    DilateKernel kernel(*this);
    kernel.run(dilate, nn);
    return this->isValid();
}

template <typename SdfGridT, typename ExtValueT>
template<typename MaskTreeT>
bool FastSweeping<SdfGridT, ExtValueT>::initMask(const SdfGridT &sdfGrid, const Grid<MaskTreeT> &mask, bool ignoreActiveTiles)
{
    this->clear();
    mSdfGrid = sdfGrid.deepCopy();//very fast

    if (mSdfGrid->transform() != mask.transform()) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: Mask not aligned with the grid!");
    }

    if (mask.getGridClass() == GRID_LEVEL_SET) {
        using T = typename MaskTreeT::template ValueConverter<bool>::Type;
        typename Grid<T>::Ptr tmp = sdfInteriorMask(mask);//might have active tiles
        tmp->tree().voxelizeActiveTiles();//multi-threaded
        MaskKernel<T> kernel(*this);
        kernel.run(tmp->tree());//multi-threaded
    } else {
        if (ignoreActiveTiles || !mask.tree().hasActiveTiles()) {
            MaskKernel<MaskTreeT> kernel(*this);
            kernel.run(mask.tree());//multi-threaded
        } else {
            using T = typename MaskTreeT::template ValueConverter<ValueMask>::Type;
            T tmp(mask.tree(), false, TopologyCopy());//multi-threaded
            tmp.voxelizeActiveTiles(true);//multi-threaded
            MaskKernel<T> kernel(*this);
            kernel.run(tmp);//multi-threaded
        }
    }
    return this->isValid();
}// FastSweeping::initMask

template <typename SdfGridT, typename ExtValueT>
void FastSweeping<SdfGridT, ExtValueT>::sweep(int nIter, bool finalize)
{
    if (!mSdfGrid) {
        OPENVDB_THROW(RuntimeError, "FastSweeping::sweep called before initialization!");
    }
    if (mExtGrid && mSweepDirection != FastSweepingDomain::SWEEP_ALL && !mExtGridInput) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: Trying to extend a field in one direction needs"
                                     " a non-null reference extension grid input.");
    }
    if (this->boundaryVoxelCount() == 0) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: No boundary voxels found!");
    } else if (this->sweepingVoxelCount() == 0) {
        OPENVDB_THROW(RuntimeError, "FastSweeping: No computing voxels found!");
    }

    // note: Sweeping kernel is non copy-constructible, so use a deque instead of a vector
    std::deque<SweepingKernel> kernels;
    for (int i = 0; i < 4; i++) kernels.emplace_back(*this);

    { // compute voxel slices
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Computing voxel slices");
#endif

        // Exploiting nested parallelism - all voxel slice data is precomputed
        tbb::task_group tasks;
        tasks.run([&] { kernels[0].computeVoxelSlices([](const Coord &a){ return a[0]+a[1]+a[2]; });/*+++ & ---*/ });
        tasks.run([&] { kernels[1].computeVoxelSlices([](const Coord &a){ return a[0]+a[1]-a[2]; });/*++- & --+*/ });
        tasks.run([&] { kernels[2].computeVoxelSlices([](const Coord &a){ return a[0]-a[1]+a[2]; });/*+-+ & -+-*/ });
        tasks.run([&] { kernels[3].computeVoxelSlices([](const Coord &a){ return a[0]-a[1]-a[2]; });/*+-- & -++*/ });
        tasks.wait();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }

    // perform nIter iterations of bi-directional sweeping in all directions
    for (int i = 0; i < nIter; ++i) {
        for (SweepingKernel& kernel : kernels) kernel.sweep();
    }

    if (finalize) {
#ifdef BENCHMARK_FAST_SWEEPING
      util::CpuTimer timer("Computing extrema values");
#endif
      MinMaxKernel kernel;
      auto e = kernel.run(*mSdfGrid);//multi-threaded
      //auto e = extrema(mGrid->beginValueOn());// 100x slower!!!!
      if (kernel.mFltMinExists || kernel.mFltMaxExists) {
          tree::NodeManager<SdfTreeT> nodeManager(mSdfGrid->tree());
          PruneMinMaxFltKernel op(e.min(), e.max());
          nodeManager.foreachTopDown(op, true /* = threaded*/, 1 /* = grainSize*/);
      }
#ifdef BENCHMARK_FAST_SWEEPING
      std::cerr << "Min = " << e.min() << " Max = " << e.max() << std::endl;
      timer.restart("Changing asymmetric background value");
#endif
      changeAsymmetricLevelSetBackground(mSdfGrid->tree(), e.max(), e.min());//multi-threaded

#ifdef BENCHMARK_FAST_SWEEPING
      timer.stop();
#endif
    }
}// FastSweeping::sweep

/// Private class of FastSweeping to quickly compute the extrema
/// values of the active voxels in the leaf nodes. Several orders
/// of magnitude faster than tools::extrema!
/// Also determines whether there is float max or float min stored
/// in a voxel.
template <typename SdfGridT, typename ExtValueT>
struct FastSweeping<SdfGridT, ExtValueT>::MinMaxKernel
{
    using LeafMgr = tree::LeafManager<const SdfTreeT>;
    using LeafRange = typename LeafMgr::LeafRange;
    MinMaxKernel() : mMin(std::numeric_limits<SdfValueT>::max()), mMax(-mMin), mFltMinExists(false), mFltMaxExists(true) {}
    MinMaxKernel(MinMaxKernel& other, tbb::split) : mMin(other.mMin), mMax(other.mMax), mFltMinExists(other.mFltMinExists), mFltMaxExists(other.mFltMaxExists) {}

    math::MinMax<SdfValueT> run(const SdfGridT &grid)
    {
        LeafMgr mgr(grid.tree());// super fast
        tbb::parallel_reduce(mgr.leafRange(), *this);
        return math::MinMax<SdfValueT>(mMin, mMax);
    }

    void operator()(const LeafRange& r)
    {
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            for (auto voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                const SdfValueT v = *voxelIter;
                const bool vEqFltMin = v == -std::numeric_limits<SdfValueT>::max();
                const bool vEqFltMax = v == std::numeric_limits<SdfValueT>::max();
                if (v < mMin && !vEqFltMin) mMin = v;
                if (v > mMax && !vEqFltMax) mMax = v;
                if (vEqFltMin) mFltMinExists = true;
                if (vEqFltMax) mFltMaxExists = true;
            }
        }
    }

    void join(const MinMaxKernel& other)
    {
        if (other.mMin < mMin) mMin = other.mMin;
        if (other.mMax > mMax) mMax = other.mMax;
        mFltMinExists = other.mFltMinExists || mFltMinExists;
        mFltMaxExists = other.mFltMaxExists || mFltMaxExists;
    }

    SdfValueT mMin, mMax;
    bool mFltMinExists, mFltMaxExists;
};// FastSweeping::MinMaxKernel

/// Private class of FastSweeping to prune sdf value that is equal to float max or
/// float min.
template <typename SdfGridT, typename ExtValueT>
struct FastSweeping<SdfGridT, ExtValueT>::PruneMinMaxFltKernel {
    PruneMinMaxFltKernel(SdfValueT min, SdfValueT max) : mMin(min), mMax(max) {}

    // Root node
    void operator()(typename SdfTreeT::RootNodeType& node, size_t = 1) const {
        for (auto iter = node.beginValueAll(); iter; ++iter) {
            if (*iter == -std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMin);
            }
            if (*iter == std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMax);
            }
        }
    }

    // Internal nodes
    template<typename NodeT>
    void operator()(NodeT& node, size_t = 1) const
    {
        for (auto iter = node.beginValueAll(); iter; ++iter) {
            if (*iter == -std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMin);
            }
            if (*iter == std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMax);
            }
        }
    }

    // Leaf nodes
    void operator()(typename SdfTreeT::LeafNodeType& leaf, size_t = 1) const
    {
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            if (*iter == -std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMin);
            }
            if (*iter == std::numeric_limits<SdfValueT>::max()) {
                iter.setValue(mMax);
            }
        }
    }
  SdfValueT mMin, mMax;
};// FastSweeping::PruneMinMaxFltKernel

////////////////////////////////////////////////////////////////////////////////

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename SdfGridT, typename ExtValueT>
struct FastSweeping<SdfGridT, ExtValueT>::DilateKernel
{
    using LeafRange = typename tree::LeafManager<SdfTreeT>::LeafRange;
    DilateKernel(FastSweeping &parent)
        : mParent(&parent),
          mBackground(parent.mSdfGrid->background())
    {
        mSdfGridInput = mParent->mSdfGrid->deepCopy();
    }
    DilateKernel(const DilateKernel &parent) = default;// for tbb::parallel_for
    DilateKernel& operator=(const DilateKernel&) = delete;

    void run(int dilation, NearestNeighbors nn)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Construct LeafManager");
#endif
        tree::LeafManager<SdfTreeT> mgr(mParent->mSdfGrid->tree());// super fast

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Changing background value");
#endif
        static const SdfValueT Unknown = std::numeric_limits<SdfValueT>::max();
        changeLevelSetBackground(mgr, Unknown);//multi-threaded

 #ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Dilating and updating mgr (parallel)");
        //timer.restart("Dilating and updating mgr (serial)");
#endif

        const int delta = 5;
        for (int i=0, d = dilation/delta; i<d; ++i) dilateActiveValues(mgr, delta, nn, IGNORE_TILES);
        dilateActiveValues(mgr, dilation % delta, nn, IGNORE_TILES);
        //for (int i=0, n=5, d=dilation/n; i<d; ++i) dilateActiveValues(mgr, n, nn, IGNORE_TILES);
        //dilateVoxels(mgr, dilation, nn);

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initializing grid and sweep mask");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mSdfGrid->constTree());

        using LeafManagerT = tree::LeafManager<typename SdfGridT::TreeType>;
        using LeafT = typename SdfGridT::TreeType::LeafNodeType;

        const FastSweepingDomain mode = mParent->mSweepDirection;

        LeafManagerT leafManager(mParent->mSdfGrid->tree());

        auto kernel = [&](LeafT& leaf, size_t /*leafIdx*/) {
            static const SdfValueT Unknown = std::numeric_limits<SdfValueT>::max();
            const SdfValueT background = mBackground;//local copy
            auto* maskLeaf = mParent->mSweepMask.probeLeaf(leaf.origin());
            SdfConstAccT sdfInputAcc(mSdfGridInput->tree());
            assert(maskLeaf);
            for (auto voxelIter = leaf.beginValueOn(); voxelIter; ++voxelIter) {
                const SdfValueT value = *voxelIter;
                SdfValueT inputValue;
                const Coord ijk = voxelIter.getCoord();

                if (math::Abs(value) < background) {// disable boundary voxels from the mask tree
                    maskLeaf->setValueOff(voxelIter.pos());
                } else {
                    switch (mode) {
                        case FastSweepingDomain::SWEEP_ALL :
                            voxelIter.setValue(value > 0 ? Unknown : -Unknown);
                            break;
                        case FastSweepingDomain::SWEEP_GREATER_THAN_ISOVALUE :
                            if (value > 0) voxelIter.setValue(Unknown);
                            else {
                                maskLeaf->setValueOff(voxelIter.pos());
                                bool isInputOn = sdfInputAcc.probeValue(ijk, inputValue);
                                if ( !isInputOn ) voxelIter.setValueOff();
                                else voxelIter.setValue(inputValue);
                            }
                            break;
                        case FastSweepingDomain::SWEEP_LESS_THAN_ISOVALUE :
                            if (value < 0) voxelIter.setValue(-Unknown);
                            else {
                                maskLeaf->setValueOff(voxelIter.pos());
                                bool isInputOn = sdfInputAcc.probeValue(ijk, inputValue);
                                if ( !isInputOn ) voxelIter.setValueOff();
                                else voxelIter.setValue(inputValue);
                            }
                            break;
                    }
                }
            }
        };

        leafManager.foreach( kernel );

        // cache the leaf node origins for fast lookup in the sweeping kernels
        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }// FastSweeping::DilateKernel::run

    // Private member data of DilateKernel
    FastSweeping                *mParent;
    const SdfValueT             mBackground;
    typename SdfGridT::ConstPtr mSdfGridInput;
};// FastSweeping::DilateKernel

////////////////////////////////////////////////////////////////////////////////

template <typename SdfGridT, typename ExtValueT>
struct FastSweeping<SdfGridT, ExtValueT>::InitSdf
{
    using LeafRange = typename tree::LeafManager<SdfTreeT>::LeafRange;
    InitSdf(FastSweeping &parent): mParent(&parent),
      mSdfGrid(parent.mSdfGrid.get()), mIsoValue(0), mAboveSign(0) {}
    InitSdf(const InitSdf&) = default;// for tbb::parallel_for
    InitSdf& operator=(const InitSdf&) = delete;

    void run(SdfValueT isoValue)
    {
        mIsoValue   = isoValue;
        mAboveSign  = mParent->mIsInputSdf ? SdfValueT(1) : SdfValueT(-1);
        SdfTreeT &tree = mSdfGrid->tree();//sdf
        const bool hasActiveTiles = tree.hasActiveTiles();

        if (mParent->mIsInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif
        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mSdfGrid->constTree());

        {// Process all voxels
          tree::LeafManager<SdfTreeT> mgr(tree, 1);// we need one auxiliary buffer
          tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
          mgr.swapLeafBuffer(1);//swap voxel values
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initialize tiles - new");
#endif
        // Process all tiles
        tree::NodeManager<SdfTreeT, SdfTreeT::RootNodeType::LEVEL-1> mgr(tree);
        mgr.foreachBottomUp(*this);//multi-threaded
        tree.root().setBackground(std::numeric_limits<SdfValueT>::max(), false);
        if (hasActiveTiles) tree.voxelizeActiveTiles();//multi-threaded

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();
    }// FastSweeping::InitSdf::run

    void operator()(const LeafRange& r) const
    {
        SweepMaskAccT sweepMaskAcc(mParent->mSweepMask);
        math::GradStencil<SdfGridT, false> stencil(*mSdfGrid);
        const SdfValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<SdfValueT>::max();//local copy
        const SdfValueT h = mAboveSign*static_cast<SdfValueT>(mSdfGrid->voxelSize()[0]);//Voxel size
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            SdfValueT* sdf = leafIter.buffer(1).data();
            for (auto voxelIter = leafIter->beginValueAll(); voxelIter; ++voxelIter) {
                const SdfValueT value = *voxelIter;
                const bool isAbove = value > isoValue;
                if (!voxelIter.isValueOn()) {// inactive voxels
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                } else {// active voxels
                    const Coord ijk = voxelIter.getCoord();
                    stencil.moveTo(ijk, value);
                    const auto mask = stencil.intersectionMask( isoValue );
                    if (mask.none()) {// most common case
                        sdf[voxelIter.pos()] = isAbove ? above : -above;
                    } else {// compute distance to iso-surface
                        // disable boundary voxels from the mask tree
                        sweepMaskAcc.setValueOff(ijk);
                        const SdfValueT delta = value - isoValue;//offset relative to iso-value
                        if (math::isApproxZero(delta)) {//voxel is on the iso-surface
                            sdf[voxelIter.pos()] = 0;
                        } else {//voxel is neighboring the iso-surface
                            SdfValueT sum = 0;
                            for (int i=0; i<6;) {
                                SdfValueT d = std::numeric_limits<SdfValueT>::max(), d2;
                                if (mask.test(i++)) d = math::Abs(delta/(value-stencil.getValue(i)));
                                if (mask.test(i++)) {
                                    d2 = math::Abs(delta/(value-stencil.getValue(i)));
                                    if (d2 < d) d = d2;
                                }
                                if (d < std::numeric_limits<SdfValueT>::max()) sum += 1/(d*d);
                            }
                            sdf[voxelIter.pos()] = isAbove ? h / math::Sqrt(sum) : -h / math::Sqrt(sum);
                        }// voxel is neighboring the iso-surface
                    }// intersecting voxels
                }// active voxels
            }// loop over voxels
        }// loop over leaf nodes
    }// FastSweeping::InitSdf::operator(const LeafRange&)

    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const SdfValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<SdfValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          SdfValueT& v = const_cast<SdfValueT&>(*it);
          v = v > isoValue ? above : -above;
        }//loop over all tiles
    }// FastSweeping::InitSdf::operator()(const RootOrInternalNodeT&)

    // Public member data
    FastSweeping *mParent;
    SdfGridT     *mSdfGrid;//raw pointer, i.e. lock free
    SdfValueT      mIsoValue;
    SdfValueT      mAboveSign;//sign of distance values above the iso-value
};// FastSweeping::InitSdf


/// Private class of FastSweeping to perform multi-threaded initialization
template <typename SdfGridT, typename ExtValueT>
template <typename OpT>
struct FastSweeping<SdfGridT, ExtValueT>::InitExt
{
    using LeafRange = typename tree::LeafManager<SdfTreeT>::LeafRange;
    using OpPoolT = tbb::enumerable_thread_specific<OpT>;
    InitExt(FastSweeping &parent) : mParent(&parent),
      mOpPool(nullptr), mSdfGrid(parent.mSdfGrid.get()),
      mExtGrid(parent.mExtGrid.get()), mIsoValue(0), mAboveSign(0) {}
    InitExt(const InitExt&) = default;// for tbb::parallel_for
    InitExt& operator=(const InitExt&) = delete;
    void run(SdfValueT isoValue, const OpT &opPrototype)
    {
        static_assert(std::is_convertible<decltype(opPrototype(Vec3d(0))),ExtValueT>::value, "Invalid return type of functor");
        if (!mExtGrid) {
          OPENVDB_THROW(RuntimeError, "FastSweeping::InitExt expected an extension grid!");
        }

        mAboveSign  = mParent->mIsInputSdf ? SdfValueT(1) : SdfValueT(-1);
        mIsoValue = isoValue;
        auto &tree1 = mSdfGrid->tree();
        auto &tree2 = mExtGrid->tree();
        const bool hasActiveTiles = tree1.hasActiveTiles();//very fast

        if (mParent->mIsInputSdf && hasActiveTiles) {
          OPENVDB_THROW(RuntimeError, "FastSweeping: A SDF should not have active tiles!");
        }

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer  timer("Initialize voxels");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mSdfGrid->constTree());

        {// Process all voxels
          // Define thread-local operators
          OpPoolT opPool(opPrototype);
          mOpPool = &opPool;

          tree::LeafManager<SdfTreeT> mgr(tree1, 1);// we need one auxiliary buffer
          tbb::parallel_for(mgr.leafRange(32), *this);//multi-threaded
          mgr.swapLeafBuffer(1);//swap out auxiliary buffer
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initialize tiles");
#endif
        {// Process all tiles
          tree::NodeManager<SdfTreeT, SdfTreeT::RootNodeType::LEVEL-1> mgr(tree1);
          mgr.foreachBottomUp(*this);//multi-threaded
          tree1.root().setBackground(std::numeric_limits<SdfValueT>::max(), false);
          if (hasActiveTiles) {
#ifdef BENCHMARK_FAST_SWEEPING
            timer.restart("Voxelizing active tiles");
#endif
            tree1.voxelizeActiveTiles();//multi-threaded
            tree2.voxelizeActiveTiles();//multi-threaded
          }
        }

        // cache the leaf node origins for fast lookup in the sweeping kernels

        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }// FastSweeping::InitExt::run

    // int implements an update if minD needs to be updated
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<std::is_same<ExtT, int>::value, int>::type = 0>
    void sumHelper(ExtT& sum2, ExtT ext, bool update, const SdfT& /* d2 */) const { if (update) sum2 = ext; }// int implementation

    // non-int implements a weighted sum
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<!std::is_same<ExtT, int>::value, int>::type = 0>
    void sumHelper(ExtT& sum2, ExtT ext, bool /* update */, const SdfT&  d2) const { sum2 += static_cast<ExtValueT>(d2 * ext); }// non-int implementation

    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT extValHelper(ExtT extSum, const SdfT& /* sdfSum */) const { return extSum; }// int implementation

    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<!std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT extValHelper(ExtT extSum, const SdfT& sdfSum) const {return ExtT((SdfT(1) / sdfSum) * extSum); }// non-int implementation

    void operator()(const LeafRange& r) const
    {
        ExtAccT acc(mExtGrid->tree());
        SweepMaskAccT sweepMaskAcc(mParent->mSweepMask);
        math::GradStencil<SdfGridT, false> stencil(*mSdfGrid);
        const math::Transform& xform = mExtGrid->transform();
        typename OpPoolT::reference op = mOpPool->local();
        const SdfValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<SdfValueT>::max();//local copy
        const SdfValueT h = mAboveSign*static_cast<SdfValueT>(mSdfGrid->voxelSize()[0]);//Voxel size
        for (auto leafIter = r.begin(); leafIter; ++leafIter) {
            SdfValueT *sdf = leafIter.buffer(1).data();
            ExtValueT *ext = acc.probeLeaf(leafIter->origin())->buffer().data();//should be safe!
            for (auto voxelIter = leafIter->beginValueAll(); voxelIter; ++voxelIter) {
                const SdfValueT value = *voxelIter;
                const bool isAbove = value > isoValue;
                if (!voxelIter.isValueOn()) {// inactive voxels
                    sdf[voxelIter.pos()] = isAbove ? above : -above;
                } else {// active voxels
                    const Coord ijk = voxelIter.getCoord();
                    stencil.moveTo(ijk, value);
                    const auto mask = stencil.intersectionMask( isoValue );
                    if (mask.none()) {// no zero-crossing neighbors, most common case
                        sdf[voxelIter.pos()] = isAbove ? above : -above;
                        // the ext grid already has its active values set to the background value
                    } else {// compute distance to iso-surface
                        // disable boundary voxels from the mask tree
                        sweepMaskAcc.setValueOff(ijk);
                        const SdfValueT delta = value - isoValue;//offset relative to iso-value
                        if (math::isApproxZero(delta)) {//voxel is on the iso-surface
                            sdf[voxelIter.pos()] = 0;
                            ext[voxelIter.pos()] = ExtValueT(op(xform.indexToWorld(ijk)));
                        } else {//voxel is neighboring the iso-surface
                            SdfValueT sum1 = 0;
                            ExtValueT sum2 = zeroVal<ExtValueT>();
                            // minD is used to update sum2 in the integer case,
                            // where we choose the value of the extension grid corresponding to
                            // the smallest value of d in the 6 direction neighboring the current
                            // voxel.
                            SdfValueT minD = std::numeric_limits<SdfValueT>::max();
                            for (int n=0, i=0; i<6;) {
                                SdfValueT d = std::numeric_limits<SdfValueT>::max(), d2;
                                if (mask.test(i++)) {
                                    d = math::Abs(delta/(value-stencil.getValue(i)));
                                    n = i - 1;
                                }
                                if (mask.test(i++)) {
                                    d2 = math::Abs(delta/(value-stencil.getValue(i)));
                                    if (d2 < d) {
                                        d = d2;
                                        n = i - 1;
                                    }
                                }
                                if (d < std::numeric_limits<SdfValueT>::max()) {
                                    d2 = 1/(d*d);
                                    sum1 += d2;
                                    const Vec3R xyz(static_cast<SdfValueT>(ijk[0])+d*static_cast<SdfValueT>(FastSweeping::mOffset[n][0]),
                                                    static_cast<SdfValueT>(ijk[1])+d*static_cast<SdfValueT>(FastSweeping::mOffset[n][1]),
                                                    static_cast<SdfValueT>(ijk[2])+d*static_cast<SdfValueT>(FastSweeping::mOffset[n][2]));
                                    // If current d is less than minD, update minD
                                    sumHelper(sum2, ExtValueT(op(xform.indexToWorld(xyz))), d < minD, d2);
                                    if (d < minD) minD = d;
                                }
                            }//look over six cases
                            ext[voxelIter.pos()] = extValHelper(sum2, sum1);
                            sdf[voxelIter.pos()] = isAbove ? h / math::Sqrt(sum1) : -h / math::Sqrt(sum1);
                        }// voxel is neighboring the iso-surface
                    }// intersecting voxels
                }// active voxels
            }// loop over voxels
        }// loop over leaf nodes
    }// FastSweeping::InitExt::operator(const LeafRange& r)

    template<typename RootOrInternalNodeT>
    void operator()(const RootOrInternalNodeT& node) const
    {
        const SdfValueT isoValue = mIsoValue, above = mAboveSign*std::numeric_limits<SdfValueT>::max();
        for (auto it = node.cbeginValueAll(); it; ++it) {
          SdfValueT& v = const_cast<SdfValueT&>(*it);
          v = v > isoValue ? above : -above;
        }//loop over all tiles
    }
    // Public member data
    FastSweeping *mParent;
    OpPoolT      *mOpPool;
    SdfGridT     *mSdfGrid;
    ExtGridT     *mExtGrid;
    SdfValueT      mIsoValue;
    SdfValueT      mAboveSign;//sign of distance values above the iso-value
};// FastSweeping::InitExt

/// Private class of FastSweeping to perform multi-threaded initialization
template <typename SdfGridT, typename ExtValueT>
template <typename MaskTreeT>
struct FastSweeping<SdfGridT, ExtValueT>::MaskKernel
{
    using LeafRange = typename tree::LeafManager<const MaskTreeT>::LeafRange;
    MaskKernel(FastSweeping &parent) : mParent(&parent),
      mSdfGrid(parent.mSdfGrid.get()) {}
    MaskKernel(const MaskKernel &parent) = default;// for tbb::parallel_for
    MaskKernel& operator=(const MaskKernel&) = delete;

    void run(const MaskTreeT &mask)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer;
#endif
        auto &lsTree = mSdfGrid->tree();

        static const SdfValueT Unknown = std::numeric_limits<SdfValueT>::max();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Changing background value");
#endif
        changeLevelSetBackground(lsTree, Unknown);//multi-threaded

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Union with mask");//multi-threaded
#endif
        lsTree.topologyUnion(mask);//multi-threaded

        // ignore active tiles since the input grid is assumed to be a level set
        tree::LeafManager<const MaskTreeT> mgr(mask);// super fast

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Initializing grid and sweep mask");
#endif

        mParent->mSweepMask.clear();
        mParent->mSweepMask.topologyUnion(mParent->mSdfGrid->constTree());

        using LeafManagerT = tree::LeafManager<SweepMaskTreeT>;
        using LeafT = typename SweepMaskTreeT::LeafNodeType;
        LeafManagerT leafManager(mParent->mSweepMask);

        auto kernel = [&](LeafT& leaf, size_t /*leafIdx*/) {
            static const SdfValueT Unknown = std::numeric_limits<SdfValueT>::max();
            SdfAccT acc(mSdfGrid->tree());
            // The following hack is safe due to the topology union in
            // init and the fact that SdfValueT is known to be a floating point!
            SdfValueT *data = acc.probeLeaf(leaf.origin())->buffer().data();
            for (auto voxelIter = leaf.beginValueOn(); voxelIter; ++voxelIter) {// mask voxels
                if (math::Abs( data[voxelIter.pos()] ) < Unknown ) {
                    // disable boundary voxels from the mask tree
                    voxelIter.setValue(false);
                }
            }
        };
        leafManager.foreach( kernel );

        // cache the leaf node origins for fast lookup in the sweeping kernels
        mParent->computeSweepMaskLeafOrigins();

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }// FastSweeping::MaskKernel::run

    // Private member data of MaskKernel
    FastSweeping *mParent;
    SdfGridT     *mSdfGrid;//raw pointer, i.e. lock free
};// FastSweeping::MaskKernel

/// @brief Private class of FastSweeping to perform concurrent fast sweeping in two directions
template <typename SdfGridT, typename ExtValueT>
struct FastSweeping<SdfGridT, ExtValueT>::SweepingKernel
{
    SweepingKernel(FastSweeping &parent) : mParent(&parent) {}
    SweepingKernel(const SweepingKernel&) = delete;
    SweepingKernel& operator=(const SweepingKernel&) = delete;

    /// Main method that performs concurrent bi-directional sweeps
    template<typename HashOp>
    void computeVoxelSlices(HashOp hash)
    {
#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer;
#endif

        // mask of the active voxels to be solved for, i.e. excluding boundary voxels
        const SweepMaskTreeT& maskTree = mParent->mSweepMask;

        using LeafManagerT = typename tree::LeafManager<const SweepMaskTreeT>;
        using LeafT = typename SweepMaskTreeT::LeafNodeType;
        LeafManagerT leafManager(maskTree);

        // compute the leaf node slices that have active voxels in them
        // the sliding window of the has keys is -14 to 21 (based on an 8x8x8 leaf node
        // and the extrema hash values i-j-k and  i+j+k), but we use a larger mask window here to
        // easily accommodate any leaf dimension. The mask offset is used to be able to
        // store this in a fixed-size byte array
        constexpr int maskOffset = LeafT::DIM * 3;
        constexpr int maskRange = maskOffset * 2;

        // mark each possible slice in each leaf node that has one or more active voxels in it
        std::vector<int8_t> leafSliceMasks(leafManager.leafCount()*maskRange);
        auto kernel1 = [&](const LeafT& leaf, size_t leafIdx) {
            const size_t leafOffset = leafIdx * maskRange;
            for (auto voxelIter = leaf.cbeginValueOn(); voxelIter; ++voxelIter) {
                const Coord ijk = LeafT::offsetToLocalCoord(voxelIter.pos());
                leafSliceMasks[leafOffset + hash(ijk) + maskOffset] = uint8_t(1);
            }
        };
        leafManager.foreach( kernel1 );

        // compute the voxel slice map using a thread-local-storage hash map
        // the key of the hash map is the slice index of the voxel coord (ijk.x() + ijk.y() + ijk.z())
        // the values are an array of indices for every leaf that has active voxels with this slice index
        using ThreadLocalMap = std::unordered_map</*voxelSliceKey=*/int64_t, /*leafIdx=*/std::deque<size_t>>;
        tbb::enumerable_thread_specific<ThreadLocalMap> pool;
        auto kernel2 = [&](const LeafT& leaf, size_t leafIdx) {
            ThreadLocalMap& map = pool.local();
            const Coord& origin = leaf.origin();
            const int64_t leafKey = hash(origin);
            const size_t leafOffset = leafIdx * maskRange;
            for (int sliceIdx = 0; sliceIdx < maskRange; sliceIdx++) {
                if (leafSliceMasks[leafOffset + sliceIdx] == uint8_t(1)) {
                    const int64_t voxelSliceKey = leafKey+sliceIdx-maskOffset;
                    map[voxelSliceKey].emplace_back(leafIdx);
                }
            }
        };
        leafManager.foreach( kernel2 );

        // combine into a single ordered map keyed by the voxel slice key
        // note that this is now stored in a map ordered by voxel slice key,
        // so sweep slices can be processed in order
        for (auto poolIt = pool.begin(); poolIt != pool.end(); ++poolIt) {
            const ThreadLocalMap& map = *poolIt;
            for (const auto& it : map) {
                for (const size_t leafIdx : it.second) {
                    mVoxelSliceMap[it.first].emplace_back(leafIdx, NodeMaskPtrT());
                }
            }
        }

        // extract the voxel slice keys for random access into the map
        mVoxelSliceKeys.reserve(mVoxelSliceMap.size());
        for (const auto& it : mVoxelSliceMap) {
            mVoxelSliceKeys.push_back(it.first);
        }

        // allocate the node masks in parallel, as the map is populated in serial
        auto kernel3 = [&](tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                const int64_t key = mVoxelSliceKeys[i];
                for (auto& it : mVoxelSliceMap[key]) {
                    it.second = std::make_unique<NodeMaskT>();
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceKeys.size()), kernel3);

        // each voxel slice contains a leafIdx-nodeMask pair,
        // this routine populates these node masks to select only the active voxels
        // from the mask tree that have the same voxel slice key
        // TODO: a small optimization here would be to union this leaf node mask with
        // a pre-computed one for this particular slice pattern
        auto kernel4 = [&](tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                const int64_t voxelSliceKey = mVoxelSliceKeys[i];
                LeafSliceArray& leafSliceArray = mVoxelSliceMap[voxelSliceKey];
                for (LeafSlice& leafSlice : leafSliceArray) {
                    const size_t leafIdx = leafSlice.first;
                    NodeMaskPtrT& nodeMask = leafSlice.second;
                    const LeafT& leaf = leafManager.leaf(leafIdx);
                    const Coord& origin = leaf.origin();
                    const int64_t leafKey = hash(origin);
                    for (auto voxelIter = leaf.cbeginValueOn(); voxelIter; ++voxelIter) {
                        const Index voxelIdx = voxelIter.pos();
                        const Coord ijk = LeafT::offsetToLocalCoord(voxelIdx);
                        const int64_t key = leafKey + hash(ijk);
                        if (key == voxelSliceKey) {
                            nodeMask->setOn(voxelIdx);
                        }
                    }
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceKeys.size()), kernel4);
    }// FastSweeping::SweepingKernel::computeVoxelSlices

    // Private struct for nearest neighbor grid points (very memory light!)
    struct NN {
        SdfValueT v;
        int    n;
        inline static Coord ijk(const Coord &p, int i) { return p + FastSweeping::mOffset[i]; }
        NN() : v(), n() {}
        NN(const SdfAccT &a, const Coord &p, int i) : v(math::Abs(a.getValue(ijk(p,i)))), n(i) {}
        inline Coord operator()(const Coord &p) const { return ijk(p, n); }
        inline bool operator<(const NN &rhs) const { return v < rhs.v; }
        inline operator bool() const { return v < SdfValueT(1000); }
    };// NN

    /// @note   Extending an integer field is based on the nearest-neighbor interpolation
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT twoNghbr(const NN& d1, const NN& d2, const SdfT& /* w */, const ExtT& v1, const ExtT& v2) const { return d1.v < d2.v ? v1 : v2; }// int implementation

    /// @note   Extending a non-integer field is based on a weighted interpolation
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<!std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT twoNghbr(const NN& d1, const NN& d2, const SdfT& w, const ExtT& v1, const ExtT& v2) const { return ExtT(w*(d1.v*v1 + d2.v*v2)); }// non-int implementation

    /// @note   Extending an integer field is based on the nearest-neighbor interpolation
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT threeNghbr(const NN& d1, const NN& d2, const NN& d3, const SdfT& /* w */, const ExtT& v1, const ExtT& v2, const ExtT& v3) const {
        math::Vec3<SdfT> d(d1.v, d2.v, d3.v);
        math::Vec3<ExtT> v(v1, v2, v3);
        return v[static_cast<int>(math::MinIndex(d))];
    }// int implementation

    /// @note   Extending a non-integer field is based on a weighted interpolation
    template<typename ExtT = ExtValueT, typename SdfT = SdfValueT, typename std::enable_if<!std::is_same<ExtT, int>::value, int>::type = 0>
    ExtT threeNghbr(const NN& d1, const NN& d2, const NN& d3, const SdfT& w, const ExtT& v1, const ExtT& v2, const ExtT& v3) const {
        return ExtT(w*(d1.v*v1 + d2.v*v2 + d3.v*v3));
    }// non-int implementation

    void sweep()
    {
        typename ExtGridT::TreeType *tree2 = mParent->mExtGrid ? &mParent->mExtGrid->tree() : nullptr;
        typename ExtGridT::TreeType *tree3 = mParent->mExtGridInput ? &mParent->mExtGridInput->tree() : nullptr;

        const SdfValueT h = static_cast<SdfValueT>(mParent->mSdfGrid->voxelSize()[0]);
        const SdfValueT sqrt2h = math::Sqrt(SdfValueT(2))*h;
        const FastSweepingDomain mode = mParent->mSweepDirection;
        const bool isInputSdf = mParent->mIsInputSdf;

        // If we are using an extension in one direction, we need a reference grid
        // for the default value of the extension for the voxels that are not
        // intended to be updated by the sweeping algorithm.
        if (tree2 && mode != FastSweepingDomain::SWEEP_ALL) assert(tree3);

        const std::vector<Coord>& leafNodeOrigins = mParent->mSweepMaskLeafOrigins;

        int64_t voxelSliceIndex(0);

        auto kernel = [&](const tbb::blocked_range<size_t>& range) {
            using LeafT = typename SdfGridT::TreeType::LeafNodeType;

            SdfAccT acc1(mParent->mSdfGrid->tree());
            auto acc2 = std::unique_ptr<ExtAccT>(tree2 ? new ExtAccT(*tree2) : nullptr);
            auto acc3 = std::unique_ptr<ExtAccT>(tree3 ? new ExtAccT(*tree3) : nullptr);
            SdfValueT absV, sign, update, D;
            NN d1, d2, d3;//distance values and coordinates of closest neighbor points

            const LeafSliceArray& leafSliceArray = mVoxelSliceMap[voxelSliceIndex];

            // Solves Godunov's scheme: [x-d1]^2 + [x-d2]^2  + [x-d3]^2 = h^2
            // where [X] = (X>0?X:0) and ai=min(di+1,di-1)
            for (size_t i = range.begin(); i < range.end(); ++i) {

                // iterate over all leafs in the slice and extract the leaf
                // and node mask for each slice pattern

                const LeafSlice& leafSlice = leafSliceArray[i];
                const size_t leafIdx = leafSlice.first;
                const NodeMaskPtrT& nodeMask = leafSlice.second;

                const Coord& origin = leafNodeOrigins[leafIdx];

                Coord ijk;
                for (auto indexIter = nodeMask->beginOn(); indexIter; ++indexIter) {

                    // Get coordinate of center point of the FD stencil
                    ijk = origin + LeafT::offsetToLocalCoord(indexIter.pos());

                    // Find the closes neighbors in the three axial directions
                    d1 = std::min(NN(acc1, ijk, 0), NN(acc1, ijk, 1));
                    d2 = std::min(NN(acc1, ijk, 2), NN(acc1, ijk, 3));
                    d3 = std::min(NN(acc1, ijk, 4), NN(acc1, ijk, 5));

                    if (!(d1 || d2 || d3)) continue;//no valid neighbors

                    // Get the center point of the FD stencil (assumed to be an active voxel)
                    // Note this const_cast is normally unsafe but by design we know the tree
                    // to be static, of floating-point type and containing active voxels only!
                    SdfValueT &value = const_cast<SdfValueT&>(acc1.getValue(ijk));

                    // Extract the sign
                    sign = value >= SdfValueT(0) ? SdfValueT(1) : SdfValueT(-1);

                    // Absolute value
                    absV = math::Abs(value);

                    // sort values so d1 <= d2 <= d3
                    if (d2 < d1) std::swap(d1, d2);
                    if (d3 < d2) std::swap(d2, d3);
                    if (d2 < d1) std::swap(d1, d2);

                    // Test if there is a solution depending on ONE of the neighboring voxels
                    // if d2 - d1 >= h  => d2 >= d1 + h  then:
                    // (x-d1)^2=h^2 => x = d1 + h
                    update = d1.v + h;
                    if (update <= d2.v) {
                        if (update < absV) {
                            value = sign * update;
                            if (acc2) {
                                // There is an assert upstream to check if mExtGridInput exists if mode != SWEEP_ALL
                                ExtValueT updateExt = acc2->getValue(d1(ijk));
                                if (mode == FastSweepingDomain::SWEEP_GREATER_THAN_ISOVALUE) {
                                    if (isInputSdf) updateExt = (value >= SdfValueT(0)) ? acc2->getValue(d1(ijk)) : acc3->getValue(ijk);
                                    else updateExt = (value <= SdfValueT(0)) ? acc2->getValue(d1(ijk)) : acc3->getValue(ijk);
                                } // SWEEP_GREATER_THAN_ISOVALUE
                                else if (mode == FastSweepingDomain::SWEEP_LESS_THAN_ISOVALUE) {
                                    if (isInputSdf) updateExt = (value <= SdfValueT(0)) ? acc2->getValue(d1(ijk)) : acc3->getValue(ijk);
                                    else updateExt = (value >= SdfValueT(0)) ? acc2->getValue(d1(ijk)) : acc3->getValue(ijk);
                                } // SWEEP_LESS_THAN_ISOVALUE
                                acc2->setValue(ijk, updateExt);
                            }//update ext?
                        }//update sdf?
                        continue;
                    }// one neighbor case

                    // Test if there is a solution depending on TWO of the neighboring voxels
                    // (x-d1)^2 + (x-d2)^2 = h^2
                    //D = SdfValueT(2) * h * h - math::Pow2(d1.v - d2.v);// = 2h^2-(d1-d2)^2
                    //if (D >= SdfValueT(0)) {// non-negative discriminant
                    if (d2.v <= sqrt2h + d1.v) {
                        D = SdfValueT(2) * h * h - math::Pow2(d1.v - d2.v);// = 2h^2-(d1-d2)^2
                        update = SdfValueT(0.5) * (d1.v + d2.v + std::sqrt(D));
                        if (update > d2.v && update <= d3.v) {
                            if (update < absV) {
                                value = sign * update;
                                if (acc2) {
                                    d1.v -= update;
                                    d2.v -= update;
                                    // affine combination of two neighboring extension values
                                    const SdfValueT w = SdfValueT(1)/(d1.v+d2.v);
                                    const ExtValueT v1 = acc2->getValue(d1(ijk));
                                    const ExtValueT v2 = acc2->getValue(d2(ijk));
                                    const ExtValueT extVal = twoNghbr(d1, d2, w, v1, v2);

                                    ExtValueT updateExt = extVal;
                                    if (mode == FastSweepingDomain::SWEEP_GREATER_THAN_ISOVALUE) {
                                        if (isInputSdf) updateExt = (value >= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                        else updateExt = (value <= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                    } // SWEEP_GREATER_THAN_ISOVALUE
                                    else if (mode == FastSweepingDomain::SWEEP_LESS_THAN_ISOVALUE) {
                                        if (isInputSdf) updateExt = (value <= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                        else updateExt = (value >= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                    } // SWEEP_LESS_THAN_ISOVALUE
                                    acc2->setValue(ijk, updateExt);
                                }//update ext?
                            }//update sdf?
                            continue;
                        }//test for two neighbor case
                    }//test for non-negative determinant

                    // Test if there is a solution depending on THREE of the neighboring voxels
                    // (x-d1)^2 + (x-d2)^2  + (x-d3)^2 = h^2
                    // 3x^2 - 2(d1 + d2 + d3)x + d1^2 + d2^2 + d3^2 = h^2
                    // ax^2 + bx + c=0, a=3, b=-2(d1+d2+d3), c=d1^2 + d2^2 + d3^2 - h^2
                    const SdfValueT d123 = d1.v + d2.v + d3.v;
                    D = d123*d123 - SdfValueT(3)*(d1.v*d1.v + d2.v*d2.v + d3.v*d3.v - h * h);
                    if (D >= SdfValueT(0)) {// non-negative discriminant
                        update = SdfValueT(1.0/3.0) * (d123 + std::sqrt(D));//always passes test
                        //if (update > d3.v) {//disabled due to round-off errors
                        if (update < absV) {
                            value = sign * update;
                            if (acc2) {
                                d1.v -= update;
                                d2.v -= update;
                                d3.v -= update;
                                // affine combination of three neighboring extension values
                                const SdfValueT w = SdfValueT(1)/(d1.v+d2.v+d3.v);
                                const ExtValueT v1 = acc2->getValue(d1(ijk));
                                const ExtValueT v2 = acc2->getValue(d2(ijk));
                                const ExtValueT v3 = acc2->getValue(d3(ijk));
                                const ExtValueT extVal = threeNghbr(d1, d2, d3, w, v1, v2, v3);

                                ExtValueT updateExt = extVal;
                                if (mode == FastSweepingDomain::SWEEP_GREATER_THAN_ISOVALUE) {
                                    if (isInputSdf) updateExt = (value >= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                    else updateExt = (value <= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                } // SWEEP_GREATER_THAN_ISOVALUE
                                else if (mode == FastSweepingDomain::SWEEP_LESS_THAN_ISOVALUE) {
                                    if (isInputSdf) updateExt = (value <= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                    else updateExt = (value >= SdfValueT(0)) ? extVal : acc3->getValue(ijk);
                                } // SWEEP_LESS_THAN_ISOVALUE
                                acc2->setValue(ijk, updateExt);
                            }//update ext?
                        }//update sdf?
                    }//test for non-negative determinant
                }//loop over coordinates
            }
        };

#ifdef BENCHMARK_FAST_SWEEPING
        util::CpuTimer timer("Forward  sweep");
#endif

        for (size_t i = 0; i < mVoxelSliceKeys.size(); i++) {
            voxelSliceIndex = mVoxelSliceKeys[i];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceMap[voxelSliceIndex].size()), kernel);
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.restart("Backward sweeps");
#endif
        for (size_t i = mVoxelSliceKeys.size(); i > 0; i--) {
            voxelSliceIndex = mVoxelSliceKeys[i-1];
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mVoxelSliceMap[voxelSliceIndex].size()), kernel);
        }

#ifdef BENCHMARK_FAST_SWEEPING
        timer.stop();
#endif
    }// FastSweeping::SweepingKernel::sweep

private:
    using NodeMaskT = typename SweepMaskTreeT::LeafNodeType::NodeMaskType;
    using NodeMaskPtrT = std::unique_ptr<NodeMaskT>;
    // using a unique ptr for the NodeMask allows for parallel allocation,
    // but makes this class not copy-constructible
    using LeafSlice = std::pair</*leafIdx=*/size_t, /*leafMask=*/NodeMaskPtrT>;
    using LeafSliceArray = std::deque<LeafSlice>;
    using VoxelSliceMap = std::map</*voxelSliceKey=*/int64_t, LeafSliceArray>;

    // Private member data of SweepingKernel
    FastSweeping *mParent;
    VoxelSliceMap mVoxelSliceMap;
    std::vector<int64_t> mVoxelSliceKeys;
};// FastSweeping::SweepingKernel

////////////////////////////////////////////////////////////////////////////////

template<typename GridT>
typename GridT::Ptr
fogToSdf(const GridT &fogGrid,
         typename GridT::ValueType isoValue,
         int nIter)
{
    FastSweeping<GridT> fs;
    if (fs.initSdf(fogGrid, isoValue, /*isInputSdf*/false)) fs.sweep(nIter);
    return fs.sdfGrid();
}

template<typename GridT>
typename GridT::Ptr
sdfToSdf(const GridT &sdfGrid,
         typename GridT::ValueType isoValue,
         int nIter)
{
    FastSweeping<GridT> fs;
    if (fs.initSdf(sdfGrid, isoValue, /*isInputSdf*/true)) fs.sweep(nIter);
    return fs.sdfGrid();
}

template<typename FogGridT, typename ExtOpT, typename ExtValueT>
typename FogGridT::template ValueConverter<ExtValueT>::Type::Ptr
fogToExt(const FogGridT &fogGrid,
         const ExtOpT &op,
         const ExtValueT& background,
         typename FogGridT::ValueType isoValue,
         int nIter,
         FastSweepingDomain mode,
         const typename FogGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid)
{
  FastSweeping<FogGridT, ExtValueT> fs;
  if (fs.initExt(fogGrid, op, background, isoValue, /*isInputSdf*/false, mode, extGrid))
      fs.sweep(nIter, /*finalize*/true);
  return fs.extGrid();
}

template<typename SdfGridT, typename OpT, typename ExtValueT>
typename SdfGridT::template ValueConverter<ExtValueT>::Type::Ptr
sdfToExt(const SdfGridT &sdfGrid,
         const OpT &op,
         const ExtValueT &background,
         typename SdfGridT::ValueType isoValue,
         int nIter,
         FastSweepingDomain mode,
         const typename SdfGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid)
{
  FastSweeping<SdfGridT, ExtValueT> fs;
  if (fs.initExt(sdfGrid, op, background, isoValue, /*isInputSdf*/true, mode, extGrid))
      fs.sweep(nIter, /*finalize*/true);
  return fs.extGrid();
}

template<typename FogGridT, typename ExtOpT, typename ExtValueT>
std::pair<typename FogGridT::Ptr, typename FogGridT::template ValueConverter<ExtValueT>::Type::Ptr>
fogToSdfAndExt(const FogGridT &fogGrid,
               const ExtOpT &op,
               const ExtValueT &background,
               typename FogGridT::ValueType isoValue,
               int nIter,
               FastSweepingDomain mode,
               const typename FogGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid)
{
  FastSweeping<FogGridT, ExtValueT> fs;
  if (fs.initExt(fogGrid, op, background, isoValue, /*isInputSdf*/false, mode, extGrid))
      fs.sweep(nIter, /*finalize*/true);
  return std::make_pair(fs.sdfGrid(), fs.extGrid());
}

template<typename SdfGridT, typename ExtOpT, typename ExtValueT>
std::pair<typename SdfGridT::Ptr, typename SdfGridT::template ValueConverter<ExtValueT>::Type::Ptr>
sdfToSdfAndExt(const SdfGridT &sdfGrid,
               const ExtOpT &op,
               const ExtValueT &background,
               typename SdfGridT::ValueType isoValue,
               int nIter,
               FastSweepingDomain mode,
               const typename SdfGridT::template ValueConverter<ExtValueT>::Type::ConstPtr extGrid)
{
  FastSweeping<SdfGridT, ExtValueT> fs;
  if (fs.initExt(sdfGrid, op, background, isoValue, /*isInputSdf*/true, mode, extGrid))
      fs.sweep(nIter, /*finalize*/true);
  return std::make_pair(fs.sdfGrid(), fs.extGrid());
}

template<typename GridT>
typename GridT::Ptr
dilateSdf(const GridT &sdfGrid,
          int dilation,
          NearestNeighbors nn,
          int nIter,
          FastSweepingDomain mode)
{
    FastSweeping<GridT> fs;
    if (fs.initDilate(sdfGrid, dilation, nn, /*sweep direction*/ mode)) fs.sweep(nIter);
    return fs.sdfGrid();
}

template<typename GridT, typename MaskTreeT>
typename GridT::Ptr
maskSdf(const GridT &sdfGrid,
        const Grid<MaskTreeT> &mask,
        bool ignoreActiveTiles,
        int nIter)
{
    FastSweeping<GridT> fs;
    if (fs.initMask(sdfGrid, mask, ignoreActiveTiles)) fs.sweep(nIter);
    return fs.sdfGrid();
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_FASTSWEEPING
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr fogToSdf(const Grid<TreeT>&, TreeT::ValueType, int)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr sdfToSdf(const Grid<TreeT>&, TreeT::ValueType, int)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr dilateSdf(const Grid<TreeT>&, int, NearestNeighbors, int, FastSweepingDomain)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_FASTSWEEPING_HAS_BEEN_INCLUDED
