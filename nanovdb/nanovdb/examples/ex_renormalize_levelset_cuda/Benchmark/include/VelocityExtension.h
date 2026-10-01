// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file    VelocityExtension.h
///
/// @author  Ken Museth
///
/// @brief   Defines velExt, which performs velocity extension off a given narrow-band
///          level set surface. This is performed by one of two algorithms: 1) closest-
///          point-transform (CPT), or 2) via an iterative pseudo-time dependent PDE.

#ifndef OPENVDB_TOOLS_VELOCITY_EXTENSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VELOCITY_EXTENSION_HAS_BEEN_INCLUDED

//#define BENCHMARK_FAST_SWEEPING

#include <openvdb/openvdb.h>
#include <openvdb/Platform.h>
#include <openvdb/math/Math.h> // for Abs() and isExactlyEqual()
#include <openvdb/math/Stencils.h> // for GradStencil
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h> // for PruneMinMaxFltKernel
#include <openvdb/util/Assert.h>

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

/// @brief Computes the extension of a field (scalar, vector, or int are supported), defined
///        by the specified functor, off an iso-surface from an input SDF volume.
///
/// @return A shared pointer to the extension field defined on the active values in the
///         input signed distance field.
///
/// @param sdfGrid  A signed distance field to a surface on which values are to be extended.
///
/// @param op       Functor with signature [](const Vec3R &xyz)->VelValueT that
///                 defines the Dirichlet boundary condition, on the zero iso-surface,
///                 of the field to be extended.
///
/// @param background Background value of return grid with the extension field.
///
/// @param nIter    Number of iterations of doe the pseudo-time-dependent extension PDE.
///                 The default value of zero, means extension will used closest-point-transform.
///
/// @details Topology of output grid is identical to that of the input grid, except
///          active tiles in the input grid will be converted to active voxels
///          in the output grid!
template<typename SdfGridT, typename VelOpT, typename VelValueT>
typename SdfGridT::template ValueConverter<VelValueT>::Type::Ptr
velExt(const SdfGridT &sdfGrid,
       const VelOpT &op,
       const VelValueT &background,
       int nIter = 0);

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
template<typename SdfGridT, typename VelValueT = typename SdfGridT::ValueType>
class VelocityExtension
{
    static_assert(std::is_floating_point<typename SdfGridT::ValueType>::value,
                  "VelocityExtension requires SdfGridT to have floating-point values");
    // Defined types related to the signed distance (or fog) grid
    using SdfValueT = typename SdfGridT::ValueType;
    using SdfTreeT = typename SdfGridT::TreeType;

    // define types related to the extension field
    using VelGridT = typename SdfGridT::template ValueConverter<VelValueT>::Type;
    using VelTreeT = typename VelGridT::TreeType;
    using VelMgrT  = tree::LeafManager<VelTreeT>;

public:

    /// @brief Constructor
    VelocityExtension(const SdfGridT &sdfGrid);

    /// @brief Disallow copy construction.
    VelocityExtension(const VelocityExtension&) = delete;

    /// @brief Disallow copy assignment.
    VelocityExtension& operator=(const VelocityExtension&) = delete;

    /// @brief Returns a const reference to the signed distance field computed
    ///        by this class.
    const SdfGridT &sdfGrid() { return mSdfGrid; }

    /// @brief Returns a shared pointer to the extension field computed
    ///        by this class.
    ///
    /// @warning This shared pointer might point to NULL if the grid has not been
    ///          initialize (by one of the init methods) or computed (by the sweep
    ///          method).
    typename VelGridT::Ptr velGrid() { return mVelGrid; }
    
    /// @brief 
    /// @tparam ExtOpT 
    /// @param op Functor with signature [](const Vec3R &ijk)->VelValueT that
    ///           defines the Dirichlet boundary condition, on the iso-surface,
    ///           of the field to be extended. Strictly the return type of this functor
    ///           is only required to be convertible to VelValueT! Note, ijk is floating
    ///           point index coordinates.
    /// @param background 
    template <typename ExtOpT>
    void initDense(const ExtOpT &op,
                   const VelValueT &background);

    /// @brief Use CPT to initialize velocity values sparsely around the zero crossing
    /// @tparam ExtOpT 
    /// @param sdfGrid 
    /// @param op 
    /// @param background 
    template <typename ExtOpT>
    void initSparse(const ExtOpT &op,
                    const VelValueT &background);

    /// @brief Perform @a nIter iterations of the fast sweeping algorithm.
    ///
    /// @param nIter Number of iterations of the fast sweeping algorithm.
    ///              Each iteration performs 2^3 = 8 individual sweeps.
    void iterate(int nIter = 3);

    /// @brief Return the number of voxels that will be solved for.
    size_t solveVoxelCount() const { return mSolveVoxelCount; }

    /// @brief Return the number of voxels that defined the boundary condition.
    size_t initVoxelCount() const { return mInitVoxelCount; }

private:

    // Private member data of VelocityExtension
    const SdfGridT &mSdfGrid;
    typename VelGridT::Ptr mVelGrid;
    std::atomic<size_t> mSolveVoxelCount, mInitVoxelCount;
};// VelocityExtension

////////////////////////////////////////////////////////////////////////////////

template <typename SdfGridT, typename VelValueT>
VelocityExtension<SdfGridT, VelValueT>::VelocityExtension(const SdfGridT &sdfGrid)
    : mSdfGrid(sdfGrid), mVelGrid(nullptr), mSolveVoxelCount(0), mInitVoxelCount(0)
{
    if ( !sdfGrid.hasUniformVoxels() ) {
        OPENVDB_THROW(RuntimeError,
            "The transform must have uniform scale for the VelocityExtension to function");
    }
    if ( sdfGrid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
            "VelocityExtension expected a level set, got a grid of class \""
            + sdfGrid.gridClassToString(sdfGrid.getGridClass())
            + "\" [hint: Grid::setGridClass(openvdb::GRID_LEVEL_SET)]");
    }
}

template <typename SdfGridT, typename VelValueT>
template <typename OpT>
void VelocityExtension<SdfGridT, VelValueT>::initDense(const OpT &op, const VelValueT &background)
{
    mSolveVoxelCount = mInitVoxelCount = 0;
    mVelGrid = createGrid<VelGridT>( background );
    mVelGrid->setTransform(mSdfGrid.transform().copy());
    mVelGrid->setGridClass(GRID_UNKNOWN);
    mVelGrid->topologyUnion( mSdfGrid );// very fast
    VelMgrT leafMgr(mVelGrid->tree(), 0);// we need zero auxiliary buffers

    auto kernel = [&](const typename VelMgrT::LeafRange& range) {
        const OpT func = op;// Important: create a deep copy since op(ijk) is not thread-safe
        size_t sum = 0;
        math::GradStencil<SdfGridT, false> stencil(mSdfGrid);// don't register accessor
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            sum += leafIter->valueMask().countOn();
            for (auto voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                stencil.moveTo(voxelIter.getCoord());
                auto grad = stencil.gradient();
                // Using gradient from cpt-projected point
                // auto valueCpt = func(stencil.cpt());
                // Simplification: Use gradient at current point
                auto valueAlt = func.fromGradient(grad);
                voxelIter.setValue(valueAlt);
            }// loop over active voxels in leaf
            mInitVoxelCount += sum;
        }// loop over leaf nodes in range
    };// kernel

    tbb::parallel_for(leafMgr.leafRange(32), kernel);//multi-threaded
}

template <typename SdfGridT, typename VelValueT>
template <typename OpT>
void VelocityExtension<SdfGridT, VelValueT>::initSparse(const OpT &op, const VelValueT &background)
{
    mSolveVoxelCount = mInitVoxelCount = 0;
    mVelGrid = createGrid<VelGridT>( background );
    mVelGrid->setTransform(mSdfGrid.transform().copy());
    mVelGrid->setGridClass(GRID_UNKNOWN);
    mVelGrid->topologyUnion( mSdfGrid );// very fast
    VelMgrT leafMgr(mVelGrid->tree(), 0);// we need zero auxiliary buffers
    const float dx = mSdfGrid.voxelSize()[0];

    auto kernel = [&](const typename VelMgrT::LeafRange& range) {
        const OpT func = op;// Important: create a deep copy since op(ijk) is not thread-safe
        size_t sum = 0;
        math::GradStencil<SdfGridT, false> stencil(mSdfGrid);// don't register accessor
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            const auto *sdfLeaf = mSdfGrid.tree().probeLeaf(leafIter->origin());
            if (sdfLeaf==nullptr) OPENVDB_THROW(ValueError, "VelocityExtension::initSparse: NULL leaf pointer");
            for (auto voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                if (math::Abs(sdfLeaf->getValue(voxelIter.pos())) < dx) {
                  stencil.moveTo(voxelIter.getCoord());
                  voxelIter.setValue(func(stencil.cpt()));
                  ++sum;
                }
            }// loop over active voxels in leaf
            mInitVoxelCount  += sum;
            mSolveVoxelCount += leafIter->valueMask().countOn() - sum;
        }// loop over leaf nodes in range
    };// kernel

    tbb::parallel_for(leafMgr.leafRange(32), kernel);//multi-threaded
}

template <typename SdfGridT, typename VelValueT>
void VelocityExtension<SdfGridT, VelValueT>::iterate(int nIter)
{
    OPENVDB_THROW(ValueError, "VelocityExtension::iterate has not been implemented yet!");
    // iteratively solve pseudo-time-dependent PDE to sweep out velocity values in
    // the remaining narrow-band
}// VelocityExtension::sweep
   
////////////////////////////////////////////////////////////////////////////////

template<typename SdfGridT, typename OpT, typename VelValueT>
typename SdfGridT::template ValueConverter<VelValueT>::Type::Ptr
velExt(const SdfGridT &sdfGrid,
         const OpT &op,
         const VelValueT &background,
         int nIter)
{
  VelocityExtension<SdfGridT, VelValueT> tmp(sdfGrid);
  if (nIter > 0) {
    tmp.initSparse(op, background);
    tmp.iterate(nIter);
  } else {
    tmp.initDense(op, background);
  }
  return tmp.velGrid();
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VELOCITY_EXTENSION_HAS_BEEN_INCLUDED
