///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
//
/// @file    MaskToLevelSet.h
///
/// @brief   This tool generated a narrob band level set from the
///          interface between the active and inactive voxels of an
///          input grid.
///
/// @par Example:
/// Combine with @c tools::createPointMaskGrid for fast point cloud to level set conversion.
///
/// @author FX R&D OpenVDB team

#ifndef OPENVDB_TOOLS_MASK_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MASK_TO_LEVELSET_HAS_BEEN_INCLUDED

// Keep this undefined - the corresponding method appears to generate
// artifacts when used on very narrow bands, i.e. 2 x one voxel width
//#define EXPERIMENTAL_METHOD

#if __cplusplus <= 201103L
#include <tbb/task_group.h>
#else
#include <thread>
#endif
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/math/Math.h> // for isNegative
#include <openvdb/tree/LeafManager.h>
#include "LevelSetFilter.h"
#include "Morphology.h" // for erodeVoxels and dilateVoxels
#include "SignedFloodFill.h" // for signedFloodFill

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Compute the narrow band level set to the interface
/// between active and inactive voxels in the input grid.
///
/// @return a shared pointer to a new narrow band level set (SDF) of type @c float
///
/// @param grid       an incoming grid of arbitrary data type whose active voxels
///                   are used in constructing the narrow band level set
/// @param halfWidth  half of the narrow band level set width in voxel units
/// @param dilation   amount of dilation in voxel units
/// @param erosion    amount of erosion in voxel units
/// @param interrupt  optional interrupter (see NullInterrupter.h)    
///
template<typename GridT,
         math::BiasedGradientScheme Scheme,
         typename InterrupterT>
inline typename Grid<typename GridT::TreeType::template ValueConverter<float>::Type>::Ptr
maskToLevelSet(const GridT& grid,
               int halfWidth = 3,
               int dilation = 1,
               int erosion = 1,
               InterrupterT* interrupt = NULL);       
    
/// @brief Compute the narrow band level set to the interface
/// between active and inactive voxels in the input grid.
///
/// @return a shared pointer to a new narrow band level set (SDF) of type @c float
///
/// @param grid       an incoming grid of arbitrary data type whose active voxels
///                   are used in constructing the narrow band level set
/// @param halfWidth  half of the narrow band level set width in voxel units
/// @param dilation   amount of dilation in voxel units
/// @param erosion    amount of erosion in voxel units
///
/// @note This template specialized version uses first order upwinding and no interrupter.
/// C++03 compilers do not allow functions to have default template arguments -
/// hence this workaround.
///    
template<typename GridT>
inline typename Grid<typename GridT::TreeType::template ValueConverter<float>::Type>::Ptr
maskToLevelSet(const GridT& grid,
               int halfWidth = 3,
               int dilation = 1,
               int erosion = 1)
{
    typedef util::NullInterrupter T;
    return maskToLevelSet<GridT, math::FIRST_BIAS, T>(grid, halfWidth, dilation, erosion);
}

  

#if __cplusplus <= 201103L
namespace {

template<typename TreeT>
struct DilateOp
{
    DilateOp(TreeT& t, int n) : tree(&t), size(n) {}
    void operator()() const { dilateVoxels( *tree, size); }
    TreeT* tree;
    const int size;
};


template<typename TreeT>
struct ErodeOp
{
    ErodeOp(TreeT& t, int n) : tree(&t), size(n) {}
    void operator()() const { erodeVoxels( *tree, size); }
    TreeT* tree;
    const int size;
};        
  
}// unnamed namespace    
#endif
    

template<typename GridT, math::BiasedGradientScheme Scheme, typename InterrupterT>
inline typename Grid<typename GridT::TreeType::template ValueConverter<float>::Type>::Ptr
maskToLevelSet(const GridT& grid, int halfWidth, int dilation, int erosion, InterrupterT* interrupt)
{
    typedef typename GridT::TreeType::template ValueConverter<ValueMask>::Type MaskTreeT;
    typedef typename GridT::TreeType::template ValueConverter<float>::Type     FloatTreeT;
    typedef Grid<FloatTreeT>                                                   FloatGridT;

    // Check input.
    if ( halfWidth <= 0 ) {
        OPENVDB_THROW(ValueError, "Narrow band width must be non-zero!");
    }
    if ( !grid.hasUniformVoxels() ) {
        OPENVDB_THROW(ValueError, "Non-uniform voxels are not supported!");
    }

    // background value = outside value
    const float outside = static_cast<float>(grid.voxelSize()[0]) * halfWidth;
    
    // Copy the topology into a MaskGrid.
    MaskTreeT maskTree( grid.tree(), false/*background*/, openvdb::TopologyCopy() );

    // Morphological closing operation.
    dilateVoxels( maskTree, dilation);
    erodeVoxels(  maskTree, erosion);

#ifdef EXPERIMENTAL_METHOD    
    // Construct mask of the interior narrow band
    MaskTreeT coreMask( maskTree );//deep copy
    erodeVoxels( coreMask, halfWidth );
    maskTree.topologyDifference( coreMask );
#endif

    // Generate a volume with an implicit zero crossing at the boundary
    // between active and inactive values in the input grid.
    typename FloatTreeT::Ptr lsTree( new FloatTreeT(maskTree,
                                                    outside,// = inactive
                                                    -outside,//= active
                                                    openvdb::TopologyCopy()) );
    
#ifdef EXPERIMENTAL_METHOD
    // Construct mask of the exterior narrow band 
    dilateVoxels( maskTree, halfWidth );
    maskTree.topologyDifference( coreMask );

    // Activate outside band
    lsTree->topologyUnion( maskTree );

    // Propagate the correct sign into the interior of the level set
    signedFloodFill( *lsTree );
#else
#if __cplusplus <= 201103L
    tbb::task_group pool;
    pool.run( ErodeOp< MaskTreeT >( maskTree, halfWidth ) );
    pool.run( DilateOp<FloatTreeT>( *lsTree , halfWidth ) );
    pool.wait();// wait for both tasks to complete
#else
    std::thread erode( [&maskTree, &halfWidth](){ erodeVoxels( maskTree, halfWidth); });
    std::thread dilate([&lsTree,   &halfWidth](){ dilateVoxels( *lsTree, halfWidth); });
    erode.join(); dilate.join();
#endif
    lsTree->topologyDifference( maskTree );
#endif
    
    // Create a level set grid from the tree
    typename FloatGridT::Ptr lsGrid = FloatGridT::create( lsTree );
    lsGrid->setTransform( grid.transform().copy() );
    lsGrid->setGridClass( openvdb::GRID_LEVEL_SET );

    // Normalize and prune level set
    LevelSetTracker<FloatGridT, InterrupterT> tracker( *lsGrid, interrupt );
    tracker.setSpatialScheme( Scheme );
    tracker.setNormCount( 3 * halfWidth );
    tracker.normalize();
    tracker.prune();
    
    return lsGrid;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVDB_TOOLS_MASK_TO_LEVELSET_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
