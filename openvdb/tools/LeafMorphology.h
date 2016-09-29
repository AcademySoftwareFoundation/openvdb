///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file   LeafMorphology.h
///
/// @author Fredrik Salomonsson <fredriks@d2.com>
///
/// @brief  Topology leaf dilation


#ifndef OPENVDB_TOOLS_LEAF_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEAF_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <vector>

#include <openvdb/math/Coord.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/ValueAccessor.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Topologically dilate all faces of the leafs in the given tree,
/// similar to what dilateVoxels(TreeType& tree, int count=1) does,
/// i.e. expanding the set of active voxels in the +x, -x, +y, -y,
/// +z and -z directions, but this work on one level up expanding
/// the set of active leafs
template< typename _TreeType >
void dilateLeafs6N( _TreeType& tree, const size_t iterations = 1 );

/// Topologically dilate all corners of leafs in the given tree, i.e
/// expanding the set of active leafs in the (-x,-y,-z), ( x,-y,-z),
/// (-x, y,-z), ( x, y,-z),(-x,-y, z) ,( x,-y, z) ,(-x, y, z) and
/// (x, y, z) directions.
template< typename _TreeType >
void dilateLeafs8N( _TreeType& tree, const size_t iterations = 1 );

/// Topologically dilate all edges of leafs in the given tree.
template< typename _TreeType >
void dilateLeafs12N( _TreeType& tree, const size_t iterations = 1 );

/// Topologically dilate all edges and faces of the leafs in the
/// given tree.
template< typename _TreeType >
void dilateLeafs18N( _TreeType& tree, const size_t iterations = 1 );

/// Topologically dilate of the leafs in the given tree in all directions.
template< typename _TreeType >
void dilateLeafs26N( _TreeType& tree, const size_t iterations = 1 );

template< typename _TreeType >
class LeafMorphology
{
    typedef openvdb::Coord                         Coord;
    typedef size_t                                 Size;
    typedef _TreeType                              TreeType;
    typedef typename TreeType::LeafNodeType        LeafType;
    typedef typename TreeType::LeafIter            LeafIterType;
    typedef openvdb::tree::ValueAccessor<TreeType> AccessorType;

private:
    static const int LEAF_DIM = LeafType::DIM;
    TreeType& my_tree;
    AccessorType my_access;

    std::vector<LeafType*> my_leafs; /** Array containing leafs to
                                         iterate over. */
    std::vector<LeafType*> my_next_leafs; /** Array containing leafs
                                              for the next
                                              iteration. */
public:
    LeafMorphology( TreeType& tree );
    /**
     * Dilate face connected neighbors
     */
    void dilate6N( const Size count = 1 );

    /**
     * Dilate corner connected neighbors
     */
    void dilate8N( const Size count = 1 );

    /**
     * Dilate edge connected neighbors
     */
    void dilate12N( const Size count = 1 );

    /**
     * Dilate face + edge connected neighbors
     */
    void dilate18N( const Size count = 1 );

    /**
     * Dilate face + edge + corner connected neighbors
     */
    void dilate26N( const Size count = 1 );

private:
    void init();
    template<int DX, int DY, int DZ>
    void allocate( const Coord &xyz );
};

template< typename _T >
LeafMorphology<_T>::LeafMorphology( TreeType& tree ) :
    my_tree( tree ),
    my_access( tree )
{}

template< typename _T >
template<int DX, int DY, int DZ>
void LeafMorphology<_T>::   
allocate( const Coord &xyz )
{
    const Coord orig = xyz.offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);

    LeafType* leaf = my_access.probeLeaf( orig );

    if (leaf == NULL ) {
        // Leaf not allocated, allocate it 
        leaf = my_access.touchLeaf( orig );
        // And activate all values
        leaf->setValuesOn();

        my_next_leafs.push_back( leaf );
    }
}
  
template< typename _T >
void LeafMorphology<_T>::   
init()
{
    const Size leaf_count = my_tree.leafCount();
    my_leafs.resize( leaf_count );
    LeafIterType iter = my_tree.beginLeaf();

    for (Size n = 0; n != leaf_count; ++n, ++iter) {
        my_leafs[ n ] = iter.getLeaf();
        my_leafs[ n ]->setValuesOn();
    }
}

template< typename _T >
void  LeafMorphology<_T>::   
dilate6N( const Size count )
{
    init();
    for( Size i = 0; i < count; ++i ) {
        my_next_leafs.clear();

        for( Size n = 0, size = my_leafs.size(); n < size; ++n ) {
            LeafType* leaf = my_leafs[n];
    
            const Coord orig = leaf->origin();

            allocate< 0, 0,-1 >( orig );
            allocate< 0, 0, 1 >( orig );

            allocate< 0,-1, 0 >( orig );
            allocate< 0, 1, 0 >( orig );

            allocate<-1, 0, 0 >( orig );
            allocate< 1, 0, 0 >( orig );
        }

        my_leafs.swap( my_next_leafs );
    }
}

template< typename _T >
void  LeafMorphology<_T>::   
dilate8N( const Size count )
{
    init();
    for( Size i = 0; i < count; ++i ) {
        my_next_leafs.clear();

        for( Size n = 0, size = my_leafs.size(); n < size; ++n ) {
            LeafType* leaf = my_leafs[n];
    
            const Coord orig = leaf->origin();

            allocate<-1,-1,-1 >( orig );
            allocate< 1,-1,-1 >( orig );
            allocate<-1, 1,-1 >( orig );
            allocate< 1, 1,-1 >( orig );

            allocate<-1,-1, 1 >( orig );
            allocate< 1,-1, 1 >( orig );
            allocate<-1, 1, 1 >( orig );
            allocate< 1, 1, 1 >( orig );
        }

        my_leafs.swap( my_next_leafs );
    }
}

template< typename _T >
void  LeafMorphology<_T>::   
dilate12N( const Size count )
{
    init();
    for( Size i = 0; i < count; ++i ) {
        my_next_leafs.clear();

        for( Size n = 0, size = my_leafs.size(); n < size; ++n ) {
            LeafType* leaf = my_leafs[n];
    
            const Coord orig = leaf->origin();

            allocate< 0,-1,-1 >( orig );
            allocate<-1, 0,-1 >( orig );
            allocate< 1, 0,-1 >( orig );
            allocate< 0, 1,-1 >( orig );

            allocate<-1,-1, 0 >( orig );
            allocate< 1,-1, 0 >( orig );
            allocate<-1, 1, 0 >( orig );
            allocate< 1, 1, 0 >( orig );

            allocate< 0,-1, 1 >( orig );
            allocate<-1, 0, 1 >( orig );
            allocate< 1, 0, 1 >( orig );
            allocate< 0, 1, 1 >( orig );
        }

        my_leafs.swap( my_next_leafs );
    }
}

template< typename _T >
void  LeafMorphology<_T>::   
dilate18N( const Size count )
{
    init();
    for( Size i = 0; i < count; ++i ) {
        my_next_leafs.clear();

        for( Size n = 0, size = my_leafs.size(); n < size; ++n ) {
            LeafType* leaf = my_leafs[n];
    
            const Coord orig = leaf->origin();

            allocate< 0,-1,-1 >( orig );
            allocate<-1, 0,-1 >( orig );
            allocate< 0, 0,-1 >( orig );
            allocate< 1, 0,-1 >( orig );
            allocate< 0, 1,-1 >( orig );

            allocate<-1,-1, 0 >( orig );
            allocate< 0,-1, 0 >( orig );
            allocate< 1,-1, 0 >( orig );
            allocate<-1, 0, 0 >( orig );
    
            allocate< 1, 0, 0 >( orig );
            allocate<-1, 1, 0 >( orig );
            allocate< 0, 1, 0 >( orig );
            allocate< 1, 1, 0 >( orig );

            allocate< 0,-1, 1 >( orig );
            allocate<-1, 0, 1 >( orig );
            allocate< 0, 0, 1 >( orig );
            allocate< 1, 0, 1 >( orig );
            allocate< 0, 1, 1 >( orig );
        }

        my_leafs.swap( my_next_leafs );
    }
}

template< typename _T >
void  LeafMorphology<_T>::   
dilate26N( const Size count )
{
    init();
    for( Size i = 0; i < count; ++i ) {
        my_next_leafs.clear();

        for( Size n = 0, size = my_leafs.size(); n < size; ++n ) {
            LeafType* leaf = my_leafs[n];
    
            const Coord orig = leaf->origin();

            allocate<-1,-1,-1 >( orig );
            allocate< 0,-1,-1 >( orig );
            allocate< 1,-1,-1 >( orig );
            allocate<-1, 0,-1 >( orig );
            allocate< 0, 0,-1 >( orig );
            allocate< 1, 0,-1 >( orig );
            allocate<-1, 1,-1 >( orig );
            allocate< 0, 1,-1 >( orig );
            allocate< 1, 1,-1 >( orig );

            allocate<-1,-1, 0 >( orig );
            allocate< 0,-1, 0 >( orig );
            allocate< 1,-1, 0 >( orig );
            allocate<-1, 0, 0 >( orig );
    
            allocate< 1, 0, 0 >( orig );
            allocate<-1, 1, 0 >( orig );
            allocate< 0, 1, 0 >( orig );
            allocate< 1, 1, 0 >( orig );

            allocate<-1,-1, 1 >( orig );
            allocate< 0,-1, 1 >( orig );
            allocate< 1,-1, 1 >( orig );
            allocate<-1, 0, 1 >( orig );
            allocate< 0, 0, 1 >( orig );
            allocate< 1, 0, 1 >( orig );
            allocate<-1, 1, 1 >( orig );
            allocate< 0, 1, 1 >( orig );
            allocate< 1, 1, 1 >( orig );
        }

        my_leafs.swap( my_next_leafs );
    }
}

template< typename _TreeType >
void dilateLeafs6N( _TreeType& tree, const size_t iterations = 1 )
{
    LeafMorphology<_TreeType>( tree ).dilate6N( iterations );
}

template< typename _TreeType >
void dilateLeafs8N( _TreeType& tree, const size_t iterations = 1 )
{
    LeafMorphology<_TreeType>( tree ).dilate8N( iterations );
}

template< typename _TreeType >
void dilateLeafs12N( _TreeType& tree, const size_t iterations = 1 )
{
    LeafMorphology<_TreeType>( tree ).dilate12N( iterations );
}

template< typename _TreeType >
void dilateLeafs18N( _TreeType& tree, const size_t iterations = 1 )
{
    LeafMorphology<_TreeType>( tree ).dilate18N( iterations );
}

template< typename _TreeType >
void dilateLeafs26N( _TreeType& tree, const size_t iterations = 1 )
{
    LeafMorphology<_TreeType>( tree ).dilate26N( iterations );
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif
