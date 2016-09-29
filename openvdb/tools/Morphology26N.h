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
///
/// @author Fredrik Salomonsson ( fredriks@d2.com)
/// 
/// @date   July 2013
/// 
/// @brief  26-neighborhood topology dilation and erosion
///

#ifndef OPENVDB_TOOLS_MORPHOLOGY_26N_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_26N_HAS_BEEN_INCLUDED

#include <openvdb/tools/Morphology.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {
//@{
/// Topologically dilate all leaf-level active voxels in the given
/// tree, similar to what dilateVoxels(TreeType& tree, int count=1)
/// does except this expands in all 26 directions.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template< typename TreeType>
inline void dilateVoxels26N( TreeType& tree, const int count );

template< typename TreeType>
inline void 
dilateVoxels26N( openvdb::tree::LeafManager<TreeType>& manager,
                 const int count );
//@}

//@{
/// Topologically erode all leaf-level active voxels in the given
/// tree, similar to what erodeVoxels(TreeType& tree, int count=1)
/// does except this shrinks the set of active voxels in all
/// directions.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template< typename TreeType>
inline void erodeVoxels26N( TreeType& tree, 
                            const int count,
                            const bool threaded = true );

template< typename TreeType>
inline void 
erodeVoxels26N( openvdb::tree::LeafManager<TreeType>& manager,
                const int count,
                const bool threaded = true );
//@}
template<typename TreeType>
class Morphology26N
    : private Morphology<TreeType>
{
public:
    typedef Morphology<TreeType>                 Parent;
    typedef typename Parent::ManagerType         ManagerType;
    typedef typename Parent::Word                Word;
    typedef typename Parent::Neighbor            Neighbor;
    typedef typename Parent::MaskManager         MaskManager;

    typedef typename TreeType::LeafNodeType      LeafType;
    typedef typename LeafType::NodeMaskType      MaskType;
    typedef tree::ValueAccessor<TreeType>        AccessorType;

private:
    using Parent::mManager;
    using Parent::mAcc;
    using Parent::LEAF_DIM;
    using Parent::LEAF_LOG2DIM;
    static const int LEAF_DIM_1 = LEAF_DIM - 1;

public:
    Morphology26N( TreeType& tree ): Parent( tree ) {}
    Morphology26N( ManagerType* mgr ): Parent( mgr ) {}
    void dilateVoxels26N();
    void dilateVoxels26N( const int count );

    void erodeVoxels26N( const int count = 1, const bool threaded = true  );

private:
    void dilate( const openvdb::Coord& origin, 
                 const int x, 
                 const int y, 
                 const int n,
                 Neighbor* NN, 
                 LeafType& leaf,
                 const Word oldWord );
    void erode();

    struct ErodeVoxels26nOp
    {
        ErodeVoxels26nOp( std::vector<MaskType>& masks, ManagerType& manager) : 
            mSavedMasks( masks ), 
            mManager( manager )
            {}

        void runParallel() { tbb::parallel_for( mManager.getRange(), *this); }
        void runSerial() { (*this)( mManager.getRange() ); }
        void operator()( const tbb::blocked_range<size_t>& range ) const;
        void safeErode( Word& w,
                        const int nOff,
                        Neighbor* NN, 
                        const openvdb::Coord& origin,
                        AccessorType& acc,
                        LeafType& leaf ) const;
        template< int DX, int DY >
        void edgeErode( Word& w,
                        const int n,
                        Neighbor* NN, 
                        const openvdb::Coord& origin,
                        AccessorType& acc ) const;

    private:
        std::vector<MaskType>& mSavedMasks;
        ManagerType& mManager;
    };
};

template<typename TreeType>
inline void Morphology26N<TreeType>::dilateVoxels26N( const int count )
{
    for( int i = 0; i < count; ++i)
        dilateVoxels26N();
}  

/// Extend this to include 8 and 18?
template<typename TreeType>
inline void Morphology26N<TreeType>::dilateVoxels26N()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = mManager->leafCount();

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    MaskManager masks(savedMasks, *mManager);
    masks.save();

    Neighbor NN[26];
    openvdb::Coord origin; 
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        // Original bit-mask of current leaf node.
        const MaskType& oldMask = savedMasks[leafIdx];
        // Current leaf node.
        LeafType& leaf = mManager->leaf(leafIdx);
        // Origin of the current leaf node.
        leaf.getOrigin(origin);
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {

                // Extract the portion of the original mask that corresponds
                // to a row in z.
                const Word oldWord = oldMask.template getWord<Word>(n);
                if (oldWord == 0) continue; // no active voxels
                
                // Take care of the negative z-direction.
                if( const Word w = oldWord<<(LEAF_DIM-1) ) {
                    NN[24].template scatter< 0, 0,-1>(mAcc, origin, n, w);
                     
                    const openvdb::Coord leftOrigin =
                        origin.offsetBy(0, 0, -LEAF_DIM); 
                     
                    if( NN[24].leaf )
                        dilate( leftOrigin, x, y, n, NN, *NN[24].leaf, w );
                }

                // Take care of the positive z-direction.
                if( const Word w = oldWord>>(LEAF_DIM-1) ) {
                    NN[25].template scatter< 0, 0, 1>(mAcc, origin, n, w);

                    const openvdb::Coord rightOrigin =
                        origin.offsetBy(0, 0, LEAF_DIM); 
                    // Offset NN to use the correct neighbor leafs
                    if( NN[25].leaf )
                        dilate( rightOrigin, x, y, n, NN + 8, *NN[25].leaf, w );
                }

                // Expand old word

                // Dilate the current leaf node in the z direction
                // by ORing its mask with itself shifted first
                // left and then right by one bit.
                const Word newWord = oldWord << 1 | oldWord | oldWord >> 1;

                leaf.getValueMask().template getWord<Word>(n) |= newWord;

                // Dilate the leaf in x,y direction.
                // Offset NN to use the correct neighbor leafs
                dilate( origin, x, y, n, NN + 16, leaf, newWord );
            }// loop over y 
        }//loop over x
        for (int i=0; i<26; ++i) NN[i].clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}

template<typename TreeType>
inline void Morphology26N<TreeType>::
dilate( const openvdb::Coord& origin, 
        const int x, 
        const int y, 
        const int n,
        Neighbor* NN,
        LeafType& leaf,
        const Word word)
{
    // dilate current leaf or neighbor in negative x-direction
    if (x > 0) {
        leaf.getValueMask().template getWord<Word>(n-LEAF_DIM) |= word;
    } else {
        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n, word);
    }
    // dilate current leaf or neighbor in positive x-direction
    if (x < LEAF_DIM_1) {
        leaf.getValueMask().template getWord<Word>(n+LEAF_DIM) |= word;
    } else {
        NN[1].template scatter< 1, 0, 0>(mAcc, origin, n, word);
    }
    // dilate current leaf or neighbor in negative y-direction
    if (y > 0) {
        leaf.getValueMask().template getWord<Word>(n-1) |= word;
    } else {
        NN[2].template scatter< 0,-1, 0>(mAcc, origin, n, word);
    }
    // dilate current leaf or neighbor in positive y-direction
    if (y < LEAF_DIM_1) {
        leaf.getValueMask().template getWord<Word>(n+1) |= word;
    } else {
        NN[3].template scatter< 0, 1, 0>(mAcc, origin, n, word);
    }
   
    // dilate current leaf or neighbor in negative xy-direction
    if (x > 0 && y > 0) {
        static const int off = -LEAF_DIM - 1;
        leaf.getValueMask().template getWord<Word>(n+off) |= word;
    } else if( x > 0 && y == 0 ) {
        NN[2].template scatter< 0,-1, 0>(mAcc, origin, n-LEAF_DIM, word);
    } else if( x == 0 && y > 0 ) {
        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n-1, word);
    } else {
        NN[4].template scatter<-1,-1, 0>(mAcc, origin, n, word);
    }

    // dilate current leaf or neighbor in positive xy-direction
    if ( (x < LEAF_DIM_1) && (y < LEAF_DIM_1) ) {
        static const int off = LEAF_DIM + 1;
        leaf.getValueMask().template getWord<Word>(n+off) |= word;
    } else if ( (x < LEAF_DIM_1) && (y == LEAF_DIM_1) ) {
        NN[3].template scatter< 0, 1, 0>(mAcc, origin, n+LEAF_DIM, word);
    } else if ( (x == LEAF_DIM_1) && (y < LEAF_DIM_1) ) {
        NN[1].template scatter< 1, 0, 0>(mAcc, origin, n+1, word);
    } else {
        NN[5].template scatter< 1, 1, 0>(mAcc, origin, n, word);
    }
    // dilate current leaf or neighbor in positive x-direction and
    // negative y-direction
    if ((x < LEAF_DIM_1) && y > 0 ) {
        static const int off = LEAF_DIM_1;
        leaf.getValueMask().template getWord<Word>(n+off) |= word;
    } else if ((x < LEAF_DIM_1) && y == 0 ) {
        NN[2].template scatter< 0,-1, 0>(mAcc, origin, n+LEAF_DIM, word);
    } else if ((x == LEAF_DIM_1) && y > 0 ) {
        NN[1].template scatter< 1, 0, 0>(mAcc, origin, n-1, word);
    } else {
        NN[6].template scatter< 1,-1, 0>(mAcc, origin, n, word);
    }
    // dilate current leaf or neighbor in negative x-direction and
    // positive y-direction
    if ( x > 0 && (y < LEAF_DIM_1) ) {
        static const int off = -LEAF_DIM_1;
        leaf.getValueMask().template getWord<Word>(n+off) |= word;
    } else if ( x > 0 && (y == LEAF_DIM_1) ) {
        NN[3].template scatter< 0, 1, 0>(mAcc, origin, n-LEAF_DIM, word);
    } else if ( x == 0 && (y < LEAF_DIM_1) ) {
        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n+1, word);
    } else {
        NN[7].template scatter<-1, 1, 0>(mAcc, origin, n, word);
    }
}

template<typename TreeType>
inline void Morphology26N<TreeType>::
erodeVoxels26N( const int count, const bool threaded )
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = mManager->leafCount();

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks( leafCount );
    MaskManager masks( savedMasks, *mManager);
    masks.save();

    void (ErodeVoxels26nOp::*run)() = threaded ? &ErodeVoxels26nOp::runParallel 
        : &ErodeVoxels26nOp::runSerial;

    ErodeVoxels26nOp erode( savedMasks, *mManager );
    for (int i = 0; i < count; ++i) {
        (erode.*run)();
        masks.update();
    }

    tools::pruneLevelSet(mManager->tree());
}

template<typename TreeType>
inline void Morphology26N<TreeType>::
ErodeVoxels26nOp::safeErode( Word& w,
                             const int n,
                             Neighbor* NN, 
                             const openvdb::Coord& origin,
                             AccessorType& acc,
                             LeafType& leaf ) const
{
    const Word neighbor = leaf.getValueMask().template getWord<Word>( n );
    const Word leftNeighbor = NN[24].template gather<0, 0,-1>(acc, origin, n);
    const Word rightNeighbor = NN[25].template gather<0, 0, 1>(acc, origin, n);

    w &= neighbor &
        ( ( neighbor << 1 ) | ( leftNeighbor  >> LEAF_DIM_1 ) ) 
        &
        ( ( neighbor >> 1 ) | ( rightNeighbor << LEAF_DIM_1 ) );
}

template<typename TreeType>
template< int DX, int DY >
inline void Morphology26N<TreeType>::ErodeVoxels26nOp::
edgeErode( Word& w,
           const int n,
           Neighbor* NN, 
           const openvdb::Coord& origin,
           AccessorType& acc ) const
{
    const Word neighbor = NN[0].template gather<DX, DY, 0>(acc, origin, n);
    const Word leftNeighbor = NN[8].template gather<DX, DY,-1>(acc, origin, n);
    const Word rightNeighbor = NN[16].template gather<DX, DY, 1>(acc, origin, n);

    w &= neighbor &
        ( ( neighbor << 1 ) | ( leftNeighbor  >> LEAF_DIM_1 ) ) 
        &
        ( ( neighbor >> 1 ) | ( rightNeighbor << LEAF_DIM_1 ) );
}
  
template<typename TreeType>
inline void Morphology26N<TreeType>::
ErodeVoxels26nOp::operator()( const tbb::blocked_range<size_t>& range ) const
{
    AccessorType acc(mManager.tree());
    Neighbor NN[26];
    Coord origin;

    for (size_t leafIdx = range.begin(); leafIdx < range.end(); ++leafIdx) 
    {
        LeafType& leaf = mManager.leaf( leafIdx );//current leaf node
        if (leaf.isEmpty()) continue;

        // Original bit-mask of current leaf node
        MaskType& newMask = mSavedMasks[ leafIdx ];
        // Origin of the current leaf node.
        leaf.getOrigin( origin );
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that
                // corresponds to a row in z.
                Word& w = newMask.template getWord<Word>(n); 
                if (w == 0) continue; // no active voxels

                // Erode in two z directions (this is first since
                // it uses the original w)
                w &= (w<<1 | (NN[24].template gather<0,0,-1>(acc, origin, n)
                              >>LEAF_DIM_1)) &
                    (w>>1 | (NN[25].template gather<0,0, 1>(acc, origin, n)
                             <<LEAF_DIM_1) );

                // Erode current leaf or neighbor in negative x-direction
                if( x > 0 )
                    safeErode       ( w, n-LEAF_DIM,   NN, origin, acc, leaf );
                else
                    edgeErode<-1, 0>( w, n,            NN, origin, acc );
                // Add checks if w is emtpy
                // Erode current leaf or neighbor in positive x-direction
                if( x < LEAF_DIM_1 )
                    safeErode       ( w, n+LEAF_DIM,   NN,   origin, acc, leaf );
                else
                    edgeErode< 1, 0>( w, n,            NN+1, origin, acc );

                // Erode current leaf or neighbor in negative y-direction
                if( y > 0 )
                    safeErode       ( w, n-1,          NN,   origin, acc, leaf );
                else
                    edgeErode< 0,-1>( w, n,            NN+2, origin, acc );
  
                // Erode current leaf or neighbor in positive y-direction
                if( y < LEAF_DIM_1 )
                    safeErode       ( w, n+1,          NN,   origin, acc, leaf );
                else
                    edgeErode< 0, 1>( w, n,            NN+3, origin, acc );

                // Erode current leaf or neighbor in negative xy-direction
                if (x > 0 && y > 0)
                    safeErode       ( w, n-LEAF_DIM-1, NN,   origin, acc, leaf );
                else if( x > 0 && y == 0 )
                    edgeErode< 0,-1>( w, n-LEAF_DIM,   NN+2, origin, acc );
                else if( x == 0 && y > 0 )
                    edgeErode<-1, 0>( w, n-1,          NN,   origin, acc );
                else 
                    edgeErode<-1,-1>( w, n,            NN+4, origin, acc );

                // Erode current leaf or neighbor in positive xy-direction
                if ( x < LEAF_DIM_1 && y < LEAF_DIM_1 )
                    safeErode       ( w, n+LEAF_DIM+1, NN,   origin, acc, leaf );
                else if ( (x < LEAF_DIM_1) && (y == LEAF_DIM_1) )
                    edgeErode< 0, 1>( w, n+LEAF_DIM,   NN+3, origin, acc );
                else if ( (x == LEAF_DIM_1) && (y < LEAF_DIM_1) )
                    edgeErode< 1, 0>( w, n+1,          NN+1, origin, acc );
                else
                    edgeErode< 1, 1>( w, n,            NN+5, origin, acc );

                // Erode current leaf or neighbor in positive x-direction and
                // negative y-direction
                if (x < LEAF_DIM_1 && y > 0 )
                    safeErode( w, n+LEAF_DIM_1,        NN,   origin, acc, leaf );
                else if ((x < LEAF_DIM_1) && y == 0 )
                    edgeErode< 0,-1>( w, n+LEAF_DIM,   NN+2, origin, acc );
                else if ((x == LEAF_DIM_1) && y > 0 )
                    edgeErode< 1, 0>( w, n-1,          NN+1, origin, acc );
                else
                    edgeErode< 1,-1>( w, n,            NN+6, origin, acc );

                // Erode current leaf or neighbor in negative x-direction and
                // positive y-direction
                if ( x > 0 && y < LEAF_DIM_1 )
                    safeErode       ( w, n-LEAF_DIM+1, NN,   origin, acc, leaf );
                else if ( x > 0 && (y == LEAF_DIM_1) )
                    edgeErode< 0, 1>( w, n-LEAF_DIM,   NN+3, origin, acc );
                else if ( x == 0 && (y < LEAF_DIM_1) )
                    edgeErode<-1, 0>( w, n+1,          NN,   origin, acc );
                else
                    edgeErode<-1, 1>( w, n,            NN+7, origin, acc );

            }// loop over y
        }//loop over x
        for (int i=0; i<26; ++i) NN[i].clear();
    }//loop over leafs
}

template< typename TreeType>
inline void dilateVoxels26N( TreeType& tree, const int count )
{
    Morphology26N<TreeType> m(tree);
    m.dilateVoxels26N( count );
}

template< typename TreeType>
inline void 
dilateVoxels26N( openvdb::tree::LeafManager<TreeType>& manager,
                 const int count )
{
    Morphology26N<TreeType> m( manager );
    m.dilateVoxels26N( count );
}

template< typename TreeType>
inline void erodeVoxels26N( TreeType& tree, 
                            const int count,
                            const bool threaded = true )
{
    Morphology26N<TreeType> m(tree);
    m.erodeVoxels26N( count, threaded );
}

template< typename TreeType>
inline void 
erodeVoxels26N( openvdb::tree::LeafManager<TreeType>& manager,
                const int count,
                const bool threaded = true )
{
    Morphology26N<TreeType> m( manager );
    m.erodeVoxels26N( count, threaded );
}
  
} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif
