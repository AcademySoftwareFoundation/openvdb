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

#ifndef OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for isApproxEqual()
#include <openvdb/tree/TreeIterator.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>
#include <boost/scoped_array.hpp>
#include "ValueTransformer.h" // for foreach()

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Voxel topology of nearest neighbors
/// @details
/// <dl>
/// <dt><b>NN_FACE</b>
/// <dd>face adjacency (6 nearest neighbors, defined as all neighbor
/// voxels connected along one of the primary axes)
///
/// <dt><b>NN_FACE_EDGE</b>
/// <dd>face and edge adjacency (18 nearest neighbors, defined as all
/// neighbor voxels connected along either one or two of the primary axes)
///
/// <dt><b>NN_FACE_EDGE_VERTEX</b>
/// <dd>face, edge and vertex adjacency (26 nearest neighbors, defined
/// as all neighbor voxels connected along either one, two or all
/// three of the primary axes)
/// </dl>
enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };


/// @brief Topologically dilate all leaf-level active voxels in a tree
/// using one of three nearest neighbor connectivity patterns.
///
/// @param tree          tree to be dilated
/// @param iterations    number of iterations to apply the dilation
/// @param nn            connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
///
/// @note The values of any voxels are unchanged.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(TreeType& tree,
                         int iterations = 1,
                         NearestNeighbors nn = NN_FACE);

/// @brief Topologically dilate all leaf-level active voxels in a tree
/// using one of three nearest neighbor connectivity patterns.
///
/// @param manager       LeafManager containing the tree to be dilated.
/// @param iterations    number of iterations to apply the dilation
/// @param nn           connectivity pattern of the dilation: either
///     face-adjacent (6 nearest neighbors), face- and edge-adjacent
///     (18 nearest neighbors) or face-, edge- and vertex-adjacent (26
///     nearest neighbors).
///
/// @note The values of any voxels are unchanged.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(tree::LeafManager<TreeType>& manager,
                         int iterations = 1,
                         NearestNeighbors nn = NN_FACE);


//@{
/// @brief Topologically erode all leaf-level active voxels in the given tree.
/// @details That is, shrink the set of active voxels by @a iterations voxels
/// in the +x, -x, +y, -y, +z and -z directions, but don't change the values
/// of any voxels, only their active states.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(TreeType& tree, int iterations=1);

template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(tree::LeafManager<TreeType>& manager, int iterations = 1);
//@}


/// @brief Mark as active any inactive tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void activate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


/// @brief Mark as inactive any active tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void deactivate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


////////////////////////////////////////


/// Mapping from a Log2Dim to a data type of size 2^Log2Dim bits
template<Index Log2Dim> struct DimToWord {};
template<> struct DimToWord<3> { typedef uint8_t Type; };
template<> struct DimToWord<4> { typedef uint16_t Type; };
template<> struct DimToWord<5> { typedef uint32_t Type; };
template<> struct DimToWord<6> { typedef uint64_t Type; };


////////////////////////////////////////


template<typename TreeType>
class Morphology
{
public:
    typedef tree::LeafManager<TreeType> ManagerType;

    Morphology(TreeType& tree):
        mOwnsManager(true), mManager(new ManagerType(tree)), mAcc(tree), mSteps(1) {}
    Morphology(ManagerType* mgr):
        mOwnsManager(false), mManager(mgr), mAcc(mgr->tree()), mSteps(1) {}
    virtual ~Morphology() { if (mOwnsManager) delete mManager; }
    /// @brief Face-adjacent dilation pattern
    void dilateVoxels6();
    /// @brief Face- and edge-adjacent dilation pattern.
    void dilateVoxels18();
    /// @brief Face-, edge- and vertex-adjacent dilation pattern.
    void dilateVoxels26();
    void dilateVoxels(int iterations = 1, NearestNeighbors nn = NN_FACE);
    /// @brief Face-adjacent erosion pattern.
    inline void erodeVoxels(int iterations = 1) { mSteps = iterations; this->doErosion(); }

private:
    void doErosion();

    typedef typename TreeType::LeafNodeType LeafType;
    typedef typename LeafType::NodeMaskType MaskType;
    typedef tree::ValueAccessor<TreeType>   AccessorType;

    const bool   mOwnsManager;
    ManagerType* mManager;
    AccessorType mAcc;
    int mSteps;

    static const int LEAF_DIM     = LeafType::DIM;
    static const int END          = LEAF_DIM - 1;
    static const int LEAF_LOG2DIM = LeafType::LOG2DIM;
    typedef typename DimToWord<LEAF_LOG2DIM>::Type Word;

    struct Neighbor {
        LeafType* leaf;//null if a tile
        bool      init;//true if initialization is required
        bool      isOn;//true if an active tile
        Neighbor() : leaf(NULL), init(true) {}
        inline void clear() { init = true; }
        template<int DX, int DY, int DZ>
        void scatter(AccessorType& acc, const Coord &xyz, int indx, Word oldWord)
        {
            if (init) {
                init = false;
                Coord orig = xyz.offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leaf = acc.probeLeaf(orig);
                if (leaf==NULL && !acc.isValueOn(orig)) leaf = acc.touchLeaf(orig);
            }
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
            static
#endif
            const int N = (LEAF_DIM - 1)*(DY + DX*LEAF_DIM);
            if (leaf) leaf->getValueMask().template getWord<Word>(indx-N) |= oldWord;
        }
        template<int DX, int DY, int DZ>
        Word gather(AccessorType& acc, const Coord &xyz, int indx)
        {
            if (init) {
                init = false;
                Coord orig = xyz.offsetBy(DX*LEAF_DIM, DY*LEAF_DIM, DZ*LEAF_DIM);
                leaf = acc.probeLeaf(orig);
                isOn = leaf ? false : acc.isValueOn(orig);
            }
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
            static
#endif
            const int N = (LEAF_DIM -1 )*(DY + DX*LEAF_DIM);
            return leaf ? leaf->getValueMask().template getWord<Word>(indx-N)
                : isOn ? ~Word(0) : Word(0);
        }
    };// Neighbor


    struct ErodeVoxelsOp {
        ErodeVoxelsOp(std::vector<MaskType>& masks, ManagerType& manager)
            : mSavedMasks(masks) , mManager(manager) {}

        void runParallel() { tbb::parallel_for(mManager.getRange(), *this); }
        void operator()(const tbb::blocked_range<size_t>& range) const;

    private:
        std::vector<MaskType>& mSavedMasks;
        ManagerType& mManager;
    };// ErodeVoxelsOp


    struct MaskManager {
        MaskManager(std::vector<MaskType>& masks, ManagerType& manager)
            : mMasks(masks) , mManager(manager), mSaveMasks(true) {}

        void save() { mSaveMasks = true; tbb::parallel_for(mManager.getRange(), *this); }
        void update() { mSaveMasks = false; tbb::parallel_for(mManager.getRange(), *this); }
        void operator()(const tbb::blocked_range<size_t>& range) const
        {
            if (mSaveMasks) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    mMasks[i] = mManager.leaf(i).getValueMask();
                }
            } else {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    mManager.leaf(i).setValueMask(mMasks[i]);
                }
            }
        }

    private:
        std::vector<MaskType>& mMasks;
        ManagerType& mManager;
        bool mSaveMasks;
    };// MaskManager
};

template<typename TreeType>
void
Morphology<TreeType>::dilateVoxels(int iterations, NearestNeighbors nn)
{
    for (int i=0; i<iterations; ++i) {
        switch (nn) {
        case NN_FACE_EDGE: this->dilateVoxels18(); break;
        case NN_FACE_EDGE_VERTEX: this->dilateVoxels26(); break;
        default: this->dilateVoxels6();
        }
    }
}

// Classification of adjacency patterns
// Face-adjacent cases: 0-5
// Case 0:  (-1, 0, 0)
// Case 1:  ( 1, 0, 0)
// Case 2:  ( 0,-1, 0)
// Case 3:  ( 0, 1, 0)
// Case 4:  ( 0, 0,-1)
// Case 5:  ( 0, 0, 1)
//
// Edge-adjacent cases: 6-17
// Case 6:  (-1,-1, 0)
// Case 7:  (-1, 1, 0)
// Case 8:  (-1, 0,-1)
// Case 9:  (-1, 0, 1)
// Case 10: ( 1,-1, 0)
// Case 11: ( 1, 1, 0)
// Case 12: ( 1, 0,-1)
// Case 13: ( 1, 0, 1)
// Case 14: ( 0,-1,-1)
// Case 15: ( 0,-1, 1)
// Case 16: ( 0, 1,-1)
// Case 17: ( 0, 1, 1)
//
// Vertex-adjacent cases: 18-25
// Case 18: (-1,-1,-1)
// Case 19: (-1,-1, 1)
// Case 20: (-1, 1,-1)
// Case 21: (-1, 1, 1)
// Case 22: ( 1,-1,-1)
// Case 23: ( 1,-1, 1)
// Case 24: ( 1, 1,-1)
// Case 25: ( 1, 1, 1)

template<typename TreeType>
void
Morphology<TreeType>::dilateVoxels6()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    MaskManager masks(savedMasks, *mManager);
    masks.save();
    Neighbor NN[6];
    Coord origin;
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        LeafType& leaf = mManager->leaf(leafIdx);//current leaf node
        leaf.getOrigin(origin);// origin of the current leaf node.
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                const Word oldWord = oldMask.template getWord<Word>(n);
                if (oldWord == 0) continue; // no active voxels

                const Word mask = Word(oldWord | (oldWord >> 1) | (oldWord << 1));

                // Dilate the current leaf node in the z direction by ORing its mask
                // with itself shifted first left and then right by one bit.
                leaf.getValueMask().template getWord<Word>(n) |= mask;

                // -x, cases: 0
                if (x > 0) {
                    leaf.getValueMask().template getWord<Word>(n-LEAF_DIM) |= oldWord;//  #0: (-1,0,0)
                } else {
                    NN[0].template scatter<-1, 0, 0>(mAcc, origin, n, oldWord);//         #0: (-1,0,0)
                }
                // +x, cases: 1
                if (x < END) {
                    leaf.getValueMask().template getWord<Word>(n+LEAF_DIM) |= oldWord;//  #1: (+1,0,0)
                } else {
                    NN[1].template scatter< 1, 0, 0>(mAcc, origin, n, oldWord);//         #1: (+1,0,0)
                }
                // -y, cases: 2
                if (y > 0) {
                    leaf.getValueMask().template getWord<Word>(n-1) |= oldWord;//         #2: (0,-1,0)
                } else {
                    NN[2].template scatter< 0,-1, 0>(mAcc, origin, n, oldWord);//         #2: (0,-1,0)
                }
                // +y, cases: 3
                if (y < END)  {
                    leaf.getValueMask().template getWord<Word>(n+1) |= oldWord;//         #3: (0,+1,0)
                } else {
                    NN[3].template scatter< 0, 1, 0>(mAcc, origin, n, oldWord);//         #3: (0,+1,0)
                }
                // -z, cases: 4
                if (static_cast<Word>(oldWord<<END)) {
                    NN[4].template scatter< 0, 0,-1>(mAcc, origin, n, 1<<END);//          #4: (0,0,-1)
                }
                // +z: cases: 5
                if (oldWord>>END) NN[5].template scatter< 0, 0, 1>(mAcc, origin, n, 1);// #5: (0,0,+1)
            }// loop over y
        }//loop over x
        for (int i=0; i<6; ++i) NN[i].clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}//dilateVoxels6

template<typename TreeType>
void
Morphology<TreeType>::dilateVoxels18()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    MaskManager masks(savedMasks, *mManager);
    masks.save();
    Neighbor NN[18];
    Coord origin;
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        LeafType& leaf = mManager->leaf(leafIdx);//current leaf node
        leaf.getOrigin(origin);// origin of the current leaf node.
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                const Word oldWord = oldMask.template getWord<Word>(n);
                if (oldWord == 0) continue; // no active voxels
                const Word mask = Word(oldWord | (oldWord >> 1) | (oldWord << 1));

                // Dilate the current leaf node in the z direction by ORing its mask
                // with itself shifted first left and then right by one bit.
                leaf.getValueMask().template getWord<Word>(n) |= mask;

                // -x, cases: 0, 6, 7
                if (x > 0) {
                    const Index m = n - LEAF_DIM;//-x
                    leaf.getValueMask().template getWord<Word>(m) |= mask;//            #0: (-1,0,0)
                    if (y > 0) {
                        leaf.getValueMask().template getWord<Word>(m-1) |= oldWord;//   #6: (-1,-1,0)
                    } else {//y=0
                        NN[2].template scatter< 0,-1, 0>(mAcc, origin, m, oldWord);//   #6: (-1,-1,0)
                    }
                    if (y < END) {
                        leaf.getValueMask().template getWord<Word>(m+1) |= oldWord;//   #7: (-1,+1,0)
                    } else {//y=END
                        NN[3].template scatter< 0, 1, 0>(mAcc, origin, m, oldWord);//   #7: (-1,+1,0)
                    }
                } else {//x=0
                    NN[0].template scatter<-1, 0, 0>(mAcc, origin, n, mask);//          #0: (-1,0,0)
                    if (y > 0) {
                        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n-1, oldWord);// #6: (-1,-1,0)
                    } else {//y=0
                        NN[6].template scatter<-1,-1, 0>(mAcc, origin, n, oldWord);//   #6: (-1,-1,0)
                    }
                    if (y < END) {
                        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n+1, oldWord);// #7: (-1,+1,0)
                    } else {//y=END
                        NN[7].template scatter<-1, 1, 0>(mAcc, origin, n, oldWord);//   #7: (-1,+1,0)
                    }
                }

                // +x, cases: 1, 10, 11
                if (x < END) {
                    const Index m = n + LEAF_DIM;//+x
                    leaf.getValueMask().template getWord<Word>(m) |= mask;//           #1 : (+1, 0, 0)
                    if (y > 0) {
                        leaf.getValueMask().template getWord<Word>(m-1) |= oldWord;//  #10: (+1,-1,0)
                    } else {//y=0
                        NN[2].template scatter<0,-1, 0>(mAcc, origin, m, oldWord);//   #10: (+1,-1,0)
                    }
                    if (y < END) {
                        leaf.getValueMask().template getWord<Word>(m+1) |= oldWord;//  #11: (+1,+1,0)
                    } else {//y=END
                         NN[3].template scatter<0, 1, 0>(mAcc, origin, m, oldWord);//  #11: (+1,+1,0)
                    }
                } else {
                    NN[1].template scatter< 1, 0, 0>(mAcc, origin, n, mask);//          #1 : (+1,0,0)
                    if (y>0) {
                        NN[ 1].template scatter<1, 0, 0>(mAcc, origin, n-1, oldWord);// #10: (+1,-1,0)
                    } else {//y=0
                        NN[10].template scatter<1,-1, 0>(mAcc, origin, n, oldWord);//   #10: (+1,-1,0)
                    }
                    if (y<END) {
                        NN[ 1].template scatter<1, 0, 0>(mAcc, origin, n+1, oldWord);// #11: (+1,+1,0)
                    } else {//y=END
                        NN[11].template scatter<1, 1, 0>(mAcc, origin, n, oldWord);//   #11: (+1,+1,0)
                    }
                }

                // -y, cases: 2
                if (y > 0) {
                    leaf.getValueMask().template getWord<Word>(n-1) |= mask;// #2: (0,-1,0)
                } else {
                    NN[2].template scatter< 0,-1, 0>(mAcc, origin, n, mask);// #2: (0,-1,0)
                }

                // +y, cases: 3
                if (y < END)  {
                    leaf.getValueMask().template getWord<Word>(n+1) |= mask;// #3: (0,+1,0)
                } else {
                    NN[3].template scatter< 0, 1, 0>(mAcc, origin, n, mask);// #3: (0,+1,0)
                }

                // -z, cases: 4, 8, 12, 14, 16
                if (static_cast<Word>(oldWord << END)) {
                    NN[4].template scatter< 0, 0,-1>(mAcc, origin, n, 1<<END); //             #4:  (0,0,-1)
                    if (x > 0) {
                        NN[4].template scatter< 0, 0,-1>(mAcc, origin, n-LEAF_DIM, 1<<END);// #8:  (-1,0,-1)
                    } else {
                        NN[8].template scatter<-1, 0,-1>(mAcc, origin, n, 1<<END);//          #8:  (-1,0,-1)
                    }
                    if (x < END) {
                        NN[4].template scatter< 0, 0,-1>(mAcc, origin, n+LEAF_DIM, 1<<END);// #12: (+1,0-1)
                    } else {
                        NN[12].template scatter< 1, 0,-1>(mAcc, origin, n, 1<<END);//         #12: (+1,0-1)
                    }
                    if (y > 0) {
                        NN[4].template scatter< 0, 0,-1>(mAcc, origin, n-1, 1<<END);//        #14: (0,-1,-1)
                    } else {
                        NN[14].template scatter<0,-1,-1>(mAcc, origin, n, 1<<END);//          #14: (0,-1,-1)
                    }
                    if (y < END) {
                        NN[4].template scatter< 0, 0,-1>(mAcc, origin, n+1, 1<<END);//        #16: (0,+1,-1)
                    } else {
                        NN[16].template scatter<0, 1,-1>(mAcc, origin, n, 1<<END);//          #16: (0,+1,-1)
                    }
                }
                // +z: cases: 5, 9, 13, 15, 17
                if (oldWord >> END) {
                    NN[5].template scatter< 0, 0, 1>(mAcc, origin, n, 1);//               #5: (0,0,+1)
                    if (x > 0) {
                        NN[5].template scatter< 0, 0, 1>(mAcc, origin, n-LEAF_DIM, 1);//  #9:  (-1,0,+1)
                    } else {
                        NN[9].template scatter<-1, 0, 1>(mAcc, origin, n, 1);//           #9:  (-1,0,+1)
                    }
                    if (x < END) {
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, n+LEAF_DIM, 1);// #13: (+1,0,+1)
                    } else {
                        NN[13].template scatter<1, 0, 1>(mAcc, origin, n, 1);//           #13: (+1,0,+1)
                    }
                    if (y > 0) {
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, n-1, 1);//        #15: (0,-1,+1)
                    } else {
                        NN[15].template scatter<0,-1, 1>(mAcc, origin, n, 1);//           #15: (0,-1,+1)
                    }
                    if (y < END) {
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, n+1, 1);//        #17: (0,+1,+1)
                    } else {
                        NN[17].template scatter<0, 1, 1>(mAcc, origin, n, 1);//           #17: (0,+1,+1)
                    }
                }
            }// loop over y
        }//loop over x
        for (int i=0; i<18; ++i) NN[i].clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}// dilateVoxels18

template<typename TreeType>
void
Morphology<TreeType>::dilateVoxels26()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = static_cast<int>(mManager->leafCount());

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    MaskManager masks(savedMasks, *mManager);
    masks.save();
    Neighbor NN[26];
    Coord origin;
    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        const MaskType& oldMask = savedMasks[leafIdx];//original bit-mask of current leaf node
        LeafType& leaf = mManager->leaf(leafIdx);//current leaf node
        leaf.getOrigin(origin);// origin of the current leaf node.
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                const Word oldWord = oldMask.template getWord<Word>(n);
                if (oldWord == 0) continue; // no active voxels
                const Word mask = Word(oldWord | (oldWord >> 1) | (oldWord << 1));

                // Dilate the current leaf node in the z direction by ORing its mask
                // with itself shifted first left and then right by one bit.
                leaf.getValueMask().template getWord<Word>(n) |= mask;

                // -x, cases: 0, 6, 7
                if (x > 0) {
                    const Index m = n - LEAF_DIM;//-x
                    leaf.getValueMask().template getWord<Word>(m) |= mask;//         #0: (-1,0,0)
                    if (y > 0) {
                        leaf.getValueMask().template getWord<Word>(m-1) |= mask;//   #6: (-1,-1,0)
                    } else {//y=0
                        NN[2].template scatter< 0,-1, 0>(mAcc, origin, m, mask);//   #6: (-1,-1,0)
                    }
                    if (y < END) {
                        leaf.getValueMask().template getWord<Word>(m+1) |= mask;//   #7: (-1,+1,0)
                    } else {//y=END
                        NN[3].template scatter< 0, 1, 0>(mAcc, origin, m, mask);//   #7: (-1,+1,0)
                    }
                } else {//x=0
                    NN[0].template scatter<-1, 0, 0>(mAcc, origin, n, mask);//       #0: (-1,0,0)
                    if (y > 0) {
                        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n-1, mask);// #6: (-1,-1,0)
                    } else {//y=0
                        NN[6].template scatter<-1,-1, 0>(mAcc, origin, n, mask);//   #6: (-1,-1,0)
                    }
                    if (y < END) {
                        NN[0].template scatter<-1, 0, 0>(mAcc, origin, n+1, mask);// #7: (-1,+1,0)
                    } else {//y=END
                        NN[7].template scatter<-1, 1, 0>(mAcc, origin, n, mask);//   #7: (-1,+1,0)
                    }
                }

                // +x, cases: 1, 10, 11
                if (x < END) {
                    const Index m = n + LEAF_DIM;//+x
                    leaf.getValueMask().template getWord<Word>(m) |= mask;//         #1 : (+1, 0, 0)
                    if (y > 0) {
                        leaf.getValueMask().template getWord<Word>(m-1) |= mask;//   #10: (+1,-1, 0)
                    } else {//y=0
                        NN[2].template scatter<0,-1, 0>(mAcc, origin, m, mask);//    #10: (+1,-1, 0)
                    }
                    if (y < END) {
                        leaf.getValueMask().template getWord<Word>(m+1) |= mask;//   #11: (+1,+1, 0)
                    } else {//y=END
                        NN[3].template scatter<0, 1, 0>(mAcc, origin, m, mask);//    #11: (+1,+1, 0)
                    }
                } else {
                    NN[1].template scatter< 1, 0, 0>(mAcc, origin, n, mask);//       #1 : (+1, 0, 0)
                    if (y>0) {
                        NN[ 1].template scatter<1, 0, 0>(mAcc, origin, n-1, mask);// #10: (+1,-1, 0)
                    } else {//y=0
                        NN[10].template scatter<1,-1, 0>(mAcc, origin, n, mask);//   #10: (+1,-1, 0)
                    }
                    if (y<END) {
                        NN[ 1].template scatter<1, 0, 0>(mAcc, origin, n+1, mask);// #11: (+1,+1, 0)
                    } else {//y=END
                        NN[11].template scatter<1, 1, 0>(mAcc, origin, n, mask);//   #11: (+1,+1, 0)
                    }
                }

                // -y, cases: 2
                if (y > 0) {
                    leaf.getValueMask().template getWord<Word>(n-1) |= mask;// #2: (0,-1,0)
                } else {
                    NN[2].template scatter< 0,-1, 0>(mAcc, origin, n, mask);// #2: (0,-1,0)
                }

                // +y, cases: 3
                if (y < END)  {
                    leaf.getValueMask().template getWord<Word>(n+1) |= mask;// #3: (0,+1,0)
                } else {
                    NN[3].template scatter< 0, 1, 0>(mAcc, origin, n, mask);// #3: (0,+1,0)
                }

                // -z, cases: 4, 8, 12, 14, 16, 18, 20, 22, 24
                if (static_cast<Word>(oldWord << END)) {
                    NN[4].template scatter< 0, 0,-1>(mAcc, origin, n, 1<<END); //           #4: (0,0,-1)
                    if (x > 0) {
                        const Index m = n - LEAF_DIM;//-x
                        NN[4].template scatter< 0, 0,-1>(mAcc, origin, m, 1<<END);//        #8:  (-1,0,-1)
                        if (y>0) {
                            NN[ 4].template scatter<0, 0,-1>(mAcc, origin, m-1, 1<<END);//  #18: (-1,-1,-1)
                        } else {//y=0
                            NN[14].template scatter<0,-1,-1>(mAcc, origin, m, 1<<END);//    #18: (-1,-1,-1)
                        }
                        if (y<END) {
                            NN[ 4].template scatter<0, 0,-1>(mAcc, origin, m+1, 1<<END);//  #20: (-1,+1,-1)
                        } else {//y=END
                            NN[16].template scatter<0, 1,-1>(mAcc, origin, m, 1<<END);//    #20: (-1,+1,-1)
                        }
                    } else {//x=0
                        NN[8].template scatter<-1, 0,-1>(mAcc, origin, n, 1<<END);//        #8:  (-1,0,-1)
                        if (y>0) {
                            NN[ 8].template scatter<-1, 0,-1>(mAcc, origin, n-1, 1<<END);// #18: (-1,-1,-1)
                        } else {//y=0
                            NN[18].template scatter<-1,-1,-1>(mAcc, origin, n, 1<<END);//   #18: (-1,-1,-1)
                        }
                        if (y<END) {
                            NN[ 8].template scatter<-1, 0,-1>(mAcc, origin, n+1, 1<<END);// #20: (-1,+1,-1)
                        } else {//y=END
                            NN[20].template scatter<-1, 1,-1>(mAcc, origin, n, 1<<END);//   #20: (-1,+1,-1)
                        }
                    }
                    if (x < END) {
                        const Index m = n + LEAF_DIM;//+x
                        NN[ 4].template scatter< 0, 0,-1>(mAcc, origin, m, 1<<END);//      #12: (+1,0-1)
                        if (y>0) {
                            NN[ 4].template scatter<0, 0,-1>(mAcc, origin, m-1, 1<<END);// #22: (+1,-1,-1)
                        } else {//y=0
                            NN[14].template scatter<0,-1,-1>(mAcc, origin, m, 1<<END);//   #22: (+1,-1,-1)
                        }
                        if (y<END) {
                            NN[ 4].template scatter<0, 0,-1>(mAcc, origin, m+1, 1<<END);// #24: (+1,+1,-1)
                        } else {//y=END
                            NN[16].template scatter<0, 1,-1>(mAcc, origin, m, 1<<END);//   #24: (+1,+1,-1)
                        }
                    } else {//x=END
                        NN[12].template scatter< 1, 0,-1>(mAcc, origin, n, 1<<END);//      #12: (+1,0-1)
                        if (y>0) {
                            NN[12].template scatter<1, 0,-1>(mAcc, origin, n-1, 1<<END);// #22: (+1,-1,-1)
                        } else {//y=0
                            NN[22].template scatter<1,-1,-1>(mAcc, origin, n, 1<<END);//   #22: (+1,-1,-1)
                        }
                        if (y<END) {
                            NN[12].template scatter<1, 0,-1>(mAcc, origin, n+1, 1<<END);// #24: (+1,+1,-1)
                        } else {//y=END
                            NN[24].template scatter<1, 1,-1>(mAcc, origin, n, 1<<END);//   #24: (+1,+1,-1)
                        }
                    }
                    if (y > 0) {
                        NN[ 4].template scatter< 0, 0,-1>(mAcc, origin, n-1, 1<<END);//    #14: (0,-1,-1)
                    } else {
                        NN[14].template scatter<0,-1,-1>(mAcc, origin, n, 1<<END);//       #14: (0,-1,-1)
                    }
                    if (y < END) {
                        NN[ 4].template scatter< 0, 0,-1>(mAcc, origin, n+1, 1<<END);//    #16: (0,+1,-1)
                    } else {
                        NN[16].template scatter<0, 1,-1>(mAcc, origin, n, 1<<END);//       #16: (0,+1,-1)
                    }
                }
                // +z: cases: 5, 9, 13, 15, 17, 19, 21, 23, 25
                if (oldWord >> END) {
                    NN[5].template scatter< 0, 0, 1>(mAcc, origin, n, 1);//            #5 : (0,0,+1)
                    if (x > 0) {
                        const Index m = n - LEAF_DIM;//-x
                        NN[5].template scatter< 0, 0, 1>(mAcc, origin, m, 1);//        #9 : (-1,0,+1)
                        if (y>0) {
                            NN[ 5].template scatter<0, 0, 1>(mAcc, origin, m-1, 1);//  #19: (-1,-1,+1)
                        } else {//y=0
                            NN[15].template scatter<0,-1, 1>(mAcc, origin, m, 1);//    #19: (-1,-1,+1)
                        }
                        if (y<END) {
                            NN[ 5].template scatter<0, 0, 1>(mAcc, origin, m+1, 1);//  #21: (-1,+1,+1)
                        } else {//y=END
                            NN[17].template scatter<0, 1, 1>(mAcc, origin, m, 1);//    #21: (-1,+1,+1)
                        }
                    } else {//x=0
                        NN[9].template scatter<-1, 0, 1>(mAcc, origin, n, 1);//        #9 : (-1, 0,+1)
                        if (y>0) {
                            NN[ 9].template scatter<-1, 0, 1>(mAcc, origin, n-1, 1);// #19: (-1,-1,+1)
                        } else {//y=0
                            NN[19].template scatter<-1,-1, 1>(mAcc, origin, n, 1);//   #19: (-1,-1,+1)
                        }
                        if (y<END) {
                            NN[ 9].template scatter<-1, 0, 1>(mAcc, origin, n+1, 1);// #21: (-1,+1,+1)
                        } else {//y=END
                            NN[21].template scatter<-1, 1, 1>(mAcc, origin, n, 1);//   #21: (-1,+1,+1)
                        }
                    }
                    if (x < END) {
                        const Index m = n + LEAF_DIM;//+x
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, m, 1);//       #13: (+1, 0,+1)
                        if (y>0) {
                            NN[ 5].template scatter<0, 0, 1>(mAcc, origin, m-1, 1);//  #23: (+1,-1,+1)
                        } else {//y=0
                            NN[15].template scatter<0,-1, 1>(mAcc, origin, m, 1);//    #23: (+1,-1,+1)
                        }
                        if (y<END) {
                            NN[ 5].template scatter<0, 0, 1>(mAcc, origin, m+1, 1);//  #25: (+1,+1,+1)
                        } else {//y=END
                            NN[17].template scatter<0, 1, 1>(mAcc, origin, m, 1);//    #25: (+1,+1,+1)
                        }
                    } else {//x=END
                        NN[13].template scatter<1, 0, 1>(mAcc, origin, n, 1);//        #13: (+1, 0,+1)
                        if (y>0) {
                            NN[13].template scatter< 1, 0, 1>(mAcc, origin, n-1, 1);// #23: (+1,-1,+1)
                        } else {//y=0
                            NN[23].template scatter< 1,-1, 1>(mAcc, origin, n, 1);//   #23: (+1,-1,+1)
                        }
                        if (y<END) {
                            NN[13].template scatter< 1, 0, 1>(mAcc, origin, n+1, 1);// #25: (+1,+1,+1)
                        } else {//y=END
                            NN[25].template scatter< 1, 1, 1>(mAcc, origin, n, 1);//   #25: (+1,+1,+1)
                        }
                    }
                    if (y > 0) {
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, n-1, 1);//     #15: ( 0,-1,+1)
                    } else {
                        NN[15].template scatter<0,-1, 1>(mAcc, origin, n, 1);//        #15: ( 0,-1,+1)
                    }
                    if (y < END) {
                        NN[ 5].template scatter< 0, 0, 1>(mAcc, origin, n+1, 1);//     #17: ( 0,+1,+1)
                    } else {
                        NN[17].template scatter<0, 1, 1>(mAcc, origin, n, 1);//        #17: ( 0,+1,+1)
                    }
                }
            }// loop over y
        }//loop over x
        for (int i=0; i<26; ++i) NN[i].clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}// dilateVoxels26

template <typename TreeType>
void
Morphology<TreeType>::ErodeVoxelsOp::operator()(const tbb::blocked_range<size_t>& range) const
{
    AccessorType acc(mManager.tree());
    Neighbor NN[6];
    Coord origin;
    for (size_t leafIdx = range.begin(); leafIdx < range.end(); ++leafIdx) {
        LeafType& leaf = mManager.leaf(leafIdx);//current leaf node
        if (leaf.isEmpty()) continue;
        MaskType& newMask = mSavedMasks[leafIdx];//original bit-mask of current leaf node
        leaf.getOrigin(origin);// origin of the current leaf node.
        for (int x = 0; x < LEAF_DIM; ++x ) {
            for (int y = 0, n = (x << LEAF_LOG2DIM); y < LEAF_DIM; ++y, ++n) {
                // Extract the portion of the original mask that corresponds to a row in z.
                Word& w = newMask.template getWord<Word>(n);
                if (w == 0) continue; // no active voxels

                // Erode in two z directions (this is first since it uses the original w)
                w = static_cast<Word>(w & (
                    (w<<1 | (NN[4].template gather<0,0,-1>(acc, origin, n)>>END)) &
                    (w>>1 | (NN[5].template gather<0,0, 1>(acc, origin, n)<<END))));

                // dilate current leaf or neighbor in negative x-direction
                w = static_cast<Word>(w & ((x == 0) ?
                    NN[0].template gather<-1, 0, 0>(acc, origin, n) :
                    leaf.getValueMask().template getWord<Word>(n-LEAF_DIM)));

                // dilate current leaf or neighbor in positive x-direction
                w = static_cast<Word>(w & ((x == END) ?
                    NN[1].template gather< 1, 0, 0>(acc, origin, n) :
                    leaf.getValueMask().template getWord<Word>(n+LEAF_DIM)));

                // dilate current leaf or neighbor in negative y-direction
                w = static_cast<Word>(w & ((y == 0) ?
                    NN[2].template gather< 0,-1, 0>(acc, origin, n) :
                    leaf.getValueMask().template getWord<Word>(n-1)));

                // dilate current leaf or neighbor in positive y-direction
                w = static_cast<Word>(w & ((y == END) ?
                    NN[3].template gather< 0, 1, 0>(acc, origin, n) :
                    leaf.getValueMask().template getWord<Word>(n+1)));
            }// loop over y
        }//loop over x
        for (int i=0; i<6; ++i) NN[i].clear();
    }//loop over leafs
}


template<typename TreeType>
void
Morphology<TreeType>::doErosion()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const size_t leafCount = mManager->leafCount();

    // Save the value masks of all leaf nodes.
    std::vector<MaskType> savedMasks(leafCount);
    MaskManager masks(savedMasks, *mManager);
    masks.save();

    ErodeVoxelsOp erode(savedMasks, *mManager);
    for (int i = 0; i < mSteps; ++i) {
        erode.runParallel();
        masks.update();
    }

    mManager->tree().pruneLevelSet();
}


////////////////////////////////////////


template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateVoxels(tree::LeafManager<TreeType>& manager, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(&manager);
        m.dilateVoxels(iterations, nn);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateVoxels(TreeType& tree, int iterations, NearestNeighbors nn)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(tree);
        m.dilateVoxels(iterations, nn);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(tree::LeafManager<TreeType>& manager, int iterations)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(&manager);
        m.erodeVoxels(iterations);
    }
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(TreeType& tree, int iterations)
{
    if (iterations > 0 ) {
        Morphology<TreeType> m(tree);
        m.erodeVoxels(iterations);
    }
}


////////////////////////////////////////


namespace activation {

template<typename TreeType>
class ActivationOp
{
public:
    typedef typename TreeType::ValueType ValueT;

    ActivationOp(bool state, const ValueT& val, const ValueT& tol)
        : mActivate(state)
        , mValue(val)
        , mTolerance(tol)
    {}

    void operator()(const typename TreeType::ValueOnIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setValueOff();
        }
    }

    void operator()(const typename TreeType::ValueOffIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setActiveState(/*on=*/true);
        }
    }

    void operator()(const typename TreeType::LeafIter& lit) const
    {
        typedef typename TreeType::LeafNodeType LeafT;
        LeafT& leaf = *lit;
        if (mActivate) {
            for (typename LeafT::ValueOffIter it = leaf.beginValueOff(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOn(it.pos());
                }
            }
        } else {
            for (typename LeafT::ValueOnIter it = leaf.beginValueOn(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOff(it.pos());
                }
            }
        }
    }

private:
    bool mActivate;
    const ValueT mValue, mTolerance;
}; // class ActivationOp

} // namespace activation


template<typename GridOrTree>
inline void
activate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    typedef TreeAdapter<GridOrTree> Adapter;
    typedef typename Adapter::TreeType TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/true, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other inactive values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOffIter it = tree.beginValueOff();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}


template<typename GridOrTree>
inline void
deactivate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    typedef TreeAdapter<GridOrTree> Adapter;
    typedef typename Adapter::TreeType TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/false, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other active values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOnIter it = tree.beginValueOn();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
