///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
///////////////////////////////////////////////////////////////////////////

#ifndef OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/tree/TreeIterator.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/tree/LeafManager.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

//@{
/// Topologically dilate all leaf-level active voxels in the given tree,
/// i.e., expand the set of active voxels by @a count voxels in the +x, -x,
/// +y, -y, +z and -z directions, but don't change the values of any voxels,
/// only their active states.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(TreeType& tree, int count=1);

template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void dilateVoxels(tree::LeafManager<TreeType>& manager, int count = 1);
//@}

//@{
/// Topologically erode all leaf-level active voxels in the given tree,
/// i.e., shrink the set of active voxels by @a count voxels in the +x, -x,
/// +y, -y, +z and -z directions, but don't change the values of any voxels,
/// only their active states.
/// @todo Currently operates only on leaf voxels; need to extend to tiles.
template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(TreeType& tree, int count=1);

template<typename TreeType> OPENVDB_STATIC_SPECIALIZATION
inline void erodeVoxels(tree::LeafManager<TreeType>& manager, int count = 1);
//@}

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
    void dilateVoxels();
    void dilateVoxels(int count) { for (int i=0; i<count; ++i) this->dilateVoxels(); }
    void erodeVoxels(int count = 1) { mSteps = count; this->doErosion(); }

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
            static const int N = (LEAF_DIM -1 )*(DY + DX*LEAF_DIM);
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
            static const int N = (LEAF_DIM -1 )*(DY + DX*LEAF_DIM);
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
Morphology<TreeType>::dilateVoxels()
{
    /// @todo Currently operates only on leaf voxels; need to extend to tiles.
    const int leafCount = mManager->leafCount();

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

                // dilate current leaf or neighbor in negative x-direction
                if (x > 0) {
                    leaf.getValueMask().template getWord<Word>(n-LEAF_DIM) |= oldWord;
                } else {
                    NN[0].template scatter<-1, 0, 0>(mAcc, origin, n, oldWord);
                }
                // dilate current leaf or neighbor in positive x-direction
                if (x < LEAF_DIM - 1) {
                    leaf.getValueMask().template getWord<Word>(n+LEAF_DIM) |= oldWord;
                } else {
                    NN[1].template scatter< 1, 0, 0>(mAcc, origin, n, oldWord);
                }
                // dilate current leaf or neighbor in negative y-direction
                if (y > 0) {
                    leaf.getValueMask().template getWord<Word>(n-1) |= oldWord;
                } else {
                    NN[2].template scatter< 0,-1, 0>(mAcc, origin, n, oldWord);
                }
                // dilate current leaf or neighbor in positive y-direction
                if (y < LEAF_DIM - 1)  {
                    leaf.getValueMask().template getWord<Word>(n+1) |= oldWord;
                } else {
                    NN[3].template scatter< 0, 1, 0>(mAcc, origin, n, oldWord);
                }
                // Dilate the current leaf node in the z direction by ORing its mask
                // with itself shifted first left and then right by one bit.
                leaf.getValueMask().template getWord<Word>(n) |= (oldWord >> 1) | (oldWord << 1);
                // dilate neighbor in negative z-direction
                if (Word w = oldWord<<(LEAF_DIM-1)) {
                    NN[4].template scatter< 0, 0,-1>(mAcc, origin, n, w);
                }
                // dilate neighbot in positive z-direction
                if (Word w = oldWord>>(LEAF_DIM-1)) {
                    NN[5].template scatter< 0, 0, 1>(mAcc, origin, n, w);
                }
            }// loop over y
        }//loop over x
        for (int i=0; i<6; ++i) NN[i].clear();
    }//loop over leafs

    mManager->rebuildLeafArray();
}


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
                w &= (w<<1 | (NN[4].template gather<0,0,-1>(acc, origin, n)>>(LEAF_DIM-1))) &
                     (w>>1 | (NN[5].template gather<0,0, 1>(acc, origin, n)<<(LEAF_DIM-1)));

                // dilate current leaf or neighbor in negative x-direction
                w &= (x == 0)          ? NN[0].template gather<-1, 0, 0>(acc, origin, n) :
                     leaf.getValueMask().template getWord<Word>(n-LEAF_DIM);

                // dilate current leaf or neighbor in positive x-direction
                w &= (x == LEAF_DIM-1) ? NN[1].template gather< 1, 0, 0>(acc, origin, n) :
                     leaf.getValueMask().template getWord<Word>(n+LEAF_DIM);

                // dilate current leaf or neighbor in negative y-direction
                w &= (y == 0)          ? NN[2].template gather< 0,-1, 0>(acc, origin, n) :
                     leaf.getValueMask().template getWord<Word>(n-1);

                // dilate current leaf or neighbor in positive y-direction
                w &= (y == LEAF_DIM-1) ? NN[3].template gather< 0, 1, 0>(acc, origin, n) :
                     leaf.getValueMask().template getWord<Word>(n+1);
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
    const int leafCount = mManager->leafCount();

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
dilateVoxels(tree::LeafManager<TreeType>& manager, int count)
{
    Morphology<TreeType> m(&manager);
    m.dilateVoxels(count);
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
dilateVoxels(TreeType& tree, int count)
{
    Morphology<TreeType> m(tree);
    m.dilateVoxels(count);
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(tree::LeafManager<TreeType>& manager, int count)
{
    Morphology<TreeType> m(&manager);
    m.erodeVoxels(count);
}

template<typename TreeType>
OPENVDB_STATIC_SPECIALIZATION inline void
erodeVoxels(TreeType& tree, int count)
{
    Morphology<TreeType> m(tree);
    m.erodeVoxels(count);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MORPHOLOGY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
