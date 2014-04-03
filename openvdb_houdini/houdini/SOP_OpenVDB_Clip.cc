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
/// @file SOP_OpenVDB_Clip.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Clip grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Morphology.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Clip: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Clip(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Clip() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input == 1); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
};



////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Grids")
        .setHelpText("Specify a subset of the input VDB grids to be clip.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setSpareData(&SOP_Node::theSecondInput));

    hvdb::OpenVDBOpFactory("OpenVDB Clip", SOP_OpenVDB_Clip::factory, parms, *table)
        .addInput("VDBs")
        .addInput("Mask VDB or bounding geometry");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Clip::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Clip(net, name, op);
}


SOP_OpenVDB_Clip::SOP_OpenVDB_Clip(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {


struct MaskOp
{
    template<typename GridType>
    void operator()(const GridType& grid)
    {
        if (openvdb::GRID_LEVEL_SET == grid.getGridClass()) {
            mMaskGrid = openvdb::tools::sdfInteriorMask(grid);
        } else {
            mMaskGrid = openvdb::BoolGrid::create(false);
            mMaskGrid->setTransform(grid.transform().copy());
            mMaskGrid->tree().topologyUnion(grid.tree());
        }
    }

    openvdb::BoolGrid::Ptr mMaskGrid;
};


template<typename TreeT>
class MaskInteriorVoxels
{
public:
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType LeafNodeT;

    MaskInteriorVoxels(const TreeT& tree) : mAcc(tree) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t /*leafIndex*/) const
    {
        const LeafNodeT *refLeaf = mAcc.probeConstLeaf(leaf.origin());
        if (refLeaf) {
            typename LeafNodeType::ValueOffIter iter = leaf.beginValueOff();
            for (; iter; ++iter) {
                const openvdb::Index pos = iter.pos();
                leaf.setActiveState(pos, refLeaf->getValue(pos) < ValueT(0));
            }
        }
    }

private:
     openvdb::tree::ValueAccessor<const TreeT> mAcc;
};


template<typename TreeT>
class CopyLeafs
{
public:
    typedef typename TreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef openvdb::tree::LeafManager<const BoolTreeT> BoolLeafManagerT;

    //////////

    CopyLeafs(const TreeT& tree, const BoolLeafManagerT& leafs);

    void run(bool threaded = true);

    typename TreeT::Ptr tree() const { return mNewTree; }

    //////////

    CopyLeafs(CopyLeafs&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const CopyLeafs& rhs)
    {
        mNewTree->merge(*rhs.mNewTree);
    }

private:
    const BoolTreeT* mClipMask;
    const TreeT* mTree;
    const BoolLeafManagerT* mLeafs;
    typename TreeT::Ptr mNewTree;
};

template<typename TreeT>
CopyLeafs<TreeT>::CopyLeafs(const TreeT& tree, const BoolLeafManagerT& leafs)
    : mTree(&tree)
    , mLeafs(&leafs)
    , mNewTree(new TreeT(mTree->background()))
{
}

template<typename TreeT>
CopyLeafs<TreeT>::CopyLeafs(CopyLeafs& rhs, tbb::split)
    : mTree(rhs.mTree)
    , mLeafs(rhs.mLeafs)
    , mNewTree(new TreeT(mTree->background()))
{
}

template<typename TreeT>
void
CopyLeafs<TreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafs->getRange(), *this);
    else (*this)(mLeafs->getRange());
}

template<typename TreeT>
void
CopyLeafs<TreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    typedef typename TreeT::LeafNodeType LeafT;
    typedef typename openvdb::BoolTree::LeafNodeType BoolLeafT;
    typename BoolLeafT::ValueOnCIter it;

    openvdb::tree::ValueAccessor<TreeT> acc(*mNewTree);
    openvdb::tree::ValueAccessor<const TreeT> refAcc(*mTree);

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const BoolLeafT& maskLeaf = mLeafs->leaf(n);
        const openvdb::Coord& ijk = maskLeaf.origin();
        const LeafT* refLeaf = refAcc.probeConstLeaf(ijk);

        LeafT* newLeaf = acc.touchLeaf(ijk);

        if (refLeaf) {

            for (it = maskLeaf.cbeginValueOn(); it; ++it) {
                const openvdb::Index pos = it.pos();
                newLeaf->setValueOnly(pos, refLeaf->getValue(pos));
                newLeaf->setActiveState(pos, refLeaf->isValueOn(pos));
            }

        } else {

            typename TreeT::ValueType value;
            bool isActive = refAcc.probeValue(ijk, value);

            for (it = maskLeaf.cbeginValueOn(); it; ++it) {
                const openvdb::Index pos = it.pos();
                newLeaf->setValueOnly(pos, value);
                newLeaf->setActiveState(pos, isActive);
            }
        }
    }
}


struct BoolSampler
{
    static const char* name() { return "bin"; }
    static int radius() { return 2; }
    static bool mipmap() { return false; }
    static bool consistent() { return true; }

    template<class TreeT>
    static bool sample(const TreeT& inTree,
        const openvdb::Vec3R& inCoord, typename TreeT::ValueType& result)
    {
        openvdb::Coord ijk;
        ijk[0] = int(std::floor(inCoord[0]));
        ijk[1] = int(std::floor(inCoord[1]));
        ijk[2] = int(std::floor(inCoord[2]));
        return inTree.probeValue(ijk, result);
    }
};


struct ClipOp
{
    ClipOp() : mMaskGrid(NULL), mBBox(NULL) {}

    ClipOp(const openvdb::BoolGrid& mask) : mMaskGrid(&mask), mBBox(NULL) {}
    ClipOp(const openvdb::BBoxd& bbox) : mMaskGrid(NULL), mBBox(&bbox) {}


    template<typename GridType>
    void operator()(const GridType& grid)
    {
        if (!mMaskGrid && !mBBox) return;

        const openvdb::GridClass gridClass = grid.getGridClass();
        typedef typename GridType::TreeType TreeType;
        typedef openvdb::BoolTree BoolTree;
        const TreeType& tree = grid.tree();


        openvdb::BoolTree mask(false);
        mask.topologyUnion(tree);

        if (openvdb::GRID_LEVEL_SET == gridClass) {

            openvdb::tree::LeafManager<openvdb::BoolTree> leafs(mask);
            leafs.foreach(MaskInteriorVoxels<TreeType>(tree));

            openvdb::tree::ValueAccessor<const TreeType> acc(tree);

            typename BoolTree::ValueAllIter iter(mask);
            iter.setMaxDepth(BoolTree::ValueAllIter::LEAF_DEPTH - 1);

            for ( ; iter; ++iter) {
                iter.setActiveState(acc.getValue(iter.getCoord()) < typename TreeType::ValueType(0));
            }
        }

        if (!mBBox) {

            openvdb::BoolGrid clipMask;
            clipMask.setTransform(grid.transform().copy());
            openvdb::tools::resampleToMatch<BoolSampler>(*mMaskGrid, clipMask);
            clipMask.tree().prune();
            mask.topologyIntersection(clipMask.tree());

        } else {
            openvdb::Vec3d minIS, maxIS;
            openvdb::math::calculateBounds(grid.transform(), mBBox->min(), mBBox->max(), minIS, maxIS);

            openvdb::CoordBBox region;
            region.min()[0] = int(std::floor(minIS[0]));
            region.min()[1] = int(std::floor(minIS[1]));
            region.min()[2] = int(std::floor(minIS[2]));

            region.max()[0] = int(std::floor(maxIS[0]));
            region.max()[1] = int(std::floor(maxIS[1]));
            region.max()[2] = int(std::floor(maxIS[2]));

            openvdb::BoolTree clipMask(false);
            clipMask.fill(region, true, true);

            mask.topologyIntersection(clipMask);
        }

        typename GridType::Ptr newGrid;

        { // Copy voxel data and state
            openvdb::tree::LeafManager<const openvdb::BoolTree> leafs(mask);
            CopyLeafs<TreeType> maskOp(tree, leafs);
            maskOp.run();
            newGrid = GridType::create(maskOp.tree());
        }

        { // Copy tile data and state
            openvdb::tree::ValueAccessor<const TreeType> refAcc(tree);
            openvdb::tree::ValueAccessor<const openvdb::BoolTree> maskAcc(mask);

            typename TreeType::ValueAllIter it(newGrid->tree());
            it.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 1);
            for ( ; it; ++it) {
                openvdb::Coord ijk = it.getCoord();

                if (maskAcc.isValueOn(ijk)) {
                    typename TreeType::ValueType value;
                    bool isActive = refAcc.probeValue(ijk, value);

                    it.setValue(value);
                    if (!isActive) it.setValueOff();
                }
            }
        }


        if (openvdb::GRID_LEVEL_SET != gridClass) {
            newGrid->setGridClass(gridClass);
        }

        mGrid = newGrid;
        mGrid->setTransform(grid.transform().copy());
    }

    const openvdb::BoolGrid *mMaskGrid;
    const openvdb::BBoxd *mBBox;
    hvdb::GridPtr mGrid;
};


} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Clip::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);


        openvdb::BoolGrid::Ptr maskGrid;
        const GU_Detail* maskGeo = inputGeo(1);
        bool gridNameEntered = false;

        if (maskGeo) {
            UT_String maskStr;
            evalString(maskStr, "mask", 0, time);

            gridNameEntered = maskStr.length() > 0;

            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskStr.buffer(), const_cast<GU_Detail*>(maskGeo));

            hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
            if (maskIt) {
                MaskOp op;
                UTvdbProcessTypedGridScalar(maskIt->getStorageType(), maskIt->getGrid(), op);
                maskGrid = op.mMaskGrid;
            }
        }

        ClipOp clipOp;

        if (maskGrid) {
            clipOp = ClipOp(*maskGrid);

        } else if (gridNameEntered) {
            addError(SOP_MESSAGE, "Mask VDB not found.");
            return error();

        } else {

            UT_BoundingBox box;
            maskGeo->computeQuickBounds(box);

            openvdb::BBoxd bbox;
            bbox.min()[0] = box.xmin();
            bbox.min()[1] = box.ymin();
            bbox.min()[2] = box.zmin();
            bbox.max()[0] = box.xmax();
            bbox.max()[1] = box.ymax();
            bbox.max()[2] = box.zmax();

            clipOp = ClipOp(bbox);
        }


        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            //UTvdbProcessTypedGridScalar(it->getStorageType(), it->getGrid(), clipOp);
            GEOvdbProcessTypedGridTopology(**it, clipOp);

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, clipOp.mGrid, **it, true);
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
