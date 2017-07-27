///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Occlusion_Mask.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Masks the occluded regions behind objects in the camera frustum

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Morphology.h>

#include <OBJ/OBJ_Camera.h>

#include <cmath> // for std::floor()

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Occlusion_Mask: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Occlusion_Mask(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Occlusion_Mask() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    OP_ERROR cookMyGuide1(OP_Context&) override;

private:
    openvdb::math::Transform::Ptr mFrustum;
};



////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Grids")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB grids to be processed.")
        .setDocumentation(
            "A subset of the input VDBs to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_STRING, "camera", "Camera")
        .setTypeExtended(PRM_TYPE_DYNAMIC_PATH)
        .setSpareData(&PRM_SpareData::objCameraPath)
        .setTooltip("Reference camera path")
        .setDocumentation("The path to the camera (e.g., `/obj/cam1`)"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "voxelCount", "Voxel Count")
        .setDefault(PRM100Defaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 200)
        .setTooltip("The desired width in voxels of the camera's near plane"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelDepthSize", "Voxel Depth Size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setTooltip("The depth of a voxel in world units (all voxels have equal depth)"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "depth", "Mask Depth")
        .setDefault(100)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 1000.0)
        .setTooltip(
            "The desired depth of the mask in world units"
            " from the near plane to the far plane"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "erode", "Erode")
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setDefault(PRMzeroDefaults)
        .setTooltip("The number of voxels by which to shrink the mask"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "zoffset", "Z Offset")
        .setRange(PRM_RANGE_UI, -10, PRM_RANGE_UI, 10)
        .setDefault(PRMzeroDefaults)
        .setTooltip("The number of voxels by which to offset the near plane"));

    hvdb::OpenVDBOpFactory("OpenVDB Occlusion Mask",
        SOP_OpenVDB_Occlusion_Mask::factory, parms, *table)
        .addInput("VDBs")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Identify voxels of a VDB volume that are in shadow from a given camera.\"\"\"\n\
\n\
@overview\n\
\n\
This node outputs a VDB volume whose active voxels denote the voxels\n\
of an input volume inside a camera frustum that would be occluded\n\
when viewed through the camera.\n\
\n\
@related\n\
- [OpenVDB Clip|Node:sop/DW_OpenVDBClip]\n\
- [OpenVDB Create|Node:sop/DW_OpenVDBCreate]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Occlusion_Mask::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Occlusion_Mask(net, name, op);
}


SOP_OpenVDB_Occlusion_Mask::SOP_OpenVDB_Occlusion_Mask(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
    , mFrustum()
{
}

OP_ERROR
SOP_OpenVDB_Occlusion_Mask::cookMyGuide1(OP_Context&)
{
    myGuide1->clearAndDestroy();
    if (mFrustum) {
        UT_Vector3 color(0.9f, 0.0f, 0.0f);
        hvdb::drawFrustum(*myGuide1, *mFrustum, &color, nullptr, false, false);
    }
    return error();
}


////////////////////////////////////////


namespace {


template<typename BoolTreeT>
class VoxelShadow
{
public:
    using BoolLeafManagerT = openvdb::tree::LeafManager<const BoolTreeT>;

    //////////

    VoxelShadow(const BoolLeafManagerT& leafs, int zMax, int offset);
    void run(bool threaded = true);

    BoolTreeT& tree() const { return *mNewTree; }

    //////////

    VoxelShadow(VoxelShadow&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const VoxelShadow& rhs)
    {
        mNewTree->merge(*rhs.mNewTree);
        mNewTree->prune();
    }

private:
    typename BoolTreeT::Ptr mNewTree;
    const BoolLeafManagerT* mLeafs;
    const int mOffset, mZMax;
};

template<typename BoolTreeT>
VoxelShadow<BoolTreeT>::VoxelShadow(const BoolLeafManagerT& leafs, int zMax, int offset)
    : mNewTree(new BoolTreeT(false))
    , mLeafs(&leafs)
    , mOffset(offset)
    , mZMax(zMax)
{
}

template<typename BoolTreeT>
VoxelShadow<BoolTreeT>::VoxelShadow(VoxelShadow& rhs, tbb::split)
    : mNewTree(new BoolTreeT(false))
    , mLeafs(rhs.mLeafs)
    , mOffset(rhs.mOffset)
    , mZMax(rhs.mZMax)
{
}

template<typename BoolTreeT>
void
VoxelShadow<BoolTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafs->getRange(), *this);
    else (*this)(mLeafs->getRange());
}

template<typename BoolTreeT>
void
VoxelShadow<BoolTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename BoolTreeT::LeafNodeType::ValueOnCIter it;
    openvdb::CoordBBox bbox;

    bbox.max()[2] = mZMax;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        for (it = mLeafs->leaf(n).cbeginValueOn(); it; ++it) {

            bbox.min() = it.getCoord();
            bbox.min()[2] += mOffset;
            bbox.max()[0] = bbox.min()[0];
            bbox.max()[1] = bbox.min()[1];

            mNewTree->fill(bbox, true, true);
        }

        mNewTree->prune();
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


struct ConstructShadow
{
    ConstructShadow(const openvdb::math::Transform& frustum, int erode, int zoffset)
        : mGrid(openvdb::BoolGrid::create(false))
        , mFrustum(frustum)
        , mErode(erode)
        , mZOffset(zoffset)
    {
    }


    template<typename GridType>
    void operator()(const GridType& grid)
    {
        using TreeType = typename GridType::TreeType;

        const TreeType& tree = grid.tree();

        // Resample active tree topology into camera frustum space.

        openvdb::BoolGrid frustumMask(false);
        frustumMask.setTransform(mFrustum.copy());

        {
            openvdb::BoolGrid topologyMask(false);
            topologyMask.setTransform(grid.transform().copy());

            if (openvdb::GRID_LEVEL_SET == grid.getGridClass()) {

                openvdb::BoolGrid::Ptr tmpGrid = openvdb::tools::sdfInteriorMask(grid);

                topologyMask.tree().merge(tmpGrid->tree());

                if (mErode > 3) {
                    openvdb::tools::erodeVoxels(topologyMask.tree(), (mErode - 3));
                }

            } else {
                topologyMask.tree().topologyUnion(tree);

                if (mErode > 0) {
                    openvdb::tools::erodeVoxels(topologyMask.tree(), mErode);
                }
            }


            if (grid.transform().voxelSize()[0] < mFrustum.voxelSize()[0]) {
                openvdb::tools::resampleToMatch<openvdb::tools::PointSampler>(
                    topologyMask, frustumMask);
            } else {
                openvdb::tools::resampleToMatch<BoolSampler>(topologyMask, frustumMask);
            }

        }


        // Create shadow volume

        mGrid = openvdb::BoolGrid::create(false);
        mGrid->setTransform(mFrustum.copy());
        openvdb::BoolTree& shadowTree = mGrid->tree();

        const openvdb::math::NonlinearFrustumMap& map =
            *mFrustum.map<openvdb::math::NonlinearFrustumMap>();
        int zCoord = int(std::floor(map.getBBox().max()[2]));

        // Voxel shadows
        openvdb::tree::LeafManager<const openvdb::BoolTree> leafs(frustumMask.tree());
        VoxelShadow<openvdb::BoolTree> shadowOp(leafs, zCoord, mZOffset);
        shadowOp.run();

        shadowTree.merge(shadowOp.tree());

        // Tile shadows
        openvdb::CoordBBox bbox;
        openvdb::BoolTree::ValueOnIter it(frustumMask.tree());
        it.setMaxDepth(openvdb::BoolTree::ValueAllIter::LEAF_DEPTH - 1);
        for ( ; it; ++it) {

            it.getBoundingBox(bbox);
            bbox.min()[2] += mZOffset;
            bbox.max()[2] = zCoord;

            shadowTree.fill(bbox, true, true);
        }

        shadowTree.prune();
    }

    openvdb::BoolGrid::Ptr& grid() { return mGrid; }

private:
    openvdb::BoolGrid::Ptr mGrid;
    const openvdb::math::Transform mFrustum;
    const int mErode, mZOffset;
};


} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Occlusion_Mask::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);


        // Camera reference
        mFrustum.reset();

        UT_String cameraPath;
        evalString(cameraPath, "camera", 0, time);
        cameraPath.harden();

        if (cameraPath.isstring()) {

            OBJ_Node *camobj = findOBJNode(cameraPath);
            if (!camobj) {
                addError(SOP_MESSAGE, "Camera not found");
                return error();
            }

            OBJ_Camera* cam = camobj->castToOBJCamera();
            if (!cam) {
                addError(SOP_MESSAGE, "Camera not found");
                return error();
            }

            // Register
            this->addExtraInput(cam, OP_INTEREST_DATA);

            const float nearPlane = static_cast<float>(cam->getNEAR(time));
            const float farPlane = static_cast<float>(nearPlane + evalFloat("depth", 0, time));
            const float voxelDepthSize = static_cast<float>(evalFloat("voxelDepthSize", 0, time));
            const int voxelCount = static_cast<int>(evalInt("voxelCount", 0, time));

            mFrustum = hvdb::frustumTransformFromCamera(*this, context, *cam,
                0, nearPlane, farPlane, voxelDepthSize, voxelCount);
        } else {
            addError(SOP_MESSAGE, "No camera referenced.");
            return error();
        }


        ConstructShadow shadowOp(*mFrustum,
            static_cast<int>(evalInt("erode", 0, time)),
            static_cast<int>(evalInt("zoffset", 0, time)));


        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            UTvdbProcessTypedGridScalar(it->getStorageType(), it->getGrid(), shadowOp);
            //GEOvdbProcessTypedGridTopology(**it, clipOp);

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, shadowOp.grid(), **it, true);
        }



    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
