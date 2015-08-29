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

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Occlusion_Mask: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Occlusion_Mask(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Occlusion_Mask() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);

    virtual OP_ERROR cookMyGuide1(OP_Context&);

private:
    openvdb::math::Transform::Ptr mFrustum;
};



////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Grids")
        .setHelpText("Specify a subset of the input VDB grids to be clip.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_STRING, "camera", "Camera")
        .setHelpText("Reference camera path")
        .setTypeExtended(PRM_TYPE_DYNAMIC_PATH)
        .setSpareData(&PRM_SpareData::objCameraPath));

    parms.add(hutil::ParmFactory(PRM_INT_J, "voxelCount", "Voxel count")
        .setDefault(PRM100Defaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 200)
        .setHelpText("Horizontal voxel count for the near plane."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelDepthSize", "Voxel depth size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setHelpText("The voxel depth (uniform z-size) in world units."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "depth", "Mask Depth")
        .setDefault(100)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 1000.0)
        .setHelpText("Specify mask depth"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "erode", "Erode")
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setDefault(PRMzeroDefaults));

    parms.add(hutil::ParmFactory(PRM_INT_J, "zoffset", "Z Offset")
        .setRange(PRM_RANGE_UI, -10, PRM_RANGE_UI, 10)
        .setDefault(PRMzeroDefaults));

    hvdb::OpenVDBOpFactory("OpenVDB Occlusion Mask",
        SOP_OpenVDB_Occlusion_Mask::factory, parms, *table)
        .addInput("VDBs");
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
        hvdb::drawFrustum(*myGuide1, *mFrustum, &color, NULL, false, false);
    }
    return error();
}


////////////////////////////////////////


namespace {


template<typename BoolTreeT>
class VoxelShadow
{
public:
    typedef openvdb::tree::LeafManager<const BoolTreeT> BoolLeafManagerT;

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
        typedef typename GridType::TreeType TreeType;

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
            const int voxelCount = evalInt("voxelCount", 0, time);

            mFrustum = hvdb::frustumTransformFromCamera(*this, context, *cam,
                0, nearPlane, farPlane, voxelDepthSize, voxelCount);
        } else {
            addError(SOP_MESSAGE, "No camera referenced.");
            return error();
        }


        ConstructShadow shadowOp(*mFrustum,
            evalInt("erode", 0, time), evalInt("zoffset", 0, time));


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

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
