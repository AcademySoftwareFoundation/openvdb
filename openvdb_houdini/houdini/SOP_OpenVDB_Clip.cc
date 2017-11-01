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

/// @file SOP_OpenVDB_Clip.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Clip grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/GeometryUtil.h> // for drawFrustum(), frustumTransformFromCamera()
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Clip.h> // for tools::clip()
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfInteriorMask()
#include <openvdb/points/PointDataGrid.h>
#include <OBJ/OBJ_Camera.h>
#include <exception>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Clip: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Clip(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Clip() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned input) const override { return (input == 1); }

protected:
    void resolveObsoleteParms(PRM_ParmList*) override;
    bool updateParmsFlags() override;
    OP_ERROR cookMyGuide1(OP_Context&) override;
    OP_ERROR cookMySop(OP_Context&) override;

private:
    void getFrustum(OP_Context&);

    openvdb::math::Transform::Ptr mFrustum;
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of VDBs from the first input to be clipped.")
        .setDocumentation(
            "A subset of VDBs from the first input to be clipped"
            " (see [specifying volumes|/model/volumes#group])"));

    {
        char const * const items[] = {
            "camera",   "Camera",
            "geometry", "Geometry",
            "mask",     "Mask VDB",
            nullptr
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "clipper", "Clip To")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault("geometry")
            .setTooltip("Specify how the clipping region should be defined.")
            .setDocumentation("\
How to define the clipping region\n\
\n\
Camera:\n\
    Use a camera frustum as the clipping region.\n\
Geometry:\n\
    Use the bounding box of geometry from the second input as the clipping region.\n\
Mask VDB:\n\
    Use the active voxels of a VDB volume from the second input as a clipping mask.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Specify a VDB whose active voxels are to be used as a clipping mask.")
        .setDocumentation(
            "A VDB from the second input whose active voxels are to be used as a clipping mask"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_STRING, "camera", "Camera")
        .setTypeExtended(PRM_TYPE_DYNAMIC_PATH)
        .setSpareData(&PRM_SpareData::objCameraPath)
        .setTooltip("Specify the path to a reference camera")
        .setDocumentation(
            "The path to the camera whose frustum is to be used as a clipping region"
            " (e.g., `/obj/cam1`)"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "inside", "Keep Inside")
        .setDefault(PRMoneDefaults)
        .setTooltip(
            "If enabled, keep voxels that lie inside the clipping region.\n"
            "If disabled, keep voxels that lie outside the clipping region.")
        .setDocumentation(
            "If enabled, keep voxels that lie inside the clipping region,"
            " otherwise keep voxels that lie outside the clipping region."));


    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "usemask", "").setDefault(PRMzeroDefaults));


    hvdb::OpenVDBOpFactory("OpenVDB Clip", SOP_OpenVDB_Clip::factory, parms, *table)
        .addInput("VDBs")
        .addOptionalInput("Mask VDB or bounding geometry")
        .setObsoleteParms(obsoleteParms)
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Clip VDB volumes using a camera frustum, a bounding box, or another VDB as a mask.\"\"\"\n\
\n\
@overview\n\
\n\
This node clips VDB volumes, that is, it removes voxels that lie outside\n\
(or, optionally, inside) a given region by deactivating them and setting them\n\
to the background value.\n\
The clipping region may be one of the following:\n\
* the frustum of a camera\n\
* the bounding box of reference geometry\n\
* the active voxels of another VDB.\n\
\n\
When the clipping region is defined by a VDB, the operation\n\
is similar to [activity intersection|Node:sop/DW_OpenVDBCombine],\n\
except that clipped voxels are not only deactivated but also set\n\
to the background value.\n\
\n\
@related\n\
\n\
- [OpenVDB Combine|Node:sop/DW_OpenVDBCombine]\n\
- [OpenVDB Occlusion Mask|Node:sop/DW_OpenVDBOcclusionMask]\n\
- [Node:sop/vdbactivate]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


void
SOP_OpenVDB_Clip::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    auto* parm = obsoleteParms->getParmPtr("usemask");
    if (parm && !parm->isFactoryDefault()) { // factory default was Off
        setString("clipper", CH_STRING_LITERAL, "mask", 0, 0.0);
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


bool
SOP_OpenVDB_Clip::updateParmsFlags()
{
    bool changed = false;

    UT_String clipper;
    evalString(clipper, "clipper", 0, 0);

    changed |= enableParm("camera", clipper == "camera");
    changed |= enableParm("mask", clipper == "mask");

    return changed;
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


struct LevelSetMaskOp
{
    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid = openvdb::tools::sdfInteriorMask(grid);
    }

    hvdb::GridPtr outputGrid;
};


struct BBoxClipOp
{
    BBoxClipOp(const openvdb::BBoxd& bbox_, bool inside_ = true):
        bbox(bbox_), inside(inside_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid = openvdb::tools::clip(grid, bbox, inside);
    }

    openvdb::BBoxd bbox;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};


struct FrustumClipOp
{
    FrustumClipOp(const openvdb::math::Transform::Ptr& frustum_, bool inside_ = true):
        frustum(frustum_), inside(inside_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        openvdb::math::NonlinearFrustumMap::ConstPtr mapPtr;
        if (frustum) mapPtr = frustum->constMap<openvdb::math::NonlinearFrustumMap>();
        if (mapPtr) {
            outputGrid = openvdb::tools::clip(grid, *mapPtr, inside);
        }
    }

    const openvdb::math::Transform::ConstPtr frustum;
    const bool inside = true;
    hvdb::GridPtr outputGrid;
};


template<typename GridType>
struct MaskClipDispatchOp
{
    MaskClipDispatchOp(const GridType& grid_, bool inside_ = true):
        grid(&grid_), inside(inside_)
    {}

    template<typename MaskGridType>
    void operator()(const MaskGridType& mask)
    {
        outputGrid.reset();
        if (grid) outputGrid = openvdb::tools::clip(*grid, mask, inside);
    }

    const GridType* grid;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};


struct MaskClipOp
{
    MaskClipOp(hvdb::GridCPtr mask_, bool inside_ = true):
        mask(mask_), inside(inside_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid.reset();
        if (mask) {
            // Dispatch on the mask grid type, now that the source grid type is resolved.
            MaskClipDispatchOp<GridType> op(grid, inside);
            UTvdbProcessTypedGridTopology(UTvdbGetGridType(*mask), *mask, op);
            outputGrid = op.outputGrid;
        }
    }

    hvdb::GridCPtr mask;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};

} // unnamed namespace


////////////////////////////////////////


/// Get the selected camera's frustum transform.
void
SOP_OpenVDB_Clip::getFrustum(OP_Context& context)
{
    mFrustum.reset();

    const auto time = context.getTime();

    UT_String cameraPath;
    evalString(cameraPath, "camera", 0, time);
    if (!cameraPath.isstring()) {
        throw std::runtime_error{"no camera path was specified"};
    }

    OBJ_Camera* camera = nullptr;
    if (auto* obj = findOBJNode(cameraPath)) {
        camera = obj->castToOBJCamera();
    }
    if (!camera) {
        throw std::runtime_error{"camera \"" + cameraPath.toStdString() + "\" was not found"};
    }
    this->addExtraInput(camera, OP_INTEREST_DATA);

    OBJ_CameraParms cameraParms;
    camera->getCameraParms(cameraParms, time);
    if (cameraParms.projection != OBJ_PROJ_PERSPECTIVE) {
        throw std::runtime_error{cameraPath.toStdString() + " is not a perspective camera"};
        /// @todo support ortho and other cameras?
    }

    const float
        nearPlane = static_cast<float>(camera->getNEAR(time)),
        farPlane = static_cast<float>(camera->getFAR(time));
    mFrustum = hvdb::frustumTransformFromCamera(*this, context, *camera,
        /*offset=*/0.f, nearPlane, farPlane, /*voxelDepth=*/1.f, /*voxelCountX=*/100);

    if (!mFrustum || !mFrustum->constMap<openvdb::math::NonlinearFrustumMap>()) {
        throw std::runtime_error{
            "failed to compute frustum bounds for camera " + cameraPath.toStdString()};
    }
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Clip::cookMyGuide1(OP_Context&)
{
    myGuide1->clearAndDestroy();
    if (mFrustum) {
        const UT_Vector3 color{0.9f, 0.0f, 0.0f};
        hvdb::drawFrustum(*myGuide1, *mFrustum, &color,
            /*tickColor=*/nullptr, /*shaded=*/false, /*ticks=*/false);
    }
    return error();
}


OP_ERROR
SOP_OpenVDB_Clip::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock{*this, context};
        const fpreal time = context.getTime();

        UT_AutoInterrupt progress{"Clipping VDBs"};

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        const GU_Detail* maskGeo = inputGeo(1);

        UT_String clipper;
        evalString(clipper, "clipper", 0, time);

        const bool
            useCamera = (clipper == "camera"),
            useMask = (clipper == "mask"),
            inside = evalInt("inside", 0, time);

        mFrustum.reset();

        openvdb::BBoxd clipBox;
        hvdb::GridCPtr maskGrid;

        if (useCamera) {
            getFrustum(context);
        } else if (maskGeo) {
            if (useMask) {
                UT_String maskStr;
                evalString(maskStr, "mask", 0, time);
#if (UT_MAJOR_VERSION_INT >= 15)
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    maskStr.buffer(), GroupCreator{maskGeo});
#else
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    maskStr.buffer(), const_cast<GU_Detail*>(maskGeo));
#endif
                hvdb::VdbPrimCIterator maskIt{maskGeo, maskGroup};
                if (maskIt) {
                    if (maskIt->getConstGrid().getGridClass() == openvdb::GRID_LEVEL_SET) {
                        // If the mask grid is a level set, extract an interior mask from it.
                        LevelSetMaskOp op;
                        GEOvdbProcessTypedGridScalar(**maskIt, op);
                        maskGrid = op.outputGrid;
                    } else {
                        maskGrid = maskIt->getConstGridPtr();
                    }
                }
                if (!maskGrid) {
                    addError(SOP_MESSAGE, "mask VDB not found");
                    return error();
                }
            } else {
                UT_BoundingBox box;
                maskGeo->computeQuickBounds(box);

                clipBox.min()[0] = box.xmin();
                clipBox.min()[1] = box.ymin();
                clipBox.min()[2] = box.zmin();
                clipBox.max()[0] = box.xmax();
                clipBox.max()[1] = box.ymax();
                clipBox.max()[2] = box.zmax();
            }
        } else {
            addError(SOP_MESSAGE, "Not enough sources specified.");
            return error();
        }

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        int numLevelSets = 0;
        for (hvdb::VdbPrimIterator it{gdp, group}; it; ++it) {
            if (progress.wasInterrupted()) { throw std::runtime_error{"interrupted"}; }

            const auto& inGrid = it->getConstGrid();

            hvdb::GridPtr outGrid;

            if (inGrid.getGridClass() == openvdb::GRID_LEVEL_SET) {
                ++numLevelSets;
            }

            progress.getInterrupt()->setAppTitle(
                ("Clipping VDB " + it.getPrimitiveIndexAndName().toStdString()).c_str());

            if (maskGrid) {
                MaskClipOp op{maskGrid, inside};
                if (GEOvdbProcessTypedGridTopology(**it, op)) { // all Houdini-supported grid types
                    outGrid = op.outputGrid;
                } else if (inGrid.isType<openvdb::points::PointDataGrid>()) {
                    addWarning(SOP_MESSAGE,
                        "only bounding box clipping is currently supported for point data grids");
                }
            } else if (useCamera) {
                FrustumClipOp op{mFrustum, inside};
                if (GEOvdbProcessTypedGridTopology(**it, op)) { // all Houdini-supported grid types
                    outGrid = op.outputGrid;
                } else if (inGrid.isType<openvdb::points::PointDataGrid>()) {
                    addWarning(SOP_MESSAGE,
                        "only bounding box clipping is currently supported for point data grids");
                }
            } else {
                BBoxClipOp op{clipBox, inside};
                if (GEOvdbProcessTypedGridTopology(**it, op)) { // all Houdini-supported grid types
                    outGrid = op.outputGrid;
                } else if (inGrid.isType<openvdb::points::PointDataGrid>()) {
                    if (inside) {
                        outGrid = inGrid.deepCopyGrid();
                        outGrid->clipGrid(clipBox);
                    } else {
                        addWarning(SOP_MESSAGE,
                            "only Keep Inside mode is currently supported for point data grids");
                    }
                }
            }

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, outGrid, **it, true);
        }

        if (numLevelSets > 0) {
            if (numLevelSets == 1) {
                addWarning(SOP_MESSAGE, "a level set grid was clipped;"
                    " the resulting grid might not be a valid level set");
            } else {
                addWarning(SOP_MESSAGE, "some level sets were clipped;"
                    " the resulting grids might not be valid level sets");
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
