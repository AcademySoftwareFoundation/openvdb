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
/// @file SOP_OpenVDB_Create.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <UT/UT_Interrupt.h>
#include <UT/UT_WorkArgs.h>
#include <OBJ/OBJ_Camera.h>

#include <sstream>
#include <string>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;
namespace cvdb = openvdb;


////////////////////////////////////////


namespace {

// Add new items to the *end* of this list, and update NUM_DATA_TYPES.
enum DataType {
    TYPE_FLOAT = 0,
    TYPE_DOUBLE,
    TYPE_INT,
    TYPE_BOOL,
    TYPE_VEC3S,
    TYPE_VEC3D,
    TYPE_VEC3I
};

enum { NUM_DATA_TYPES = TYPE_VEC3I + 1 };

std::string
dataTypeToString(DataType dts)
{
    std::string ret;
    switch (dts) {
        case TYPE_FLOAT:  ret = "float"; break;
        case TYPE_DOUBLE: ret = "double"; break;
        case TYPE_INT:    ret = "int"; break;
        case TYPE_BOOL:   ret = "bool"; break;
        case TYPE_VEC3S:  ret = "vec3s"; break;
        case TYPE_VEC3D:  ret = "vec3d"; break;
        case TYPE_VEC3I:  ret = "vec3i"; break;
    }
    return ret;
}

std::string
dataTypeToMenuItems(DataType dts)
{
    std::string ret;
    switch (dts) {
        case TYPE_FLOAT:  ret = "float"; break;
        case TYPE_DOUBLE: ret = "double"; break;
        case TYPE_INT:    ret = "int"; break;
        case TYPE_BOOL:   ret = "bool"; break;
        case TYPE_VEC3S:  ret = "vec3s (float)"; break;
        case TYPE_VEC3D:  ret = "vec3d (double)"; break;
        case TYPE_VEC3I:  ret = "vec3i (int)"; break;
    }
    return ret;
}

DataType
stringToDataType(const std::string& s)
{
    DataType ret = TYPE_FLOAT;
    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);
    if (str == dataTypeToString(TYPE_FLOAT)) {
        ret = TYPE_FLOAT;
    } else if (str == dataTypeToString(TYPE_DOUBLE)) {
        ret = TYPE_DOUBLE;
    } else if (str == dataTypeToString(TYPE_INT)) {
        ret = TYPE_INT;
    } else if (str == dataTypeToString(TYPE_BOOL)) {
        ret = TYPE_BOOL;
    } else if (str == dataTypeToString(TYPE_VEC3S)) {
        ret = TYPE_VEC3S;
    } else if (str == dataTypeToString(TYPE_VEC3D)) {
        ret = TYPE_VEC3D;
    } else if (str == dataTypeToString(TYPE_VEC3I)) {
        ret = TYPE_VEC3I;
    }
    return ret;
}

} // unnamed namespace


////////////////////////////////////////


class SOP_OpenVDB_Create : public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Create(OP_Network *net, const char *name, OP_Operator *op);
    virtual ~SOP_OpenVDB_Create() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i == 0); }

    int updateNearFar(float time);
    int updateFarPlane(float time);
    int updateNearPlane(float time);

protected:
    virtual OP_ERROR cookMySop(OP_Context &context);
    virtual bool updateParmsFlags();

private:
    inline cvdb::Vec3i voxelToIndex(const cvdb::Vec3R& V) const
    {
        return cvdb::Vec3i(cvdb::Int32(V[0]), cvdb::Int32(V[1]), cvdb::Int32(V[2]));
    }

    template<typename GridType>
    void createNewGrid(
        const UT_String& gridNameStr,
        const typename GridType::ValueType& background,
        const cvdb::math::Transform::Ptr&,
        GA_PrimitiveGroup* group = NULL,
        int gridClass = 0,
        int vecType = -1);

    OP_ERROR buildTransform(OP_Context&, openvdb::math::Transform::Ptr&);
};


////////////////////////////////////////


// Callback functions that update the near and far parameters

int
updateNearFarCallback(void* data, int /*idx*/, float time, const PRM_Template*)
{
   SOP_OpenVDB_Create* sop = static_cast<SOP_OpenVDB_Create*>(data);
   if (sop == NULL) return 0;
   return sop->updateNearFar(time);
}


int
SOP_OpenVDB_Create::updateNearFar(float time)
{
    UT_String cameraPath;
    evalString(cameraPath, "camera", 0, time);
    cameraPath.harden();
    if (!cameraPath.isstring()) return 1;

    OBJ_Node *camobj = findOBJNode(cameraPath);
    if (!camobj) return 1;

    OBJ_Camera* cam = camobj->castToOBJCamera();
    if (!cam) return 1;

    fpreal nearPlane = cam->getNEAR(time);
    fpreal farPlane = cam->getFAR(time);

    setFloat("nearPlane", 0, time, nearPlane);
    setFloat("farPlane", 0, time, farPlane);

    return 1;
}


int
updateNearPlaneCallback(void* data, int /*idx*/, float time, const PRM_Template*)
{
   SOP_OpenVDB_Create* sop = static_cast<SOP_OpenVDB_Create*>(data);
   if (sop == NULL) return 0;
   return sop->updateNearPlane(time);
}


int
SOP_OpenVDB_Create::updateNearPlane(float time)
{
    fpreal
        nearPlane = evalFloat("nearPlane", 0, time),
        farPlane = evalFloat("farPlane", 0, time),
        voxelDepthSize = evalFloat("voxelDepthSize", 0, time);

    if (!(voxelDepthSize > 0.0)) voxelDepthSize = 1e-6;

    farPlane -= voxelDepthSize;

    if (farPlane < nearPlane) {
        setFloat("nearPlane", 0, time, farPlane);
    }

    return 1;
}


int
updateFarPlaneCallback(void* data, int /*idx*/, float time, const PRM_Template*)
{
   SOP_OpenVDB_Create* sop = static_cast<SOP_OpenVDB_Create*>(data);
   if (sop == NULL) return 0;
   return sop->updateFarPlane(time);
}


int
SOP_OpenVDB_Create::updateFarPlane(float time)
{
    fpreal
        nearPlane = evalFloat("nearPlane", 0, time),
        farPlane = evalFloat("farPlane", 0, time),
        voxelDepthSize = evalFloat("voxelDepthSize", 0, time);

    if (!(voxelDepthSize > 0.0)) voxelDepthSize = 1e-6;

    nearPlane += voxelDepthSize;

    if (farPlane < nearPlane) {
        setFloat("farPlane", 0, time, nearPlane);
    }

    return 1;
}


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable *table)
{
    if (table == NULL) return;

    hutil::ParmList parms;


    // Group name
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a name for this group of grids."));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR,"sep1", "Sep"));


    {   // Transform type
        const char* items[] = {
            "linear",  "Linear",
            "frustum", "Frustum",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD | PRM_TYPE_JOIN_NEXT, "transform", "Transform")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    // Toggle to preview the frustum
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "previewFrustum", "Preview")
        .setDefault(PRMoneDefaults));

    // Uniform voxel size
    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelSize", "Voxel size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setHelpText("Uniform voxel size in world units.\n(Ignored by frustum-aligned grids.)"));

    // Rotation
    parms.add(hutil::ParmFactory(PRM_XYZ_J, "rotation", "Rotation")
        .setHelpText("Rotation specified in ZYX order")
        .setVectorSize(3)
        .setDefault(PRMzeroDefaults));

    // Frustum settings
    // {

    parms.add(hutil::ParmFactory(PRM_STRING, "camera", "Camera")
        .setHelpText("Reference camera path")
        .setTypeExtended(PRM_TYPE_DYNAMIC_PATH)
        .setCallbackFunc(&updateNearFarCallback)
        .setSpareData(&PRM_SpareData::objCameraPath));

    parms.add(hutil::ParmFactory(PRM_FLT_J | PRM_TYPE_JOIN_NEXT, "nearPlane", "Near/far planes")
        .setHelpText("Near plane distance, should always be <= farPlane - voxelDepthSize\n"
            "Far plane distance, should always be >= nearPlane + voxelDepthSize")
        .setDefault(PRMzeroDefaults)
        .setCallbackFunc(&updateNearPlaneCallback)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 20));

    parms.add(hutil::ParmFactory(
        PRM_FLT_J | PRM_Type(PRM_Type::PRM_INTERFACE_LABEL_NONE), "farPlane", "")
        .setHelpText("Far plane distance, should always be >= nearPlane + voxelDepthSize")
        .setDefault(PRMoneDefaults)
        .setCallbackFunc(&updateFarPlaneCallback)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 20));

    parms.add(hutil::ParmFactory(PRM_INT_J, "voxelCount", "Voxel count")
        .setDefault(PRM100Defaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 200)
        .setHelpText("Horizontal voxel count for the near plane."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelDepthSize", "Voxel depth size")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setHelpText("The voxel depth (uniform z-size) in world units."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "cameraOffset", "Camera offset")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 20.0)
        .setHelpText("Adds padding to the view frustum without changing \n"
        "the near and far plane positions. Offsets the camera position \n"
        "in the reversed view direction."));

    // }

    // Grids Heading
    parms.add(hutil::ParmFactory(PRM_HEADING, "gridsHeading", ""));

    // Dynamic grid menu
    hutil::ParmList gridParms;
    {
        {   // Grid class menu
            std::vector<std::string> items;

            for (int i = 0; i < openvdb::NUM_GRID_CLASSES; ++i) {
                openvdb::GridClass cls = openvdb::GridClass(i);
                items.push_back(openvdb::GridBase::gridClassToString(cls)); // token
                items.push_back(openvdb::GridBase::gridClassToMenuName(cls)); // label
            }

            gridParms.add(hutil::ParmFactory(PRM_STRING, "gridClass#", "Grid class")
                .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        }

        {   // Element type menu
            std::vector<std::string> items;
            for (int i = 0; i < NUM_DATA_TYPES; ++i) {
                items.push_back(dataTypeToString(DataType(i))); // token
                items.push_back(dataTypeToMenuItems(DataType(i))); // label
            }
            gridParms.add(hutil::ParmFactory(PRM_STRING, "elementType#",  "   Type")
                .setHelpText("Set the voxel value type.")
                .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        }

        // Optional grid name string
        gridParms.add(hutil::ParmFactory(PRM_STRING, "gridName#", "Grid name")
            .setHelpText("Specify a name for this grid."));

        // Default background values
        // {
        const char* bgHelpStr =
            "The background value is a unique value that is\n"
            "returned when accessing any location in space\n"
            "that does not resolve to either a tile or a voxel.";
        gridParms.add(hutil::ParmFactory(PRM_FLT_J, "bgFloat#", "Background value")
            .setHelpText(bgHelpStr));
        gridParms.add(hutil::ParmFactory(PRM_INT_J, "bgInt#", "Background value")
            .setDefault(PRMoneDefaults)
            .setHelpText(bgHelpStr));
        gridParms.add(hutil::ParmFactory(PRM_INT_J, "bgBool#", "Background value")
            .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_RESTRICTED, 1)
            .setDefault(PRMoneDefaults)
            .setHelpText(bgHelpStr));
        gridParms.add(hutil::ParmFactory(PRM_FLT_J, "bgVec3f#", "Background value")
            .setVectorSize(3)
            .setHelpText(bgHelpStr));
        gridParms.add(hutil::ParmFactory(PRM_INT_J, "bgVec3i#", "Background value")
            .setVectorSize(3)
            .setHelpText(bgHelpStr));
        gridParms.add(hutil::ParmFactory(PRM_FLT_J, "width#", "Half voxel width")
            .setDefault(PRMthreeDefaults)
            .setRange(PRM_RANGE_RESTRICTED, 1.0, PRM_RANGE_UI, 10)
            .setHelpText("Half the width of the narrow band in voxel units, "
                "3 is optimal for level set operations."));
        // }

        // Vec type menu
        {
            std::vector<std::string> items;
            for (int i = 0; i < openvdb::NUM_VEC_TYPES ; ++i) {
                items.push_back(openvdb::GridBase::vecTypeToString(openvdb::VecType(i)));
                items.push_back(openvdb::GridBase::vecTypeExamples(openvdb::VecType(i)));
            }

            gridParms.add(hutil::ParmFactory(PRM_ORD, "vecType#", "Vector type")
                .setDefault(PRMzeroDefaults)
                .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        }
    }

    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "gridList", "Grids")
        .setMultiparms(gridParms)
        .setDefault(PRMoneDefaults));


    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING,
        "propertiesHeading", "Shared grid properties"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "frustumHeading", "Frustum grid settings"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "padding", "Padding"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Create", SOP_OpenVDB_Create::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addOptionalInput("Optional input to merge with");
}


////////////////////////////////////////


OP_Node *
SOP_OpenVDB_Create::factory(OP_Network *net, const char *name, OP_Operator *op)
{
    return new SOP_OpenVDB_Create(net, name, op);
}


SOP_OpenVDB_Create::SOP_OpenVDB_Create(OP_Network *net,
    const char *name, OP_Operator *op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


bool
SOP_OpenVDB_Create::updateParmsFlags()
{
    bool changed = false;
    UT_String tmpStr;

    bool frustum = evalInt("transform", 0, 0) == 1;

    for (int i = 1, N = evalInt("gridList", 0, 0); i <= N; ++i) {

        evalStringInst("gridClass#", &i, tmpStr, 0, 0);
        openvdb::GridClass gridClass = openvdb::GridBase::stringToGridClass(tmpStr.toStdString());

        evalStringInst("elementType#", &i, tmpStr, 0, 0);
        DataType eType = stringToDataType(tmpStr.toStdString());
        bool isLevelSet = false;

        // Force a specific data type for some of the grid classes
        if (gridClass == openvdb::GRID_LEVEL_SET) {
            eType = TYPE_FLOAT;
            isLevelSet = true;
        } else if (gridClass == openvdb::GRID_FOG_VOLUME) {
            eType = TYPE_FLOAT;
        } else if (gridClass == openvdb::GRID_STAGGERED) {
            eType = TYPE_VEC3S;
        }

        /// Disbale unused bg value options
        changed |= enableParmInst("bgFloat#", &i,
            !isLevelSet && (eType == TYPE_FLOAT || eType == TYPE_DOUBLE));
        changed |= enableParmInst("width#",   &i, isLevelSet);
        changed |= enableParmInst("bgInt#",   &i, eType == TYPE_INT || eType == TYPE_BOOL);
        changed |= enableParmInst("bgVec3f#", &i, eType == TYPE_VEC3S || eType == TYPE_VEC3D);
        changed |= enableParmInst("bgVec3i#", &i, eType == TYPE_VEC3I);
        changed |= enableParmInst("vecType#", &i, eType >= TYPE_VEC3S);

        // Hide unused bg value options.
        changed |= setVisibleStateInst("bgFloat#", &i,
            !isLevelSet && (eType == TYPE_FLOAT || eType == TYPE_DOUBLE));
        changed |= setVisibleStateInst("width#",   &i, isLevelSet);
        changed |= setVisibleStateInst("bgInt#",   &i, eType == TYPE_INT);
        changed |= setVisibleStateInst("bgBool#",  &i, eType == TYPE_BOOL);
        changed |= setVisibleStateInst("bgVec3f#", &i, eType == TYPE_VEC3S || eType == TYPE_VEC3D);
        changed |= setVisibleStateInst("bgVec3i#", &i, eType == TYPE_VEC3I);
        changed |= setVisibleStateInst("vecType#", &i, eType >= TYPE_VEC3S);

        // Enable different data types
        changed |= enableParmInst("elementType#", &i, gridClass == openvdb::GRID_UNKNOWN);
        changed |= setVisibleStateInst("elementType#", &i, gridClass == openvdb::GRID_UNKNOWN);
    }

    // linear transform
    changed |= enableParm("voxelSize", !frustum);
    changed |= enableParm("rotation", !frustum);

    changed |= setVisibleState("voxelSize", !frustum);
    changed |= setVisibleState("rotation", !frustum);

    // frustum transform
    UT_String cameraPath;
    evalString(cameraPath, "camera", 0, 0);
    cameraPath.harden();

    bool enableFrustumSettings = cameraPath.isstring() &&
        findOBJNode(cameraPath) != NULL;

    changed |= enableParm("camera", frustum);
    changed |= enableParm("voxelCount", frustum & enableFrustumSettings);
    changed |= enableParm("voxelDepthSize", frustum & enableFrustumSettings);
    changed |= enableParm("offset", frustum & enableFrustumSettings);
    changed |= enableParm("nearPlane", frustum & enableFrustumSettings);
    changed |= enableParm("farPlane", frustum & enableFrustumSettings);
    changed |= enableParm("cameraOffset", frustum & enableFrustumSettings);
    changed |= enableParm("previewFrustum", frustum & enableFrustumSettings);

    changed |= setVisibleState("camera", frustum);
    changed |= setVisibleState("voxelCount", frustum);
    changed |= setVisibleState("voxelDepthSize", frustum);
    changed |= setVisibleState("offset", frustum);
    changed |= setVisibleState("nearPlane", frustum);
    changed |= setVisibleState("farPlane", frustum);
    changed |= setVisibleState("cameraOffset", frustum);
    changed |= setVisibleState("previewFrustum", frustum);

    return changed;
}


////////////////////////////////////////


template<typename GridType>
void
SOP_OpenVDB_Create::createNewGrid(
    const UT_String& gridNameStr,
    const typename GridType::ValueType& background,
    const cvdb::math::Transform::Ptr& transform,
    GA_PrimitiveGroup* group,
    int gridClass,
    int vecType)
{
    // Create a grid of a pre-registered type and assign it a transform.
    hvdb::GridPtr newGrid = GridType::create(background);
    newGrid->setTransform(transform);

    newGrid->setGridClass(openvdb::GridClass(gridClass));
    if (vecType != -1) newGrid->setVectorType(openvdb::VecType(vecType));

    // Store the grid in a new VDB primitive and add the primitive
    // to the output geometry detail.
    GEO_PrimVDB* vdb = hvdb::createVdbPrimitive(*gdp, newGrid,
        gridNameStr.toStdString().c_str());

    // Add the primitive to the group.
    if (group) group->add(vdb);
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Create::cookMySop(OP_Context &context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        gdp->clearAndDestroy();
        if (nInputs() == 1) duplicateSource(0, context);

        fpreal time = context.getTime();

        // Create a group for the grid primitives.
        GA_PrimitiveGroup* group = NULL;
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        if(groupStr.isstring()) {
            group = gdp->newPrimitiveGroup(groupStr.buffer());
        }

        // Create a shared transform
        cvdb::math::Transform::Ptr transform;
        if (buildTransform(context, transform) >= UT_ERROR_ABORT) return error();


        // Create the grids
        UT_String gridNameStr, tmpStr;

        for (int i = 1, N = evalInt("gridList", 0, 0); i <= N; ++i) {

            evalStringInst("gridName#", &i, gridNameStr, 0, time);

            evalStringInst("gridClass#", &i, tmpStr, 0, time);
            openvdb::GridClass gridClass =
                openvdb::GridBase::stringToGridClass(tmpStr.toStdString());

            evalStringInst("elementType#", &i, tmpStr, 0, time);
            DataType eType = stringToDataType(tmpStr.toStdString());

            // Force a specific data type for some of the grid classes
            if (gridClass == openvdb::GRID_LEVEL_SET ||
                gridClass == openvdb::GRID_FOG_VOLUME) {
                eType = TYPE_FLOAT;
            } else if (gridClass == openvdb::GRID_STAGGERED) {
                eType = TYPE_VEC3S;
            }

            switch(eType) {
                case TYPE_FLOAT:
                {
                    float voxelSize = float(transform->voxelSize()[0]);
                    float background = 0.0;

                    if (gridClass == openvdb::GRID_LEVEL_SET) {
                        background = float(evalFloatInst("width#", &i, 0, time) * voxelSize);
                    } else {
                        background = float(evalFloatInst("bgFloat#", &i, 0, time));
                    }

                    createNewGrid<cvdb::FloatGrid>(gridNameStr, background,
                                                   transform, group, gridClass);
                    break;
                }
                case TYPE_DOUBLE:
                {
                    double background = double(evalFloatInst("bgFloat#", &i, 0, time));
                    createNewGrid<cvdb::DoubleGrid>(gridNameStr, background,
                                                   transform, group, gridClass);
                    break;
                }
                case TYPE_INT:
                {
                    int background = evalIntInst("bgInt#", &i, 0, time);
                    createNewGrid<cvdb::Int32Grid>(gridNameStr, background,
                                                   transform, group, gridClass);
                    break;
                }
                case TYPE_BOOL:
                {
                    bool background = evalIntInst("bgBool#", &i, 0, time);
                    createNewGrid<cvdb::BoolGrid>(gridNameStr, background,
                                                  transform, group, gridClass);
                    break;
                }
                case TYPE_VEC3S:
                {
                    cvdb::Vec3f background(float(evalFloatInst("bgVec3f#", &i, 0, time)),
                                           float(evalFloatInst("bgVec3f#", &i, 1, time)),
                                           float(evalFloatInst("bgVec3f#", &i, 2, time)));

                    int vecType = evalIntInst("vecType#", &i, 0, time);

                    createNewGrid<cvdb::Vec3SGrid>(gridNameStr, background,
                                                   transform, group, gridClass, vecType);
                    break;
                }
                case TYPE_VEC3D:
                {
                    cvdb::Vec3d background(double(evalFloatInst("bgVec3f#", &i, 0, time)),
                                           double(evalFloatInst("bgVec3f#", &i, 1, time)),
                                           double(evalFloatInst("bgVec3f#", &i, 2, time)));

                    int vecType = evalIntInst("vecType#", &i, 0, time);

                    createNewGrid<cvdb::Vec3DGrid>(gridNameStr, background,
                                                   transform, group, gridClass, vecType);
                    break;
                }
                case TYPE_VEC3I:
                {
                    cvdb::Vec3i background(evalIntInst("bgVec3i#", &i, 0, time),
                                           evalIntInst("bgVec3i#", &i, 1, time),
                                           evalIntInst("bgVec3i#", &i, 2, time));
                    int vecType = evalIntInst("vecType#", &i, 0, time);
                    createNewGrid<cvdb::Vec3IGrid>(gridNameStr, background,
                                                   transform, group, gridClass, vecType);
                    break;
                }
            } // eType switch
        } // grid create loop

    } catch ( std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Create::buildTransform(OP_Context& context, openvdb::math::Transform::Ptr& transform)
{
    fpreal time = context.getTime();
    bool frustum = evalInt("transform", 0, time) == 1;

    if (frustum) { // nonlinear frustum transform

        UT_String cameraPath;
        evalString(cameraPath, "camera", 0, time);
        cameraPath.harden();
        if (!cameraPath.isstring()) {
            addError(SOP_MESSAGE, "No camera selected");
            return error();
        }

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

        const float
            offset = static_cast<float>(evalFloat("cameraOffset", 0, time)),
            nearPlane = static_cast<float>(evalFloat("nearPlane", 0, time)),
            farPlane = static_cast<float>(evalFloat("farPlane", 0, time)),
            voxelDepthSize = static_cast<float>(evalFloat("voxelDepthSize", 0, time));
        const int voxelCount = evalInt("voxelCount", 0, time);

        transform = hvdb::frustumTransformFromCamera(*this, context, *cam,
            offset, nearPlane, farPlane, voxelDepthSize, voxelCount);

        if (bool(evalInt("previewFrustum", 0, time))) {
            UT_Vector3 boxColor(0.6f, 0.6f, 0.6f);
            UT_Vector3 tickColor(0.0f, 0.0f, 0.0f);
            hvdb::drawFrustum(*gdp, *transform,
                &boxColor, &tickColor, /*shaded*/true);
        }

    } else { // linear affine transform

        const double voxelSize = double(evalFloat("voxelSize", 0, time));

        openvdb::Vec3d rotation(
            evalFloat("rotation", 0, time),
            evalFloat("rotation", 1, time),
            evalFloat("rotation", 2, time));

        if (std::abs(rotation.x()) < 0.00001 && std::abs(rotation.y()) < 0.00001
            && std::abs(rotation.z()) < 0.00001) {
            transform = openvdb::math::Transform::createLinearTransform(voxelSize);
        } else {

            openvdb::math::Mat4d xform(openvdb::math::Mat4d::identity());

            xform.preRotate(openvdb::math::X_AXIS, rotation.x());
            xform.preRotate(openvdb::math::Y_AXIS, rotation.y());
            xform.preRotate(openvdb::math::Z_AXIS, rotation.z());
            xform.preScale(openvdb::Vec3d(voxelSize));

            transform = openvdb::math::Transform::createLinearTransform(xform);
        }
    }

    return error();
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
