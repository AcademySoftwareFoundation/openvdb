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
/// @file SOP_OpenVDB_From_Polygons.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Converts a closed mesh of trinagles and/or quads into different VDB volumes.
/// The supported volumes are: Signed distance field / level-set, closest primitive grid
/// and grids with different mesh attributes (closest UVW, Normal etc.)

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/util/Util.h>

#include <CH/CH_Manager.h>
#include <PRM/PRM_Parm.h>
#include <PRM/PRM_SharedFunc.h>

#include <algorithm> // for std::max()
#include <sstream>
#include <string>
#include <limits>
#include <vector>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


namespace {

enum AttributeClass { POINT_ATTR, VERTEX_ATTR, PRIMITIVE_ATTR };

inline bool
evalAttrType(const UT_String& attrStr, UT_String& attrName, int& attrClass)
{
    std::string str = attrStr.toStdString();

    const size_t idx = str.find_first_of(".");
    if (idx == std::string::npos) return false;

    attrName = str.substr(idx + 1, str.size() - 1);

    str = str.substr(0, 2);

    if (str == "po") attrClass = POINT_ATTR;
    else if (str == "ve") attrClass = VERTEX_ATTR;
    else if (str == "pr") attrClass = PRIMITIVE_ATTR;
    else return false;

    return true;
}

inline int
lookupAttrInput(const PRM_SpareData* spare)
{
    const char  *istring;
    if (!spare) return 0;
    istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}

inline void
sopBuildAttrMenu(void* data, PRM_Name* menuEntries, int themenusize,
    const PRM_SpareData* spare, const PRM_Parm*)
{
    if (data == nullptr || menuEntries == nullptr || spare == nullptr) return;

    SOP_Node* sop = CAST_SOPNODE(static_cast<OP_Node*>(data));

    if (sop == nullptr) {
        // terminate and quit
        menuEntries[0].setToken(0);
        menuEntries[0].setLabel(0);
        return;
    }

    int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());
    size_t menuIdx = 0, menuEnd(themenusize - 2);

    if (gdp) {

        // null object
        menuEntries[menuIdx].setToken("");
        menuEntries[menuIdx++].setLabel("- no attribute selected -");

        // point attribute names
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);

        if(!iter.atEnd() && menuIdx != menuEnd) {

            if (menuIdx > 0) {
                menuEntries[menuIdx].setToken(PRM_Name::mySeparator);
                menuEntries[menuIdx++].setLabel(PRM_Name::mySeparator);
            }

            for (; !iter.atEnd() && menuIdx != menuEnd; ++iter) {

                std::ostringstream token;
                token << "point." << (*iter)->getName();

                menuEntries[menuIdx].setToken(token.str().c_str());
                menuEntries[menuIdx++].setLabel(token.str().c_str());
            }
        }


        // vertex attribute names
        iter = gdp->vertexAttribs().begin(GA_SCOPE_PUBLIC);

        if(!iter.atEnd() && menuIdx != menuEnd) {

            if (menuIdx > 0) {
                menuEntries[menuIdx].setToken(PRM_Name::mySeparator);
                menuEntries[menuIdx++].setLabel(PRM_Name::mySeparator);
            }

            for (; !iter.atEnd() && menuIdx != menuEnd; ++iter) {

                std::ostringstream token;
                token << "vertex." << (*iter)->getName();

                menuEntries[menuIdx].setToken(token.str().c_str());
                menuEntries[menuIdx++].setLabel(token.str().c_str());
            }
        }


        // primitive attribute names
        iter = gdp->primitiveAttribs().begin(GA_SCOPE_PUBLIC);

        if(menuIdx != menuEnd) {

            if (menuIdx > 0) {
                menuEntries[menuIdx].setToken(PRM_Name::mySeparator);
                menuEntries[menuIdx++].setLabel(PRM_Name::mySeparator);
            }

            for (; !iter.atEnd() && menuIdx != menuEnd; ++iter) {

                std::ostringstream token;
                token << "primitive." << (*iter)->getName();

                menuEntries[menuIdx].setToken(token.str().c_str());
                menuEntries[menuIdx++].setLabel(token.str().c_str());
            }

            // Special case
            menuEntries[menuIdx].setToken("primitive.primitive_list_index");
            menuEntries[menuIdx++].setLabel("primitive.primitive_list_index");
        }

    }

    // terminator
    menuEntries[menuIdx].setToken(0);
    menuEntries[menuIdx].setLabel(0);
}

const PRM_ChoiceList PrimAttrMenu(
    PRM_ChoiceListType(PRM_CHOICELIST_EXCLUSIVE | PRM_CHOICELIST_REPLACE), sopBuildAttrMenu);

} // unnamed namespace


////////////////////////////////////////


class SOP_OpenVDB_From_Polygons: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_From_Polygons(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_From_Polygons() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i ) const override { return (i == 1); }

    int convertUnits();

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
    void resolveObsoleteParms(PRM_ParmList*) override;

    int constructGenericAtttributeLists(
        hvdb::AttributeDetailList &pointAttributes,
        hvdb::AttributeDetailList &vertexAttributes,
        hvdb::AttributeDetailList &primitiveAttributes,
        const GU_Detail&,
        const openvdb::Int32Grid& closestPrimGrid,
        const float time);

    template <class ValueType>
    void addAttributeDetails(
        hvdb::AttributeDetailList &attributeList,
        const GA_Attribute *attribute,
        const GA_AIFTuple *tupleAIF,
        const int attrTupleSize,
        const openvdb::Int32Grid& closestPrimGrid,
        std::string& customName,
        int vecType = -1);

    void transferAttributes(
        hvdb::AttributeDetailList &pointAttributes,
        hvdb::AttributeDetailList &vertexAttributes,
        hvdb::AttributeDetailList &primitiveAttributes,
        const openvdb::Int32Grid&,
        openvdb::math::Transform::Ptr& transform,
        const GU_Detail&);

private:
    float mVoxelSize;
};


////////////////////////////////////////


namespace
{

// Callback to convert from voxel to world space units
int
convertUnitsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
   SOP_OpenVDB_From_Polygons* sop = static_cast<SOP_OpenVDB_From_Polygons*>(data);
   if (sop == nullptr) return 0;
   return sop->convertUnits();
}

} // unnamed namespace


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;


    //////////
    // Output grids

    //  distance field
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "distanceField", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the level set output.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_STRING, "distanceFieldGridName", "Distance VDB")
        .setDefault("surface")
        .setTooltip(
            "Output a signed distance field VDB with the given name.\n\n"
            "An SDF stores the distance to the surface in each voxel."
            " If a voxel is inside the surface, the distance is negative."));

    //  fog volume
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fogVolume", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Enable / disable the fog volume output.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_STRING, "fogVolumeGridName", "Fog VDB")
        .setDefault("density")
        .setTooltip(
            "Output a fog volume VDB with the given name.\n\n"
            "Voxels inside the surface have value one, and voxels outside"
            " have value zero.  Within a narrow band centered on the surface,"
            " voxel values vary linearly from zero to one.\n\n"
            "Turn on __Fill Interior__ to create a solid VDB"
            " (from an airtight surface) instead of a narrow band."));

    //////////
    // Conversion settings

    parms.add(hutil::ParmFactory(PRM_HEADING, "conversionHeading", "Conversion settings"));

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip(
            "Give the output VDB the same orientation and voxel size as the selected VDB,"
            " and match the narrow band width if the reference VDB is a level set.")
        .setDocumentation(
            "Give the output VDB the same orientation and voxel size as"
            " the selected VDB (see [specifying volumes|/model/volumes#group])"
            " and match the narrow band width if the reference VDB is a level set."));

    {// Voxel size or voxel count menu
        const auto items = std::vector<std::string>{
            "worldVoxelSize",   "Size in World Units",
            "countX",           "Count Along X Axis",
            "countY",           "Count Along Y Axis",
            "countZ",           "Count Along Z Axis",
            "countLongest",     "Count Along Longest Axis"
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "sizeOrCount", "Voxel")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(items[0])
            .setTooltip(
                "How to specify the voxel size: either in world units or as"
                " a voxel count along one axis"));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelSize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setTooltip(
            "The desired voxel size in world units\n\n"
            "Surface features smaller than this will not be represented in the output VDB."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "voxelCount", "Voxel Count")
        .setDefault(100)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 500)
        .setTooltip(
            "The desired voxel count along one axis\n\n"
            "The resulting voxel count might be off by one voxel"
            " due to roundoff errors during the conversion process."));

    // Narrow-band width {
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits",
        "Use World Space Units for Narrow Band")
        .setCallbackFunc(&convertUnitsCB)
        .setTooltip(
            "If enabled, specify the narrow band width in world units,"
            " otherwise in voxels."));

    //   voxel space units
    parms.add(hutil::ParmFactory(PRM_INT_J, "exteriorBandWidth", "Exterior Band Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setTooltip(
            "The width of the exterior (distance >= 0) portion of the narrow band\n"
            "Many level set operations require a minimum of three voxels.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_INT_J, "interiorBandWidth", "Interior Band Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setTooltip(
            "The width of the interior (distance < 0) portion of the narrow band\n"
            "Many level set operations require a minimum of three voxels.")
        .setDocumentation(nullptr));

    //   world space units
    parms.add(hutil::ParmFactory(PRM_FLT_J, "exteriorBandWidthWS", "Exterior Band")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setTooltip("The width of the exterior (distance >= 0) portion of the narrow band")
        .setDocumentation(
            "The width of the exterior (_distance_ => 0) portion of the narrow band"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "interiorBandWidthWS", "Interior Band")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setTooltip("The width of the interior (distance < 0) portion of the narrow band")
        .setDocumentation(
            "The width of the interior (_distance_ < 0) portion of the narrow band"));
    // }

    // Options
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fillInterior", "Fill Interior")
        .setTooltip(
            "Extract signed distances for all interior voxels.\n\n"
            "This operation densifies the interior of the model."
            " It requires a closed, watertight surface."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "unsignedDist", "Unsigned Distance Field")
        .setTooltip(
            "Generate an unsigned distance field.\n"
            "This operation will work on any surface, whether or not it is closed or watertight.")
        .setDocumentation(
            "Generate an unsigned distance field.\n\n"
            "This operation will work on any surface, whether or not"
            " it is closed or watertight.  It is similar to the Minimum"
            " function of the [Node:sop/isooffset] node."));

    //////////
    // Mesh attribute transfer {Point, Vertex & Primitive}

    parms.add(hutil::ParmFactory(PRM_HEADING, "transferHeading", "Attribute Transfer"));

    hutil::ParmList attrParms;

    // Attribute name
    attrParms.add(hutil::ParmFactory(PRM_STRING, "attribute#",  "Attribute")
        .setChoiceList(&PrimAttrMenu)
        .setSpareData(&SOP_Node::theFirstInput)
        .setTooltip(
            "A point, vertex, or primitive attribute from which to create a VDB\n\n"
            "Supports integer and floating point attributes of arbitrary"
            " precision and tuple size."));

    attrParms.add(hutil::ParmFactory(PRM_STRING, "attributeGridName#", "VDB Name")
        .setTooltip("The name for this VDB primitive (leave blank to use the attribute's name)"));

    // Vec type menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < openvdb::NUM_VEC_TYPES ; ++i) {
            items.push_back(openvdb::GridBase::vecTypeToString(openvdb::VecType(i)));
            items.push_back(openvdb::GridBase::vecTypeExamples(openvdb::VecType(i)));
        }

        attrParms.add(hutil::ParmFactory(PRM_ORD, "vecType#", "Vector Type")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("How vector values should be interpreted"));
    }

    // Add multi parm
    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "attrList", "Surface Attributes")
        .setMultiparms(attrParms)
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Generate additional VDB primitives that store the values of"
            " primitive (face), point, or vertex attributes.")
        .setDocumentation(
            "Generate additional VDB primitives that store the values of primitive"
            " (face), point, or vertex [attributes|/model/attributes].\n\n"
            "Only voxels in the narrow band around the surface will be set."));


    //////////
    // Obsolete parameters

    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "optionsHeading", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "otherHeading", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "attrHeading", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "isoOffset",  ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "gradientWidth", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "customGradientWidth", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "sdfHeading", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "addSdfGridName", "")); // fix
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "sdfGridName", "")); // fix
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "outputClosestPrimGrid", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "closestPrimGridName", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "transformHeading", "Transform"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "outputHeading", "Output grids"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "hermiteData", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "hermiteDataGridName", "Hermite VDB"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "matchlevelset", ""));

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB From Polygons",
        SOP_OpenVDB_From_Polygons::factory, parms, *table)
        .addAlias("OpenVDB From Mesh")
        .addAlias("OpenVDB Mesh Voxelizer")
        .setObsoleteParms(obsoleteParms)
        .addInput("Polygons to Convert")
        .addOptionalInput("Optional Reference VDB (for transform matching)")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Convert polygonal surfaces and/or surface attributes into VDB volumes.\"\"\"\n\
\n\
@overview\n\
\n\
This node can create signed or unsigned distance fields\n\
and/or density fields (\"fog volumes\") from polygonal surfaces.\n\
\n\
When you create a fog volume you can choose either to fill the band of voxels\n\
on the surface or (if you have an airtight surface) to fill the interior\n\
of the surface (see the __Fill interior__ parameter).\n\
\n\
Since the resulting VDB volumes store only the voxels near the surface,\n\
they can have a much a higher effective resolution than a traditional volume\n\
created with [Node:sop/isooffset].\n\
\n\
You can connect a VDB to the second input to automatically use that VDB's\n\
orientation and voxel size (see the __Reference VDB__ parameter).\n\
\n\
NOTE:\n\
    The input geometry must be a quad or triangle mesh.\n\
    This node will convert the input surface into such a mesh if necessary.\n\
\n\
@inputs\n\
\n\
Polygonal mesh to convert:\n\
    The polygonal surface to convert.\n\
Optional reference VDB:\n\
    If connected, give the output VDB the same orientation and voxel size\n\
    as a VDB from this input.\n\
\n\
@related\n\
- [OpenVDB Create|Node:sop/DW_OpenVDBCreate]\n\
- [OpenVDB From Particles|Node:sop/DW_OpenVDBFromParticles]\n\
- [Node:sop/isooffset]\n\
- [Node:sop/vdbfrompolygons]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_From_Polygons::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_From_Polygons(net, name, op);
}


SOP_OpenVDB_From_Polygons::SOP_OpenVDB_From_Polygons(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
    , mVoxelSize(0.1f)
{
}


////////////////////////////////////////


int
SOP_OpenVDB_From_Polygons::convertUnits()
{
    const bool toWSUnits = static_cast<bool>(evalInt("worldSpaceUnits", 0, 0));
    float width;

    if (toWSUnits) {
        width = static_cast<float>(evalInt("exteriorBandWidth", 0, 0));
        setFloat("exteriorBandWidthWS", 0, 0, width * mVoxelSize);

        width = static_cast<float>(evalInt("interiorBandWidth", 0, 0));
        setFloat("interiorBandWidthWS", 0, 0, width * mVoxelSize);
        return 1;
    }

    width = static_cast<float>(evalFloat("exteriorBandWidthWS", 0, 0));
    int voxelWidth = std::max(static_cast<int>(width / mVoxelSize), 1);
    setInt("exteriorBandWidth", 0, 0, voxelWidth);

    width = static_cast<float>(evalFloat("interiorBandWidthWS", 0, 0));
    voxelWidth = std::max(static_cast<int>(width / mVoxelSize), 1);
    setInt("interiorBandWidth", 0, 0, voxelWidth);

    return 1;
}


////////////////////////////////////////

void
SOP_OpenVDB_From_Polygons::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;
    const fpreal time = CHgetEvalTime();

    PRM_Parm* parm = obsoleteParms->getParmPtr("sdfGridName");

    if (parm && !parm->isFactoryDefault()) {

        UT_String gridNameStr;
        obsoleteParms->evalString(gridNameStr, "sdfGridName", 0, time);

        if (gridNameStr.length() > 0) {

            if(bool(evalInt("fogVolume", 0, time))) {
                setString(gridNameStr, CH_STRING_LITERAL, "fogVolumeGridName", 0, time);
            } else {
                setString(gridNameStr, CH_STRING_LITERAL, "distanceFieldGridName", 0, time);
            }
        }
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}



////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_From_Polygons::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = 0; // No point using CHgetTime as that is unstable.

    int refexists = (nInputs() == 2);

    // Transform
    changed |= enableParm("group", refexists);

    // Conversion
    const bool wsUnits = bool(evalInt("worldSpaceUnits", 0, time));
    const bool fillInterior = bool(evalInt("fillInterior", 0, time));
    const bool unsignedDist = bool(evalInt("unsignedDist", 0, time));

    // Voxel size or voxel count menu
    UT_String str;
    evalString(str, "sizeOrCount", 0, time);
    const bool countMenu =  str != "worldVoxelSize";
    changed |= setVisibleState("voxelSize", !countMenu);
    changed |= setVisibleState("voxelCount", countMenu);
    changed |= enableParm("voxelSize", !countMenu && !refexists);
    changed |= enableParm("voxelCount", countMenu && !refexists);

    changed |= enableParm("interiorBandWidth",
        !wsUnits && !fillInterior && !unsignedDist);

    changed |= enableParm("interiorBandWidthWS",
        wsUnits && !fillInterior && !unsignedDist);

    changed |= enableParm("exteriorBandWidth", !wsUnits);
    changed |= enableParm("exteriorBandWidthWS", wsUnits);

    changed |= setVisibleState("interiorBandWidth", !wsUnits);
    changed |= setVisibleState("exteriorBandWidth", !wsUnits);
    changed |= setVisibleState("interiorBandWidthWS", wsUnits);
    changed |= setVisibleState("exteriorBandWidthWS", wsUnits);

    changed |= enableParm("fillInterior", !unsignedDist);

    // Output
    changed |= enableParm("distanceFieldGridName", bool(evalInt("distanceField", 0, time)));
    changed |= enableParm("fogVolumeGridName",
        bool(evalInt("fogVolume", 0, time)) && !unsignedDist);
    changed |= enableParm("fogVolume", !unsignedDist);

    // enable / diable vector type menu
    UT_String attrStr, attrName;
    GA_ROAttributeRef attrRef;
    int attrClass = POINT_ATTR;
    const GU_Detail* meshGdp = this->getInputLastGeo(0, time);
    if (meshGdp) {
        for (int i = 1, N = static_cast<int>(evalInt("attrList", 0, time)); i <= N; ++i) {

            evalStringInst("attribute#", &i, attrStr, 0, time);
            bool isVector = false;

            if (attrStr.length() != 0 && evalAttrType(attrStr, attrName, attrClass)) {

                if (attrClass == POINT_ATTR) {
                    attrRef = meshGdp->findPointAttribute(attrName);
                } else if (attrClass == VERTEX_ATTR) {
                    attrRef = meshGdp->findVertexAttribute(attrName);
                } else if (attrClass == PRIMITIVE_ATTR) {
                    attrRef = meshGdp->findPrimitiveAttribute(attrName);
                }

                if (attrRef.isValid()) {

                    const GA_Attribute *attr = attrRef.getAttribute();

                    if (attr) {
                        const GA_TypeInfo typeInfo = attr->getTypeInfo();
                        isVector = (typeInfo == GA_TYPE_HPOINT || typeInfo == GA_TYPE_POINT
                            || typeInfo == GA_TYPE_VECTOR || typeInfo == GA_TYPE_NORMAL);

                        if (!isVector) {
                            const GA_AIFTuple *tupleAIF = attr->getAIFTuple();
                            if (tupleAIF) isVector = tupleAIF->getTupleSize(attr) == 3;
                        }
                    }
                }
            }

            changed |= enableParmInst("vecType#", &i, isVector);
            changed |= setVisibleStateInst("vecType#", &i, isVector);
        }
    }
    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_From_Polygons::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();
        const fpreal time = context.getTime();

        hvdb::Interrupter boss("Converting geometry to volume");

        //////////
        // Validate the input

        const GU_Detail* inputGdp = inputGeo(0);

        if (!inputGdp || !inputGdp->getNumPrimitives()) {
            addWarning(SOP_MESSAGE, "No mesh to convert");
            // We still create the grids as later workflow
            // may be able to handle an empty grid.
        }

        // Validate geometry
        std::string warningStr;
        auto geoPtr = hvdb::convertGeometry(*inputGdp, warningStr, &boss);
        if (geoPtr) {
            inputGdp = geoPtr.get();
            if (!warningStr.empty()) addWarning(SOP_MESSAGE, warningStr.c_str());
        }

        //////////
        // Evaluate the UI parameters.

        const bool outputDistanceField = bool(evalInt("distanceField", 0, time));
        const bool unsignedDistanceFieldConversion = bool(evalInt("unsignedDist", 0, time));
        const bool outputFogVolumeGrid = bool(evalInt("fogVolume", 0, time));
        const bool outputAttributeGrid = bool(evalInt("attrList", 0, time) > 0);


        if (!outputDistanceField && !outputFogVolumeGrid && !outputAttributeGrid) {

            addWarning(SOP_MESSAGE, "No output selected");
            return error();
        }

        openvdb::math::Transform::Ptr transform;

        float inBand = std::numeric_limits<float>::max(), exBand = 0.0;

        const GU_Detail* refGdp = inputGeo(1);
        bool secondinput = refGdp != nullptr;

        if (secondinput) {

            // Get the first grid's transform

            UT_String groupStr;
            evalString(groupStr, "group", 0, time);

            const GA_PrimitiveGroup *refGroup = matchGroup(*refGdp, groupStr.toStdString());

            hvdb::VdbPrimCIterator gridIter(refGdp, refGroup);

            if (gridIter) {
                transform = (*gridIter)->getGrid().transform().copy();
                mVoxelSize = static_cast<float>(transform->voxelSize()[0]);
                ++gridIter;
            } else {
                 addError(SOP_MESSAGE, "Could not find a reference grid");
                return error();
            }

        } else {// derive the voxel size and define output grid's transform

            UT_String str;
            evalString(str, "sizeOrCount", 0, time);
            if ( str == "worldVoxelSize" ) {
                mVoxelSize = static_cast<float>(evalFloat("voxelSize", 0, time));
            } else {
                const float dim = static_cast<float>(evalInt("voxelCount", 0, time));
                UT_BoundingBox bbox;
                inputGdp->getCachedBounds(bbox);
                const float size = str == "countX" ? bbox.xsize() : str == "countY" ? bbox.ysize() :
                                   str == "countZ" ? bbox.ysize() : bbox.sizeMax();
                if ( evalInt("worldSpaceUnits", 0, time) ) {
                    const float w = static_cast<float>(evalFloat("exteriorBandWidthWS", 0, time));
                    mVoxelSize = (size + 2.0f*w)/dim;
                } else {
                    const float w = static_cast<float>(evalInt("exteriorBandWidth", 0, time));
                    mVoxelSize = size/std::max(1.0f, dim - 2.0f*w);
                }
            }
            // Create a new transform
            transform = openvdb::math::Transform::createLinearTransform(mVoxelSize);
        }

        if (mVoxelSize < 1e-5) {
            std::ostringstream ostr;
            ostr << "The voxel size ("<< mVoxelSize << ") is too small.";
            addError(SOP_MESSAGE, ostr.str().c_str());
            return error();
        }

        // Set the narrow-band parameters
        {
            const bool wsUnits = static_cast<bool>(evalInt("worldSpaceUnits", 0, time));

            if (wsUnits) {
                exBand = static_cast<float>(evalFloat("exteriorBandWidthWS", 0, time) / mVoxelSize);
            } else {
                exBand = static_cast<float>(evalInt("exteriorBandWidth", 0, time));
            }
            if (!bool(evalInt("fillInterior", 0, time))) {
                if (wsUnits) {
                    inBand = static_cast<float>(
                        evalFloat("interiorBandWidthWS", 0, time) / mVoxelSize);
                } else {
                    inBand = static_cast<float>(evalInt("interiorBandWidth", 0, time));
                }
            }
        }

        //////////
        // Copy the input mesh and transform to local grid space.

        std::vector<openvdb::Vec3s> pointList;
        std::vector<openvdb::Vec4I> primList;

        if (!boss.wasInterrupted()) {

            pointList.resize(inputGdp->getNumPoints());
            primList.resize(inputGdp->getNumPrimitives());

            UTparallelFor(GA_SplittableRange(inputGdp->getPointRange()),
                hvdb::TransformOp(inputGdp, *transform, pointList));

            UTparallelFor(GA_SplittableRange(inputGdp->getPrimitiveRange()),
                hvdb::PrimCpyOp(inputGdp, primList));
        }

        //////////
        // Mesh to volume conversion


        openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I>
            mesh(pointList, primList);

        int conversionFlags = unsignedDistanceFieldConversion ?
            openvdb::tools::UNSIGNED_DISTANCE_FIELD : 0;


        openvdb::Int32Grid::Ptr primitiveIndexGrid;

        if (outputAttributeGrid) {
            primitiveIndexGrid.reset(new openvdb::Int32Grid(0));
        }

        openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
            boss, mesh, *transform, exBand, inBand, conversionFlags, primitiveIndexGrid.get());

        //////////
        // Output

        // Distance field / level set
        if (!boss.wasInterrupted() && outputDistanceField) {
            UT_String gridNameStr;
            evalString(gridNameStr, "distanceFieldGridName", 0, time);
            hvdb::createVdbPrimitive(*gdp, grid, gridNameStr.toStdString().c_str());
        }


        // Fog volume
        if (!boss.wasInterrupted() && outputFogVolumeGrid && !unsignedDistanceFieldConversion) {

            // If no level set grid is exported the original level set
            // grid is modified in place.
            openvdb::FloatGrid::Ptr outputGrid;

            if (outputDistanceField) {
                outputGrid = grid->deepCopy();
            } else {
                outputGrid = grid;
            }

            openvdb::tools::sdfToFogVolume(*outputGrid);

            UT_String gridNameStr;
            evalString(gridNameStr, "fogVolumeGridName", 0, time);
            hvdb::createVdbPrimitive(*gdp, outputGrid, gridNameStr.toStdString().c_str());
        }

        // Transfer mesh attributes
        if (!boss.wasInterrupted() && outputAttributeGrid) {

            hvdb::AttributeDetailList pointAttributes;
            hvdb::AttributeDetailList vertexAttributes;
            hvdb::AttributeDetailList primitiveAttributes;

            int closestPrimIndexInstance =
                constructGenericAtttributeLists(pointAttributes, vertexAttributes,
                    primitiveAttributes, *inputGdp, *primitiveIndexGrid, float(time));

            transferAttributes(pointAttributes, vertexAttributes, primitiveAttributes,
                *primitiveIndexGrid, transform, *inputGdp);

            // Export the closest prim idx grid.
            if (!boss.wasInterrupted() && closestPrimIndexInstance > -1) {
                UT_String gridNameStr;
                evalStringInst("attributeGridName#", &closestPrimIndexInstance,
                    gridNameStr, 0, time);
                if (gridNameStr.length() == 0) gridNameStr = "primitive_list_index";
                hvdb::createVdbPrimitive(
                    *gdp, primitiveIndexGrid, gridNameStr.toStdString().c_str());
            }
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////


// Helper method constructs the attribute detail lists
int
SOP_OpenVDB_From_Polygons::constructGenericAtttributeLists(
    hvdb::AttributeDetailList &pointAttributes,
    hvdb::AttributeDetailList &vertexAttributes,
    hvdb::AttributeDetailList &primitiveAttributes,
    const GU_Detail& meshGdp,
    const openvdb::Int32Grid& closestPrimGrid,
    const float time)
{
    UT_String attrStr, attrName;
    GA_ROAttributeRef attrRef;
    GA_Range range;
    int attrClass = POINT_ATTR;
    int closestPrimIndexInstance = -1;

    // for each selected attribute
    for (int i = 1, N = static_cast<int>(evalInt("attrList", 0, time)); i <= N; ++i) {

        evalStringInst("attribute#", &i, attrStr, 0, time);

        if (attrStr.length() == 0) continue;

        if (!evalAttrType(attrStr, attrName, attrClass)) {
            std::ostringstream ostr;
            ostr << "Skipped attribute with unrecognized class {point/vertex/prim}: "<< attrStr;
            addWarning(SOP_MESSAGE, ostr.str().c_str());
            continue;
        }

        hvdb::AttributeDetailList* attributeList = nullptr;

        if (attrClass == POINT_ATTR) {
            attrRef = meshGdp.findPointAttribute(attrName);
            attributeList = &pointAttributes;
        } else if (attrClass == VERTEX_ATTR) {
            attrRef = meshGdp.findVertexAttribute(attrName);
            attributeList = &vertexAttributes;
        } else if (attrClass == PRIMITIVE_ATTR) {

            if (attrName == "primitive_list_index") {
                // The closest prim idx grid is a special case,
                // the converter has already generated it for us.
                closestPrimIndexInstance = i;
                continue;
            }
            attrRef = meshGdp.findPrimitiveAttribute(attrName);
            attributeList = &primitiveAttributes;
        }

        if (attrName.length() == 0 || !attrRef.isValid()) {
            std::ostringstream ostr;
            ostr << "Skipped unrecognized attribute: "<< attrName;
            addWarning(SOP_MESSAGE, ostr.str().c_str());
            continue;
        }

        evalStringInst("attributeGridName#", &i, attrStr, 0, time);
        std::string customName = attrStr.toStdString();
        int vecType = static_cast<int>(evalIntInst("vecType#", &i, 0, time));


        const GA_Attribute *attr = attrRef.getAttribute();
        if (!attr) {
            std::ostringstream ostr;
            ostr << "Skipped unrecognized attribute type for: "<< attrName;
            addWarning(SOP_MESSAGE, ostr.str().c_str());
            continue;
        }

        const GA_AIFTuple *tupleAIF = attr->getAIFTuple();
        if (!tupleAIF) {
            std::ostringstream ostr;
            ostr << "Skipped unrecognized attribute type for: "<< attrName;
            addWarning(SOP_MESSAGE, ostr.str().c_str());
            continue;
        }


        const GA_Storage attrStorage = tupleAIF->getStorage(attr);
        const int attrTupleSize = tupleAIF->getTupleSize(attr);
        const GA_TypeInfo typeInfo = attr->getTypeInfo();

        const bool interpertAsVector = (typeInfo == GA_TYPE_HPOINT || typeInfo == GA_TYPE_POINT
            || typeInfo == GA_TYPE_VECTOR || typeInfo == GA_TYPE_NORMAL);

        switch (attrStorage)
        {
            case GA_STORE_INT16:
            case GA_STORE_INT32:

                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3i>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName, vecType);
                } else {
                    addAttributeDetails<openvdb::Int32>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName);
                }

                break;
            case GA_STORE_INT64:
                addAttributeDetails<openvdb::Int64>
                    (*attributeList, attr, tupleAIF, attrTupleSize, closestPrimGrid, customName);
                break;
            case GA_STORE_REAL16:
            case GA_STORE_REAL32:

                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3s>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName, vecType);
                } else {
                    addAttributeDetails<float>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName);
                }

                break;
            case GA_STORE_REAL64:

                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3d>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName, vecType);
                } else {
                    addAttributeDetails<double>(*attributeList, attr, tupleAIF,
                        attrTupleSize, closestPrimGrid, customName);
                }

                break;
            default:
                addWarning(SOP_MESSAGE, "Skipped unrecognized attribute type");
                break;
        }
    }

    return closestPrimIndexInstance;
}


////////////////////////////////////////


template <class ValueType>
void
SOP_OpenVDB_From_Polygons::addAttributeDetails(
    hvdb::AttributeDetailList &attributeList,
    const GA_Attribute *attribute,
    const GA_AIFTuple *tupleAIF,
    const int attrTupleSize,
    const openvdb::Int32Grid& closestPrimGrid,
    std::string& customName,
    int vecType)
{
    // Defines a new type of a tree having the same hierarchy as the incoming
    // Int32Grid's tree but potentially a different value type.
    using TreeType = typename openvdb::Int32Grid::TreeType::ValueConverter<ValueType>::Type;
    using GridType = typename openvdb::Grid<TreeType>;

    if (vecType != -1) { // Vector grid
         // Get the attribute's default value.
         ValueType defValue =
             hvdb::evalAttrDefault<ValueType>(tupleAIF->getDefaults(attribute), 0);

        // Construct a new tree that matches the closestPrimGrid's active voxel topology.
        typename TreeType::Ptr tree(
            new TreeType(closestPrimGrid.tree(), defValue, openvdb::TopologyCopy()));
        typename GridType::Ptr grid(GridType::create(tree));

        grid->setVectorType(openvdb::VecType(vecType));

        attributeList.push_back(hvdb::AttributeDetailBase::Ptr(
            new hvdb::AttributeDetail<GridType>(grid, attribute, tupleAIF, 0, true)));

        if (customName.size() > 0) {
            attributeList[attributeList.size()-1]->name() = customName;
        }

    } else {
        for (int c = 0; c < attrTupleSize; ++c) {
            // Get the attribute's default value.
            ValueType defValue =
                hvdb::evalAttrDefault<ValueType>(tupleAIF->getDefaults(attribute), c);

            // Construct a new tree that matches the closestPrimGrid's active voxel topology.
            typename TreeType::Ptr tree(
                new TreeType(closestPrimGrid.tree(), defValue, openvdb::TopologyCopy()));
            typename GridType::Ptr grid(GridType::create(tree));

            attributeList.push_back(hvdb::AttributeDetailBase::Ptr(
                new hvdb::AttributeDetail<GridType>(grid, attribute, tupleAIF, c)));

            if (customName.size() > 0) {
                std::ostringstream name;
                name << customName;
                if(attrTupleSize != 1) name << "_" << c;

                attributeList[attributeList.size()-1]->name() = name.str();
            }
        }
    }
}


////////////////////////////////////////


void
SOP_OpenVDB_From_Polygons::transferAttributes(
    hvdb::AttributeDetailList &pointAttributes,
    hvdb::AttributeDetailList &vertexAttributes,
    hvdb::AttributeDetailList &primitiveAttributes,
    const openvdb::Int32Grid& closestPrimGrid,
    openvdb::math::Transform::Ptr& transform,
    const GU_Detail& meshGdp)
{

    // Threaded attribute transfer.
    hvdb::MeshAttrTransfer transferOp(pointAttributes, vertexAttributes, primitiveAttributes,
        closestPrimGrid, *transform, meshGdp);

    transferOp.runParallel();


    // Construct and add VDB primitives to the gdp

    for (size_t i = 0, N = pointAttributes.size(); i < N; ++i) {
        hvdb::AttributeDetailBase::Ptr& attrDetail = pointAttributes[i];
        attrDetail->grid()->setTransform(transform);
        hvdb::createVdbPrimitive(*gdp, attrDetail->grid(), attrDetail->name().c_str());
    }

    for (size_t i = 0, N = vertexAttributes.size(); i < N; ++i) {
        hvdb::AttributeDetailBase::Ptr& attrDetail = vertexAttributes[i];
        attrDetail->grid()->setTransform(transform);
        hvdb::createVdbPrimitive(*gdp, attrDetail->grid(), attrDetail->name().c_str());
    }

    for (size_t i = 0, N = primitiveAttributes.size(); i < N; ++i) {
        hvdb::AttributeDetailBase::Ptr& attrDetail = primitiveAttributes[i];
        attrDetail->grid()->setTransform(transform);
        hvdb::createVdbPrimitive(*gdp, attrDetail->grid(), attrDetail->name().c_str());
    }
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
