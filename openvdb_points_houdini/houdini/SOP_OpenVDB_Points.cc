///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file SOP_OpenVDB_Points.cc
///
/// @author Dan Bailey
///
/// @brief Converts points to OpenVDB points.


#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/PointGroup.h>

#include "Utils.h"
#include "SOP_NodeVDBPoints.h"

#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

#include <openvdb_houdini/AttributeTransferUtil.h>

#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT
#include <SYS/SYS_Types.h> // for int32, float32, etc

#include <boost/ptr_container/ptr_vector.hpp>

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
namespace hvdbp = openvdb_points_houdini;
namespace hutil = houdini_utils;

enum COMPRESSION_TYPE
{
    NONE = 0,
    TRUNCATE_16,
    UNIT_VECTOR,
    FIXED_POSITION_16,
    FIXED_POSITION_8
};

/// @brief Translate the type of a GA_Attribute into a position Attribute Type
inline NamePair
positionAttrTypeFromCompression(const int compression)
{
    if (compression > 0 && compression + FIXED_POSITION_16 - 1 == FIXED_POSITION_16) {
        return TypedAttributeArray<Vec3<float>,
                            FixedPointAttributeCodec<Vec3<uint16_t> > >::attributeType();
    }
    else if (compression > 0 && compression + FIXED_POSITION_16 - 1 == FIXED_POSITION_8) {
        return TypedAttributeArray<Vec3<float>,
                            FixedPointAttributeCodec<Vec3<uint8_t> > >::attributeType();
    }

    // compression == NONE

    return TypedAttributeArray<Vec3<float> >::attributeType();
}

/// @brief Translate the type of a GA_Attribute into our AttrType
inline NamePair
attrTypeFromGAAttribute(GA_Attribute const * attribute, const int compression = 0)
{
    if (!attribute) {
        std::stringstream ss; ss << "Invalid attribute - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();

    if (!tupleAIF) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_Storage storage = tupleAIF->getStorage(attribute);

    const int16_t width = static_cast<int16_t>(tupleAIF->getTupleSize(attribute));

    if (width == 1)
    {
        if (storage == GA_STORE_BOOL) {
            return TypedAttributeArray<bool>::attributeType();
        }
        else if (storage == GA_STORE_INT16) {
            return TypedAttributeArray<int16_t>::attributeType();
        }
        else if (storage == GA_STORE_INT32) {
            return TypedAttributeArray<int32_t>::attributeType();
        }
        else if (storage == GA_STORE_INT64) {
            return TypedAttributeArray<int64_t>::attributeType();
        }
        else if (storage == GA_STORE_REAL16) {
            return TypedAttributeArray<half>::attributeType();
        }
        else if (storage == GA_STORE_REAL32)
        {
            if (compression == NONE) {
                return TypedAttributeArray<float>::attributeType();
            }
            else if (compression == TRUNCATE_16) {
                return TypedAttributeArray<float, NullAttributeCodec<half> >::attributeType();
            }
        }
        else if (storage == GA_STORE_REAL64) {
            return TypedAttributeArray<double>::attributeType();
        }
    }
    else if (width == 2)
    {
        if (storage == GA_STORE_REAL16) {
            return TypedAttributeArray<Vec2<half> >::attributeType();
        }
        else if (storage == GA_STORE_REAL32)
        {
            return TypedAttributeArray<Vec2<float> >::attributeType();
        }
        else if (storage == GA_STORE_REAL64) {
            return TypedAttributeArray<Vec2<double> >::attributeType();
        }
    }
    else if (width == 3 || width == 4)
    {
        // note: process 4-component vectors as 3-component vectors for now

        if (storage == GA_STORE_REAL16) {
            return TypedAttributeArray<Vec3<half> >::attributeType();
        }
        else if (storage == GA_STORE_REAL32)
        {
            if (compression == NONE) {
                return TypedAttributeArray<Vec3<float> >::attributeType();
            }
            else if (compression == TRUNCATE_16) {
                return TypedAttributeArray<Vec3<float>, NullAttributeCodec<Vec3<half> > >::attributeType();
            }
            else if (compression == UNIT_VECTOR) {
                return TypedAttributeArray<Vec3<float>, UnitVecAttributeCodec>::attributeType();
            }
        }
        else if (storage == GA_STORE_REAL64) {
            return TypedAttributeArray<Vec3<double> >::attributeType();
        }
    }

    std::stringstream ss; ss << "Unknown attribute type - " << attribute->getName();
    throw std::runtime_error(ss.str());
}

inline Name
attrStringTypeFromGAAttribute(GA_Attribute const * attribute)
{
    if (!attribute) {
        std::stringstream ss; ss << "Invalid attribute - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();

    if (!tupleAIF) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_Storage storage = tupleAIF->getStorage(attribute);

    const int16_t width = static_cast<int16_t>(tupleAIF->getTupleSize(attribute));

    if (width == 1)
    {
        if (storage == GA_STORE_BOOL)           return "bool";
        else if (storage == GA_STORE_INT16)     return "int16";
        else if (storage == GA_STORE_INT32)     return "int32";
        else if (storage == GA_STORE_INT64)     return "int64";
        else if (storage == GA_STORE_REAL16)    return "half";
        else if (storage == GA_STORE_REAL32)    return "float";
        else if (storage == GA_STORE_REAL64)    return "double";
    }
    else if (width == 2)
    {
        if (storage == GA_STORE_REAL16)         return "vec2h";
        else if (storage == GA_STORE_REAL32)    return "vec2s";
        else if (storage == GA_STORE_REAL64)    return "vec2d";
    }
    else if (width == 3 || width == 4)
    {
        // note: process 4-component vectors as 3-component vectors for now

        if (storage == GA_STORE_REAL16)         return "vec3h";
        else if (storage == GA_STORE_REAL32)    return "vec3s";
        else if (storage == GA_STORE_REAL64)    return "vec3d";
    }

    std::stringstream ss; ss << "Unknown attribute type - " << attribute->getName();
    throw std::runtime_error(ss.str());
}

/// @brief Translate the default value of a GA_Defaults into default Metadata
template <typename ValueType>
Metadata::Ptr
defaultMetadataFromGADefault(const GA_Defaults& defaults)
{
    ValueType value = hvdb::evalAttrDefault<ValueType>(defaults, 0);

    // return empty metadata if default is zero
    if (math::isZero<ValueType>(value))     return Metadata::Ptr();

    return TypedMetadata<ValueType>(value).copy();
}

/// @brief Translate the default value of a GA_Attribute into default Metadata
inline Metadata::Ptr
defaultMetadataFromGAAttribute(GA_Attribute const * attribute)
{
    if (!attribute) {
        std::stringstream ss; ss << "Invalid attribute - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();

    if (!tupleAIF) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_Defaults& defaults = tupleAIF->getDefaults(attribute);

    const GA_Storage storage = tupleAIF->getStorage(attribute);

    const int16_t width = static_cast<int16_t>(tupleAIF->getTupleSize(attribute));

    if (width == 1)
    {
        if (storage == GA_STORE_BOOL) {
            return defaultMetadataFromGADefault<bool>(defaults);
        }
        else if (storage == GA_STORE_INT16) {
            return defaultMetadataFromGADefault<int16_t>(defaults);
        }
        else if (storage == GA_STORE_INT32) {
            return defaultMetadataFromGADefault<int32_t>(defaults);
        }
        else if (storage == GA_STORE_INT64) {
            return defaultMetadataFromGADefault<int64_t>(defaults);
        }
        else if (storage == GA_STORE_REAL16) {
            return defaultMetadataFromGADefault<half>(defaults);
        }
        else if (storage == GA_STORE_REAL32) {
            return defaultMetadataFromGADefault<float>(defaults);
        }
        else if (storage == GA_STORE_REAL64) {
            return defaultMetadataFromGADefault<double>(defaults);
        }
    }
    else if (width == 2)
    {
        if (storage == GA_STORE_REAL16) {
            return defaultMetadataFromGADefault<Vec2<half> >(defaults);
        }
        else if (storage == GA_STORE_REAL32) {
            return defaultMetadataFromGADefault<Vec2<float> >(defaults);
        }
        else if (storage == GA_STORE_REAL64) {
            return defaultMetadataFromGADefault<Vec2<double> >(defaults);
        }
    }
    else if (width == 3 || width == 4)
    {
        // note: process 4-component vectors as 3-component vectors for now

        if (storage == GA_STORE_REAL16) {
            return defaultMetadataFromGADefault<Vec3<half> >(defaults);
        }
        else if (storage == GA_STORE_REAL32) {
            return defaultMetadataFromGADefault<Vec3<float> >(defaults);
        }
        else if (storage == GA_STORE_REAL64) {
            return defaultMetadataFromGADefault<Vec3<double> >(defaults);
        }
    }

    std::stringstream ss; ss << "Unknown attribute type - " << attribute->getName();
    throw std::runtime_error(ss.str());
}

////////////////////////////////////////


struct AttributeInfo
{
    AttributeInfo(const Name& name,
                  const int valueCompression,
                  const bool bloscCompression)
        : name(name)
        , valueCompression(valueCompression)
        , bloscCompression(bloscCompression) { }

    Name name;
    int valueCompression;
    bool bloscCompression;
}; // struct AttributeInfo

typedef std::vector<AttributeInfo> AttributeInfoVec;

///////////////////////////////////////


inline
PointDataGrid::Ptr
createPointDataGrid(const GU_Detail& ptGeo, const openvdb::NamePair& positionAttributeType,
                    const AttributeInfoVec& attributes, const openvdb::math::Transform& transform)
{
    // store point group information

    const GA_ElementGroupTable& elementGroups = ptGeo.getElementGroupTable(GA_ATTRIB_POINT);

    // Create PointPartitioner compatible P attribute wrapper (for now no offset filtering)

    hvdbp::OffsetListPtr offsets;
    hvdbp::HoudiniReadAttribute<openvdb::Vec3d> points(*ptGeo.getP(), offsets);

    // Create PointIndexGrid used for consistent index ordering in all attribute conversion

    PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(points, transform);

    // Create PointDataGrid using position attribute

    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(
                            *pointIndexGrid, points, positionAttributeType, transform);

    PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();

    // Append (empty) groups to tree

    std::vector<Name> groupNames;
    groupNames.reserve(elementGroups.entries());

    for (GA_ElementGroupTable::iterator it = elementGroups.beginTraverse(),
                                        itEnd = elementGroups.endTraverse(); it != itEnd; ++it)
    {
        groupNames.push_back((*it)->getName().toStdString());
    }

    appendGroups(tree, groupNames);

    // Set group membership in tree

    const int64_t numPoints = ptGeo.getNumPoints();
    std::vector<short> inGroup(numPoints, short(0));

    for (GA_ElementGroupTable::iterator it = elementGroups.beginTraverse(),
                                        itEnd = elementGroups.endTraverse(); it != itEnd; ++it)
    {
        // insert group offsets

        GA_Offset start, end;
        GA_Range range(**it);
        for (GA_Iterator rangeIt = range.begin(); rangeIt.blockAdvance(start, end); ) {
            end = std::min(end, numPoints);
            for (GA_Offset off = start; off < end; ++off) {
                assert(off < numPoints);
                inGroup[off] = short(1);
            }
        }

        const Name groupName = (*it)->getName().toStdString();
        setGroup(tree, indexTree, inGroup, groupName);

        std::fill(inGroup.begin(), inGroup.end(), short(0));
    }

    // Add other attributes to PointDataGrid

    for (AttributeInfoVec::const_iterator it = attributes.begin(),
                                          it_end = attributes.end(); it != it_end; ++it)
    {
        const openvdb::Name name = it->name;
        const int compression = it->valueCompression;

        // skip position as this has already been added

        if (name == "P")  continue;

        GA_ROAttributeRef attrRef = ptGeo.findPointAttribute(name.c_str());

        if (!attrRef.isValid())     continue;

        GA_Attribute const * gaAttribute = attrRef.getAttribute();

        if (!gaAttribute)             continue;

        // Append the new attribute to the PointDataGrid
        AttributeSet::Util::NameAndType nameAndType(name,
                                attrTypeFromGAAttribute(gaAttribute, compression));
        Metadata::Ptr defaultMetadata = defaultMetadataFromGAAttribute(gaAttribute);

        appendAttribute(tree, nameAndType, defaultMetadata);

        const openvdb::Name type = nameAndType.type.first;

        hvdbp::OffsetListPtr offsets;

        if (type == "bool") {
            hvdbp::HoudiniReadAttribute<bool> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "int16") {
            hvdbp::HoudiniReadAttribute<int16_t> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "int32") {
            hvdbp::HoudiniReadAttribute<int32_t> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "int64") {
            hvdbp::HoudiniReadAttribute<int64_t> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "half") {
            hvdbp::HoudiniReadAttribute<half> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "float") {
            hvdbp::HoudiniReadAttribute<float> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "double") {
            hvdbp::HoudiniReadAttribute<double> attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec2h") {
            hvdbp::HoudiniReadAttribute<Vec2<half> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec2s") {
            hvdbp::HoudiniReadAttribute<Vec2<float> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec2d") {
            hvdbp::HoudiniReadAttribute<Vec2<double> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec3h") {
            hvdbp::HoudiniReadAttribute<Vec3<half> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec3s") {
            hvdbp::HoudiniReadAttribute<Vec3<float> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else if (type == "vec3d") {
            hvdbp::HoudiniReadAttribute<Vec3<double> > attribute(*gaAttribute, offsets);
            populateAttribute(tree, indexTree, name, attribute);
        }
        else {
            throw std::runtime_error("Unknown Attribute Type for Conversion: " + type);
        }
    }

    // Attempt to compact attributes

    compactAttributes(tree);

    // Apply blosc compression to attributes

    for (AttributeInfoVec::const_iterator   it = attributes.begin(),
                                            it_end = attributes.end(); it != it_end; ++it)
    {
        if (!it->bloscCompression)  continue;

        bloscCompressAttribute(tree, it->name);
    }

    return pointDataGrid;
}


////////////////////////////////////////


class SOP_OpenVDB_Points: public hvdb::SOP_NodeVDBPoints
{
public:
    SOP_OpenVDB_Points(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Points() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i == 1); }

protected:

    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

private:
    hvdb::Interrupter mBoss;
}; // class SOP_OpenVDB_Points



////////////////////////////////////////

namespace {

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
    if (data == NULL || menuEntries == NULL || spare == NULL) return;

    SOP_Node* sop = CAST_SOPNODE((OP_Node *)data);

    if (sop == NULL) {
        // terminate and quit
        menuEntries[0].setToken(0);
        menuEntries[0].setLabel(0);
        return;
    }


    int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    size_t menuIdx = 0, menuEnd(themenusize - 2);

    // null object
    menuEntries[menuIdx].setToken("0");
    menuEntries[menuIdx++].setLabel("- no attribute selected -");

    if (gdp) {

        // point attribute names
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);

        if (!iter.atEnd() && menuIdx != menuEnd) {

            if (menuIdx > 0) {
                menuEntries[menuIdx].setToken(PRM_Name::mySeparator);
                menuEntries[menuIdx++].setLabel(PRM_Name::mySeparator);
            }

            for (; !iter.atEnd() && menuIdx != menuEnd; ++iter) {

                const char* str = (*iter)->getName();

                if (str) {
                    Name name = str;
                    if (name != "P") {
                        menuEntries[menuIdx].setToken(name.c_str());
                        menuEntries[menuIdx++].setLabel(name.c_str());
                    }
                }
            }
        }
    }

    // terminator
    menuEntries[menuIdx].setToken(0);
    menuEntries[menuIdx].setLabel(0);
}

const PRM_ChoiceList PrimAttrMenu(
    PRM_ChoiceListType(PRM_CHOICELIST_REPLACE), sopBuildAttrMenu);

} // unnamed namespace

////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    points::initialize();

    if (table == NULL) return;

    hutil::ParmList parms;

    {
        const char* items[] = {
            "vdb", "Houdini points to VDB points",
            "hdk", "VDB points to Houdini points",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "conversion", "Conversion")
            .setDefault(PRMzeroDefaults)
            .setHelpText("The conversion method for the expected input types.")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input point data grids to convert.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroup", "VDB Points Group")
        .setHelpText("Specify VDB Points Groups to use as an input."));

    //  point grid name
    parms.add(hutil::ParmFactory(PRM_STRING, "name", "VDB Name")
        .setDefault("points")
        .setHelpText("Output grid name."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setHelpText("The desired voxel size of the new VDB Points grid.")
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5));

    // Group name (Transform reference)
    parms.add(hutil::ParmFactory(PRM_STRING, "refvdb", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setSpareData(&SOP_Node::theSecondInput)
        .setHelpText("References the first/selected grid's transform."));

    //////////

    // Point attribute transfer

    {
        const char* items[] = {
            "none", "None",
            "int16", "16-bit fixed point",
            "int8", "8-bit fixed point",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "poscompression", "Position Compression")
            .setDefault(PRMzeroDefaults)
            .setHelpText("The position attribute compression setting.")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    parms.add(hutil::ParmFactory(PRM_HEADING, "transferHeading", "Attribute transfer"));

     // Mode. Either convert all or convert specifc attributes

    {
        const char* items[] = {
            "all", "All Attributes",
            "spec", "Specific Attributes",
            NULL
    };

    parms.add(hutil::ParmFactory(PRM_ORD, "mode", "Mode")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Whether to transfer only specific attributes or all attributes found.")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    hutil::ParmList attrParms;

    // Attribute name
    attrParms.add(hutil::ParmFactory(PRM_STRING, "attribute#", "Attribute")
        .setChoiceList(&PrimAttrMenu)
        .setSpareData(&SOP_Node::theFirstInput)
        .setHelpText("Select a point attribute to transfer. "
            "Supports integer and floating point attributes of "
            "arbitrary precisions and tuple sizes."));

    {
        const char* items[] = {
            "none", "None",
            "truncate", "16-bit Truncate",
            UnitVecAttributeCodec::name(), "Unit Vector",
            NULL
        };

        attrParms.add(hutil::ParmFactory(PRM_ORD, "valuecompression#", "Value Compression")
            .setDefault(PRMzeroDefaults)
            .setHelpText("Value Compression to use for specific attributes.")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    attrParms.add(hutil::ParmFactory(PRM_TOGGLE, "blosccompression#", "Blosc Compression")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Enable Blosc Compression."));

    // Add multi parm
    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "attrList", "Point Attributes")
        .setHelpText("Transfer point attributes to each voxel in the level set's narrow band")
        .setMultiparms(attrParms)
        .setDefault(PRMzeroDefaults));

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB Points",
        SOP_OpenVDB_Points::factory, parms, *table)
        .addInput("Points to Convert")
        .addOptionalInput("Optional Reference VDB "
            "(for transform)");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Points::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Points(net, name, op);
}


SOP_OpenVDB_Points::SOP_OpenVDB_Points(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDBPoints(net, name, op)
    , mBoss("Converting points")
{
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Points::updateParmsFlags()
{
    bool changed = false;

    const bool toVdbPoints = evalInt("conversion", 0, 0) == 0;
    const bool convertAll = evalInt("mode", 0, 0) == 0;

    changed |= enableParm("group", !toVdbPoints);
    changed |= setVisibleState("group", !toVdbPoints);

    changed |= enableParm("name", toVdbPoints);
    changed |= setVisibleState("name", toVdbPoints);

    int refexists = (this->nInputs() == 2);

    changed |= enableParm("refvdb", refexists);
    changed |= setVisibleState("refvdb", toVdbPoints);

    changed |= enableParm("voxelsize", !refexists && toVdbPoints);
    changed |= setVisibleState("voxelsize", toVdbPoints);

    changed |= setVisibleState("transferHeading", toVdbPoints);

    changed |= enableParm("poscompression", toVdbPoints);
    changed |= setVisibleState("poscompression", toVdbPoints);

    changed |= enableParm("mode", toVdbPoints);
    changed |= setVisibleState("mode", toVdbPoints);

    changed |= enableParm("attrList", toVdbPoints && !convertAll);
    changed |= setVisibleState("attrList", toVdbPoints && !convertAll);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Points::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();

        const fpreal time = context.getTime();
        // Check for particles in the primary (left) input port
        const GU_Detail* ptGeo = inputGeo(0, context);

        if (evalInt("conversion", 0, time) != 0) {

            UT_String groupStr;
            evalString(groupStr, "group", 0, time);
            const GA_PrimitiveGroup *group =
                matchGroup(const_cast<GU_Detail&>(*ptGeo), groupStr.toStdString());

            // Extract VDB Point groups to filter

            UT_String pointsGroupStr;
            evalString(pointsGroupStr, "vdbpointsgroup", 0, time);
            const std::string pointsGroup = pointsGroupStr.toStdString();

            std::vector<std::string> includeGroups;
            std::vector<std::string> excludeGroups;
            openvdb::tools::AttributeSet::Descriptor::parseNames(includeGroups, excludeGroups, pointsGroup);

            // passing an empty vector of attribute names implies that all attributes should be converted
            const std::vector<std::string> emptyNameVector;

            // Mesh each VDB primitive independently
            for (hvdb::VdbPrimCIterator vdbIt(ptGeo, group); vdbIt; ++vdbIt) {

                GU_Detail geo;

                const GridBase& baseGrid = vdbIt->getGrid();
                if (!baseGrid.isType<PointDataGrid>()) continue;

                const PointDataGrid& grid = static_cast<const PointDataGrid&>(baseGrid);

                hvdbp::convertPointDataGridToHoudini(geo, grid, emptyNameVector, includeGroups, excludeGroups);

                gdp->merge(geo);
            }

            return error();
        }

        // Set member data

        Transform::Ptr transform;

        // Optionally copy transform parameters from reference grid.

        if (const GU_Detail* refGeo = inputGeo(1, context)) {

            UT_String refvdbStr;
            evalString(refvdbStr, "refvdb", 0, time);

            const GA_PrimitiveGroup *group =
                matchGroup(const_cast<GU_Detail&>(*refGeo), refvdbStr.toStdString());

            hvdb::VdbPrimCIterator it(refGeo, group);
            const hvdb::GU_PrimVDB* refPrim = *it;

            if (refPrim) {
                transform = refPrim->getGrid().transform().copy();
            } else {
                addError(SOP_MESSAGE, "Second input has no VDB primitives.");
                return error();
            }
        }
        else {
            float voxelSize = evalFloat("voxelsize", 0, time);
            transform = Transform::createLinearTransform(voxelSize);
        }

        UT_String attrName;
        AttributeInfoVec attributes;

        if (evalInt("mode", 0, time) != 0) {
            // Transfer point attributes.
            if (evalInt("attrList", 0, time) > 0) {
                for (int i = 1, N = evalInt("attrList", 0, 0); i <= N; ++i) {
                    evalStringInst("attribute#", &i, attrName, 0, 0);
                    Name attributeName = Name(attrName);

                    GA_ROAttributeRef attrRef = ptGeo->findPointAttribute(attributeName.c_str());

                    if (!attrRef.isValid()) continue;

                    GA_Attribute const * attribute = attrRef.getAttribute();

                    if (!attribute) continue;

                    const Name type(attrStringTypeFromGAAttribute(attribute));

                    int valueCompression = 0;

                    // when converting specific attributes apply chosen compression.

                    valueCompression = evalIntInst("valuecompression#", &i, 0, 0);

                    std::stringstream ss;
                    ss <<   "Invalid value compression for attribute - " << attributeName << ". " <<
                            "Disabling compression for this attribute.";

                    if (valueCompression == TRUNCATE_16)
                    {
                        if (type != "float" && type != "vec3s") {
                            valueCompression = 0;
                            addWarning(SOP_MESSAGE, ss.str().c_str());
                        }
                    }
                    else if (valueCompression == UNIT_VECTOR)
                    {
                        if (type != "vec3s") {
                            valueCompression = 0;
                            addWarning(SOP_MESSAGE, ss.str().c_str());
                        }
                    }

                    const bool bloscCompression = evalIntInst("blosccompression#", &i, 0, 0);

                    attributes.push_back(AttributeInfo(attributeName, valueCompression, bloscCompression));
                }
            }
        } else {
            // point attribute names
            GA_AttributeDict::iterator iter = ptGeo->pointAttribs().begin(GA_SCOPE_PUBLIC);

            if (!iter.atEnd()) {
                for (; !iter.atEnd(); ++iter) {
                    const char* str = (*iter)->getName();

                    if (str) {
                        Name attrName = str;

                        if (attrName == "P") continue;

                        // when converting all attributes apply no compression
                        attributes.push_back(AttributeInfo(attrName, 0, false));
                    }
                }
            }
        }

        // Determine position compression

        const int positionCompression = evalInt("poscompression", 0, time);

        const openvdb::NamePair positionAttributeType =
                    positionAttrTypeFromCompression(positionCompression);

        PointDataGrid::Ptr pointDataGrid = createPointDataGrid(*ptGeo, positionAttributeType, attributes, *transform);

        UT_String nameStr = "";
        evalString(nameStr, "name", 0, time);
        hvdb::createVdbPrimitive(*gdp, pointDataGrid, nameStr.toStdString().c_str());

        mBoss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
