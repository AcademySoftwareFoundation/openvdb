///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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

#include "SOP_NodeVDBPoints.h"

#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

enum COMPRESSION_TYPE
{
    NONE = 0,
    FIXED_POSITION_16,
    FIXED_POSITION_8,
    TRUNCATE_16,
    UNIT_VECTOR
};

/// @brief Translate the type of a GA_Attribute into a position Attribute Type
inline NamePair
positionAttrTypeFromCompression(const int compression)
{
    if (compression == FIXED_POSITION_16) {
        return TypedAttributeArray<Vec3<float>,
                            FixedPointAttributeCodec<Vec3<uint16_t> > >::attributeType();
    }
    else if (compression == FIXED_POSITION_8) {
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

inline GA_Storage
gaStorageFromAttrString(const Name& type)
{
    if (type == "bool")             return GA_STORE_BOOL;
    else if (type == "int16")       return GA_STORE_INT16;
    else if (type == "int32")         return GA_STORE_INT32;
    else if (type == "int64")        return GA_STORE_INT64;
    else if (type == "half")        return GA_STORE_REAL16;
    else if (type == "float")       return GA_STORE_REAL32;
    else if (type == "double")      return GA_STORE_REAL64;
    else if (type == "vec3h")       return GA_STORE_REAL16;
    else if (type == "vec3s")       return GA_STORE_REAL32;
    else if (type == "vec3d")       return GA_STORE_REAL64;

    return GA_STORE_INVALID;
}

inline unsigned
widthFromAttrString(const Name& type)
{
    if (type == "bool" ||
        type == "int16" ||
        type == "int32" ||
        type == "int64" ||
        type == "half" ||
        type == "float" ||
        type == "double")
    {
        return 1;
    }
    else if (type == "vec3h" ||
             type == "vec3s" ||
             type == "vec3d")
    {
        return 3;
    }

    return 0;
}

////////////////////////////////////////


typedef std::vector<GA_Offset> OffsetList;
typedef boost::shared_ptr<OffsetList> OffsetListPtr;

OffsetListPtr computeOffsets(GA_PointGroup* group)
{
    if (!group) return OffsetListPtr();

    OffsetListPtr offsets = OffsetListPtr(new OffsetList());

    size_t size = group->entries();
    offsets->reserve(size);

    GA_Offset start, end;
    GA_Range range(*group);
    for (GA_Iterator it = range.begin(); it.blockAdvance(start, end); ) {
        for (GA_Offset off = start; off < end; ++off) {
            offsets->push_back(off);
        }
    }

    return offsets;
}


////////////////////////////////////////


template <typename T, typename T0>
T attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i)
{
    T0 tmp;
    attribute->getAIFTuple()->get(attribute, n, tmp, i);
    return static_cast<T>(tmp);
}

template <typename T>
T attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i) {
    return attributeValue<T, T>(attribute, n, i);
}
template <>
bool attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i) {
    return attributeValue<bool, int>(attribute, n, i);
}
template <>
short attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i) {
    return attributeValue<short, int>(attribute, n, i);
}
template <>
long attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i) {
    return attributeValue<long, int>(attribute, n, i);
}
template <>
half attributeValue(GA_Attribute const * const attribute, GA_Offset n, unsigned i) {
    return attributeValue<half, float>(attribute, n, i);
}


////////////////////////////////////////


/// @brief Wrapper class around Houdini point attributes which hold a pointer to the
/// GA_Attribute to access the data and optionally a list of offsets
template <typename AttributeType>
class PointAttribute
{
public:
    typedef AttributeType value_type;

    PointAttribute(GA_Attribute const * attribute, OffsetListPtr offsets)
        : mAttribute(attribute)
        , mOffsets(offsets) { }

    size_t size() const
    {
        return mAttribute->getIndexMap().indexSize();
    }

    GA_Offset getOffset(size_t n) const
    {
        return mOffsets ? (*mOffsets)[n] : mAttribute->getIndexMap().offsetFromIndex(GA_Index(n));
    }

    // Return the value of the nth point in the array (scalar type only)
    template <typename T> typename boost::disable_if_c<VecTraits<T>::IsVec, void>::type
    get(size_t n, T& value) const
    {
        value = attributeValue<T>(mAttribute, getOffset(n), 0);
    }

    // Return the value of the nth point in the array (vector type only)
    template <typename T> typename boost::enable_if_c<VecTraits<T>::IsVec, void>::type
    get(size_t n, T& value) const
    {
        for (unsigned i = 0; i < VecTraits<T>::Size; ++i) {
            value[i] = attributeValue<typename VecTraits<T>::ElementType>(mAttribute, getOffset(n), i);
        }
    }

    // Only provided to match the required interface for the PointPartitioner
    void getPos(size_t n, AttributeType& xyz) const { return this->get<AttributeType>(n, xyz); }

private:
    GA_Attribute const * const mAttribute;
    OffsetListPtr mOffsets;
}; // PointAttribute


////////////////////////////////////////


/// @brief Populate a VDB Points attribute using the PointAttribute wrapper
void populateAttributeFromHoudini(  PointDataTree& tree, const PointIndexTree& indexTree, const openvdb::Name& name,
                                    const NamePair& attributeType, GA_Attribute const * attribute, OffsetListPtr offsets)
{
    const openvdb::Name type = attributeType.first;

    if (type == "bool") {
        populateAttribute(tree, indexTree, name, PointAttribute<bool>(attribute, offsets));
    }
    else if (type == "int16") {
        populateAttribute(tree, indexTree, name, PointAttribute<int16_t>(attribute, offsets));
    }
    else if (type == "int32") {
        populateAttribute(tree, indexTree, name, PointAttribute<int32_t>(attribute, offsets));
    }
    else if (type == "int64") {
        populateAttribute(tree, indexTree, name, PointAttribute<int64_t>(attribute, offsets));
    }
    else if (type == "half") {
        populateAttribute(tree, indexTree, name, PointAttribute<half>(attribute, offsets));
    }
    else if (type == "float") {
        populateAttribute(tree, indexTree, name, PointAttribute<float>(attribute, offsets));
    }
    else if (type == "double") {
        populateAttribute(tree, indexTree, name, PointAttribute<double>(attribute, offsets));
    }
    else if (type == "vec3h") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec3<half> >(attribute, offsets));
    }
    else if (type == "vec3s") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec3<float> >(attribute, offsets));
    }
    else if (type == "vec3d") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec3<double> >(attribute, offsets));
    }
    else {
        throw std::runtime_error("Unknown Attribute Type for Conversion: " + type);
    }
}


////////////////////////////////////////


template <typename TypedAttributeType>
inline void
convertPointDataGridPosition(const PointDataGrid& grid, GU_Detail& detail)
{
    typedef PointDataTree                           PointDataTree;
    typedef PointDataAccessor<const PointDataTree>  PointDataAccessor;
    typedef AttributeSet                            AttributeSet;

    const PointDataTree& tree = grid.tree();

    PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    // determine the position index
    const size_t positionIndex = iter->attributeSet().find("P");
    const bool hasPosition = positionIndex != AttributeSet::INVALID_POS;

    PointDataAccessor acc(tree);

#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245
    for (size_t n = 0, N = acc.totalPointCount(); n < N; ++n) detail.appendPointOffset();
#else
    detail.appendPointBlock(acc.totalPointCount());
#endif

    const Transform& xform = grid.transform();
    AttributeHandle<Vec3f>::Ptr handle;

    GA_Offset offset = 0;
    for (; iter; ++iter) {

        if (hasPosition) {
            handle = iter->attributeHandle<Vec3f>(positionIndex);
        }

        Coord ijk;
        Vec3d xyz, pos;

        for (PointDataTree::LeafNodeType::ValueOnCIter vIt = iter->cbeginValueOn(); vIt; ++vIt) {

            ijk = vIt.getCoord();
            xyz = ijk.asVec3d();

            PointDataAccessor::PointDataIndex pointIndexRange = acc.get(ijk);
            for (Index64 n = pointIndexRange.first, N = pointIndexRange.second; n < N; ++n) {

                if (hasPosition) {
                    pos = Vec3d(handle->get(n)) + xyz;
                    pos = xform.indexToWorld(pos);
                    detail.setPos3(offset++, pos.x(), pos.y(), pos.z());
                } else {
                    pos = xform.indexToWorld(xyz);
                    detail.setPos3(offset++, pos.x(), pos.y(), pos.z());
                }
            }
        }
    }
}


template <typename VDBType, typename AttrHandle>
inline void
setAttributeValue(const VDBType value, AttrHandle& handle, const GA_Offset& offset)
{
    handle.set(offset, value);
}


template <typename VDBElementType, typename AttrHandle>
inline void
setAttributeValue(const Vec3<VDBElementType>& v, AttrHandle& handle, const GA_Offset& offset)
{
    handle.set(offset, UT_Vector3(v.x(), v.y(), v.z()));
}


template <typename T> struct GAHandleTraits { typedef GA_RWHandleF RW; };
template <typename T> struct GAHandleTraits<Vec3<T> > { typedef GA_RWHandleV3 RW; };
template <> struct GAHandleTraits<bool> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int16_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int32_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int64_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<half> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<float> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<double> { typedef GA_RWHandleF RW; };


template <typename Type>
inline void
convertPointDataGridAttribute(const PointDataTree& tree,
    const unsigned arrayIndex, GA_Attribute& attribute, GU_Detail&)
{
    typedef PointDataTree                           PointDataTree;
    typedef PointDataAccessor<const PointDataTree>  PointDataAccessor;
    typedef typename GAHandleTraits<Type>::RW GAHandleType;

    GAHandleType attributeHandle(&attribute);
    PointDataAccessor acc(tree);

    GA_Offset offset = 0;
    for (PointDataTree::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {

        typename AttributeHandle<Type>::Ptr handle = iter->attributeHandle<Type>(arrayIndex);

        for (PointDataTree::LeafNodeType::ValueOnCIter vIt = iter->cbeginValueOn(); vIt; ++vIt) {

            PointDataAccessor::PointDataIndex pointIndexRange = acc.get(vIt.getCoord());
            for (Index64 n = pointIndexRange.first, N = pointIndexRange.second; n < N; ++n) {
                setAttributeValue(handle->get(n), attributeHandle, offset++);
            }
        }
    }
}


inline void
convertPointDataGrid(GU_Detail& detail, openvdb_houdini::VdbPrimCIterator& vdbIt)
{
    typedef PointDataGrid   PointDataGrid;
    typedef PointDataGrid::TreeType         PointDataTree;
    typedef AttributeSet    AttributeSet;

    GU_Detail geo;

    // Mesh each VDB primitive independently
    for (; vdbIt; ++vdbIt) {

        const GridBase& baseGrid = vdbIt->getGrid();
        if (!baseGrid.isType<PointDataGrid>()) continue;

        const PointDataGrid& grid = static_cast<const PointDataGrid&>(baseGrid);
        const PointDataTree& tree = grid.tree();

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
        if (!leafIter) continue;

        const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
        { //
            const size_t positionIndex = descriptor.find("P");

            // determine whether position and color exist

            const bool hasPosition = positionIndex != AttributeSet::INVALID_POS;

            if (hasPosition) {
                convertPointDataGridPosition<Vec3f>(grid, geo);
            }
        }

        // add other point attributes to the hdk detail
        const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

        for (AttributeSet::Descriptor::ConstIterator it = nameToPosMap.begin(), it_end = nameToPosMap.end();
            it != it_end; ++it) {

            const Name& name = it->first;

            const Name& type = descriptor.type(it->second).first;

            // position handled explicitly
            if (name == "P")    continue;

            const unsigned index = it->second;

            const GA_Storage storage = gaStorageFromAttrString(type);
            const unsigned width = widthFromAttrString(type);

            GA_RWAttributeRef attributeRef = geo.addTuple(storage, GA_ATTRIB_POINT, UT_String(name).buffer(), width);
            if (attributeRef.isInvalid()) continue;

            GA_Attribute& attribute = *attributeRef.getAttribute();
            attribute.hardenAllPages();

            if (type == "bool") {
                convertPointDataGridAttribute<bool>(tree, index, attribute, geo);
            }
            else if (type == "int16") {
                convertPointDataGridAttribute<int16_t>(tree, index, attribute, geo);
            }
            else if (type == "int32") {
                convertPointDataGridAttribute<int32_t>(tree, index, attribute, geo);
            }
            else if (type == "int64") {
                convertPointDataGridAttribute<int64_t>(tree, index, attribute, geo);
            }
            else if (type == "half") {
                convertPointDataGridAttribute<half>(tree, index, attribute, geo);
            }
            else if (type == "float") {
                convertPointDataGridAttribute<float>(tree, index, attribute, geo);
            }
            else if (type == "double") {
                convertPointDataGridAttribute<double>(tree, index, attribute, geo);
            }
            else if (type == "vec3h") {
                convertPointDataGridAttribute<Vec3<half> >(tree, index, attribute, geo);
            }
            else if (type == "vec3s") {
                convertPointDataGridAttribute<Vec3<float> >(tree, index, attribute, geo);
            }
            else if (type == "vec3d") {
                convertPointDataGridAttribute<Vec3<double> >(tree, index, attribute, geo);
            }
            else {
                throw std::runtime_error("Unknown Attribute Type for Conversion: " + type);
            }

            attribute.tryCompressAllPages();
        }
    }

    detail.merge(geo);
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
    PRM_ChoiceListType(PRM_CHOICELIST_EXCLUSIVE | PRM_CHOICELIST_REPLACE), sopBuildAttrMenu);

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
        const char* itemsA[] = {
            "none", "None",
            "truncate", "16-bit truncate",
            NULL
    };

    attrParms.add(hutil::ParmFactory(PRM_ORD, "valuecompressionA#", "Value Compression")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Value Compression to use for specific attributes.")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, itemsA));
    }

    {
        const char* itemsB[] = {
            "none", "None",
            "truncate", "16-bit truncate",
            UnitVecAttributeCodec::name(), "Unit Vector",
            NULL
    };

    attrParms.add(hutil::ParmFactory(PRM_ORD, "valuecompressionB#", "Value Compression")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Value Compression to use for specific attributes.")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, itemsB));
    }

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

    UT_String tmpStr;

    const GU_Detail* gdp = this->getInputLastGeo(0, CHgetEvalTime());

    if(!toVdbPoints || !gdp)
    {
        for (int i = 1, N = evalInt("attrList", 0, 0); i <= N; ++i) {
            changed |= setVisibleStateInst("valuecompressionA#", &i, false);
            changed |= setVisibleStateInst("valuecompressionB#", &i, false);
        }
    }
    else
    {
        for (int i = 1, N = evalInt("attrList", 0, 0); i <= N; ++i) {
            evalStringInst("attribute#", &i, tmpStr, 0, 0);
            Name attributeName = Name(tmpStr);

            GA_ROAttributeRef attrRef = gdp->findPointAttribute(attributeName.c_str());

            if (!attrRef.isValid()) {
                changed |= setVisibleStateInst("valuecompressionA#", &i, false);
                changed |= setVisibleStateInst("valuecompressionB#", &i, false);
                continue;
            }

            GA_Attribute const * attribute = attrRef.getAttribute();

            if (!attribute) {
                changed |= setVisibleStateInst("valuecompressionA#", &i, false);
                changed |= setVisibleStateInst("valuecompressionB#", &i, false);
                continue;
            }

            Name type;

            try {
                type = attrStringTypeFromGAAttribute(attribute);
            }
            catch (std::exception& e) {
                continue;
            }

            changed |= setVisibleStateInst("valuecompressionA#", &i, (type == "float"));
            changed |= setVisibleStateInst("valuecompressionB#", &i, (type == "vec3s"));
        }
    }

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Points::cookMySop(OP_Context& context)
{
    typedef std::pair<Name, int> NameAndCompression;
    typedef std::vector<NameAndCompression> NameAndCompressionVec;

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

            hvdb::VdbPrimCIterator vdbIt(ptGeo, group);

            if (vdbIt) {
                convertPointDataGrid(*gdp, vdbIt);
            } else {
                addError(SOP_MESSAGE, "No VDBs found");
            }

            return error();
        }

        // Set member data
        float voxelSize = evalFloat("voxelsize", 0, time);

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
                voxelSize = transform->voxelSize()[0];

            } else {
                addError(SOP_MESSAGE, "Second input has no VDB primitives.");
                return error();
            }
        }
        else {
            transform = Transform::createLinearTransform(voxelSize);
        }

        UT_String attrName;
        NameAndCompressionVec attributes;

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

                    int attributeCompression = 0;

                    // when converting specific attributes apply chosen compression.

                    if (type == "float") {
                        attributeCompression = evalIntInst("valuecompressionA#", &i, 0, 0);
                    }
                    if (type == "vec3s") {
                        attributeCompression = evalIntInst("valuecompressionB#", &i, 0, 0);
                    }

                    // compression types for these attributes are offset from the position
                    // compression types at the start of the compression enum in PointUtil.h
                    if (attributeCompression != 0) attributeCompression += FIXED_POSITION_8;
                    attributes.push_back(NameAndCompression(attributeName, attributeCompression));
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
                         attributes.push_back(NameAndCompression(attrName, 0));
                    }
                }
            }
        }

        // Determine position compression

        const int positionCompression = evalInt("poscompression", 0, time);

        const openvdb::NamePair positionAttributeType =
                    positionAttrTypeFromCompression(positionCompression);

        // compute list of offsets (if group supplied)

        GA_PointGroup* group = NULL;

        const OffsetListPtr offsets = computeOffsets(group);

        // Create PointPartitioner compatible P attribute wrapper

        PointAttribute<openvdb::Vec3f> points(ptGeo->getP(), offsets);

        // Create PointIndexGrid used for consistent index ordering in all attribute conversion

        PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(points, *transform);

        // Create PointDataGrid using position attribute

        PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(
                                *pointIndexGrid, points, positionAttributeType, *transform);

        PointIndexTree& indexTree = pointIndexGrid->tree();
        PointDataTree& tree = pointDataGrid->tree();

        // Add other attributes to PointDataGrid

        for (NameAndCompressionVec::const_iterator it = attributes.begin(),
                    it_end = attributes.end(); it != it_end; ++it)
        {
            const openvdb::Name name = it->first;
            const int compression = it->second;

            // skip position as this has already been added

            if (name == "P")  continue;

            GA_ROAttributeRef attrRef = ptGeo->findPointAttribute(name.c_str());

            if (!attrRef.isValid())     continue;

            GA_Attribute const * attribute = attrRef.getAttribute();

            if (!attribute)             continue;

            // Append the new attribute to the PointDataGrid
            AttributeSet::Util::NameAndType nameAndType(name,
                                    attrTypeFromGAAttribute(attribute, compression));

            appendAttribute(tree, nameAndType);

            // Now populate the attribute using the Houdini attribute
            populateAttributeFromHoudini(tree, indexTree, nameAndType.name, nameAndType.type, attribute, offsets);
        }

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

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
