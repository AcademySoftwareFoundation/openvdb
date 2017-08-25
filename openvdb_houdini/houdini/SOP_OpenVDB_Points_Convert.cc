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

/// @file SOP_OpenVDB_Points_Convert.cc
///
/// @authors Dan Bailey, Nick Avramoussis, James Bird
///
/// @brief Converts points to OpenVDB points.

#include <openvdb/openvdb.h>
#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointGroup.h>

#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/AttributeTransferUtil.h>

#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

#if (UT_MAJOR_VERSION_INT >= 15)
    #include <GU/GU_PackedContext.h>
#endif

#if (UT_MAJOR_VERSION_INT >= 14)
    #include <GU/GU_PrimPacked.h>
    #include <GU/GU_PackedGeometry.h>
    #include <GU/GU_PackedFragment.h>
    #include <GU/GU_DetailHandle.h>
#endif

#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT
#include <SYS/SYS_Types.h> // for int32, float32, etc

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility> // for std::pair
#include <vector>

using namespace openvdb;
using namespace openvdb::points;
using namespace openvdb::math;

namespace openvdb_houdini {
    template <> inline openvdb::math::Quat<float>
    evalAttrDefault<openvdb::math::Quat<float>>(const GA_Defaults& defaults, int)
    {
        openvdb::math::Quat<float> quat;
        fpreal32 value;

        for (int i = 0; i < 4; i++) {
            defaults.get(i, value);
            quat[i] = float(value);
        }

        return quat;
    }

    template <> inline openvdb::math::Quat<double>
    evalAttrDefault<openvdb::math::Quat<double>>(const GA_Defaults& defaults, int)
    {
        openvdb::math::Quat<double> quat;
        fpreal64 value;

        for (int i = 0; i < 4; i++) {
            defaults.get(i, value);
            quat[i] = double(value);
        }

        return quat;
    }

    template <> inline openvdb::math::Mat4<float>
    evalAttrDefault<openvdb::math::Mat4<float>>(const GA_Defaults& defaults, int)
    {
        openvdb::math::Mat4<float> mat;
        fpreal64 value;
        float* data = mat.asPointer();

        for (int i = 0; i < 16; i++) {
            defaults.get(i, value);
            data[i] = float(value);
        }

        return mat;
    }

    template <> inline openvdb::math::Mat4<double>
    evalAttrDefault<openvdb::math::Mat4<double>>(const GA_Defaults& defaults, int)
    {
        openvdb::math::Mat4<double> mat;
        fpreal64 value;
        double* data = mat.asPointer();

        for (int i = 0; i < 16; i++) {
            defaults.get(i, value);
            data[i] = double(value);
        }

        return mat;
    }
}

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

enum COMPRESSION_TYPE
{
    NONE = 0,
    TRUNCATE,
    UNIT_VECTOR,
    UNIT_FIXED_POINT_8,
    UNIT_FIXED_POINT_16,
};

/// @brief Returns supported Storage types for conversion from GA_Attribute
///
inline GA_Storage
attributeStorageType(const GA_Attribute* const attribute)
{
    if (!attribute) return GA_STORE_INVALID;

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();
    if (!tupleAIF)
    {
        if (attribute->getAIFStringTuple())
        {
            return GA_STORE_STRING;
        }
    }
    else
    {
        return tupleAIF->getStorage(attribute);
    }

    return GA_STORE_INVALID;
}

inline int16_t
attributeTupleSize(const GA_Attribute* const attribute)
{
    if (!attribute) return int16_t(0);

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();
    if (!tupleAIF)
    {
        const GA_AIFStringTuple* tupleAIFString = attribute->getAIFStringTuple();
        if (tupleAIFString)
        {
            return static_cast<int16_t>(tupleAIFString->getTupleSize(attribute));
        }
    }
    else
    {
        return static_cast<int16_t>(tupleAIF->getTupleSize(attribute));
    }

    return int16_t(0);
}

template <typename ValueType, typename CodecType = NullCodec>
inline void
convertAttributeFromHoudini(PointDataTree& tree, const tools::PointIndexTree& indexTree,
    const openvdb::Name& name, const GA_Attribute* const attribute,
    const GA_Defaults& defaults, const Index stride = 1)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");
    static_assert(!std::is_same<ValueType, openvdb::Name>::value,
        "ValueType must not be openvdb::Name/std::string");

    using HoudiniAttribute = hvdb::HoudiniReadAttribute<ValueType>;

    ValueType value = hvdb::evalAttrDefault<ValueType>(defaults, 0);

    // empty metadata if default is zero
    Metadata::Ptr defaultValue;
    if (!math::isZero<ValueType>(value)) {
        defaultValue = TypedMetadata<ValueType>(value).copy();
    }

    appendAttribute<ValueType, CodecType>(tree, name, zeroVal<ValueType>(),
        stride, /*constantstride=*/true, defaultValue);

    HoudiniAttribute houdiniAttribute(*attribute);
    populateAttribute<PointDataTree, tools::PointIndexTree, HoudiniAttribute>(
        tree, indexTree, name, houdiniAttribute, stride);
}

inline void
convertAttributeFromHoudini(PointDataTree& tree, const tools::PointIndexTree& indexTree,
    const openvdb::Name& name, const GA_Attribute* const attribute, const int compression = 0)
{
    using HoudiniStringAttribute = hvdb::HoudiniReadAttribute<openvdb::Name>;

    if (!attribute) {
        std::stringstream ss; ss << "Invalid attribute - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_Storage storage(attributeStorageType(attribute));

    if (storage == GA_STORE_INVALID) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const int16_t width(attributeTupleSize(attribute));
    assert(width > 0);

    // explicitly handle string attributes

    if (storage == GA_STORE_STRING) {
        appendAttribute<Name>(tree, name);
        HoudiniStringAttribute houdiniAttribute(*attribute);
        populateAttribute<PointDataTree, tools::PointIndexTree, HoudiniStringAttribute>(
            tree, indexTree, name, houdiniAttribute);
        return;
    }

    const GA_AIFTuple* tupleAIF = attribute->getAIFTuple();
    if (!tupleAIF) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    GA_Defaults defaults = tupleAIF->getDefaults(attribute);
    const GA_TypeInfo typeInfo(attribute->getOptions().typeInfo());

    const bool isVector = width == 3 && (typeInfo == GA_TYPE_VECTOR ||
                                         typeInfo == GA_TYPE_NORMAL ||
                                         typeInfo == GA_TYPE_COLOR);
    const bool isQuaternion = width == 4 && (typeInfo == GA_TYPE_QUATERNION);
    const bool isMatrix = width == 16 && (typeInfo == GA_TYPE_TRANSFORM);

    if (isVector)
    {
        if (storage == GA_STORE_INT32) {
            convertAttributeFromHoudini<Vec3<int>>(tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL16)
        {
            // implicitly convert 16-bit float into truncated 32-bit float

            convertAttributeFromHoudini<Vec3<float>, TruncateCodec>(
                tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL32)
        {
            if (compression == NONE) {
                convertAttributeFromHoudini<Vec3<float>>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == TRUNCATE) {
                convertAttributeFromHoudini<Vec3<float>, TruncateCodec>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == UNIT_VECTOR) {
                convertAttributeFromHoudini<Vec3<float>, UnitVecCodec>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == UNIT_FIXED_POINT_8) {
                convertAttributeFromHoudini<Vec3<float>, FixedPointCodec<true, UnitRange>>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == UNIT_FIXED_POINT_16) {
                convertAttributeFromHoudini<Vec3<float>, FixedPointCodec<false, UnitRange>>(
                    tree, indexTree, name, attribute, defaults);
            }
        }
        else if (storage == GA_STORE_REAL64) {
            convertAttributeFromHoudini<Vec3<double>>(tree, indexTree, name, attribute, defaults);
        }
        else {
            std::stringstream ss; ss << "Unknown vector attribute type - " << name;
            throw std::runtime_error(ss.str());
        }
    }
    else if (isQuaternion)
    {
        if (storage == GA_STORE_REAL16)
        {
            // implicitly convert 16-bit float into 32-bit float

            convertAttributeFromHoudini<Quat<float>>(tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL32)
        {
            convertAttributeFromHoudini<Quat<float>>(tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL64) {
            convertAttributeFromHoudini<Quat<double>>(tree, indexTree, name, attribute, defaults);
        }
        else {
            std::stringstream ss; ss << "Unknown quaternion attribute type - " << name;
            throw std::runtime_error(ss.str());
        }
    }
    else if (isMatrix)
    {
        if (storage == GA_STORE_REAL16)
        {
            // implicitly convert 16-bit float into 32-bit float

            convertAttributeFromHoudini<Mat4<float>>(tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL32)
        {
            convertAttributeFromHoudini<Mat4<float>>(tree, indexTree, name, attribute, defaults);
        }
        else if (storage == GA_STORE_REAL64) {
            convertAttributeFromHoudini<Mat4<double>>(tree, indexTree, name, attribute, defaults);
        }
        else {
            std::stringstream ss; ss << "Unknown matrix attribute type - " << name;
            throw std::runtime_error(ss.str());
        }
    }
    else {
        if (storage == GA_STORE_BOOL) {
            convertAttributeFromHoudini<bool>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_INT16) {
            convertAttributeFromHoudini<int16_t>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_INT32) {
            convertAttributeFromHoudini<int32_t>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_INT64) {
            convertAttributeFromHoudini<int64_t>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL16) {
            convertAttributeFromHoudini<float, TruncateCodec>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == NONE) {
            convertAttributeFromHoudini<float>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == TRUNCATE) {
            convertAttributeFromHoudini<float, TruncateCodec>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == UNIT_FIXED_POINT_8) {
            convertAttributeFromHoudini<float, FixedPointCodec<true, UnitRange>>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == UNIT_FIXED_POINT_16) {
            convertAttributeFromHoudini<float, FixedPointCodec<false, UnitRange>>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL64) {
            convertAttributeFromHoudini<double>(tree, indexTree, name, attribute, defaults, width);
        } else {
            std::stringstream ss; ss << "Unknown attribute type - " << name;
            throw std::runtime_error(ss.str());
        }
    }
}


////////////////////////////////////////


using AttributeInfoMap = std::map<Name, std::pair<int, bool>>;


///////////////////////////////////////


inline
PointDataGrid::Ptr
createPointDataGrid(const GU_Detail& ptGeo, const int compression,
                    const AttributeInfoMap& attributes, const openvdb::math::Transform& transform)
{
    using HoudiniPositionAttribute = hvdb::HoudiniReadAttribute<openvdb::Vec3d>;

    // store point group information

    const GA_ElementGroupTable& elementGroups = ptGeo.getElementGroupTable(GA_ATTRIB_POINT);

    // Create PointPartitioner compatible P attribute wrapper (for now no offset filtering)

    const GA_Attribute& positionAttribute = *ptGeo.getP();

    hvdb::OffsetListPtr offsets;
    hvdb::OffsetPairListPtr offsetPairs;

    size_t vertexCount = 0;

    for (GA_Iterator primitiveIt(ptGeo.getPrimitiveRange()); !primitiveIt.atEnd(); ++primitiveIt) {
        const GA_Primitive* primitive = ptGeo.getPrimitiveList().get(*primitiveIt);

        if (primitive->getTypeId() != GA_PRIMNURBCURVE) continue;

        vertexCount = primitive->getVertexCount();

        if (vertexCount == 0)  continue;

        if (!offsets)   offsets.reset(new hvdb::OffsetList);

        GA_Offset firstOffset = primitive->getPointOffset(0);
        offsets->push_back(firstOffset);

        if (vertexCount > 1) {
            if (!offsetPairs)   offsetPairs.reset(new hvdb::OffsetPairList);

            for (size_t i = 1; i < vertexCount; i++) {
                GA_Offset offset = primitive->getPointOffset(i);
                offsetPairs->push_back(hvdb::OffsetPair(firstOffset, offset));
            }
        }
    }

    HoudiniPositionAttribute points(positionAttribute, offsets);

    // Create PointIndexGrid used for consistent index ordering in all attribute conversion

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(points, transform);

    // Create PointDataGrid using position attribute

    PointDataGrid::Ptr pointDataGrid;

    if (compression == 1 /*FIXED_POSITION_16*/) {
        pointDataGrid = createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(
            *pointIndexGrid, points, transform);
    }
    else if (compression == 2 /*FIXED_POSITION_8*/) {
        pointDataGrid = createPointDataGrid<FixedPointCodec<true>, PointDataGrid>(
            *pointIndexGrid, points, transform);
    }
    else /*NONE*/ {
        pointDataGrid = createPointDataGrid<NullCodec, PointDataGrid>(
            *pointIndexGrid, points, transform);
    }

    tools::PointIndexTree& indexTree = pointIndexGrid->tree();
    PointDataTree& tree = pointDataGrid->tree();
    PointDataTree::LeafIter leafIter = tree.beginLeaf();

    if (!leafIter)  return pointDataGrid;

    // Append (empty) groups to tree

    std::vector<Name> groupNames;
    groupNames.reserve(elementGroups.entries());

    for (auto it = elementGroups.beginTraverse(), itEnd = elementGroups.endTraverse();
        it != itEnd; ++it)
    {
        groupNames.push_back((*it)->getName().toStdString());
    }

    appendGroups(tree, groupNames);

    // Set group membership in tree

    const int64_t numPoints = ptGeo.getNumPoints();
    std::vector<short> inGroup(numPoints, short(0));

    for (auto it = elementGroups.beginTraverse(), itEnd = elementGroups.endTraverse();
        it != itEnd; ++it)
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

    for (const auto& attrInfo : attributes)
    {
        const openvdb::Name& name = attrInfo.first;

        // skip position as this has already been added

        if (name == "P")  continue;

        GA_ROAttributeRef attrRef = ptGeo.findPointAttribute(name.c_str());

        if (!attrRef.isValid())     continue;

        GA_Attribute const * gaAttribute = attrRef.getAttribute();

        if (!gaAttribute)             continue;

        const GA_AIFSharedStringTuple* sharedStringTupleAIF =
            gaAttribute->getAIFSharedStringTuple();
        const bool isString = bool(sharedStringTupleAIF);

        // Extract all the string values from the string table and insert them
        // into the Descriptor Metadata
        if (isString)
        {
            // Iterate over the strings in the table and insert them into the Metadata
            MetaMap& metadata = makeDescriptorUnique(tree)->getMetadata();
            StringMetaInserter inserter(metadata);
            for (auto it = sharedStringTupleAIF->begin(gaAttribute),
                itEnd = sharedStringTupleAIF->end(); !(it == itEnd); ++it)
            {
                Name str(it.getString());
                if (!str.empty())   inserter.insert(str);
            }
        }

        convertAttributeFromHoudini(tree, indexTree, name, gaAttribute,
            /*compression=*/attrInfo.second.first);
    }

    // Attempt to compact attributes

    compactAttributes(tree);

    // Apply blosc compression to attributes

    for (const auto& attrInfo : attributes)
    {
        if (!attrInfo.second.second)  continue;

        bloscCompressAttribute(tree, attrInfo.first);
    }

    return pointDataGrid;
}

///////////////////////////////////////


template<typename ValueType>
Metadata::Ptr
createTypedMetadataFromAttribute(const GA_Attribute* const attribute, const uint32_t component = 0)
{
    using HoudiniAttribute = hvdb::HoudiniReadAttribute<ValueType>;

    ValueType value;
    HoudiniAttribute::get(*attribute, value, /*offset*/0, component);
    return openvdb::TypedMetadata<ValueType>(value).copy();
}

template <typename ValueType>
void
populateHoudiniDetailAttribute(GA_RWAttributeRef& attrib, const openvdb::MetaMap& metaMap,
                               const Name& key, const int index)
{
    using WriteHandleType = typename hvdb::GAHandleTraits<ValueType>::RW;
    using TypedMetadataT = TypedMetadata<ValueType>;

    typename TypedMetadataT::ConstPtr typedMetadata = metaMap.getMetadata<TypedMetadataT>(key);
    if (!typedMetadata) return;

    const ValueType& value = typedMetadata->value();
    WriteHandleType handle(attrib.getAttribute());
    hvdb::writeAttributeValue<WriteHandleType, ValueType>(handle, GA_Offset(0), index, value);
}

inline void
convertGlobalMetadataToHoudini(GU_Detail& detail, const openvdb::MetaMap& metaMap,
                               std::vector<std::string>& warnings)
{
    struct Local {
        static bool isGlobalMetadata(const Name& name) {
            return name.compare(0, 7, "global:") == 0;
        }

        static Name toDetailName(const Name& name) {
            Name detailName(name);
            detailName.erase(0, 7);
            const size_t open = detailName.find('[');
            if (open != std::string::npos) {
                detailName = detailName.substr(0, open);
            }
            return detailName;
        }

        static int toDetailIndex(const Name& name) {
            const size_t open = name.find('[');
            const size_t close = name.find(']');
            int index = 0;
            if (open != std::string::npos && close != std::string::npos &&
                close == name.length()-1 && open > 0 && open+1 < close) {
                try { // parse array index
                    index = std::stoi(name.substr(open+1, close-open-1));
                }
                catch (const std::exception&) {}
            }
            return index;
        }
    };

    using DetailInfo = std::pair<Name, int>;
    using DetailMap = std::map<Name, DetailInfo>;

    DetailMap detailCreate;
    DetailMap detailPopulate;

    for(MetaMap::ConstMetaIterator iter = metaMap.beginMeta(); iter != metaMap.endMeta(); ++iter)
    {
        const Metadata::Ptr metadata = iter->second;
        if (!metadata) continue;

        const Name& key = iter->first;

        if (!Local::isGlobalMetadata(key)) continue;

        Name name = Local::toDetailName(key);
        int index = Local::toDetailIndex(key);

        // add to creation map

        if (detailCreate.find(name) == detailCreate.end()) {
            detailCreate[name] = DetailInfo(metadata->typeName(), index);
        }
        else {
            if (index > detailCreate[name].second)   detailCreate[name].second = index;
        }

        // add to populate map

        detailPopulate[key] = DetailInfo(name, index);
    }

    // add all detail attributes

    for (const auto& item : detailCreate) {
        const Name& name = item.first;
        const DetailInfo& info = item.second;
        const Name& type = info.first;
        const int size = info.second;
        GA_RWAttributeRef attribute = detail.findGlobalAttribute(name);

        if (attribute.isInvalid())
        {
            const GA_Storage storage = hvdb::gaStorageFromAttrString(type);

            if (storage == GA_STORE_INVALID) {
                throw std::runtime_error("Invalid attribute storage type \"" + name + "\".");
            }

            if (type == "vec3s" || type == "vec3d") {
                attribute = detail.addTuple(storage, GA_ATTRIB_GLOBAL, name.c_str(), 3);
                attribute.setTypeInfo(GA_TYPE_VECTOR);
            }
            else {
                attribute = detail.addTuple(storage, GA_ATTRIB_GLOBAL, name.c_str(), size+1);
            }

            if (!attribute.isValid()) {
                throw std::runtime_error("Error creating attribute with name \"" + name + "\".");
            }
        }
    }

    // populate the values

    for (const auto& item : detailPopulate) {
        const Name& key = item.first;
        const DetailInfo& info = item.second;
        const Name& name = info.first;
        const int index = info.second;
        const Name& type = metaMap[key]->typeName();

        GA_RWAttributeRef attrib = detail.findGlobalAttribute(name);
        assert(!attrib.isInvalid());

        if (type == openvdb::typeNameAsString<bool>())                 populateHoudiniDetailAttribute<bool>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<int16_t>())         populateHoudiniDetailAttribute<int16_t>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<int32_t>())         populateHoudiniDetailAttribute<int32_t>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<int64_t>())         populateHoudiniDetailAttribute<int64_t>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<float>())           populateHoudiniDetailAttribute<float>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<double>())          populateHoudiniDetailAttribute<double>(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<Vec3<int32_t> >())  populateHoudiniDetailAttribute<Vec3<int32_t> >(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<Vec3<float> >())    populateHoudiniDetailAttribute<Vec3<float> >(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<Vec3<double> >())   populateHoudiniDetailAttribute<Vec3<double> >(attrib, metaMap, key, index);
        else if (type == openvdb::typeNameAsString<Name>())            populateHoudiniDetailAttribute<Name>(attrib, metaMap, key, index);
        else {
            std::stringstream ss;
            ss << "Metadata value \"" << key
                << "\" unsupported type for detail attribute conversion.";
            warnings.push_back(ss.str());
        }
    }
}


////////////////////////////////////////


class SOP_OpenVDB_Points_Convert: public hvdb::SOP_NodeVDB
{
public:
    enum { TRANSFORM_TARGET_POINTS = 0, TRANSFORM_VOXEL_SIZE, TRANSFORM_REF_GRID };

    SOP_OpenVDB_Points_Convert(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Points_Convert() override = default;

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i ) const override { return (i == 1); }

protected:

    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;

private:
    hvdb::Interrupter mBoss;
}; // class SOP_OpenVDB_Points_Convert



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

    // null object
    menuEntries[menuIdx].setToken("0");
    menuEntries[menuIdx++].setLabel("- no attribute selected -");

    if (gdp) {

        // point attribute names
        auto iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);

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
    openvdb::initialize();

    if (table == nullptr) return;

    hutil::ParmList parms;

    {
        const char* items[] = {
            "vdb", "Pack Points into VDB Points",
            "hdk", "Extract Points from VDB Points",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "conversion", "Conversion")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("The conversion method for the expected input types.")
            .setDocumentation(
                "Whether to pack points into a VDB Points primitive"
                " or to extract points from such a primitive "));
    }

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setTooltip("Specify a subset of the input point data grids to convert.")
        .setDocumentation(
            "A subset of the input VDB Points primitives to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroup", "VDB Points Group")
        .setChoiceList(&hvdb::VDBPointsGroupMenuInput1)
        .setTooltip("Specify VDB Points Groups to use as an input.")
        .setDocumentation(
            "The point group inside the VDB Points primitive to extract\n\n"
            "This may be a normal point group that was collapsed into the"
            " VDB Points primitive when it was created, or a new group created"
            " with the [OpenVDB Points Group node|Node:sop/DW_OpenVDBPointsGroup]."));

    //  point grid name
    parms.add(hutil::ParmFactory(PRM_STRING, "name", "VDB Name")
        .setDefault("points")
        .setTooltip("The name of the VDB Points primitive to be created"));

    {   // Transform
        const char* items[] = {
            "targetpointspervoxel",  "Using Target Points Per Voxel",
            "voxelsizeonly",         "Using Voxel Size Only",
            "userefvdb",             "To Match Reference VDB",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "transform", "Define Transform")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip(
                "Specify how to construct the PointDataGrid transform. If\n"
                "an optional transform input is provided for the first two\n"
                "options, the rotate and translate components are preserved.\n"
                "Using Target Points Per Voxel:\n"
                "    Automatically calculates a voxel size based off the input\n"
                "    point set and a target amount of points per voxel.\n"
                "Using Voxel Size Only:\n"
                "    Explicitly sets a voxel size.\n"
                "To Match Reference VDB:\n"
                "    Uses the complete transform provided from the second input.")
            .setDocumentation("\
How to construct the VDB Points primitive's transform\n\n\
An important consideration is how big to make the grid cells\n\
that contain the points.  Too large and there are too many points\n\
per cell and little optimization occurs.  Too small and the cost\n\
of the cells outweighs the points.\n\
\n\
Using Target Points Per Voxel:\n\
    Automatically calculate a voxel size so that the given number\n\
    of points ends up in each voxel.  This will assume uniform\n\
    distribution of points.\n\
    \n\
    If an optional transform input is provided, use its rotation\n\
    and translation.\n\
Using Voxel Size Only:\n\
    Provide an explicit voxel size, and if an optional transform input\n\
    is provided, use its rotation and translation.\n\
To Match Reference VDB:\n\
    Use the complete transform provided from the second input.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setTooltip("The desired voxel size of the new VDB Points grid"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "pointspervoxel", "Points Per Voxel")
        .setDefault(8)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 16)
        .setTooltip(
            "The number of points per voxel to use as the target for "
            "automatic voxel size computation"));

    // Group name (Transform reference)
    parms.add(hutil::ParmFactory(PRM_STRING, "refvdb", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setSpareData(&SOP_Node::theSecondInput)
        .setTooltip("References the first/selected grid's transform.")
        .setDocumentation(
            "Which VDB in the second input to use as the reference for the transform\n\n"
            "If this is not set, use the first VDB found."));

    //////////

    // Point attribute transfer

    {
        char const * const items[] = {
            "none", "None",
            "int16", "16-bit Fixed Point",
            "int8", "8-bit Fixed Point",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "poscompression", "Position Compression")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("The position attribute compression setting.")
            .setDocumentation(
                "The position can be stored relative to the center of the voxel.\n"
                "This means it does not require the full 32-bit float representation,\n"
                "but can be quantized to a smaller fixed-point value."));
    }

    parms.add(hutil::ParmFactory(PRM_HEADING, "transferHeading", "Attribute Transfer"));

     // Mode. Either convert all or convert specifc attributes

    {
        char const * const items[] = {
            "all", "All Attributes",
            "spec", "Specific Attributes",
            nullptr
    };

    parms.add(hutil::ParmFactory(PRM_ORD, "mode", "Mode")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Whether to transfer only specific attributes or all attributes found")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    hutil::ParmList attrParms;

    // Attribute name
    attrParms.add(hutil::ParmFactory(PRM_STRING, "attribute#", "Attribute")
        .setChoiceList(&PrimAttrMenu)
        .setSpareData(&SOP_Node::theFirstInput)
        .setTooltip("Select a point attribute to transfer.\n\n"
            "Supports integer and floating-point attributes of "
            "arbitrary precisions and tuple sizes."));

    {
        char const * const items[] = {
            "none", "None",
            "truncate", "16-bit Truncate",
            UnitVecCodec::name(), "Unit Vector",
            FixedPointCodec<true, UnitRange>::name(), "8-bit Unit",
            FixedPointCodec<false, UnitRange>::name(), "16-bit Unit",
            nullptr
        };

        attrParms.add(hutil::ParmFactory(PRM_ORD, "valuecompression#", "Value Compression")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("Value compression to use for specific attributes.")
            .setDocumentation("\
How to compress attribute values\n\
\n\
None:\n\
    Values are stored with their full precision.\n\
\n\
16-bit Truncate:\n\
    Values are stored at half precision, truncating lower-order bits.\n\
\n\
Unit Vector:\n\
    Values are treated as unit vectors, so that if two components\n\
    are known, the third is implied and need not be stored.\n\
\n\
8-bit Unit:\n\
    Values are treated as lying in the 0..1 range and are quantized to 8 bits.\n\
\n\
16-bit Unit:\n\
    Values are treated as lying in the 0..1 range and are quantized to 16 bits.\n"));
    }

    attrParms.add(hutil::ParmFactory(PRM_TOGGLE, "blosccompression#", "Blosc Compression")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Enable Blosc compression\n\n"
            "Blosc is a lossless compression codec that is effective with"
            " floating-point data and is very fast to compress and decompress."));

    // Add multi parm
    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "attrList", "Point Attributes")
        .setTooltip("Transfer point attributes to each voxel in the level set's narrow band")
        .setMultiparms(attrParms)
        .setDefault(PRMzeroDefaults));

    parms.add(hutil::ParmFactory(PRM_LABEL, "attributespacer", ""));

    {
        char const * const items[] = {
            "none", "None",
            UnitVecCodec::name(), "Unit Vector",
            "truncate", "16-bit Truncate",
            nullptr
    };

    parms.add(hutil::ParmFactory(PRM_ORD, "normalcompression", "Normal Compression")
        .setDefault(PRMzeroDefaults)
        .setTooltip("All normal attributes will use this compression codec.")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    {
        char const * const items[] = {
            "none", "None",
            FixedPointCodec<false, UnitRange>::name(), "16-bit Unit",
            FixedPointCodec<true, UnitRange>::name(), "8-bit Unit",
            "truncate", "16-bit Truncate",
            nullptr
    };

    parms.add(hutil::ParmFactory(PRM_ORD, "colorcompression", "Color Compression")
        .setDefault(PRMzeroDefaults)
        .setTooltip("All color attributes will use this compression codec.")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB Points Convert",
        SOP_OpenVDB_Points_Convert::factory, parms, *table)
        .addInput("Points to Convert")
        .addOptionalInput("Optional Reference VDB (for transform)")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Convert a point cloud into a VDB Points primitive, or vice versa.\"\"\"\n\
\n\
@overview\n\
\n\
This node converts an unstructured cloud of points to and from a single\n\
[VDB Points|http://www.openvdb.org/documentation/doxygen/points.html] primitive.\n\
The resulting primitive will reorder the points to place spatially\n\
close points close together.\n\
It is then able to efficiently unpack regions of interest within that primitive.\n\
The [OpenVDB Points Group node|Node:sop/DW_OpenVDBPointsGroup] can be used\n\
to create regions of interest.\n\
\n\
Because nearby points often have similar data, there is the possibility\n\
of aggressively compressing attribute data to minimize data size.\n\
\n\
@related\n\
- [OpenVDB Points Group|Node:sop/DW_OpenVDBPointsGroup]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Points_Convert::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Points_Convert(net, name, op);
}


SOP_OpenVDB_Points_Convert::SOP_OpenVDB_Points_Convert(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
    , mBoss("Converting points")
{
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Points_Convert::updateParmsFlags()
{
    bool changed = false;

    const bool toVdbPoints = evalInt("conversion", 0, 0) == 0;
    const bool convertAll = evalInt("mode", 0, 0) == 0;
    const auto transform = evalInt("transform", 0, 0);

    changed |= enableParm("group", !toVdbPoints);
    changed |= setVisibleState("group", !toVdbPoints);

    changed |= enableParm("vdbpointsgroup", !toVdbPoints);
    changed |= setVisibleState("vdbpointsgroup", !toVdbPoints);

    changed |= enableParm("name", toVdbPoints);
    changed |= setVisibleState("name", toVdbPoints);

    const int refexists = (this->nInputs() == 2);

    changed |= enableParm("transform", toVdbPoints);
    changed |= setVisibleState("transform", toVdbPoints);

    changed |= enableParm("refvdb", refexists);
    changed |= setVisibleState("refvdb", toVdbPoints);

    changed |= enableParm("voxelsize", toVdbPoints && transform == TRANSFORM_VOXEL_SIZE);
    changed |= setVisibleState("voxelsize", toVdbPoints && transform == TRANSFORM_VOXEL_SIZE);

    changed |= enableParm("pointspervoxel", toVdbPoints && transform == TRANSFORM_TARGET_POINTS);
    changed |= setVisibleState("pointspervoxel",
        toVdbPoints && transform == TRANSFORM_TARGET_POINTS);

    changed |= setVisibleState("transferHeading", toVdbPoints);

    changed |= enableParm("poscompression", toVdbPoints);
    changed |= setVisibleState("poscompression", toVdbPoints);

    changed |= enableParm("mode", toVdbPoints);
    changed |= setVisibleState("mode", toVdbPoints);

    changed |= enableParm("attrList", toVdbPoints && !convertAll);
    changed |= setVisibleState("attrList", toVdbPoints && !convertAll);

    changed |= enableParm("normalcompression", toVdbPoints && convertAll);
    changed |= setVisibleState("normalcompression", toVdbPoints && convertAll);

    changed |= enableParm("colorcompression", toVdbPoints && convertAll);
    changed |= setVisibleState("colorcompression", toVdbPoints && convertAll);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Points_Convert::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        if (evalInt("conversion", 0, time) != 0) {

            // Duplicate primary (left) input geometry and convert the VDB points inside

            if (duplicateSourceStealable(0, context) >= UT_ERROR_ABORT) return error();

            UT_String groupStr;
            evalString(groupStr, "group", 0, time);
            const GA_PrimitiveGroup *group =
                matchGroup(const_cast<GU_Detail&>(*gdp), groupStr.toStdString());

            // Extract VDB Point groups to filter

            UT_String pointsGroupStr;
            evalString(pointsGroupStr, "vdbpointsgroup", 0, time);
            const std::string pointsGroup = pointsGroupStr.toStdString();

            std::vector<std::string> includeGroups;
            std::vector<std::string> excludeGroups;
            openvdb::points::AttributeSet::Descriptor::parseNames(
                includeGroups, excludeGroups, pointsGroup);

            // passing an empty vector of attribute names implies that
            // all attributes should be converted
            const std::vector<std::string> emptyNameVector;

            UT_Array<GEO_Primitive*> primsToDelete;
            primsToDelete.clear();

            // Convert each VDB primitive independently
            for (hvdb::VdbPrimIterator vdbIt(gdp, group); vdbIt; ++vdbIt) {

                GU_Detail geo;

                const GridBase& baseGrid = vdbIt->getGrid();
                if (!baseGrid.isType<PointDataGrid>()) continue;

                const PointDataGrid& grid = static_cast<const PointDataGrid&>(baseGrid);

                // if all point data is being converted, sequentially pre-fetch any out-of-core
                // data for faster performance when using delayed-loading

                const bool allData =    emptyNameVector.empty() &&
                                        includeGroups.empty() &&
                                        excludeGroups.empty();

                if (allData) {
                    prefetch(grid.tree());
                }

                // perform conversion

                hvdb::convertPointDataGridToHoudini(
                    geo, grid, emptyNameVector, includeGroups, excludeGroups);

                const MetaMap& metaMap = grid;
                std::vector<std::string> warnings;
                convertGlobalMetadataToHoudini(geo, metaMap, warnings);
                if (warnings.size() > 0) {
                    for (const auto& warning: warnings) {
                        addWarning(SOP_MESSAGE, warning.c_str());
                    }
                }

                gdp->merge(geo);
                primsToDelete.append(*vdbIt);
            }

            gdp->deletePrimitives(primsToDelete, true);
            return error();
        }

        // if we're here, we're converting Houdini points to OpenVDB. Clear gdp entirely
        // before proceeding, then check for particles in the primary (left) input port

        gdp->clearAndDestroy();

        const GU_Detail* ptGeo = inputGeo(0, context);

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

            if (!refPrim) {
                addError(SOP_MESSAGE, "Second input has no VDB primitives.");
                return error();
            }

            transform = refPrim->getGrid().transform().copy();
        }

        const auto transformMode = evalInt("transform", 0, time);

        math::Mat4d matrix(math::Mat4d::identity());

        if (transform && transformMode != TRANSFORM_REF_GRID) {
            const math::AffineMap::ConstPtr affineMap = transform->baseMap()->getAffineMap();
            matrix = affineMap->getMat4();
        }
        else if (!transform && transformMode == TRANSFORM_REF_GRID) {
            addError(SOP_MESSAGE, "No target VDB transform found on second input.");
            return error();
        }

        if (transformMode == TRANSFORM_TARGET_POINTS) {
            using HoudiniPositionAttribute = hvdb::HoudiniReadAttribute<openvdb::Vec3R>;

            const int pointsPerVoxel = static_cast<int>(evalInt("pointspervoxel", 0, time));
            HoudiniPositionAttribute positions(*(ptGeo->getP()));

            const float voxelSize =
                openvdb::points::computeVoxelSize<HoudiniPositionAttribute, hvdb::Interrupter>(
                    positions, pointsPerVoxel, matrix, /*rounding*/ 5, &mBoss);

            matrix.preScale(Vec3d(voxelSize) / math::getScale(matrix));
            transform = Transform::createLinearTransform(matrix);
        } else if (transformMode == TRANSFORM_VOXEL_SIZE) {
            const auto voxelSize = evalFloat("voxelsize", 0, time);
            matrix.preScale(Vec3d(voxelSize) / math::getScale(matrix));
            transform = Transform::createLinearTransform(matrix);
        }

        UT_String attrName;
        AttributeInfoMap attributes;

        GU_Detail nonConstDetail;
        const GU_Detail* detail;

        // unpack any packed primitives

        mBoss.start();

        for (GA_Iterator it(ptGeo->getPrimitiveRange()); !it.atEnd(); ++it)
        {
            GA_Offset offset = *it;

            const GA_Primitive* primitive = ptGeo->getPrimitive(offset);
            if (!primitive || !GU_PrimPacked::isPackedPrimitive(*primitive)) continue;

            const GU_PrimPacked* packedPrimitive = static_cast<const GU_PrimPacked*>(primitive);

            packedPrimitive->unpack(nonConstDetail);
        }

        if (ptGeo->getNumPoints() > 0 && nonConstDetail.getNumPoints() == 0) {
            // only unpacked points exist so just use the input gdp

            detail = ptGeo;
        }
        else {
            // merge unpacked and packed point data

            nonConstDetail.mergePoints(*ptGeo);
            detail = &nonConstDetail;
        }

        if (evalInt("mode", 0, time) != 0) {
            // Transfer point attributes.
            if (evalInt("attrList", 0, time) > 0) {
                for (int i = 1, N = static_cast<int>(evalInt("attrList", 0, 0)); i <= N; ++i) {
                    evalStringInst("attribute#", &i, attrName, 0, 0);
                    const Name attributeName = Name(attrName);

                    const GA_ROAttributeRef attrRef =
                        detail->findPointAttribute(attributeName.c_str());

                    if (!attrRef.isValid()) continue;

                    const GA_Attribute* const attribute = attrRef.getAttribute();

                    if (!attribute) continue;

                    const GA_Storage storage(attributeStorageType(attribute));

                    // only tuple and string tuple attributes are supported

                    if (storage == GA_STORE_INVALID) {
                        std::stringstream ss; ss << "Invalid attribute type - " << attributeName;
                        throw std::runtime_error(ss.str());
                    }

                    const int16_t width(attributeTupleSize(attribute));
                    assert(width > 0);

                    const GA_TypeInfo typeInfo(attribute->getOptions().typeInfo());

                    const bool isVector = width == 3 && (typeInfo == GA_TYPE_VECTOR ||
                                                         typeInfo == GA_TYPE_NORMAL ||
                                                         typeInfo == GA_TYPE_COLOR);
                    const bool isQuaternion = width == 4 && (typeInfo == GA_TYPE_QUATERNION);
                    const bool isMatrix = width == 16 && (typeInfo == GA_TYPE_TRANSFORM);

                    const bool bloscCompression = evalIntInst("blosccompression#", &i, 0, 0);
                    int valueCompression = static_cast<int>(
                        evalIntInst("valuecompression#", &i, 0, 0));

                    // check value compression compatibility with attribute type

                    if (valueCompression != NONE) {
                        if (storage == GA_STORE_STRING) {
                            // disable value compression for strings and add a SOP warning

                            std::stringstream ss;
                            ss << "Value compression not supported on string attributes. "
                                "Disabling compression for attribute \"" << attributeName << "\".";
                            valueCompression = NONE;
                            addWarning(SOP_MESSAGE, ss.str().c_str());
                        } else {
                            // disable value compression for incompatible types
                            // and add a SOP warning

                            if (valueCompression == TRUNCATE &&
                                (storage != GA_STORE_REAL32 || isQuaternion || isMatrix))
                            {
                                std::stringstream ss;
                                ss << "Truncate value compression only supported for 32-bit"
                                    " floating-point attributes. Disabling compression for"
                                    " attribute \"" << attributeName << "\".";
                                valueCompression = NONE;
                                addWarning(SOP_MESSAGE, ss.str().c_str());
                            }

                            if (valueCompression == UNIT_VECTOR &&
                                (storage != GA_STORE_REAL32 || !isVector))
                            {
                                std::stringstream ss;
                                ss << "Unit Vector value compression only supported for"
                                    " vector 3 x 32-bit floating-point attributes. "
                                    "Disabling compression for attribute \""
                                    << attributeName << "\".";
                                valueCompression = NONE;
                                addWarning(SOP_MESSAGE, ss.str().c_str());
                            }

                            const bool isUnit = (valueCompression == UNIT_FIXED_POINT_8
                                || valueCompression == UNIT_FIXED_POINT_16);
                            if (isUnit && (storage != GA_STORE_REAL32 || (width != 1 && !isVector)))
                            {
                                std::stringstream ss;
                                ss << "Unit compression only supported for scalar and vector"
                                    " 3 x 32-bit floating-point attributes. "
                                    "Disabling compression for attribute \""
                                    << attributeName << "\".";
                                valueCompression = NONE;
                                addWarning(SOP_MESSAGE, ss.str().c_str());
                            }
                        }
                    }

                    attributes[attributeName] =
                        std::pair<int, bool>(valueCompression, bloscCompression);
                }
            }
        } else {

            // point attribute names
            auto iter = detail->pointAttribs().begin(GA_SCOPE_PUBLIC);

            const auto normalCompression = evalInt("normalcompression", 0, time);
            const auto colorCompression = evalInt("colorcompression", 0, time);

            if (!iter.atEnd()) {
                for (; !iter.atEnd(); ++iter) {
                    const char* str = (*iter)->getName();
                    if (!str) continue;

                    const Name attributeName = str;

                    if (attributeName == "P") continue;

                    const GA_ROAttributeRef attrRef =
                        detail->findPointAttribute(attributeName.c_str());

                    if (!attrRef.isValid()) continue;

                    const GA_Attribute* const attribute = attrRef.getAttribute();

                    if (!attribute) continue;

                    const GA_Storage storage(attributeStorageType(attribute));

                    // only tuple and string tuple attributes are supported

                    if (storage == GA_STORE_INVALID) {
                        std::stringstream ss; ss << "Invalid attribute type - " << attributeName;
                        throw std::runtime_error(ss.str());
                    }

                    const int16_t width(attributeTupleSize(attribute));
                    assert(width > 0);

                    const GA_TypeInfo typeInfo(attribute->getOptions().typeInfo());

                    const bool isNormal = width == 3 && typeInfo == GA_TYPE_NORMAL;
                    const bool isColor = width == 3 && typeInfo == GA_TYPE_COLOR;

                    int valueCompression = NONE;

                    if (isNormal) {
                        if (normalCompression == 1)             valueCompression = UNIT_VECTOR;
                        else if (normalCompression == 2)        valueCompression = TRUNCATE;
                    }
                    else if (isColor) {
                        if (colorCompression == 1)              valueCompression = UNIT_FIXED_POINT_16;
                        else if (colorCompression == 2)         valueCompression = UNIT_FIXED_POINT_8;
                        else if (colorCompression == 3)         valueCompression = TRUNCATE;
                    }

                    // when converting all attributes apply no compression
                    attributes[attributeName] = std::pair<int, bool>(valueCompression, false);
                }
            }
        }

        // Determine position compression

        const int positionCompression = static_cast<int>(evalInt("poscompression", 0, time));

        PointDataGrid::Ptr pointDataGrid = createPointDataGrid(
            *detail, positionCompression, attributes, *transform);

        for (GA_AttributeDict::iterator iter = detail->attribs().begin(GA_SCOPE_PUBLIC);
            !iter.atEnd(); ++iter)
        {
            const GA_Attribute* const attribute = *iter;
            if (!attribute) continue;

            const Name name("global:" + Name(attribute->getName()));
            Metadata::Ptr metadata = (*pointDataGrid)[name];
            if (metadata) continue;

            const GA_Storage storage(attributeStorageType(attribute));
            const int16_t width(attributeTupleSize(attribute));
            const GA_TypeInfo typeInfo(attribute->getOptions().typeInfo());

            const bool isVector = width == 3 && (typeInfo == GA_TYPE_VECTOR ||
                                                 typeInfo == GA_TYPE_NORMAL ||
                                                 typeInfo == GA_TYPE_COLOR);
            const bool isQuaternion = width == 4 && (typeInfo == GA_TYPE_QUATERNION);
            const bool isMatrix = width == 16 && (typeInfo == GA_TYPE_TRANSFORM);

            if (isVector) {
                if (storage == GA_STORE_REAL16) {
                    metadata = createTypedMetadataFromAttribute<Vec3<float> >(attribute);
                } else if (storage == GA_STORE_REAL32) {
                    metadata = createTypedMetadataFromAttribute<Vec3<float> >(attribute);
                } else if (storage == GA_STORE_REAL64) {
                    metadata = createTypedMetadataFromAttribute<Vec3<double> >(attribute);
                } else {
                    std::stringstream ss;
                    ss << "Detail attribute \"" << attribute->getName() << "\" " <<
                        "unsupported vector type for metadata conversion.";
                    addWarning(SOP_MESSAGE, ss.str().c_str());
                    continue;
                }
                assert(metadata);
                pointDataGrid->insertMeta(name, *metadata);
            } else if (isQuaternion) {
                if (storage == GA_STORE_REAL16) {
                    metadata = createTypedMetadataFromAttribute<Quat<float>>(attribute);
                } else if (storage == GA_STORE_REAL32) {
                    metadata = createTypedMetadataFromAttribute<Quat<float>>(attribute);
                } else if (storage == GA_STORE_REAL64) {
                    metadata = createTypedMetadataFromAttribute<Quat<double>>(attribute);
                } else {
                    std::stringstream ss;
                    ss << "Detail attribute \"" << attribute->getName() << "\" " <<
                        "unsupported quaternion type for metadata conversion.";
                    addWarning(SOP_MESSAGE, ss.str().c_str());
                    continue;
                }
            } else if (isMatrix) {
                if (storage == GA_STORE_REAL16) {
                    metadata = createTypedMetadataFromAttribute<Mat4<float>>(attribute);
                } else if (storage == GA_STORE_REAL32) {
                    metadata = createTypedMetadataFromAttribute<Mat4<float>>(attribute);
                } else if (storage == GA_STORE_REAL64) {
                    metadata = createTypedMetadataFromAttribute<Mat4<double>>(attribute);
                } else {
                    std::stringstream ss;
                    ss << "Detail attribute \"" << attribute->getName() << "\" " <<
                        "unsupported matrix type for metadata conversion.";
                    addWarning(SOP_MESSAGE, ss.str().c_str());
                    continue;
                }
            } else {
                for (int i = 0; i < width; i++) {
                    if (storage == GA_STORE_BOOL) {
                        metadata = createTypedMetadataFromAttribute<bool>(attribute, i);
                    } else if (storage == GA_STORE_INT16) {
                        metadata = createTypedMetadataFromAttribute<int16_t>(attribute, i);
                    } else if (storage == GA_STORE_INT32) {
                        metadata = createTypedMetadataFromAttribute<int32_t>(attribute, i);
                    } else if (storage == GA_STORE_INT64) {
                        metadata = createTypedMetadataFromAttribute<int64_t>(attribute, i);
                    } else if (storage == GA_STORE_REAL16) {
                        metadata = createTypedMetadataFromAttribute<float>(attribute, i);
                    } else if (storage == GA_STORE_REAL32) {
                        metadata = createTypedMetadataFromAttribute<float>(attribute, i);
                    } else if (storage == GA_STORE_REAL64) {
                        metadata = createTypedMetadataFromAttribute<double>(attribute, i);
                    } else if (storage == GA_STORE_STRING) {
                        metadata = createTypedMetadataFromAttribute<openvdb::Name>(attribute, i);
                    } else {
                        std::stringstream ss;
                        ss << "Detail attribute \"" << attribute->getName() << "\" " <<
                            "unsupported type for metadata conversion.";
                        addWarning(SOP_MESSAGE, ss.str().c_str());
                        continue;
                    }
                    assert(metadata);
                    if (width > 1) {
                        const Name arrayName(name + Name("[") + std::to_string(i) + Name("]"));
                        pointDataGrid->insertMeta(arrayName, *metadata);
                    }
                    else {
                        pointDataGrid->insertMeta(name, *metadata);
                    }
                }
            }
        }

        UT_String nameStr = "";
        evalString(nameStr, "name", 0, time);
        hvdb::createVdbPrimitive(*gdp, pointDataGrid, nameStr.toStdString().c_str());

        mBoss.end();

    } catch (const std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
