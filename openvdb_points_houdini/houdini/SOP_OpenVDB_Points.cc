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
#include <openvdb_points/tools/PointGroup.h>

#include "SOP_NodeVDBPoints.h"

#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT

#include <boost/ptr_container/ptr_vector.hpp>

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
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
    else if (type == "vec2h") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec2<half> >(attribute, offsets));
    }
    else if (type == "vec2s") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec2<float> >(attribute, offsets));
    }
    else if (type == "vec2d") {
        populateAttribute(tree, indexTree, name, PointAttribute<Vec2<double> >(attribute, offsets));
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


namespace sop_openvdb_points_internal {


template <typename T> struct GAHandleTraits { typedef GA_RWHandleF RW; };
template <typename T> struct GAHandleTraits<Vec3<T> > { typedef GA_RWHandleV3 RW; };
template <> struct GAHandleTraits<bool> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int16_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int32_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int64_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<half> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<float> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<double> { typedef GA_RWHandleF RW; };


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


template<typename PointDataTreeType, typename LeafOffsetArray>
struct ConvertPointDataGridPositionOp {

    typedef typename PointDataTreeType::LeafNodeType    PointDataLeaf;
    typedef typename LeafOffsetArray::const_reference   LeafOffsetPair;

    ConvertPointDataGridPositionOp( GU_Detail& detail,
                                    const PointDataTreeType& tree,
                                    const math::Transform& transform,
                                    const size_t index,
                                    const LeafOffsetArray& leafOffsets)
        : mDetail(detail)
        , mTree(tree)
        , mTransform(transform)
        , mIndex(index)
        , mLeafOffsets(leafOffsets) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        GA_RWHandleV3 pHandle(mDetail.getP());

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            assert(n < mLeafOffsets.size());

            // extract leaf and offset

            const LeafOffsetPair& leafOffset = mLeafOffsets[n];
            const PointDataLeaf& leaf = leafOffset.first;
            GA_Offset offset = leafOffset.second;

            AttributeHandle<Vec3f>::Ptr handle = AttributeHandle<Vec3f>::create(leaf.template attributeArray(mIndex));

            Vec3d uniformPos;

            const bool uniform = handle->isUniform();

            if (uniform)    uniformPos = handle->get(Index64(0));

            for (PointDataTree::LeafNodeType::ValueOnCIter iter = leaf.cbeginValueOn(); iter; ++iter) {

                Coord ijk = iter.getCoord();
                Vec3d xyz = ijk.asVec3d();

                IndexIter indexIter = leaf.beginIndex(ijk);
                for (; indexIter; ++indexIter) {

                    Vec3d pos = uniform ? uniformPos : Vec3d(handle->get(*indexIter));
                    pos = mTransform.indexToWorld(pos + xyz);
                    setAttributeValue(pos, pHandle, offset++);
                }
            }
        }
    }

    //////////

    GU_Detail&                              mDetail;
    const PointDataTreeType&                mTree;
    const math::Transform&                  mTransform;
    const size_t                            mIndex;
    const LeafOffsetArray&                  mLeafOffsets;
}; // ConvertPointDataGridPositionOp



template<typename PointDataTreeType, typename LeafOffsetArray>
struct ConvertPointDataGridGroupOp {

    typedef typename PointDataTreeType::LeafNodeType    PointDataLeaf;
    typedef typename LeafOffsetArray::const_reference   LeafOffsetPair;
    typedef AttributeSet::Descriptor::GroupIndex        GroupIndex;

    ConvertPointDataGridGroupOp(GA_PointGroup* pointGroup,
                                const LeafOffsetArray& leafOffsets,
                                const AttributeSet::Descriptor::GroupIndex index)
        : mPointGroup(pointGroup)
        , mLeafOffsets(leafOffsets)
        , mIndex(index) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const GroupType bitmask = GroupType(1) << mIndex.second;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            assert(n < mLeafOffsets.size());

            // extract leaf and offset

            const LeafOffsetPair& leafOffset = mLeafOffsets[n];
            const PointDataLeaf& leaf = leafOffset.first;
            GA_Offset offset = leafOffset.second;

            const AttributeArray& array = leaf.attributeArray(mIndex.first);

            assert(GroupAttributeArray::isGroup(array));

            const GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

            if (groupArray.isUniform() && (groupArray.get(0) & bitmask))
            {
                for (Index64 index = 0; index < groupArray.size(); index++) {
                    mPointGroup->addOffset(offset + index);
                }
            }
            else
            {
                for (Index64 index = 0; index < groupArray.size(); index++) {
                    if (groupArray.get(index) & bitmask)    mPointGroup->addOffset(offset + index);
                }
            }
        }
    }

    //////////

    GA_PointGroup*                  mPointGroup;
    const LeafOffsetArray&          mLeafOffsets;
    const GroupIndex                mIndex;
}; // ConvertPointDataGridGroupOp


template<typename PointDataTreeType, typename LeafOffsetArray, typename AttributeType>
struct ConvertPointDataGridAttributeOp {

    typedef typename PointDataTreeType::LeafNodeType    PointDataLeaf;
    typedef typename LeafOffsetArray::const_reference   LeafOffsetPair;
    typedef typename GAHandleTraits<AttributeType>::RW GAHandleType;

    ConvertPointDataGridAttributeOp(GA_Attribute& attribute,
                                    const PointDataTreeType& tree,
                                    const size_t index,
                                    const LeafOffsetArray& leafOffsets)
        : mAttribute(attribute)
        , mTree(tree)
        , mIndex(index)
        , mLeafOffsets(leafOffsets) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        GAHandleType attributeHandle(&mAttribute);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            assert(n < mLeafOffsets.size());

            // extract leaf and offset

            const LeafOffsetPair& leafOffset = mLeafOffsets[n];
            const PointDataLeaf& leaf = leafOffset.first;
            GA_Offset offset = leafOffset.second;

            typename AttributeHandle<AttributeType>::Ptr handle =
                AttributeHandle<AttributeType>::create(leaf.template attributeArray(mIndex));

            AttributeType value;

            const bool uniform = handle->isUniform();

            if (uniform)    value = handle->get(Index64(0));

            for (PointDataTree::LeafNodeType::ValueOnCIter iter = leaf.cbeginValueOn(); iter; ++iter) {

                Coord ijk = iter.getCoord();

                IndexIter indexIter = leaf.beginIndex(ijk);
                for (; indexIter; ++indexIter) {
                    setAttributeValue(uniform ? value : handle->get(*indexIter), attributeHandle, offset++);
                }
            }
        }
    }

    //////////

    GA_Attribute&                           mAttribute;
    const PointDataTreeType&                mTree;
    const size_t                            mIndex;
    const LeafOffsetArray&                  mLeafOffsets;
}; // ConvertPointDataGridAttributeOp


}


template <typename LeafOffsetArray>
inline void
convertPointDataGridPosition(   GU_Detail& detail,
                                const PointDataGrid& grid,
                                const LeafOffsetArray& leafOffsets)
{
    using sop_openvdb_points_internal::ConvertPointDataGridPositionOp;

    const PointDataTree& tree = grid.tree();
    PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    const size_t positionIndex = iter->attributeSet().find("P");

    // perform threaded conversion

    detail.getP()->hardenAllPages();
    ConvertPointDataGridPositionOp<PointDataTree, LeafOffsetArray> convert(
                    detail, tree, grid.transform(), positionIndex, leafOffsets);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafOffsets.size()), convert);
    detail.getP()->tryCompressAllPages();
}


template <typename LeafOffsetArray>
inline void
convertPointDataGridGroup(  GA_PointGroup* pointGroup,
                            const LeafOffsetArray& leafOffsets,
                            const AttributeSet::Descriptor::GroupIndex index)
{
    using sop_openvdb_points_internal::ConvertPointDataGridGroupOp;

    assert(pointGroup);

    // perform threaded point group assignment

    ConvertPointDataGridGroupOp<PointDataTree, LeafOffsetArray> convert(
                    pointGroup, leafOffsets, index);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafOffsets.size()), convert);

    // must call this after modifying point groups in parallel

    pointGroup->invalidateGroupEntries();
}


template <typename LeafOffsetArray, typename AttributeType>
inline void
convertPointDataGridAttribute(  GA_Attribute& attribute,
                                const PointDataTree& tree,
                                const unsigned arrayIndex,
                                const LeafOffsetArray& leafOffsets)
{
    using sop_openvdb_points_internal::ConvertPointDataGridAttributeOp;

    // perform threaded conversion

    ConvertPointDataGridAttributeOp<PointDataTree, LeafOffsetArray, AttributeType> convert(
                    attribute, tree, arrayIndex, leafOffsets);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafOffsets.size()), convert);
}


inline void
convertPointDataGrid(GU_Detail& detail, openvdb_houdini::VdbPrimCIterator& vdbIt)
{
    typedef PointDataTree::LeafNodeType             PointDataLeaf;

    GU_Detail geo;

    // Mesh each VDB primitive independently
    for (; vdbIt; ++vdbIt) {

        const GridBase& baseGrid = vdbIt->getGrid();
        if (!baseGrid.isType<PointDataGrid>()) continue;

        const PointDataGrid& grid = static_cast<const PointDataGrid&>(baseGrid);
        const PointDataTree& tree = grid.tree();

        PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
        if (!leafIter) continue;

        // position attribute is mandatory

        const AttributeSet& attributeSet = leafIter->attributeSet();
        const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
        const bool hasPosition = descriptor.find("P") != AttributeSet::INVALID_POS;
        if (!hasPosition)   continue;

        // allocate Houdini point array

#if (UT_VERSION_INT < 0x0c0500F5) // earlier than 12.5.245
        for (size_t n = 0, N = pointCount(tree); n < N; ++n) geo.appendPointOffset();
#else
        geo.appendPointBlock(pointCount(tree));
#endif

        // compute global point offsets for each leaf

        typedef std::pair<const PointDataLeaf&, Index64> LeafOffsetPair;
        typedef boost::ptr_vector<LeafOffsetPair> LeafOffsetArray;

        const Index64 leafCount = tree.leafCount();

        LeafOffsetArray leafOffsets;
        leafOffsets.reserve(leafCount);

        Index64 count = 0;
        for ( ; leafIter; ++leafIter) {
            leafOffsets.push_back(new LeafOffsetPair(*leafIter, count));
            count += leafIter->pointCount();
        }

        convertPointDataGridPosition<LeafOffsetArray>(geo, grid, leafOffsets);

        // add other point attributes to the hdk detail
        const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

        for (AttributeSet::Descriptor::ConstIterator it = nameToPosMap.begin(), it_end = nameToPosMap.end();
            it != it_end; ++it) {

            const Name& name = it->first;

            const Name& type = descriptor.type(it->second).first;

            // position handled explicitly
            if (name == "P")    continue;

            const unsigned index = it->second;

            // don't convert group attributes
            if (descriptor.hasGroup(name))  continue;

            const GA_Storage storage = gaStorageFromAttrString(type);
            const unsigned width = widthFromAttrString(type);

            GA_RWAttributeRef attributeRef = geo.addTuple(storage, GA_ATTRIB_POINT, UT_String(name).buffer(), width);
            if (attributeRef.isInvalid()) continue;

            GA_Attribute& attribute = *attributeRef.getAttribute();
            attribute.hardenAllPages();

            if (type == "bool") {
                convertPointDataGridAttribute<LeafOffsetArray, bool>(attribute, tree, index, leafOffsets);
            }
            else if (type == "int16") {
                convertPointDataGridAttribute<LeafOffsetArray, int16_t>(attribute, tree, index, leafOffsets);
            }
            else if (type == "int32") {
                convertPointDataGridAttribute<LeafOffsetArray, int32_t>(attribute, tree, index, leafOffsets);
            }
            else if (type == "int64") {
                convertPointDataGridAttribute<LeafOffsetArray, int64_t>(attribute, tree, index, leafOffsets);
            }
            else if (type == "half") {
                convertPointDataGridAttribute<LeafOffsetArray, half>(attribute, tree, index, leafOffsets);
            }
            else if (type == "float") {
                convertPointDataGridAttribute<LeafOffsetArray, float>(attribute, tree, index, leafOffsets);
            }
            else if (type == "double") {
                convertPointDataGridAttribute<LeafOffsetArray, double>(attribute, tree, index, leafOffsets);
            }
            else if (type == "vec3h") {
                convertPointDataGridAttribute<LeafOffsetArray, Vec3<half> >(attribute, tree, index, leafOffsets);
            }
            else if (type == "vec3s") {
                convertPointDataGridAttribute<LeafOffsetArray, Vec3<float> >(attribute, tree, index, leafOffsets);
            }
            else if (type == "vec3d") {
                convertPointDataGridAttribute<LeafOffsetArray, Vec3<double> >(attribute, tree, index, leafOffsets);
            }
            else {
                throw std::runtime_error("Unknown Attribute Type for Conversion: " + type);
            }

            attribute.tryCompressAllPages();
        }

        // add point groups to the hdk detail
        const AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor.groupMap();

        for (AttributeSet::Descriptor::ConstIterator    it = groupMap.begin(),
                                                        itEnd = groupMap.end(); it != itEnd; ++it) {
            const Name& name = it->first;

            assert(!name.empty());

            GA_PointGroup* pointGroup = geo.findPointGroup(name.c_str());
            if (!pointGroup) pointGroup = geo.newPointGroup(name.c_str());

            const AttributeSet::Descriptor::GroupIndex index = attributeSet.groupIndex(name);

            convertPointDataGridGroup(pointGroup, leafOffsets, index);
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

namespace {

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

} // namespace


OP_ERROR
SOP_OpenVDB_Points::cookMySop(OP_Context& context)
{
    typedef std::vector<AttributeInfo> AttributeInfoVec;

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

        // store point group information

        const GA_ElementGroupTable& elementGroups = ptGeo->getElementGroupTable(GA_ATTRIB_POINT);

        // Create PointPartitioner compatible P attribute wrapper (for now no offset filtering)

        PointAttribute<openvdb::Vec3f> points(ptGeo->getP(), OffsetListPtr());

        // Create PointIndexGrid used for consistent index ordering in all attribute conversion

        PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(points, *transform);

        // Create PointDataGrid using position attribute

        PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(
                                *pointIndexGrid, points, positionAttributeType, *transform);

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

        std::vector<bool> inGroup(ptGeo->getNumPoints(), false);

        for (GA_ElementGroupTable::iterator it = elementGroups.beginTraverse(),
                                            itEnd = elementGroups.endTraverse(); it != itEnd; ++it)
        {
            // insert group offsets

            GA_Offset start, end;
            GA_Range range(**it);
            for (GA_Iterator rangeIt = range.begin(); rangeIt.blockAdvance(start, end); ) {
                for (GA_Offset off = start; off < end; ++off) {
                    assert(off < inGroup.size());
                    inGroup[off] = true;
                }
            }

            const Name groupName = (*it)->getName().toStdString();
            setGroup(tree, indexTree, inGroup, groupName);

            std::fill(inGroup.begin(), inGroup.end(), false);
        }

        // Add other attributes to PointDataGrid

        for (AttributeInfoVec::const_iterator it = attributes.begin(),
                                              it_end = attributes.end(); it != it_end; ++it)
        {
            const openvdb::Name name = it->name;
            const int compression = it->valueCompression;

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
            populateAttributeFromHoudini(tree, indexTree, nameAndType.name, nameAndType.type, attribute, OffsetListPtr());
        }

        // Apply blosc compression to attributes

        for (AttributeInfoVec::const_iterator   it = attributes.begin(),
                                                it_end = attributes.end(); it != it_end; ++it)
        {
            if (!it->bloscCompression)  continue;

            bloscCompressAttribute(tree, it->name);
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
