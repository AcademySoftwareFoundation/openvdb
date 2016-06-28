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
/// @file Utils.h
///
/// @author Dan Bailey
///
/// @brief Utility classes and functions for OpenVDB Points Houdini plugins

#ifndef OPENVDB_POINTS_HOUDINI_UTILS_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_HOUDINI_UTILS_HAS_BEEN_INCLUDED


#include <openvdb/math/Vec3.h>
#include <openvdb/Types.h>
#include <openvdb_points/tools/PointCount.h>
#include <openvdb_points/tools/PointConversion.h>

#include <GA/GA_Attribute.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <GA/GA_AIFTuple.h>
#include <GA/GA_ElementGroup.h>
#include <GA/GA_Iterator.h>


namespace openvdb_points_houdini {


typedef std::vector<GA_Offset> OffsetList;
typedef boost::shared_ptr<OffsetList> OffsetListPtr;

/// @brief  Converts a VDB Points grid into Houdini points and appends to a Houdini Detail
///
/// @param  detail              GU_Detail to append the converted points and attributes to
/// @param  grid                grid containing the points that will be converted
/// @param  attributes          a vector of VDB Points attributes to be included (empty vector defaults to all)
/// @param  includeGroups       a vector of VDB Points groups to be included (empty vector defaults to all)
/// @param  excludeGroups       a vector of VDB Points groups to be excluded (empty vector defaults to none)

void
convertPointDataGridToHoudini(GU_Detail& detail,
                              const openvdb::tools::PointDataGrid& grid,
                              const std::vector<std::string>& attributes = std::vector<std::string>(),
                              const std::vector<std::string>& includeGroups = std::vector<std::string>(),
                              const std::vector<std::string>& excludeGroups = std::vector<std::string>());

namespace {

/// @brief Houdini GA Handle Traits
///
template <typename T> struct GAHandleTraits { typedef GA_RWHandleF RW; };
template <typename T> struct GAHandleTraits<openvdb::math::Vec3<T> > { typedef GA_RWHandleV3 RW; };
template <> struct GAHandleTraits<bool> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int16_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int32_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<int64_t> { typedef GA_RWHandleI RW; };
template <> struct GAHandleTraits<half> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<float> { typedef GA_RWHandleF RW; };
template <> struct GAHandleTraits<double> { typedef GA_RWHandleF RW; };


////////////////////////////////////////


template <typename T, typename T0>
inline T attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i)
{
    T0 tmp;
    attribute.getAIFTuple()->get(&attribute, n, tmp, i);
    return static_cast<T>(tmp);
}

template <typename T>
inline T attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i) {
    return attributeValue<T, T>(attribute, n, i);
}
template <>
inline bool attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i) {
    return attributeValue<bool, int>(attribute, n, i);
}
template <>
inline short attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i) {
    return attributeValue<short, int>(attribute, n, i);
}
template <>
inline long attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i) {
    return attributeValue<long, int>(attribute, n, i);
}
template <>
inline half attributeValue(const GA_Attribute& attribute, GA_Offset n, unsigned i) {
    return attributeValue<half, float>(attribute, n, i);
}

template<typename ValueType, typename HoudiniType>
typename boost::enable_if_c<openvdb::VecTraits<ValueType>::IsVec, void>::type
getValues(HoudiniType* values, const ValueType& value)
{
    for (unsigned i = 0; i < openvdb::VecTraits<ValueType>::Size; ++i) {
        values[i] = value(i);
    }
}

template<typename ValueType, typename HoudiniType>
typename boost::disable_if_c<openvdb::VecTraits<ValueType>::IsVec, void>::type
getValues(HoudiniType* values, const ValueType& value)
{
    values[0] = value;
}

template <typename ValueType, typename HoudiniType>
GA_Defaults
gaDefaultsFromDescriptorTyped(const openvdb::tools::AttributeSet::Descriptor& descriptor, const openvdb::Name& name)
{
    const int size = openvdb::VecTraits<ValueType>::Size;

    boost::scoped_array<HoudiniType> values(new HoudiniType[size]);
    ValueType defaultValue = descriptor.getDefaultValue<ValueType>(name);

    getValues<ValueType, HoudiniType>(values.get(), defaultValue);

    return GA_Defaults(values.get(), size);
}

inline GA_Defaults
gaDefaultsFromDescriptor(const openvdb::tools::AttributeSet::Descriptor& descriptor, const openvdb::Name& name)
{
    const size_t pos = descriptor.find(name);

    if (pos == openvdb::tools::AttributeSet::INVALID_POS)   return GA_Defaults(0);

    const openvdb::Name type = descriptor.type(pos).first;

    if (type == "bool")             return gaDefaultsFromDescriptorTyped<bool, int32>(descriptor, name);
    else if (type == "int16")       return gaDefaultsFromDescriptorTyped<int16_t, int32>(descriptor, name);
    else if (type == "int32")       return gaDefaultsFromDescriptorTyped<int32_t, int32>(descriptor, name);
    else if (type == "int64")       return gaDefaultsFromDescriptorTyped<int64_t, int64>(descriptor, name);
    else if (type == "half")        return gaDefaultsFromDescriptorTyped<half, fpreal32>(descriptor, name);
    else if (type == "float")       return gaDefaultsFromDescriptorTyped<float, fpreal32>(descriptor, name);
    else if (type == "double")      return gaDefaultsFromDescriptorTyped<double, fpreal64>(descriptor, name);
    else if (type == "vec3h")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<half>, fpreal32>(descriptor, name);
    else if (type == "vec3s")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<float>, fpreal32>(descriptor, name);
    else if (type == "vec3d")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<double>, fpreal64>(descriptor, name);

    return GA_Defaults(0);
}

inline GA_Storage
gaStorageFromAttrString(const openvdb::Name& type)
{
    if (type == "bool")             return GA_STORE_BOOL;
    else if (type == "int16")       return GA_STORE_INT16;
    else if (type == "int32")       return GA_STORE_INT32;
    else if (type == "int64")       return GA_STORE_INT64;
    else if (type == "half")        return GA_STORE_REAL16;
    else if (type == "float")       return GA_STORE_REAL32;
    else if (type == "double")      return GA_STORE_REAL64;
    else if (type == "vec3h")       return GA_STORE_REAL16;
    else if (type == "vec3s")       return GA_STORE_REAL32;
    else if (type == "vec3d")       return GA_STORE_REAL64;

    return GA_STORE_INVALID;
}

inline unsigned
widthFromAttrString(const openvdb::Name& type)
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

} // namespace


////////////////////////////////////////


/// @brief Writeable wrapper class around Houdini point attributes which hold
/// a reference to the GA Attribute to write
template <typename T>
struct HoudiniWriteAttribute
{
    typedef T ValueType;

    struct Handle
    {
        Handle(HoudiniWriteAttribute<T>& attribute)
            : mHandle(&attribute.mAttribute) { }

        template <typename ValueType>
        void set(openvdb::Index offset, const ValueType& value) {
            mHandle.set(GA_Offset(offset), value);
        }

        template <typename ValueType>
        void set(openvdb::Index offset, const openvdb::math::Vec3<ValueType>& value) {
            mHandle.set(GA_Offset(offset), UT_Vector3(value.x(), value.y(), value.z()));
        }

    private:
        typename GAHandleTraits<T>::RW mHandle;
    }; // struct Handle

    HoudiniWriteAttribute(GA_Attribute& attribute)
        : mAttribute(attribute) { }

    void expand() {
        mAttribute.hardenAllPages();
    }

    void compact() {
        mAttribute.tryCompressAllPages();
    }

private:
    GA_Attribute& mAttribute;
}; // struct HoudiniWriteAttribute


////////////////////////////////////////


/// @brief Readable wrapper class around Houdini point attributes which hold
/// a reference to the GA Attribute to access and optionally a list of offsets
template <typename T>
struct HoudiniReadAttribute
{
    typedef T value_type;
    typedef T PosType;

    HoudiniReadAttribute(const GA_Attribute& attribute, OffsetListPtr offsets)
        : mAttribute(attribute)
        , mOffsets(offsets) { }

    // Return the value of the nth point in the array (scalar type only)
    template <typename ValueType> typename boost::disable_if_c<openvdb::VecTraits<ValueType>::IsVec, void>::type
    get(size_t n, ValueType& value) const
    {
        value = attributeValue<ValueType>(mAttribute, getOffset(n), 0);
    }

    // Return the value of the nth point in the array (vector type only)
    template <typename ValueType> typename boost::enable_if_c<openvdb::VecTraits<ValueType>::IsVec, void>::type
    get(size_t n, ValueType& value) const
    {
        for (unsigned i = 0; i < openvdb::VecTraits<ValueType>::Size; ++i) {
            value[i] = attributeValue<typename openvdb::VecTraits<ValueType>::ElementType>(mAttribute, getOffset(n), i);
        }
    }

    // Only provided to match the required interface for the PointPartitioner
    void getPos(size_t n, T& xyz) const { return this->get<T>(n, xyz); }

    size_t size() const { return mAttribute.getIndexMap().indexSize(); }

private:
    GA_Offset getOffset(size_t n) const {
        return mOffsets ? (*mOffsets)[n] : mAttribute.getIndexMap().offsetFromIndex(GA_Index(n));
    }

    const GA_Attribute& mAttribute;
    OffsetListPtr mOffsets;
}; // HoudiniReadAttribute


////////////////////////////////////////


struct HoudiniGroup
{
    HoudiniGroup(GA_PointGroup& group)
        : mGroup(group) { }

    void setOffsetOn(openvdb::Index index) {
        mGroup.addOffset(index);
    }

    void finalize() {
        mGroup.invalidateGroupEntries();
    }

private:
    GA_PointGroup& mGroup;
}; // HoudiniGroup


///////////////////////////////////////

void
convertPointDataGridToHoudini(GU_Detail& detail,
                              const openvdb::tools::PointDataGrid& grid,
                              const std::vector<std::string>& attributes,
                              const std::vector<std::string>& includeGroups,
                              const std::vector<std::string>& excludeGroups)
{

    // sort for binary search
    std::vector<std::string> sortedAttributes(attributes);
    std::sort(sortedAttributes.begin(), sortedAttributes.end());

    const openvdb::tools::PointDataTree& tree = grid.tree();

    openvdb::tools::PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    if (!leafIter) return;

    // position attribute is mandatory
    const openvdb::tools::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::tools::AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
    const bool hasPosition = descriptor.find("P") != openvdb::tools::AttributeSet::INVALID_POS;
    if (!hasPosition)   return;

    // obtain cumulative point offsets and total points
    std::vector<openvdb::Index64> pointOffsets;
    openvdb::Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups);

    // a block's global offset is needed to transform its point offsets to global offsets
    openvdb::Index64 startOffset = detail.appendPointBlock(total);

    HoudiniWriteAttribute<openvdb::Vec3f> positionAttribute(*detail.getP());
    convertPointDataGridPosition(positionAttribute, grid, pointOffsets,
                                 startOffset, includeGroups, excludeGroups);

    // add other point attributes to the hdk detail
    const openvdb::tools::AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

    for (openvdb::tools::AttributeSet::Descriptor::ConstIterator    it = nameToPosMap.begin(),
                                                                    itEnd = nameToPosMap.end(); it != itEnd; ++it) {

        const openvdb::Name& name = it->first;
        const openvdb::Name& type = descriptor.type(it->second).first;

        // position handled explicitly
        if (name == "P")    continue;

        // filter attributes
        if (!sortedAttributes.empty() && !std::binary_search(sortedAttributes.begin(), sortedAttributes.end(), name))   continue;

        // don't convert group attributes
        if (descriptor.hasGroup(name))  continue;

        GA_RWAttributeRef attributeRef = detail.findPointAttribute(name.c_str());

        // create the attribute if it doesn't already exist in the detail
        if (attributeRef.isInvalid()) {

            const GA_Storage storage = gaStorageFromAttrString(type);

            if (storage == GA_STORE_INVALID)    continue;

            const unsigned width = widthFromAttrString(type);
            const GA_Defaults defaults = gaDefaultsFromDescriptor(descriptor, name);

            attributeRef = detail.addTuple(storage, GA_ATTRIB_POINT, name.c_str(), width, defaults);

            // '|' and ':' characters are valid in OpenVDB Points names but will make Houdini Attribute names invalid
            if (attributeRef.isInvalid()){
                OPENVDB_THROW(  openvdb::RuntimeError,
                                "Unable to create Houdini Points Attribute with name '" + name +
                                "'. '|' and ':' characters are not supported by Houdini.");
            }
        }

        const unsigned index = it->second;

        if (type == "bool") {
            HoudiniWriteAttribute<bool> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "int16") {
            HoudiniWriteAttribute<int16_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "int32") {
            HoudiniWriteAttribute<int32_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "int64") {
            HoudiniWriteAttribute<int64_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "half") {
            HoudiniWriteAttribute<half> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "float") {
            HoudiniWriteAttribute<float> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "double") {
            HoudiniWriteAttribute<double> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "vec3h") {
            HoudiniWriteAttribute<openvdb::math::Vec3<half> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "vec3s") {
            HoudiniWriteAttribute<openvdb::math::Vec3<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else if (type == "vec3d") {
            HoudiniWriteAttribute<openvdb::math::Vec3<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
        }
        else {
            throw std::runtime_error("Unknown Attribute Type for Conversion: " + type);
        }
    }

    // add point groups to the hdk detail
    const openvdb::tools::AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor.groupMap();

    for (openvdb::tools::AttributeSet::Descriptor::ConstIterator    it = groupMap.begin(),
                                                                    itEnd = groupMap.end(); it != itEnd; ++it) {
        const openvdb::Name& name = it->first;

        assert(!name.empty());

        GA_PointGroup* pointGroup = detail.findPointGroup(name.c_str());
        if (!pointGroup) pointGroup = detail.newPointGroup(name.c_str());

        const openvdb::tools::AttributeSet::Descriptor::GroupIndex index = attributeSet.groupIndex(name);

        HoudiniGroup group(*pointGroup);
        convertPointDataGridGroup(group, tree, pointOffsets, startOffset, index, includeGroups, excludeGroups);
    }
}

} // namespace openvdb_points_houdini

#endif // OPENVDB_POINTS_HOUDINI_UTILS_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
