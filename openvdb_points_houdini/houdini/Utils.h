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


#include <openvdb_points/tools/AttributeArrayString.h>
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


using OffsetList = std::vector<GA_Offset>;
using OffsetListPtr = std::shared_ptr<OffsetList>;

using OffsetPair = std::pair<GA_Offset, GA_Offset>;
using OffsetPairList = std::vector<OffsetPair>;
using OffsetPairListPtr = std::shared_ptr<OffsetPairList>;

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
                              const std::vector<std::string>& attributes = {},
                              const std::vector<std::string>& includeGroups = {},
                              const std::vector<std::string>& excludeGroups = {});

namespace {

/// @brief Houdini GA Handle Traits
///
template <typename T> struct GAHandleTraits                     { typedef GA_RWHandleF RW;   typedef GA_ROHandleF RO;  };
template <> struct GAHandleTraits<openvdb::math::Vec3<int> >    { typedef GA_RWHandleV3 RW;  typedef GA_ROHandleV3 RO; };
template <> struct GAHandleTraits<openvdb::math::Vec3<float> >  { typedef GA_RWHandleV3 RW;  typedef GA_ROHandleV3 RO; };
template <> struct GAHandleTraits<openvdb::math::Vec3<double> > { typedef GA_RWHandleV3D RW; typedef GA_ROHandleV3D RO; };
template <> struct GAHandleTraits<openvdb::math::Quat<float> >  { typedef GA_RWHandleQ RW;  typedef GA_ROHandleQ RO; };
template <> struct GAHandleTraits<openvdb::math::Quat<double> > { typedef GA_RWHandleQD RW; typedef GA_ROHandleQD RO; };
template <> struct GAHandleTraits<openvdb::math::Mat4<float> >  { typedef GA_RWHandleM4 RW;  typedef GA_ROHandleM4 RO; };
template <> struct GAHandleTraits<openvdb::math::Mat4<double> > { typedef GA_RWHandleM4D RW; typedef GA_ROHandleM4D RO; };
template <> struct GAHandleTraits<bool>          { typedef GA_RWHandleI RW; typedef GA_ROHandleI RO; };
template <> struct GAHandleTraits<int16_t>       { typedef GA_RWHandleI RW; typedef GA_ROHandleI RO; };
template <> struct GAHandleTraits<int32_t>       { typedef GA_RWHandleI RW; typedef GA_ROHandleI RO; };
template <> struct GAHandleTraits<int64_t>       { typedef GA_RWHandleI RW; typedef GA_ROHandleI RO; };
template <> struct GAHandleTraits<half>          { typedef GA_RWHandleF RW; typedef GA_ROHandleF RO; };
template <> struct GAHandleTraits<float>         { typedef GA_RWHandleF RW; typedef GA_ROHandleF RO; };
template <> struct GAHandleTraits<double>        { typedef GA_RWHandleF RW; typedef GA_ROHandleF RO; };
template <> struct GAHandleTraits<openvdb::Name> { typedef GA_RWHandleS RW; typedef GA_ROHandleS RO; };


////////////////////////////////////////

template<typename T> struct SizeTraits                          {
    static const int Size = openvdb::VecTraits<T>::Size;
};
template<typename T> struct SizeTraits<openvdb::math::Quat<T> > {
    static const int Size = 4;
};
template<unsigned SIZE, typename T> struct SizeTraits<openvdb::math::Mat<SIZE, T> > {
    static const int Size = SIZE*SIZE;
};


template <typename HandleType, typename ValueType>
inline ValueType
readAttributeValue(const HandleType& handle, const GA_Offset offset, const openvdb::Index component = 0) {
    return ValueType(handle.get(offset, component));
}
template <>
inline openvdb::math::Vec3<float>
readAttributeValue(const GA_ROHandleV3& handle, const GA_Offset offset, const openvdb::Index component) {
    openvdb::math::Vec3<float> dstValue;
    const UT_Vector3F value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2];
    return dstValue;
}
template <>
inline openvdb::math::Vec3<int>
readAttributeValue(const GA_ROHandleV3& handle, const GA_Offset offset, const openvdb::Index component) {
    openvdb::math::Vec3<int> dstValue;
    const UT_Vector3 value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2];
    return dstValue;
}
template <>
inline openvdb::math::Vec3<double>
readAttributeValue(const GA_ROHandleV3D& handle, const GA_Offset offset, const openvdb::Index component) {
    openvdb::math::Vec3<double> dstValue;
    const UT_Vector3D value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2];
    return dstValue;
}
template <>
inline openvdb::math::Quat<float>
readAttributeValue(const GA_ROHandleQ& handle, const GA_Offset offset, const openvdb::Index component) {
    openvdb::math::Quat<float> dstValue;
    const UT_QuaternionF value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2]; dstValue[3] = value[3];
    return dstValue;
}
template <>
inline openvdb::math::Quat<double>
readAttributeValue(const GA_ROHandleQD& handle, const GA_Offset offset, const openvdb::Index component) {
    openvdb::math::Quat<double> dstValue;
    const UT_QuaternionD value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2]; dstValue[3] = value[3];
    return dstValue;
}
template <>
inline openvdb::math::Mat4<float>
readAttributeValue(const GA_ROHandleM4& handle, const GA_Offset offset, const openvdb::Index component) {
    const UT_Matrix4F value(handle.get(offset, component));
    openvdb::math::Mat4<float> dstValue(value.data());
    return dstValue;
}
template <>
inline openvdb::math::Mat4<double>
readAttributeValue(const GA_ROHandleM4D& handle, const GA_Offset offset, const openvdb::Index component) {
    const UT_Matrix4D value(handle.get(offset, component));
    openvdb::math::Mat4<double> dstValue(value.data());
    return dstValue;
}
template <>
inline openvdb::Name
readAttributeValue(const GA_ROHandleS& handle, const GA_Offset offset, const openvdb::Index component) {
    return openvdb::Name(UT_String(handle.get(offset, component)).toStdString());
}


template <typename HandleType, typename ValueType>
inline void writeAttributeValue(const HandleType& handle, const GA_Offset offset, const openvdb::Index component, const ValueType& value) {
    handle.set(offset, component, value);
}
template <>
void writeAttributeValue(const GA_RWHandleV3& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Vec3<int>& value) {
    handle.set(offset, component, UT_Vector3F(value.x(), value.y(), value.z()));
}
template <>
void writeAttributeValue(const GA_RWHandleV3& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Vec3<float>& value) {
    handle.set(offset, component, UT_Vector3(value.x(), value.y(), value.z()));
}
template <>
void writeAttributeValue(const GA_RWHandleV3D& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Vec3<double>& value) {
    handle.set(offset, component, UT_Vector3D(value.x(), value.y(), value.z()));
}
template <>
void writeAttributeValue(const GA_RWHandleQ& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Quat<float>& value) {
    handle.set(offset, component, UT_QuaternionF(value.x(), value.y(), value.z(), value.w()));
}
template <>
void writeAttributeValue(const GA_RWHandleQD& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Quat<double>& value) {
    handle.set(offset, component, UT_QuaternionD(value.x(), value.y(), value.z(), value.w()));
}
template <>
void writeAttributeValue(const GA_RWHandleM4& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Mat4<float>& value) {
    const float* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4F(data[0], data[1], data[2], data[3],
                                              data[4], data[5], data[6], data[7],
                                              data[8], data[9], data[10], data[11],
                                              data[12], data[13], data[14], data[15]));
}
template <>
void writeAttributeValue(const GA_RWHandleM4D& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::math::Mat4<double>& value) {
    const double* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4D(data[0], data[1], data[2], data[3],
                                               data[4], data[5], data[6], data[7],
                                               data[8], data[9], data[10], data[11],
                                               data[12], data[13], data[14], data[15]));
}

template <>
inline void writeAttributeValue(const GA_RWHandleS& handle, const GA_Offset offset, const openvdb::Index component, const openvdb::Name& value) {
    handle.set(offset, component, value.c_str());
}


template<typename ValueType, typename HoudiniType>
void getValues(HoudiniType* values, const ValueType& value)
{
    values[0] = value;
}

template<>
void getValues(int32* values, const openvdb::math::Vec3<int>& value)
{
    for (unsigned i = 0; i < 3; ++i) {
        values[i] = value(i);
    }
}

template<>
void getValues(fpreal32* values, const openvdb::math::Vec3<float>& value)
{
    for (unsigned i = 0; i < 3; ++i) {
        values[i] = value(i);
    }
}

template<>
void getValues(fpreal64* values, const openvdb::math::Vec3<double>& value)
{
    for (unsigned i = 0; i < 3; ++i) {
        values[i] = value(i);
    }
}

template<>
void getValues(fpreal32* values, const openvdb::math::Quat<float>& value)
{
    for (unsigned i = 0; i < 4; ++i) {
        values[i] = value(i);
    }
}

template<>
void getValues(fpreal64* values, const openvdb::math::Quat<double>& value)
{
    for (unsigned i = 0; i < 4; ++i) {
        values[i] = value(i);
    }
}

template<>
void getValues(fpreal32* values, const openvdb::math::Mat4<float>& value)
{
    const float* data = value.asPointer();
    for (unsigned i = 0; i < 16; ++i) {
        values[i] = data[i];
    }
}

template<>
void getValues(fpreal64* values, const openvdb::math::Mat4<double>& value)
{
    const double* data = value.asPointer();
    for (unsigned i = 0; i < 16; ++i) {
        values[i] = data[i];
    }
}

template <typename ValueType, typename HoudiniType>
GA_Defaults
gaDefaultsFromDescriptorTyped(const openvdb::tools::AttributeSet::Descriptor& descriptor, const openvdb::Name& name)
{
    const int size = SizeTraits<ValueType>::Size;

    std::unique_ptr<HoudiniType[]> values(new HoudiniType[size]);
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
    else if (type == "float")       return gaDefaultsFromDescriptorTyped<float, fpreal32>(descriptor, name);
    else if (type == "double")      return gaDefaultsFromDescriptorTyped<double, fpreal64>(descriptor, name);
    else if (type == "vec3i")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<int>, int32>(descriptor, name);
    else if (type == "vec3s")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<float>, fpreal32>(descriptor, name);
    else if (type == "vec3d")       return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<double>, fpreal64>(descriptor, name);
    else if (type == "quats")       return gaDefaultsFromDescriptorTyped<openvdb::math::Quat<float>, fpreal32>(descriptor, name);
    else if (type == "quatd")       return gaDefaultsFromDescriptorTyped<openvdb::math::Quat<double>, fpreal64>(descriptor, name);
    else if (type == "mat4s")       return gaDefaultsFromDescriptorTyped<openvdb::math::Mat4<float>, fpreal32>(descriptor, name);
    else if (type == "mat4d")       return gaDefaultsFromDescriptorTyped<openvdb::math::Mat4<double>, fpreal64>(descriptor, name);

    return GA_Defaults(0);
}

inline GA_Storage
gaStorageFromAttrString(const openvdb::Name& type)
{
    if (type == "string")           return GA_STORE_STRING;
    else if (type == "bool")        return GA_STORE_BOOL;
    else if (type == "int16")       return GA_STORE_INT16;
    else if (type == "int32")       return GA_STORE_INT32;
    else if (type == "int64")       return GA_STORE_INT64;
    else if (type == "float")       return GA_STORE_REAL32;
    else if (type == "double")      return GA_STORE_REAL64;
    else if (type == "vec3i")       return GA_STORE_INT32;
    else if (type == "vec3s")       return GA_STORE_REAL32;
    else if (type == "vec3d")       return GA_STORE_REAL64;
    else if (type == "quats")       return GA_STORE_REAL32;
    else if (type == "quatd")       return GA_STORE_REAL64;
    else if (type == "mat4s")       return GA_STORE_REAL32;
    else if (type == "mat4d")       return GA_STORE_REAL64;

    return GA_STORE_INVALID;
}

} // namespace


////////////////////////////////////////


/// @brief Writeable wrapper class around Houdini point attributes which hold
/// a reference to the GA Attribute to write
template <typename T>
struct HoudiniWriteAttribute
{
    using ValueType = T;

    struct Handle
    {
        explicit Handle(HoudiniWriteAttribute<T>& attribute)
            : mHandle(&attribute.mAttribute) { }

        template <typename ValueType>
        void set(openvdb::Index offset, openvdb::Index stride, const ValueType& value) {
            writeAttributeValue(mHandle, GA_Offset(offset), stride, T(value));
        }

    private:
        typename GAHandleTraits<T>::RW mHandle;
    }; // struct Handle

    explicit HoudiniWriteAttribute(GA_Attribute& attribute)
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
    typedef typename GAHandleTraits<T>::RO ReadHandleType;

    explicit HoudiniReadAttribute(const GA_Attribute& attribute, OffsetListPtr offsets = OffsetListPtr())
        : mHandle(&attribute)
        , mAttribute(attribute)
        , mOffsets(offsets) { }

    static void get(const GA_Attribute& attribute, T& value, const size_t offset, const openvdb::Index component)
    {
        const ReadHandleType handle(&attribute);
        value = readAttributeValue<ReadHandleType, T>(handle, offset, component);
    }

    // Return the value of the nth point in the array (scalar type only)
    void get(T& value, const size_t n, const openvdb::Index component = 0) const
    {
        value = readAttributeValue<ReadHandleType, T>(mHandle, getOffset(n), component);
    }

    // Only provided to match the required interface for the PointPartitioner
    void getPos(size_t n, T& xyz) const { return this->get(xyz, n); }

    size_t size() const { return mOffsets ? mOffsets->size() : mAttribute.getIndexMap().indexSize(); }

private:
    GA_Offset getOffset(size_t n) const {
        return mOffsets ? (*mOffsets)[n] : mAttribute.getIndexMap().offsetFromIndex(GA_Index(n));
    }

    const ReadHandleType   mHandle;
    const GA_Attribute&    mAttribute;
    OffsetListPtr          mOffsets;
}; // HoudiniReadAttribute


////////////////////////////////////////


struct HoudiniGroup
{
    explicit HoudiniGroup(GA_PointGroup& group)
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
    using openvdb_points_houdini::HoudiniWriteAttribute;

    const openvdb::tools::PointDataTree& tree = grid.tree();

    auto leafIter = tree.cbeginLeaf();
    if (!leafIter) return;

    // position attribute is mandatory
    const openvdb::tools::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::tools::AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
    const bool hasPosition = descriptor.find("P") != openvdb::tools::AttributeSet::INVALID_POS;
    if (!hasPosition)   return;

    // sort for binary search
    std::vector<std::string> sortedAttributes(attributes);
    std::sort(sortedAttributes.begin(), sortedAttributes.end());

    // obtain cumulative point offsets and total points
    std::vector<openvdb::Index64> pointOffsets;
    const openvdb::Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups);

    // a block's global offset is needed to transform its point offsets to global offsets
    const openvdb::Index64 startOffset = detail.appendPointBlock(total);

    HoudiniWriteAttribute<openvdb::Vec3f> positionAttribute(*detail.getP());
    convertPointDataGridPosition(positionAttribute, grid, pointOffsets, startOffset, includeGroups, excludeGroups);

    // add other point attributes to the hdk detail
    const openvdb::tools::AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

    for (const auto& namePos : nameToPosMap) {

        const openvdb::Name& name = namePos.first;
        // position handled explicitly
        if (name == "P")    continue;

        // filter attributes
        if (!sortedAttributes.empty() && !std::binary_search(sortedAttributes.begin(), sortedAttributes.end(), name))   continue;

        // don't convert group attributes
        if (descriptor.hasGroup(name))  continue;

        GA_RWAttributeRef attributeRef = detail.findPointAttribute(name.c_str());

        const unsigned index = namePos.second;

        const openvdb::tools::AttributeArray& array = leafIter->constAttributeArray(index);
        const unsigned stride = array.stride();

        const openvdb::NamePair& type = descriptor.type(index);
        const openvdb::Name valueType(openvdb::tools::isString(array) ? "string" : type.first);

        // create the attribute if it doesn't already exist in the detail
        if (attributeRef.isInvalid()) {

            const bool truncate(type.second == openvdb::tools::TruncateCodec::name());

            GA_Storage storage(gaStorageFromAttrString(valueType));
            if (storage == GA_STORE_INVALID) continue;
            if (storage == GA_STORE_REAL32 && truncate) {
                storage = GA_STORE_REAL16;
            }

            unsigned width = stride;
            const bool isVector = valueType.compare(0, 4, "vec3") == 0;
            const bool isQuaternion = valueType.compare(0, 4, "quat") == 0;
            const bool isMatrix4 = valueType.compare(0, 4, "mat4") == 0;

            if (isVector)               width = 3;
            else if (isQuaternion)      width = 4;
            else if (isMatrix4)         width = 16;

            const GA_Defaults defaults = gaDefaultsFromDescriptor(descriptor, name);

            attributeRef = detail.addTuple(storage, GA_ATTRIB_POINT, name.c_str(), width, defaults);

            // apply type info to some recognised types
            if (isVector) {
                if (name == "Cd")       attributeRef->getOptions().setTypeInfo(GA_TYPE_COLOR);
                else if (name == "N")   attributeRef->getOptions().setTypeInfo(GA_TYPE_NORMAL);
                else                    attributeRef->getOptions().setTypeInfo(GA_TYPE_VECTOR);
            }

            if (isQuaternion) {
                attributeRef->getOptions().setTypeInfo(GA_TYPE_QUATERNION);
            }

            if (isMatrix4) {
                attributeRef->getOptions().setTypeInfo(GA_TYPE_TRANSFORM);
            }

            // '|' and ':' characters are valid in OpenVDB Points names but will make Houdini Attribute names invalid
            if (attributeRef.isInvalid()) {
                OPENVDB_THROW(  openvdb::RuntimeError,
                                "Unable to create Houdini Points Attribute with name '" + name +
                                "'. '|' and ':' characters are not supported by Houdini.");
            }
        }

        if (valueType == "string") {
            HoudiniWriteAttribute<openvdb::Name> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "bool") {
            HoudiniWriteAttribute<bool> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "int16") {
            HoudiniWriteAttribute<int16_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "int32") {
            HoudiniWriteAttribute<int32_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "int64") {
            HoudiniWriteAttribute<int64_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "float") {
            HoudiniWriteAttribute<float> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "double") {
            HoudiniWriteAttribute<double> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "vec3i") {
            HoudiniWriteAttribute<openvdb::math::Vec3<int> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "vec3s") {
            HoudiniWriteAttribute<openvdb::math::Vec3<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "vec3d") {
            HoudiniWriteAttribute<openvdb::math::Vec3<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "quats") {
            HoudiniWriteAttribute<openvdb::math::Quat<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "quatd") {
            HoudiniWriteAttribute<openvdb::math::Quat<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "mat4s") {
            HoudiniWriteAttribute<openvdb::math::Mat4<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else if (valueType == "mat4d") {
            HoudiniWriteAttribute<openvdb::math::Mat4<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups);
        }
        else {
            throw std::runtime_error("Unknown Attribute Type for Conversion: " + valueType);
        }
    }

    // add point groups to the hdk detail
    const openvdb::tools::AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor.groupMap();

    for (const auto& namePos : groupMap) {
        const openvdb::Name& name = namePos.first;

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
