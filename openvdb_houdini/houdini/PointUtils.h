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

/// @file PointUtils.h
///
/// @authors Dan Bailey, Nick Avramoussis, Richard Kwok
///
/// @brief Utility classes and functions for OpenVDB Points Houdini plugins

#ifndef OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED


#include <openvdb/math/Vec3.h>
#include <openvdb/Types.h>
#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <GA/GA_Attribute.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <GA/GA_AIFTuple.h>
#include <GA/GA_ElementGroup.h>
#include <GA/GA_Iterator.h>


namespace openvdb_houdini {

using OffsetList = std::vector<GA_Offset>;
using OffsetListPtr = std::shared_ptr<OffsetList>;

using OffsetPair = std::pair<GA_Offset, GA_Offset>;
using OffsetPairList = std::vector<OffsetPair>;
using OffsetPairListPtr = std::shared_ptr<OffsetPairList>;


/// Metadata name for viewport groups
const std::string META_GROUP_VIEWPORT = "group_viewport";


/// @brief Convert a VDB Points grid into Houdini points and append them to a Houdini Detail
///
/// @param  detail         GU_Detail to append the converted points and attributes to
/// @param  grid           grid containing the points that will be converted
/// @param  attributes     a vector of VDB Points attributes to be included
///                        (empty vector defaults to all)
/// @param  includeGroups  a vector of VDB Points groups to be included
///                        (empty vector defaults to all)
/// @param  excludeGroups  a vector of VDB Points groups to be excluded
///                        (empty vector defaults to none)
/// @param inCoreOnly      true if out-of-core leaf nodes are to be ignored
void
convertPointDataGridToHoudini(
    GU_Detail& detail,
    const openvdb::points::PointDataGrid& grid,
    const std::vector<std::string>& attributes = {},
    const std::vector<std::string>& includeGroups = {},
    const std::vector<std::string>& excludeGroups = {},
    const bool inCoreOnly = false);

namespace {

// @{
// Houdini GA Handle Traits

template<typename T> struct GAHandleTraits    { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<bool>        { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int16_t>     { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int32_t>     { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int64_t>     { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<half>        { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<float>       { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<double>      { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<std::string> { using RW = GA_RWHandleS; using RO = GA_ROHandleS; };
template<>
struct GAHandleTraits<openvdb::math::Vec3<int>> { using RW=GA_RWHandleV3; using RO=GA_ROHandleV3; };
template<>
struct GAHandleTraits<openvdb::Vec3s> { using RW = GA_RWHandleV3; using RO = GA_ROHandleV3; };
template<>
struct GAHandleTraits<openvdb::Vec3d> { using RW = GA_RWHandleV3D; using RO = GA_ROHandleV3D; };
template<>
struct GAHandleTraits<openvdb::Mat4s> { using RW = GA_RWHandleM4; using RO = GA_ROHandleM4; };
template<>
struct GAHandleTraits<openvdb::Mat4d> { using RW = GA_RWHandleM4D; using RO = GA_ROHandleM4D; };
template<>
struct GAHandleTraits<openvdb::math::Quats> { using RW = GA_RWHandleQ; using RO = GA_ROHandleQ; };
template<>
struct GAHandleTraits<openvdb::math::Quatd> { using RW = GA_RWHandleQD; using RO = GA_ROHandleQD; };

// @}


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


template<typename HandleType, typename ValueType>
inline ValueType
readAttributeValue(const HandleType& handle, const GA_Offset offset,
    const openvdb::Index component = 0)
{
    return ValueType(handle.get(offset, component));
}

template<>
inline openvdb::math::Vec3<float>
readAttributeValue(const GA_ROHandleV3& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    openvdb::math::Vec3<float> dstValue;
    const UT_Vector3F value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2];
    return dstValue;
}

template<>
inline openvdb::math::Vec3<int>
readAttributeValue(const GA_ROHandleV3& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    openvdb::math::Vec3<int> dstValue;
    const UT_Vector3 value(handle.get(offset, component));
    dstValue[0] = static_cast<int>(value[0]);
    dstValue[1] = static_cast<int>(value[1]);
    dstValue[2] = static_cast<int>(value[2]);
    return dstValue;
}

template<>
inline openvdb::math::Vec3<double>
readAttributeValue(const GA_ROHandleV3D& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    openvdb::math::Vec3<double> dstValue;
    const UT_Vector3D value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2];
    return dstValue;
}

template<>
inline openvdb::math::Quat<float>
readAttributeValue(const GA_ROHandleQ& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    openvdb::math::Quat<float> dstValue;
    const UT_QuaternionF value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2]; dstValue[3] = value[3];
    return dstValue;
}

template<>
inline openvdb::math::Quat<double>
readAttributeValue(const GA_ROHandleQD& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    openvdb::math::Quat<double> dstValue;
    const UT_QuaternionD value(handle.get(offset, component));
    dstValue[0] = value[0]; dstValue[1] = value[1]; dstValue[2] = value[2]; dstValue[3] = value[3];
    return dstValue;
}

template<>
inline openvdb::math::Mat4<float>
readAttributeValue(const GA_ROHandleM4& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    const UT_Matrix4F value(handle.get(offset, component));
    openvdb::math::Mat4<float> dstValue(value.data());
    return dstValue;
}

template<>
inline openvdb::math::Mat4<double>
readAttributeValue(const GA_ROHandleM4D& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    const UT_Matrix4D value(handle.get(offset, component));
    openvdb::math::Mat4<double> dstValue(value.data());
    return dstValue;
}

template<>
inline openvdb::Name
readAttributeValue(const GA_ROHandleS& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    return openvdb::Name(UT_String(handle.get(offset, component)).toStdString());
}


template<typename HandleType, typename ValueType>
inline void
writeAttributeValue(const HandleType& handle, const GA_Offset offset,
    const openvdb::Index component, const ValueType& value)
{
    handle.set(offset, component, static_cast<typename HandleType::BASETYPE>(value));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleV3& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Vec3<int>& value)
{
    handle.set(offset, component, UT_Vector3F(
        static_cast<float>(value.x()),
        static_cast<float>(value.y()),
        static_cast<float>(value.z())));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleV3& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Vec3<float>& value)
{
    handle.set(offset, component, UT_Vector3(value.x(), value.y(), value.z()));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleV3D& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Vec3<double>& value)
{
    handle.set(offset, component, UT_Vector3D(value.x(), value.y(), value.z()));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleQ& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Quat<float>& value)
{
    handle.set(offset, component, UT_QuaternionF(value.x(), value.y(), value.z(), value.w()));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleQD& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Quat<double>& value)
{
    handle.set(offset, component, UT_QuaternionD(value.x(), value.y(), value.z(), value.w()));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleM4& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Mat4<float>& value)
{
    const float* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4F(data[0], data[1], data[2], data[3],
                                              data[4], data[5], data[6], data[7],
                                              data[8], data[9], data[10], data[11],
                                              data[12], data[13], data[14], data[15]));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleM4D& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Mat4<double>& value)
{
    const double* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4D(data[0], data[1], data[2], data[3],
                                              data[4], data[5], data[6], data[7],
                                              data[8], data[9], data[10], data[11],
                                              data[12], data[13], data[14], data[15]));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleS& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::Name& value)
{
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
gaDefaultsFromDescriptorTyped(const openvdb::points::AttributeSet::Descriptor& descriptor,
    const openvdb::Name& name)
{
    const int size = SizeTraits<ValueType>::Size;

    std::unique_ptr<HoudiniType[]> values(new HoudiniType[size]);
    ValueType defaultValue = descriptor.getDefaultValue<ValueType>(name);

    getValues<ValueType, HoudiniType>(values.get(), defaultValue);

    return GA_Defaults(values.get(), size);
}

inline GA_Defaults
gaDefaultsFromDescriptor(const openvdb::points::AttributeSet::Descriptor& descriptor,
    const openvdb::Name& name)
{
    const size_t pos = descriptor.find(name);

    if (pos == openvdb::points::AttributeSet::INVALID_POS) return GA_Defaults(0);

    const openvdb::Name type = descriptor.type(pos).first;

    if (type == "bool") {
        return gaDefaultsFromDescriptorTyped<bool, int32>(descriptor, name);
    } else if (type == "int16") {
         return gaDefaultsFromDescriptorTyped<int16_t, int32>(descriptor, name);
    } else if (type == "int32") {
         return gaDefaultsFromDescriptorTyped<int32_t, int32>(descriptor, name);
    } else if (type == "int64") {
         return gaDefaultsFromDescriptorTyped<int64_t, int64>(descriptor, name);
    } else if (type == "float") {
         return gaDefaultsFromDescriptorTyped<float, fpreal32>(descriptor, name);
    } else if (type == "double") {
        return gaDefaultsFromDescriptorTyped<double, fpreal64>(descriptor, name);
    } else if (type == "vec3i") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3<int>, int32>(descriptor, name);
    } else if (type == "vec3s") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3s, fpreal32>(descriptor, name);
    } else if (type == "vec3d") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Vec3d, fpreal64>(descriptor, name);
    } else if (type == "quats") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Quats, fpreal32>(descriptor, name);
    } else if (type == "quatd") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Quatd, fpreal64>(descriptor, name);
    } else if (type == "mat4s") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Mat4s, fpreal32>(descriptor, name);
    } else if (type == "mat4d") {
         return gaDefaultsFromDescriptorTyped<openvdb::math::Mat4d, fpreal64>(descriptor, name);
    }
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

    explicit HoudiniReadAttribute(const GA_Attribute& attribute,
        OffsetListPtr offsets = OffsetListPtr())
        : mHandle(&attribute)
        , mAttribute(attribute)
        , mOffsets(offsets) { }

    static void get(const GA_Attribute& attribute, T& value, const size_t offset,
        const openvdb::Index component)
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

    size_t size() const
    {
        return mOffsets ? mOffsets->size() : size_t(mAttribute.getIndexMap().indexSize());
    }

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
    explicit HoudiniGroup(GA_PointGroup& group,
        openvdb::Index64 startOffset, openvdb::Index64 total)
        : mGroup(group)
        , mStartOffset(startOffset)
        , mTotal(total)
    {
        mBackingArray.resize(total, 0);
    }

    HoudiniGroup(const HoudiniGroup &) = delete;
    HoudiniGroup& operator=(const HoudiniGroup &) = delete;

    void setOffsetOn(openvdb::Index index) { mBackingArray[index - mStartOffset] = 1; }

    void finalize() {
        for (openvdb::Index64 i = 0, n = mTotal; i < n; i++) {
            if (mBackingArray[i]) {
                mGroup.addOffset(GA_Offset(i + mStartOffset));
            }
        }
    }

private:
    GA_PointGroup& mGroup;
    openvdb::Index64 mStartOffset;
    openvdb::Index64 mTotal;

    // This is not a bit field as we need to allow threadsafe updates:
    std::vector<unsigned char> mBackingArray;
}; // HoudiniGroup


///////////////////////////////////////


void convertPointDataGridToHoudini(
    GU_Detail&,
    const openvdb::points::PointDataGrid&,
    const std::vector<std::string>& attributes,
    const std::vector<std::string>& includeGroups,
    const std::vector<std::string>& excludeGroups,
    const bool inCoreOnly);


/// @brief If the given grid is a PointDataGrid, add node specific info text to the stream provided
void pointDataGridSpecificInfoText(std::ostream&, const openvdb::GridBase&);

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
