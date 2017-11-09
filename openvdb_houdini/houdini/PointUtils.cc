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

/// @file PointUtils.cc
/// @authors Dan Bailey, Nick Avramoussis, Richard Kwok

#include "PointUtils.h"

#include "AttributeTransferUtil.h"
#include "Utils.h"

#include <openvdb/openvdb.h>
#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>

#include <GA/GA_AIFTuple.h>
#include <GA/GA_ElementGroup.h>
#include <GA/GA_Iterator.h>

#include <CH/CH_Manager.h> // for CHgetEvalTime
#include <PRM/PRM_SpareData.h>
#include <SOP/SOP_Node.h>

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


using namespace openvdb;
using namespace openvdb::points;

namespace hvdb = openvdb_houdini;


namespace {

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

// @{
// Houdini GA Handle Traits

template<typename T> struct GAHandleTraits    { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<bool>        { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int16_t>     { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int32_t>     { using RW = GA_RWHandleI; using RO = GA_ROHandleI; };
template<> struct GAHandleTraits<int64_t>     { using RW = GA_RWHandleID; using RO = GA_ROHandleID; };
template<> struct GAHandleTraits<half>        { using RW = GA_RWHandleH; using RO = GA_ROHandleH; };
template<> struct GAHandleTraits<float>       { using RW = GA_RWHandleF; using RO = GA_ROHandleF; };
template<> struct GAHandleTraits<double>      { using RW = GA_RWHandleD; using RO = GA_ROHandleD; };
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
    // read transposed matrix because Houdini uses column-major order so as
    // to be compatible with OpenGL
    const UT_Matrix4F value(handle.get(offset, component));
    openvdb::math::Mat4<float> dstValue(value.data());
    return dstValue.transpose();
}

template<>
inline openvdb::math::Mat4<double>
readAttributeValue(const GA_ROHandleM4D& handle, const GA_Offset offset,
    const openvdb::Index component)
{
    // read transposed matrix because Houdini uses column-major order so as
    // to be compatible with OpenGL
    const UT_Matrix4D value(handle.get(offset, component));
    openvdb::math::Mat4<double> dstValue(value.data());
    return dstValue.transpose();
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
    // write transposed matrix because Houdini uses column-major order so as
    // to be compatible with OpenGL
    const float* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4F(data[0], data[4], data[8], data[12],
                                              data[1], data[5], data[9], data[13],
                                              data[2], data[6], data[10], data[14],
                                              data[3], data[7], data[11], data[15]));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleM4D& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::math::Mat4<double>& value)
{
    // write transposed matrix because Houdini uses column-major order so as
    // to be compatible with OpenGL
    const double* data(value.asPointer());
    handle.set(offset, component, UT_Matrix4D(data[0], data[4], data[8], data[12],
                                              data[1], data[5], data[9], data[13],
                                              data[2], data[6], data[10], data[14],
                                              data[3], data[7], data[11], data[15]));
}

template<>
inline void
writeAttributeValue(const GA_RWHandleS& handle, const GA_Offset offset,
    const openvdb::Index component, const openvdb::Name& value)
{
    handle.set(offset, component, value.c_str());
}


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


/// @brief Readable wrapper class around Houdini point attributes which hold
/// a reference to the GA Attribute to access and optionally a list of offsets
template <typename T>
struct HoudiniReadAttribute
{
    using value_type = T;
    using PosType = T;
    using ReadHandleType = typename GAHandleTraits<T>::RO;

    explicit HoudiniReadAttribute(const GA_Attribute& attribute,
        hvdb::OffsetListPtr offsets = hvdb::OffsetListPtr())
        : mHandle(&attribute)
        , mAttribute(attribute)
        , mOffsets(offsets) { }

    static void get(const GA_Attribute& attribute, T& value, const GA_Offset offset,
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
    hvdb::OffsetListPtr          mOffsets;
}; // HoudiniReadAttribute


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


template <typename ValueType, typename CodecType = NullCodec>
inline void
convertAttributeFromHoudini(PointDataTree& tree, const tools::PointIndexTree& indexTree,
    const Name& name, const GA_Attribute* const attribute,
    const GA_Defaults& defaults, const Index stride = 1)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");
    static_assert(!std::is_same<ValueType, Name>::value,
        "ValueType must not be Name/std::string");

    using HoudiniAttribute = HoudiniReadAttribute<ValueType>;

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
    const Name& name, const GA_Attribute* const attribute, const int compression = 0)
{
    using namespace openvdb::math;

    using HoudiniStringAttribute = HoudiniReadAttribute<Name>;

    if (!attribute) {
        std::stringstream ss; ss << "Invalid attribute - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const GA_Storage storage(hvdb::attributeStorageType(attribute));

    if (storage == GA_STORE_INVALID) {
        std::stringstream ss; ss << "Invalid attribute type - " << attribute->getName();
        throw std::runtime_error(ss.str());
    }

    const int16_t width(hvdb::attributeTupleSize(attribute));
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
            if (compression == hvdb::COMPRESSION_NONE) {
                convertAttributeFromHoudini<Vec3<float>>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == hvdb::COMPRESSION_TRUNCATE) {
                convertAttributeFromHoudini<Vec3<float>, TruncateCodec>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == hvdb::COMPRESSION_UNIT_VECTOR) {
                convertAttributeFromHoudini<Vec3<float>, UnitVecCodec>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == hvdb::COMPRESSION_UNIT_FIXED_POINT_8) {
                convertAttributeFromHoudini<Vec3<float>, FixedPointCodec<true, UnitRange>>(
                    tree, indexTree, name, attribute, defaults);
            }
            else if (compression == hvdb::COMPRESSION_UNIT_FIXED_POINT_16) {
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
        } else if (storage == GA_STORE_REAL32 && compression == hvdb::COMPRESSION_NONE) {
            convertAttributeFromHoudini<float>(tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == hvdb::COMPRESSION_TRUNCATE) {
            convertAttributeFromHoudini<float, TruncateCodec>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == hvdb::COMPRESSION_UNIT_FIXED_POINT_8) {
            convertAttributeFromHoudini<float, FixedPointCodec<true, UnitRange>>(
                tree, indexTree, name, attribute, defaults, width);
        } else if (storage == GA_STORE_REAL32 && compression == hvdb::COMPRESSION_UNIT_FIXED_POINT_16) {
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


template <typename ValueType>
void
populateHoudiniDetailAttribute(GA_RWAttributeRef& attrib, const openvdb::MetaMap& metaMap,
                               const Name& key, const int index)
{
    using WriteHandleType = typename GAHandleTraits<ValueType>::RW;
    using TypedMetadataT = TypedMetadata<ValueType>;

    typename TypedMetadataT::ConstPtr typedMetadata = metaMap.getMetadata<TypedMetadataT>(key);
    if (!typedMetadata) return;

    const ValueType& value = typedMetadata->value();
    WriteHandleType handle(attrib.getAttribute());
    writeAttributeValue<WriteHandleType, ValueType>(handle, GA_Offset(0), index, value);
}


template<typename ValueType>
Metadata::Ptr
createTypedMetadataFromAttribute(const GA_Attribute* const attribute, const uint32_t component = 0)
{
    using HoudiniAttribute = HoudiniReadAttribute<ValueType>;

    ValueType value;
    HoudiniAttribute::get(*attribute, value, GA_Offset(0), component);
    return openvdb::TypedMetadata<ValueType>(value).copy();
}


template<typename T> struct SizeTraits                          {
    static const int Size = openvdb::VecTraits<T>::Size;
};
template<typename T> struct SizeTraits<openvdb::math::Quat<T> > {
    static const int Size = 4;
};
template<unsigned SIZE, typename T> struct SizeTraits<openvdb::math::Mat<SIZE, T> > {
    static const int Size = SIZE*SIZE;
};


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


} // unnamed namespace


////////////////////////////////////////


namespace openvdb_houdini {


float
computeVoxelSizeFromHoudini(const GU_Detail& detail,
                            const uint32_t pointsPerVoxel,
                            const openvdb::math::Mat4d& matrix,
                            const Index decimalPlaces,
                            hvdb::Interrupter& interrupter)
{
    HoudiniReadAttribute<openvdb::Vec3R> positions(*(detail.getP()));
    return openvdb::points::computeVoxelSize(
            positions, pointsPerVoxel, matrix, decimalPlaces, &interrupter);
}


PointDataGrid::Ptr
convertHoudiniToPointDataGrid(const GU_Detail& ptGeo,
                              const int compression,
                              const AttributeInfoMap& attributes,
                              const math::Transform& transform)
{
    using HoudiniPositionAttribute = HoudiniReadAttribute<Vec3d>;

    // store point group information

    const GA_ElementGroupTable& elementGroups = ptGeo.getElementGroupTable(GA_ATTRIB_POINT);

    // Create PointPartitioner compatible P attribute wrapper (for now no offset filtering)

    const GA_Attribute& positionAttribute = *ptGeo.getP();

    OffsetListPtr offsets;
    OffsetPairListPtr offsetPairs;

    size_t vertexCount = 0;

    for (GA_Iterator primitiveIt(ptGeo.getPrimitiveRange()); !primitiveIt.atEnd(); ++primitiveIt) {
        const GA_Primitive* primitive = ptGeo.getPrimitiveList().get(*primitiveIt);

        if (primitive->getTypeId() != GA_PRIMNURBCURVE) continue;

        vertexCount = primitive->getVertexCount();

        if (vertexCount == 0)  continue;

        if (!offsets)   offsets.reset(new OffsetList);

        GA_Offset firstOffset = primitive->getPointOffset(0);
        offsets->push_back(firstOffset);

        if (vertexCount > 1) {
            if (!offsetPairs)   offsetPairs.reset(new OffsetPairList);

            for (size_t i = 1; i < vertexCount; i++) {
                GA_Offset offset = primitive->getPointOffset(i);
                offsetPairs->push_back(OffsetPair(firstOffset, offset));
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
        pointDataGrid = points::createPointDataGrid<FixedPointCodec<false>, PointDataGrid>(
            *pointIndexGrid, points, transform);
    }
    else if (compression == 2 /*FIXED_POSITION_8*/) {
        pointDataGrid = points::createPointDataGrid<FixedPointCodec<true>, PointDataGrid>(
            *pointIndexGrid, points, transform);
    }
    else /*NONE*/ {
        pointDataGrid = points::createPointDataGrid<NullCodec, PointDataGrid>(
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
            end = std::min(end, GA_Offset(numPoints));
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
        const Name& name = attrInfo.first;

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


void
convertPointDataGridToHoudini(
    GU_Detail& detail,
    const PointDataGrid& grid,
    const std::vector<std::string>& attributes,
    const std::vector<std::string>& includeGroups,
    const std::vector<std::string>& excludeGroups,
    const bool inCoreOnly)
{
    using namespace openvdb::math;

    const PointDataTree& tree = grid.tree();

    auto leafIter = tree.cbeginLeaf();
    if (!leafIter) return;

    // position attribute is mandatory
    const AttributeSet& attributeSet = leafIter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
    const bool hasPosition = descriptor.find("P") != AttributeSet::INVALID_POS;
    if (!hasPosition)   return;

    // sort for binary search
    std::vector<std::string> sortedAttributes(attributes);
    std::sort(sortedAttributes.begin(), sortedAttributes.end());

    // obtain cumulative point offsets and total points
    std::vector<Index64> pointOffsets;
    const Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups,
        inCoreOnly);

    // a block's global offset is needed to transform its point offsets to global offsets
    const Index64 startOffset = detail.appendPointBlock(total);

    HoudiniWriteAttribute<Vec3f> positionAttribute(*detail.getP());
    convertPointDataGridPosition(positionAttribute, grid, pointOffsets, startOffset, includeGroups,
        excludeGroups, inCoreOnly);

    // add other point attributes to the hdk detail
    const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

    for (const auto& namePos : nameToPosMap) {

        const Name& name = namePos.first;
        // position handled explicitly
        if (name == "P")    continue;

        // filter attributes
        if (!sortedAttributes.empty() &&
            !std::binary_search(sortedAttributes.begin(), sortedAttributes.end(), name)) {
            continue;
        }

        // don't convert group attributes
        if (descriptor.hasGroup(name))  continue;

        GA_RWAttributeRef attributeRef = detail.findPointAttribute(name.c_str());

        const auto index = static_cast<unsigned>(namePos.second);

        const AttributeArray& array = leafIter->constAttributeArray(index);
        const unsigned stride = array.stride();

        const NamePair& type = descriptor.type(index);
        const Name valueType(isString(array) ? "string" : type.first);

        // create the attribute if it doesn't already exist in the detail
        if (attributeRef.isInvalid()) {

            const bool truncate(type.second == TruncateCodec::name());

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

            // '|' and ':' characters are valid in OpenVDB Points names but
            // will make Houdini Attribute names invalid
            if (attributeRef.isInvalid()) {
                OPENVDB_THROW(  RuntimeError,
                                "Unable to create Houdini Points Attribute with name '" + name +
                                "'. '|' and ':' characters are not supported by Houdini.");
            }
        }

        if (valueType == "string") {
            HoudiniWriteAttribute<Name> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "bool") {
            HoudiniWriteAttribute<bool> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int16") {
            HoudiniWriteAttribute<int16_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int32") {
            HoudiniWriteAttribute<int32_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int64") {
            HoudiniWriteAttribute<int64_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "float") {
            HoudiniWriteAttribute<float> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "double") {
            HoudiniWriteAttribute<double> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3i") {
            HoudiniWriteAttribute<Vec3<int> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3s") {
            HoudiniWriteAttribute<Vec3<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3d") {
            HoudiniWriteAttribute<Vec3<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "quats") {
            HoudiniWriteAttribute<Quat<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "quatd") {
            HoudiniWriteAttribute<Quat<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "mat4s") {
            HoudiniWriteAttribute<Mat4<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "mat4d") {
            HoudiniWriteAttribute<Mat4<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride,
                includeGroups, excludeGroups, inCoreOnly);
        }
        else {
            throw std::runtime_error("Unknown Attribute Type for Conversion: " + valueType);
        }
    }

    // add point groups to the hdk detail
    const AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor.groupMap();

    for (const auto& namePos : groupMap) {
        const Name& name = namePos.first;

        assert(!name.empty());

        GA_PointGroup* pointGroup = detail.findPointGroup(name.c_str());
        if (!pointGroup) pointGroup = detail.newPointGroup(name.c_str());

        const AttributeSet::Descriptor::GroupIndex index =
            attributeSet.groupIndex(name);

        HoudiniGroup group(*pointGroup, startOffset, total);
        convertPointDataGridGroup(group, tree, pointOffsets, startOffset, index,
            includeGroups, excludeGroups, inCoreOnly);
    }
}


void
populateMetadataFromHoudini(openvdb::points::PointDataGrid& grid,
                            std::vector<std::string>& warnings,
                            const GU_Detail& detail)
{
    using namespace openvdb::math;

    for (GA_AttributeDict::iterator iter = detail.attribs().begin(GA_SCOPE_PUBLIC);
        !iter.atEnd(); ++iter)
    {
        const GA_Attribute* const attribute = *iter;
        if (!attribute) continue;

        const Name name("global:" + Name(attribute->getName()));
        Metadata::Ptr metadata = grid[name];
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
                warnings.push_back(ss.str().c_str());
                continue;
            }
            assert(metadata);
            grid.insertMeta(name, *metadata);
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
                warnings.push_back(ss.str().c_str());
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
                warnings.push_back(ss.str().c_str());
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
                    warnings.push_back(ss.str().c_str());
                    continue;
                }
                assert(metadata);
                if (width > 1) {
                    const Name arrayName(name + Name("[") + std::to_string(i) + Name("]"));
                    grid.insertMeta(arrayName, *metadata);
                }
                else {
                    grid.insertMeta(name, *metadata);
                }
            }
        }
    }
}


void
convertMetadataToHoudini(GU_Detail& detail,
                         const openvdb::MetaMap& metaMap,
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

    using namespace openvdb::math;

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
            const GA_Storage storage = gaStorageFromAttrString(type);

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


int16_t
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


GA_Storage
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


////////////////////////////////////////


void
pointDataGridSpecificInfoText(std::ostream& infoStr, const GridBase& grid)
{
    const PointDataGrid* pointDataGrid = dynamic_cast<const PointDataGrid*>(&grid);

    if (!pointDataGrid) return;

    // match native OpenVDB convention as much as possible

    infoStr << " voxel size: " << pointDataGrid->transform().voxelSize()[0] << ",";
    infoStr << " type: points,";

    if (pointDataGrid->activeVoxelCount() != 0) {
        const Coord dim = grid.evalActiveVoxelDim();
        infoStr << " dim: " << dim[0] << "x" << dim[1] << "x" << dim[2] << ",";
    } else {
        infoStr <<" <empty>,";
    }

    std::string viewportGroupName = "";
    if (StringMetadata::ConstPtr stringMeta =
        grid.getMetadata<StringMetadata>(META_GROUP_VIEWPORT))
    {
        viewportGroupName = stringMeta->value();
    }

    const PointDataTree& pointDataTree = pointDataGrid->tree();

    PointDataTree::LeafCIter iter = pointDataTree.cbeginLeaf();

    // iterate through all leaf nodes to find out if all are out-of-core
    bool allOutOfCore = true;
    for (; iter; ++iter) {
        if (!iter->buffer().isOutOfCore()) {
            allOutOfCore = false;
            break;
        }
    }

    Index64 totalPointCount = 0;

    // it is more technically correct to rely on the voxel count as this may be out of
    // sync with the attribute size, however for faster node preview when the voxel buffers
    // are all out-of-core, count up the sizes of the first attribute array instead
    if (allOutOfCore) {
        iter = pointDataTree.cbeginLeaf();
        for (; iter; ++iter) {
            if (iter->attributeSet().size() > 0) {
                totalPointCount += iter->constAttributeArray(0).size();
            }
        }
    }
    else {
        totalPointCount = pointCount(pointDataTree);
    }

    infoStr << " count: " << util::formattedInt(totalPointCount) << ",";

    iter = pointDataTree.cbeginLeaf();

    if (!iter.getLeaf()) {
        infoStr << " attributes: <none>";
    }
    else {
        const AttributeSet::DescriptorPtr& descriptor = iter->attributeSet().descriptorPtr();

        infoStr << " groups: ";

        const AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor->groupMap();

        bool first = true;
        for (AttributeSet::Descriptor::ConstIterator it = groupMap.begin(), it_end = groupMap.end();
                it != it_end; ++it) {
            if (first) {
                first = false;
            }
            else {
                infoStr << ", ";
            }

            // add an asterisk as a viewport group indicator
            if (it->first == viewportGroupName)     infoStr << "*";

            infoStr << it->first << "(";

            // for faster node preview when all the voxel buffers are out-of-core, don't load the
            // group arrays to display the group sizes, just print "out-of-core" instead
            // @todo - put the group sizes into the grid metadata on write for this use case

            if (allOutOfCore) {
                infoStr << "out-of-core";
            }
            else {
                infoStr << util::formattedInt(groupPointCount(pointDataTree, it->first));
            }

            infoStr << ")";
        }

        if (first)  infoStr << "<none>";

        infoStr << ",";

        infoStr << " attributes: ";

        const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor->map();

        first = true;
        for (auto it = nameToPosMap.begin(), it_end = nameToPosMap.end(); it != it_end; ++it) {
            const AttributeArray& array = iter->constAttributeArray(it->second);
            if (isGroup(array))    continue;

            if (first) {
                first = false;
            }
            else {
                infoStr << ", ";
            }
            const NamePair& type = descriptor->type(it->second);

            const Name valueType = type.first;
            const Name codecType = type.second;

            infoStr << it->first << "[";

            // if no value compression, hide the codec from the middle-click output

            if (codecType == "null") {
                infoStr << valueType;
            }
            else if (isString(array)) {
                infoStr << "str";
            }
            else {
                infoStr << valueType << "_" << codecType;
            }

            if (!array.hasConstantStride()) {
                infoStr << "[dynamic]";
            }
            else if (array.stride() > 1) {
                infoStr << "[" << array.stride() << "]";
            }

            infoStr << "]";
        }

        if (first)  infoStr << "<none>";
    }
}

namespace {

inline int
lookupGroupInput(const PRM_SpareData* spare)
{
    if (!spare) return 0;
    const char* istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}

void
sopBuildVDBPointsGroupMenu(void* data, PRM_Name* menuEntries, int /*themenusize*/,
    const PRM_SpareData* spare, const PRM_Parm* /*parm*/)
{
    SOP_Node* sop = CAST_SOPNODE(static_cast<OP_Node*>(data));
    int inputIndex = lookupGroupInput(spare);

    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    // const cast as iterator requires non-const access, however data is not modified
    VdbPrimIterator vdbIt(const_cast<GU_Detail*>(gdp));

    int n_entries = 0;

    for (; vdbIt; ++vdbIt) {
        GU_PrimVDB* vdbPrim = *vdbIt;

        PointDataGrid::ConstPtr grid =
                gridConstPtrCast<PointDataGrid>(vdbPrim->getConstGridPtr());

        // ignore all but point data grids
        if (!grid)      continue;
        auto leafIter = grid->tree().cbeginLeaf();
        if (!leafIter)  continue;

        const AttributeSet::Descriptor& descriptor =
            leafIter->attributeSet().descriptor();

        for (const auto& it : descriptor.groupMap()) {
            // add each VDB Points group to the menu
            menuEntries[n_entries].setToken(it.first.c_str());
            menuEntries[n_entries].setLabel(it.first.c_str());
            n_entries++;
        }
    }

    // zero value ends the menu

    menuEntries[n_entries].setToken(0);
    menuEntries[n_entries].setLabel(0);
}

} // unnamed namespace


#ifdef _MSC_VER

OPENVDB_HOUDINI_API const PRM_ChoiceList
VDBPointsGroupMenuInput1(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
OPENVDB_HOUDINI_API const PRM_ChoiceList
VDBPointsGroupMenuInput2(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
OPENVDB_HOUDINI_API const PRM_ChoiceList
VDBPointsGroupMenuInput3(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
OPENVDB_HOUDINI_API const PRM_ChoiceList
VDBPointsGroupMenuInput4(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);

OPENVDB_HOUDINI_API const PRM_ChoiceList VDBPointsGroupMenu(PRM_CHOICELIST_TOGGLE,
    sopBuildVDBPointsGroupMenu);

#else

const PRM_ChoiceList
VDBPointsGroupMenuInput1(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
const PRM_ChoiceList
VDBPointsGroupMenuInput2(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
const PRM_ChoiceList
VDBPointsGroupMenuInput3(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);
const PRM_ChoiceList
VDBPointsGroupMenuInput4(PRM_CHOICELIST_TOGGLE, sopBuildVDBPointsGroupMenu);

const PRM_ChoiceList VDBPointsGroupMenu(PRM_CHOICELIST_TOGGLE,
    sopBuildVDBPointsGroupMenu);

#endif

} // namespace openvdb_houdini

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
