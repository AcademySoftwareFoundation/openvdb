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

#include <GA/GA_Attribute.h>
#include <GA/GA_Handle.h>
#include <GA/GA_AIFTuple.h>
#include <GA/GA_ElementGroup.h>
#include <GA/GA_Iterator.h>


namespace openvdb_points_houdini {


typedef std::vector<GA_Offset> OffsetList;
typedef boost::shared_ptr<OffsetList> OffsetListPtr;


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


} // namespace openvdb_points_houdini

#endif // OPENVDB_POINTS_HOUDINI_UTILS_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
