// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file NodeUnion.h
///
/// @details NodeUnion is a templated helper class that controls access to either
/// the child node pointer or the value for a particular element of a root
/// or internal node. For space efficiency, the child pointer and the value
/// are unioned when possible, since the two are never in use simultaneously.

#ifndef OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <cstring> // for std::memcpy()
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {


// Forward declaration of traits class
template<typename T> struct CopyTraits;

// Default implementation that stores the child pointer and the value separately
// (i.e., not in a union)
// This implementation is not used for POD, math::Vec or math::Coord value types.
template<typename ValueT, typename ChildT, typename Enable = void>
class NodeUnion
{
private:
    ChildT* mChild;
    ValueT  mValue;

public:
    NodeUnion(): mChild(nullptr), mValue() {}

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


// Template specialization for values of POD types (int, float, pointer, etc.)
template<typename ValueT, typename ChildT>
class NodeUnion<ValueT, ChildT, typename std::enable_if<std::is_pod<ValueT>::value>::type>
{
private:
    union { ChildT* mChild; ValueT mValue; };

public:
    NodeUnion(): mChild(nullptr) {}

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


// Template specialization for values of types such as math::Vec3f and math::Coord
// for which CopyTraits<T>::IsCopyable is true
template<typename ValueT, typename ChildT>
class NodeUnion<ValueT, ChildT, typename std::enable_if<CopyTraits<ValueT>::IsCopyable>::type>
{
private:
    union { ChildT* mChild; ValueT mValue; };

public:
    NodeUnion(): mChild(nullptr) {}
    NodeUnion(const NodeUnion& other): mChild(nullptr)
        { std::memcpy(static_cast<void*>(this), &other, sizeof(*this)); }
    NodeUnion& operator=(const NodeUnion& rhs)
        { std::memcpy(static_cast<void*>(this), &rhs, sizeof(*this)); return *this; }

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


/// @details A type T is copyable if
/// # T stores member values by value (vs. by pointer or reference)
///   and T's true byte size is given by sizeof(T).
/// # T has a trivial destructor
/// # T has a default constructor
/// # T has an assignment operator
template<typename T> struct CopyTraits { static const bool IsCopyable = false; };
template<typename T> struct CopyTraits<math::Vec2<T>> { static const bool IsCopyable = true; };
template<typename T> struct CopyTraits<math::Vec3<T>> { static const bool IsCopyable = true; };
template<typename T> struct CopyTraits<math::Vec4<T>> { static const bool IsCopyable = true; };
template<> struct CopyTraits<math::Coord> { static const bool IsCopyable = true; };


////////////////////////////////////////


} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED
