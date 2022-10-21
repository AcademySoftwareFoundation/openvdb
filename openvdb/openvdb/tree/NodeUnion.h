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

/// @brief Default implementation of a NodeUnion that stores the child pointer
///   and the value separately (i.e., not in a union). Types which select this
///   specialization usually do not conform to the requirements of a union
///   member, that is that the type ValueT is not trivially copyable. This
///   implementation is thus NOT used for POD, math::Vec, math::Mat, math::Quat
///   or math::Coord types, but is used (for example) with std::string
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

    // Small check to ensure this class isn't
    // selected for some expected types
    static_assert(!ValueTraits<ValueT>::IsVec &&
        !ValueTraits<ValueT>::IsMat &&
        !ValueTraits<ValueT>::IsQuat &&
        !std::is_same<ValueT, math::Coord>::value &&
        !std::is_arithmetic<ValueT>::value,
        "Unexpected instantiation of NodeUnion");
};

/// @brief Template specialization of a NodeUnion that stores the child pointer
///   and the value together (int, float, pointer, etc.)
template<typename ValueT, typename ChildT>
class NodeUnion<ValueT, ChildT,
    typename std::enable_if<std::is_trivially_copyable<ValueT>::value>::type>
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


////////////////////////////////////////


} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED
