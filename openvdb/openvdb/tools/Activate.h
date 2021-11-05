// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file   Activate.h
///
/// @brief  Implementation of topological activation/deactivation
///
/// @author Ken Museth
///

#ifndef OPENVDB_TOOLS_ACTIVATE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ACTIVATE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for isApproxEqual()
#include <openvdb/tree/NodeManager.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Mark as active any inactive tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
void activate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>(),
    const bool threaded = true
);


/// @brief Mark as inactive any active tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
void deactivate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>(),
    const bool threaded = true
);


////////////////////////////////////////


/// @cond OPENVDB_DOCS_INTERNAL

namespace activate_internal {

template<typename TreeT, bool IgnoreTolerance = false>
struct ActivateOp
{
public:
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    explicit ActivateOp(const ValueT& value,
                        const ValueT& tolerance = zeroVal<ValueT>())
        : mValue(value)
        , mTolerance(tolerance) { }

    inline bool check(const ValueT& value) const {
        // math::isApproxEqual is marginally more expensive,
        // so opt to do direct comparison if tolerance is ignored
        if (IgnoreTolerance)    return value == mValue;
        return math::isApproxEqual(value, mValue, mTolerance);
    }

    bool operator()(RootT& root, size_t) const
    {
        for (auto it = root.beginValueOff(); it; ++it) {
            if (check(*it))     it.setValueOn(/*on=*/true);
        }
        return true;
    }

    template<typename NodeT>
    bool operator()(NodeT& node, size_t) const
    {
        // only iterate if there are inactive tiles
        if (!node.isValueMaskOn()) {
            for (auto it = node.beginValueOff(); it; ++it) {
                if (check(*it))     it.setValueOn(/*on=*/true);
            }
        }
        // return false if there are no child nodes below this node
        return !node.isChildMaskOff();
    }

    bool operator()(LeafT& leaf, size_t) const
    {
        // early-exit if there are no inactive values
        if (leaf.isValueMaskOn())  return true;
        for (auto it = leaf.beginValueOff(); it; ++it) {
            if (check(*it))     it.setValueOn(/*on=*/true);
        }
        return true;
    }

private:
    const ValueT mValue;
    const ValueT mTolerance;
};// ActivateOp

template<typename TreeT, bool IgnoreTolerance = false>
struct DeactivateOp
{
public:
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    explicit DeactivateOp(const ValueT& value,
                        const ValueT& tolerance = zeroVal<ValueT>())
        : mValue(value)
        , mTolerance(tolerance) { }

    inline bool check(const ValueT& value) const {
        if (IgnoreTolerance)    return value == mValue;
        return math::isApproxEqual(value, mValue, mTolerance);
    }

    bool operator()(RootT& root, size_t) const
    {
        for (auto it = root.beginValueOn(); it; ++it) {
            if (check(*it))     it.setValueOn(/*on=*/false);
        }
        return true;
    }

    template<typename NodeT>
    bool operator()(NodeT& node, size_t) const
    {
        // only iterate if there are active tiles
        if (!node.isValueMaskOff()) {
            for (auto it = node.beginValueOn(); it; ++it) {
                if (check(*it))     it.setValueOn(/*on=*/false);
            }
        }
        // return false if there are no child nodes below this node
        return !node.isChildMaskOff();
    }

    bool operator()(LeafT& leaf, size_t) const
    {
        // early-exit if there are no active values
        if (leaf.isValueMaskOff())  return true;
        for (auto it = leaf.beginValueOn(); it; ++it) {
            if (check(*it))     it.setValueOn(/*on=*/false);
        }
        return true;
    }

private:
    const ValueT mValue;
    const ValueT mTolerance;
};// DeactivateOp

} // namespace activate_internal

/// @endcond


////////////////////////////////////////


template<typename GridOrTree>
void activate(GridOrTree& gridOrTree,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance,
    const bool threaded)
{
    using Adapter = TreeAdapter<GridOrTree>;
    using TreeType = typename Adapter::TreeType;
    using ValueType = typename TreeType::ValueType;

    TreeType& tree = Adapter::tree(gridOrTree);

    tree::DynamicNodeManager<TreeType> nodeManager(tree);

    if (tolerance == zeroVal<ValueType>()) {
        activate_internal::ActivateOp<TreeType, /*IgnoreTolerance=*/true> op(value);
        nodeManager.foreachTopDown(op, threaded);
    } else {
        activate_internal::ActivateOp<TreeType> op(value, tolerance);
        nodeManager.foreachTopDown(op, threaded);
    }
}


template<typename GridOrTree>
void deactivate(GridOrTree& gridOrTree,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance,
    const bool threaded)
{
    using Adapter = TreeAdapter<GridOrTree>;
    using TreeType = typename Adapter::TreeType;
    using ValueType = typename TreeType::ValueType;

    TreeType& tree = Adapter::tree(gridOrTree);

    tree::DynamicNodeManager<TreeType> nodeManager(tree);

    if (tolerance == zeroVal<ValueType>()) {
        activate_internal::DeactivateOp<TreeType, /*IgnoreTolerance=*/true> op(value);
        nodeManager.foreachTopDown(op, threaded);
    } else {
        activate_internal::DeactivateOp<TreeType> op(value, tolerance);
        nodeManager.foreachTopDown(op, threaded);
    }
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_ACTIVATE
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    void activate(TreeT&, const TreeT::ValueType&, const TreeT::ValueType&, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void activate(Grid<TreeT>&, const TreeT::ValueType&, const TreeT::ValueType&, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void deactivate(TreeT&, const TreeT::ValueType&, const TreeT::ValueType&, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void deactivate(Grid<TreeT>&, const TreeT::ValueType&, const TreeT::ValueType&, const bool)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_ACTIVATE_HAS_BEEN_INCLUDED
