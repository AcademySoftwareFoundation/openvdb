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
#include <openvdb/tree/TreeIterator.h>
#include "ValueTransformer.h" // for foreach()


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Mark as active any inactive tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void activate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


/// @brief Mark as inactive any active tiles or voxels in the given grid or tree
/// whose values are equal to @a value (optionally to within the given @a tolerance).
template<typename GridOrTree>
inline void deactivate(
    GridOrTree&,
    const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance = zeroVal<typename GridOrTree::ValueType>()
);


////////////////////////////////////////


namespace activation {

template<typename TreeType>
class ActivationOp
{
public:
    using ValueT = typename TreeType::ValueType;

    ActivationOp(bool state, const ValueT& val, const ValueT& tol)
        : mActivate(state)
        , mValue(val)
        , mTolerance(tol)
    {}

    void operator()(const typename TreeType::ValueOnIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setValueOff();
        }
    }

    void operator()(const typename TreeType::ValueOffIter& it) const
    {
        if (math::isApproxEqual(*it, mValue, mTolerance)) {
            it.setActiveState(/*on=*/true);
        }
    }

    void operator()(const typename TreeType::LeafIter& lit) const
    {
        using LeafT = typename TreeType::LeafNodeType;
        LeafT& leaf = *lit;
        if (mActivate) {
            for (typename LeafT::ValueOffIter it = leaf.beginValueOff(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOn(it.pos());
                }
            }
        } else {
            for (typename LeafT::ValueOnIter it = leaf.beginValueOn(); it; ++it) {
                if (math::isApproxEqual(*it, mValue, mTolerance)) {
                    leaf.setValueOff(it.pos());
                }
            }
        }
    }

private:
    bool mActivate;
    const ValueT mValue, mTolerance;
}; // class ActivationOp

} // namespace activation


template<typename GridOrTree>
inline void
activate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    using Adapter = TreeAdapter<GridOrTree>;
    using TreeType = typename Adapter::TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/true, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other inactive values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOffIter it = tree.beginValueOff();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}


template<typename GridOrTree>
inline void
deactivate(GridOrTree& gridOrTree, const typename GridOrTree::ValueType& value,
    const typename GridOrTree::ValueType& tolerance)
{
    using Adapter = TreeAdapter<GridOrTree>;
    using TreeType = typename Adapter::TreeType;

    TreeType& tree = Adapter::tree(gridOrTree);

    activation::ActivationOp<TreeType> op(/*activate=*/false, value, tolerance);

    // Process all leaf nodes in parallel.
    foreach(tree.beginLeaf(), op);

    // Process all other active values serially (because changing active states
    // is not thread-safe unless no two threads modify the same node).
    typename TreeType::ValueOnIter it = tree.beginValueOn();
    it.setMaxDepth(tree.treeDepth() - 2);
    foreach(it, op, /*threaded=*/false);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_ACTIVATE_HAS_BEEN_INCLUDED
