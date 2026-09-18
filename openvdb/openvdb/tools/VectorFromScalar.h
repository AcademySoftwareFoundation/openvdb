// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file VectorFromScalar.h
///
/// @brief Functions to produce a vector-valued tree
/// from separate scalar trees
///
/// @author Tim Straubigner

#ifndef OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h> // for ScalarToVectorConverter
#include <openvdb/tree/NodeManager.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

template<typename ScalarTreeT, bool CopyInactiveValues>
struct VectorFromScalarOp
{
    using ScalarT = typename ScalarTreeT::ValueType;
    using VectorT = math::Vec3<ScalarT>;
    using VectorTreeT = typename ScalarTreeT::template ValueConverter<VectorT>::Type;

    using VectorRootT = typename VectorTreeT::RootNodeType;
    using VectorLeafT = typename VectorTreeT::LeafNodeType;

    VectorFromScalarOp(const ScalarTreeT* x, const ScalarTreeT* y, const ScalarTreeT* z)
        : mXTree(x)
        , mYTree(y)
        , mZTree(z)
    {}

    void operator()(VectorRootT& root) const {
        for (auto i = root.beginValueOn(); i; ++i)
        {
            ScalarT x, y, z;

            if constexpr (CopyInactiveValues)
            {
                x = mXTree->getValue(i.getCoord());
                y = mYTree->getValue(i.getCoord());
                z = mZTree->getValue(i.getCoord());
            }
            else
            {
                if (!mXTree->probeValue(i.getCoord(), x)) { x = mXTree->background(); }
                if (!mYTree->probeValue(i.getCoord(), y)) { y = mYTree->background(); }
                if (!mZTree->probeValue(i.getCoord(), z)) { z = mZTree->background(); }
            }


            i.setValue(VectorT(x, y, z));
        }
    }

    template<typename VectorNodeT>
    void operator()(VectorNodeT& node) const {
        using ScalarNodeT = typename VectorNodeT::template ValueConverter<ScalarT>::Type;

        // Probe for matching tiles or voxels in each of the source scalar grids.
        // The output vector grid has been topology-merged with all source grids,
        // which guarantees that none of the corresponding nodes in the source
        // grids are more finely discretized. Each source grid either has a matching
        // node of equal level, or the tree is tiled more coarsely or empty at
        // the same location.

        const ScalarNodeT* xNode = mXTree->template probeNode<ScalarNodeT>(node.origin());
        const ScalarNodeT* yNode = mYTree->template probeNode<ScalarNodeT>(node.origin());
        const ScalarNodeT* zNode = mZTree->template probeNode<ScalarNodeT>(node.origin());

        for (auto i = node.beginValueOn(); i; ++i)
        {
            ScalarT x, y, z;

            if constexpr (CopyInactiveValues)
            {
                x = xNode ? xNode->getValueUnsafe(i.offset()) : mXTree->getValue(i.getCoord());
                y = yNode ? yNode->getValueUnsafe(i.offset()) : mYTree->getValue(i.getCoord());
                z = zNode ? zNode->getValueUnsafe(i.offset()) : mZTree->getValue(i.getCoord());
            }
            else
            {
                if (xNode)
                {
                    x = xNode->isValueOn(i.offset()) ? xNode->getValueUnsafe(i.offset()) : mXTree->background();
                }
                else
                {
                    x = mXTree->getValue(i.getCoord());
                }

                if (yNode)
                {
                    y = yNode->isValueOn(i.offset()) ? yNode->getValueUnsafe(i.offset()) : mYTree->background();
                }
                else
                {
                    y = mYTree->getValue(i.getCoord());
                }

                if (zNode)
                {
                    z = zNode->isValueOn(i.offset()) ? zNode->getValueUnsafe(i.offset()) : mZTree->background();
                }
                else
                {
                    z = mZTree->getValue(i.getCoord());
                }
            }

            i.setValue(VectorT(x, y, z));
        }
    }

private:
    const ScalarTreeT* mXTree;
    const ScalarTreeT* mYTree;
    const ScalarTreeT* mZTree;
};

/// @brief Threaded method to convert three scalar-valued trees into a single vector-valued tree.
///        The new vector tree topology is the union of all scalar tree topologies.
///
/// @return A shared pointer to a new tree with the same tree configuration, transform, and
///         resolution as the input trees. Each voxel of the new tree is a vector whose
///         component values are taken from the input trees at the corresponding voxel
///         location, using background values when one of the source trees has no voxels
///         defined there.
///
/// @param x                    Tree to use as the first vector component.
/// @param y                    Tree to use as the second vector component.
/// @param z                    Tree to use as the third vector component.
/// @param copyInactiveValues   When only some of the source trees are active at a location,
///                             this decides whether inactive locations are used exactly or are
///                             replaced with background values. In all cases, locations where
///                             all source trees are inactive will result in background values.
template<typename ScalarTreeT>
typename ScalarToVectorConverter<ScalarTreeT>::Type::Ptr
vectorFromScalar(const ScalarTreeT& x, const ScalarTreeT& y, const ScalarTreeT& z, bool copyInactiveValues = false)
{
    using ScalarT = typename ScalarTreeT::ValueType;
    using VectorT = math::Vec3<ScalarT>;
    using VectorTreeT = typename ScalarToVectorConverter<ScalarTreeT>::Type;

    auto background = VectorT(x.background(), y.background(), z.background());

    // Create an empty tree
    auto vectorTree = typename VectorTreeT::Ptr(new VectorTreeT(background));

    // Perform a topology union with each of the source scalar trees.
    // This leaves the combined tree whose tile and voxel depth is the
    // maximum of all the source trees.
    vectorTree->topologyUnion(x);
    vectorTree->topologyUnion(y);
    vectorTree->topologyUnion(z);

    // Write the tree values
    auto nodeManager = tree::NodeManager<VectorTreeT>(*vectorTree);
    if (copyInactiveValues)
    {
        auto op = VectorFromScalarOp<ScalarTreeT, /* CopyInactiveValues = */ true>(&x, &y, &z);
        nodeManager.foreachTopDown(op);
    }
    else
    {
        auto op = VectorFromScalarOp<ScalarTreeT, /* CopyInactiveValues = */ false>(&x, &y, &z);
        nodeManager.foreachTopDown(op);
    }

    return vectorTree;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb



#endif // OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
