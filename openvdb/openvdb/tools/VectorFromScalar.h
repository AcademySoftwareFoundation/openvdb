// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file VectorMerge.h
///
/// @brief Functions to produce a vector-valued grid
/// from separate scalar grids
///
/// @author Tim Straubigner

#ifndef OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h> // for ScalarToVectorConverter
#include <openvdb/tree/NodeManager.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

template<typename ScalarTreeT>
struct VectorFromScalarOp
{
    using ScalarT = typename ScalarTreeT::ValueType;
    using VectorT = math::Vec3<ScalarT>;
    using VectorTreeT = typename ScalarTreeT::ValueConverter<VectorT>::Type;

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
            i.setValue(VectorT(
                mXTree->getValue(i.getCoord()),
                mYTree->getValue(i.getCoord()),
                mZTree->getValue(i.getCoord())
            ));
        }
    }

    // void operator()(VectorLeafT& leaf) const {
    //     // TODO
    // }

    template<typename VectorNodeT>
    void operator()(VectorNodeT& node) const {
        using ScalarNodeT = typename VectorNodeT::ValueConverter<ScalarT>::Type;

        const ScalarNodeT* xNode = mXTree->template probeNode<ScalarNodeT>(node.origin());
        const ScalarNodeT* yNode = mYTree->template probeNode<ScalarNodeT>(node.origin());
        const ScalarNodeT* zNode = mZTree->template probeNode<ScalarNodeT>(node.origin());

        for (auto i = node.beginValueOn(); i; ++i)
        {
            i.setValue(VectorT(
                xNode ? xNode->getValueUnsafe(i.offset()) : mXTree->background(),
                yNode ? yNode->getValueUnsafe(i.offset()) : mYTree->background(),
                zNode ? zNode->getValueUnsafe(i.offset()) : mZTree->background()
            ));
        }
    }

private:
    const ScalarTreeT* mXTree;
    const ScalarTreeT* mYTree;
    const ScalarTreeT* mZTree;
};

/// @brief Threaded method to convert three scalar-valued grids into a single vector-valued grid.
///        The transforms and resolutions of the three input grids must be equal. The new vector
///        grid topology is the union of all scalar grid topologies.
///
/// @return A shared pointer to a new grid with the same tree configuration, transform, and
///         resolution as the input grids. Each voxel of the new grid is a vector whose
///         component values are taken from the input grids at the corresponding voxel
///         location, using background values when one of the source grids has no voxels
///         defined there.
///
/// @param x                    Grid to use as the first vector component.
/// @param y                    Grid to use as the second vector component.
/// @param z                    Grid to use as the third vector component.
template<typename ScalarGridT>
typename ScalarToVectorConverter<ScalarGridT>::Type::Ptr
vectorFromScalar(const ScalarGridT& x, const ScalarGridT& y, const ScalarGridT& z)
{
    using ScalarT = typename ScalarGridT::ValueType;
    using VectorT = math::Vec3<ScalarT>;
    using ScalarTreeT = typename ScalarGridT::TreeType;
    using VectorGridT = typename ScalarToVectorConverter<ScalarGridT>::Type;
    using VectorTreeT = typename VectorGridT::TreeType;

    math::Transform transform = x.transform();
    if (transform != y.transform() || transform != z.transform())
    {
        OPENVDB_THROW(ValueError, "vectorFromScalar: all input grids must have the same transform");
    }

    auto background = VectorT(x.background(), y.background(), z.background());

    auto vectorGrid = createGrid<VectorGridT>(background);

    vectorGrid->topologyUnion(x);
    vectorGrid->topologyUnion(y);
    vectorGrid->topologyUnion(z);

    auto nodeManager = tree::NodeManager<VectorTreeT>(vectorGrid->tree());
    auto op = VectorFromScalarOp<ScalarTreeT>(&x.tree(), &y.tree(), &z.tree());
    nodeManager.foreachTopDown(op);

    return vectorGrid;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb



#endif // OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
