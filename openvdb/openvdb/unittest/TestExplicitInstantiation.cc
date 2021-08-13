// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/PointIndexGrid.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace math {

// explicit template instantiation of math classes

template class BBox<Vec3f>;
template class Mat3<float>;
template class Mat4<float>;
template class Quat<float>;
template class Vec2<float>;
template class Vec3<float>;
template class Vec4<float>;

} // namespace math

namespace tree {

// explicit template instantiation of LeafBuffer

template class LeafBuffer<float, 3>;
template class LeafBuffer<int64_t, 3>;
template class LeafBuffer<Vec3f, 3>;
template class LeafBuffer<bool, 3>;

// explicit template instantiation of LeafNode

template class LeafNode<float, 3>;
template class LeafNode<int64_t, 3>;
template class LeafNode<Vec3f, 3>;
template class LeafNode<bool, 3>;
template class LeafNode<ValueMask, 3>;

// explicit template instantiation of InternalNode

template class InternalNode<LeafNode<float, 3>, 4>;
template class InternalNode<InternalNode<LeafNode<float, 3>, 4>, 5>;

// explicit template instantiation of RootNode

template class RootNode<InternalNode<InternalNode<LeafNode<float, 3>, 4>, 5>>;

// explicit template instantiation of Tree

template class Tree<RootNode<InternalNode<InternalNode<LeafNode<float, 3>, 4>, 5>>>;

// explicit template instantiation of LeafManager and NodeManagers

template class LeafManager<FloatTree>;
template class NodeManager<FloatTree>;
template class DynamicNodeManager<FloatTree>;

// explicit template instantiation of ValueAccessor

template class ValueAccessor3<FloatTree>;

} // namespace tree

namespace points {

// explicit template instantiation of PointDataLeafNode

template class PointDataLeafNode<PointDataIndex32, 3>;

} // namespace points

namespace tools {

// explicit template instantiation of PointIndexLeafNode

template struct PointIndexLeafNode<PointIndex32, 3>;

} // namespace tools

// explicit template instantiation of Grid

template class Grid<FloatTree>;

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
