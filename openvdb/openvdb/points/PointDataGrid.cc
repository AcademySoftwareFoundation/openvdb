// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "PointDataGridImpl.h"

#include <openvdb/tree/TreeImpl.h>
#include <openvdb/tree/RootNodeImpl.h>
#include <openvdb/tree/InternalNodeImpl.h>
#include <openvdb/tree/LeafNodeImpl.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

////////////////////////////////////////

// explicit template instantiation

#define OPENVDB_TREE4(T, N1, N2, N3, LeafT) \
    template class Grid<tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode<LeafT<T, N3>, N2>, N1>>>>; \
    template class tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode<LeafT<T, N3>, N2>, N1>>>; \
    template class tree::RootNode<tree::InternalNode<tree::InternalNode<LeafT<T, N3>, N2>, N1>>; \
    template class tree::InternalNode<tree::InternalNode<LeafT<T, N3>, N2>, N1>; \
    template class tree::InternalNode<LeafT<T, N3>, N2>; \
    template class LeafT<T, N3>;

OPENVDB_TREE4_POINT_DATA_INITIALIZE()

#undef OPENVDB_TREE4

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
