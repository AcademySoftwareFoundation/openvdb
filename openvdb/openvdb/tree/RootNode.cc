// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "RootNodeImpl.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

////////////////////////////////////////

// explicit template instantiation

#define OPENVDB_TREE4(T, N1, N2, N3, LeafT) \
    template class RootNode<InternalNode<InternalNode<LeafT<T, N3>, N2>, N1>>;

OPENVDB_TREE4_VOLUME_INITIALIZE()
OPENVDB_TREE4_PRIVATE_INITIALIZE()

#undef OPENVDB_TREE4

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
