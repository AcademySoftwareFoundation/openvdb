// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TYPES_H
#define FVDB_TYPES_H

#include <fvdb/detail/TypesImpl.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAFunctions.h>
#include <torch/all.h>

namespace fvdb {

// These are union types that can be constructed from nanovdb types, torch tensors, std::vectors,
// single scalars, etc... They are used to allow the user to pass in a variety of types to the API,
// and then convert them to the correct type
using Vec3i         = detail::Coord3Impl<false>;
using Vec3iOrScalar = detail::Coord3Impl<true>;
using Vec4i         = detail::Coord4Impl<false>;
using Vec3d         = detail::Vec3dImpl<false>;
using Vec3dOrScalar = detail::Vec3dImpl<true>;

// These are union types that can be constructed from nanovdb types, torch tensors, std::vectors,
// single scalars, etc... and resolve to a batch of values. They are used to allow the user to pass
// in a single vector (or scalar) and have it be broadcast to a whole batch of values. E.g. if you
// are constructing a batch of grids, you can pass in a single scalar 1.0 to have a voxel size of
// [1, 1, 1]
//      for every grid in the batch. Or a user can pass in a vector [1, 2, 3] to have each grid have
//      a voxel
//       size of [1, 2, 3]. Alternatively, a user can specify a voxel size for each grid in the
//       batch
//       [[v1x, v1y, v1z], ..., [vnx, vny, vnz]]. The Vec3dBatchOrScalar will accept all these
//       inputs and resolve them to a batch of values.
using Vec3dBatchOrScalar =
    detail::Vec3BatchImpl<nanovdb::Vec3d, true /*AllowScalar*/, true /*AllowBroadcast*/>;
using Vec3dBatch =
    detail::Vec3BatchImpl<nanovdb::Vec3d, false /*AllowScalar*/, true /*AllowBroadcast*/>;
using Vec3iBatch =
    detail::Vec3BatchImpl<nanovdb::Coord, false /*AllowScalar*/, true /*AllowBroadcast*/>;

} // namespace fvdb

#endif // FVDB_TYPES_H
