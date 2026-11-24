// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   Primitives.h

    \brief  Geometric primitives and non-owning triangle-mesh views, suitable for use in both
            host and device code.

    \details A triangle mesh is held as an indexed mesh: an array of points and an array of
             triangles, each triangle being the indices of its three points. This is the
             representation tools::cuda::MeshToGrid consumes and tools::cuda::DualContour
             produces. TriangleMeshBatchView concatenates the meshes of a batch (for instance
             one mesh per grid of a batch of grids) with per-mesh offsets into both arrays; the
             triangles of each mesh index its own points, starting from 0.

             The views own nothing and dereference their pointers only in the accessors, so they
             can be built on the host over device arrays and passed by value to kernels.
*/

#ifndef NANOVDB_MATH_PRIMITIVES_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_PRIMITIVES_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h> // for __hostdev__
#include <nanovdb/math/Math.h> // for Vec3f, Vec3i

#include <cstdint> // for uint32_t, uint64_t

namespace nanovdb {

namespace math {

/// @brief POD triangle given by the positions of its three vertices.
struct Triangle
{
    Vec3f v[3];
    __hostdev__ const Vec3f& operator[](int i) const { return v[i]; }
    __hostdev__       Vec3f& operator[](int i)       { return v[i]; }
}; // math::Triangle

/// @brief Non-owning view of an indexed triangle mesh.
struct TriangleMeshView
{
    const Vec3f* points{nullptr};    // pointCount positions
    const Vec3i* triangles{nullptr}; // triangleCount triples of indices into points
    uint32_t     pointCount{0};
    uint32_t     triangleCount{0};

    /// @brief The positions of triangle @a t, gathered from its point indices.
    __hostdev__ Triangle triangle(uint32_t t) const
    {
        const Vec3i& tri = triangles[t];
        return Triangle{{points[tri[0]], points[tri[1]], points[tri[2]]}};
    }
}; // math::TriangleMeshView

/// @brief Non-owning view of a batch of indexed triangle meshes stored back to back.
///
/// @details Mesh i owns points [pointOffsets[i], pointOffsets[i+1]) and triangles
///          [triangleOffsets[i], triangleOffsets[i+1]), and its triangles index its own points
///          (from 0). Both offset arrays have meshCount + 1 entries and start at 0. The accessors
///          read the offset arrays, so they are callable wherever those arrays live.
struct TriangleMeshBatchView
{
    const Vec3f*    points{nullptr};
    const Vec3i*    triangles{nullptr};
    const uint64_t* pointOffsets{nullptr};    // meshCount + 1 entries
    const uint64_t* triangleOffsets{nullptr}; // meshCount + 1 entries
    uint32_t        meshCount{0};

    /// @brief Total number of points of all meshes
    __hostdev__ uint64_t pointCount() const { return meshCount ? pointOffsets[meshCount] : 0; }

    /// @brief Total number of triangles of all meshes
    __hostdev__ uint64_t triangleCount() const { return meshCount ? triangleOffsets[meshCount] : 0; }

    /// @brief Mesh @a i of the batch, as a view over its slices of the arrays.
    __hostdev__ TriangleMeshView mesh(uint32_t i) const
    {
        const uint64_t p0 = pointOffsets[i], t0 = triangleOffsets[i];
        return TriangleMeshView{points + p0, triangles + t0, uint32_t(pointOffsets[i + 1] - p0),
                                uint32_t(triangleOffsets[i + 1] - t0)};
    }
}; // math::TriangleMeshBatchView

} // namespace math

} // namespace nanovdb

#endif // NANOVDB_MATH_PRIMITIVES_H_HAS_BEEN_INCLUDED
