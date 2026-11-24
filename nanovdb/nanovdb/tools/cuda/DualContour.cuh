// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/DualContour.cuh

    \authors Jonathan Swartz

    \brief Dual-contouring (DC + QEF) triangle-mesh extraction from a scalar field on the active
           voxels of NanoVDB ValueOnIndex grids, on the device, for one grid or a batch of grids.

    \details Each cube cell whose 8 corners are active voxels with values straddling the iso value
             gets one vertex, placed by minimising a quadratic error function built from the
             cell's edge crossings and the field's gradient there (regularised toward the crossing
             centroid and clamped to the cell). One quad joins the 4 cells around every bipolar
             grid edge; quads are split into triangles, which are oriented by the field gradient so
             that they face increasing values (outward, for a signed distance field).

             The field is a float sidecar indexed by the grids' value indices: for a single grid,
             entry v (v >= 1) is the value of the voxel with value index v, as produced for instance
             by tools::cuda::MeshToGrid's UDF sidecar. Entry 0 is never read. A narrow band of at
             least ~3 voxels around the iso surface is needed for a closed mesh.

             A batch of grids is meshed together through a batched VoxelBlockManager
             (buildVoxelBlockManagerBatch), in chunks of consecutive grids holding a bounded number
             of active voxels. The kernel launches and host synchronizations are a small constant
             per chunk rather than per grid, and each grid's mesh is bitwise identical to meshing
             that grid alone.

             Method: T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite Data",
             ACM TOG 21(3), 2002. The QEF is the quadric error metric of M. Garland, P. Heckbert,
             "Surface Simplification Using Quadric Error Metrics", SIGGRAPH 1997.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_DUALCONTOUR_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_DUALCONTOUR_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Primitives.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel, blocksPerGrid
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

#include <cub/cub.cuh>
#include <cuda/iterator> // cuda::counting_iterator

#include <cmath> // std::cos
#include <cstdint>
#include <limits>
#include <stdexcept> // std::runtime_error
#include <type_traits>
#include <utility>
#include <vector>

namespace nanovdb {

namespace tools::cuda {

/// @brief Vertex transform that maps index-space vertex positions to world space with each
///        grid's own map (the default).
struct DualContourIndexToWorld
{
    template<typename GridT>
    __device__ Vec3f operator()(const GridT* grid, uint32_t /*gridIndex*/, const Vec3f& xyz) const
    {
        return grid->indexToWorldF(xyz);
    }
};

/// @brief Vertex transform that keeps vertex positions in index space.
struct DualContourIndexSpace
{
    template<typename GridT>
    __device__ Vec3f operator()(const GridT* /*grid*/, uint32_t /*gridIndex*/, const Vec3f& xyz) const
    {
        return xyz;
    }
};

/// @brief The meshes of a DualContour run, one per grid, stored back to back in device memory.
/// @details Grid g's mesh owns points [pointOffsets[g], pointOffsets[g+1]) (and the same range of
///          normals) and triangles [triangleOffsets[g], triangleOffsets[g+1]); its triangles index
///          its own points, starting from 0. Normals are unit length.
template<typename ResourceT = nanovdb::cuda::DeviceResource>
struct DualContourMesh
{
    template<typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

    BufT<Vec3f>    points;
    BufT<Vec3f>    normals;
    BufT<Vec3i>    triangles;
    BufT<uint64_t> pointOffsets;    // gridCount + 1 entries
    BufT<uint64_t> triangleOffsets; // gridCount + 1 entries
    std::vector<uint64_t> hostPointOffsets;    // host copy of pointOffsets
    std::vector<uint64_t> hostTriangleOffsets; // host copy of triangleOffsets

    /// @brief Empty meshes whose buffers allocate from @a resource on @a stream
    DualContourMesh(cudaStream_t stream, nanovdb::cuda::ResourceRef<ResourceT> resource)
        : points(stream, resource, 0, nanovdb::cuda::noInit)
        , normals(stream, resource, 0, nanovdb::cuda::noInit)
        , triangles(stream, resource, 0, nanovdb::cuda::noInit)
        , pointOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , triangleOffsets(stream, resource, 0, nanovdb::cuda::noInit)
    {
    }

    /// @brief Number of meshes, i.e. of grids meshed
    uint32_t meshCount() const { return hostPointOffsets.empty() ? 0u : uint32_t(hostPointOffsets.size() - 1); }
    uint64_t pointCount() const { return hostPointOffsets.empty() ? 0u : hostPointOffsets.back(); }
    uint64_t triangleCount() const { return hostTriangleOffsets.empty() ? 0u : hostTriangleOffsets.back(); }

    /// @brief View of the meshes over the device arrays (its accessors are device-only)
    math::TriangleMeshBatchView deviceView() const
    {
        return math::TriangleMeshBatchView{points.data(), triangles.data(), pointOffsets.data(),
                                           triangleOffsets.data(), this->meshCount()};
    }
}; // tools::cuda::DualContourMesh

namespace dualcontour::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Box-stencil geometry (compile time).
//
// A box-stencil neighbor is addressed by a spoke index in [0, 27): spoke = (di+1)*9 + (dj+1)*3 + (dk+1),
// the center (the voxel itself) being spoke 13, as in VoxelBlockManager::computeBoxStencil. The mesher
// touches only a subset of the spokes (the 8 cube corners, the 6 faces and the 3 connectivity fans), so
// the gather stores just those columns of each voxel's neighbors.

struct BoxStencilGeometry
{
    int   numColumns{0};
    int   usedSpokes[27]{};      // the numColumns gathered spokes, sorted
    int   cornerColumn[8]{};     // column of each cube corner
    int   centerColumn{0};       // column of spoke 13 (the voxel itself)
    int   fanForwardColumn[3]{}; // column of the forward neighbor along x, y, z (the minimal edges)
    int   fanColumn[3][4]{};     // the columns of the 4 cells around each minimal edge
    int   edgeCornerA[12]{};     // the 12 cube edges as corner pairs (A, B)
    int   edgeCornerB[12]{};
    int   cornerSpoke[8]{};      // spoke of each corner (the gather reads the stencil directly)
    int   faceSpoke[6]{};        // spoke of each face neighbor (-x, +x, -y, +y, -z, +z)
    float cornerOffset[8][3]{};  // the corner offsets in [0, 1]^3
};

constexpr int spokeIndex(int di, int dj, int dk) { return (di + 1) * 9 + (dj + 1) * 3 + (dk + 1); }

constexpr int columnOfSpoke(const BoxStencilGeometry& geometry, int spoke)
{
    for (int i = 0; i < geometry.numColumns; ++i)
        if (geometry.usedSpokes[i] == spoke) return i;
    return -1;
}

constexpr BoxStencilGeometry makeBoxStencilGeometry()
{
    BoxStencilGeometry geometry{};
    const int cornerOffset[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    const int cubeEdge[12][2] = {
        {0, 1}, {2, 4}, {3, 5}, {6, 7}, {0, 2}, {1, 4}, {3, 6}, {5, 7}, {0, 3}, {1, 5}, {2, 6}, {4, 7}};
    for (int corner = 0; corner < 8; ++corner) {
        for (int axis = 0; axis < 3; ++axis) geometry.cornerOffset[corner][axis] = float(cornerOffset[corner][axis]);
        geometry.cornerSpoke[corner] = spokeIndex(cornerOffset[corner][0], cornerOffset[corner][1], cornerOffset[corner][2]);
    }
    for (int edge = 0; edge < 12; ++edge) {
        geometry.edgeCornerA[edge] = cubeEdge[edge][0];
        geometry.edgeCornerB[edge] = cubeEdge[edge][1];
    }
    geometry.faceSpoke[0] = spokeIndex(-1, 0, 0);
    geometry.faceSpoke[1] = spokeIndex( 1, 0, 0);
    geometry.faceSpoke[2] = spokeIndex(0, -1, 0);
    geometry.faceSpoke[3] = spokeIndex(0,  1, 0);
    geometry.faceSpoke[4] = spokeIndex(0, 0, -1);
    geometry.faceSpoke[5] = spokeIndex(0, 0,  1);
    const int fanForward[3]  = {spokeIndex(1, 0, 0), spokeIndex(0, 1, 0), spokeIndex(0, 0, 1)};
    const int fanSpoke[3][4] = {
        {13, spokeIndex(0, 0, -1), spokeIndex(0, -1, -1), spokeIndex(0, -1, 0)},
        {13, spokeIndex(0, 0, -1), spokeIndex(-1, 0, -1), spokeIndex(-1, 0, 0)},
        {13, spokeIndex(0, -1, 0), spokeIndex(-1, -1, 0), spokeIndex(-1, 0, 0)}};

    // the gathered columns: every spoke the mesher reads, in ascending order
    bool used[27]{};
    for (int i = 0; i < 6; ++i) used[geometry.faceSpoke[i]] = true;
    for (int i = 0; i < 8; ++i) used[geometry.cornerSpoke[i]] = true;
    for (int dir = 0; dir < 3; ++dir) {
        used[fanForward[dir]] = true;
        for (int i = 0; i < 4; ++i) used[fanSpoke[dir][i]] = true;
    }
    for (int spoke = 0; spoke < 27; ++spoke)
        if (used[spoke]) geometry.usedSpokes[geometry.numColumns++] = spoke;

    for (int corner = 0; corner < 8; ++corner) geometry.cornerColumn[corner] = columnOfSpoke(geometry, geometry.cornerSpoke[corner]);
    geometry.centerColumn = columnOfSpoke(geometry, 13);
    for (int dir = 0; dir < 3; ++dir) {
        geometry.fanForwardColumn[dir] = columnOfSpoke(geometry, fanForward[dir]);
        for (int i = 0; i < 4; ++i) geometry.fanColumn[dir][i] = columnOfSpoke(geometry, fanSpoke[dir][i]);
    }
    return geometry;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// QEF and connectivity primitives.

/// @brief Solve the symmetric 3x3 system A x = b (A given by its 6 unique entries a00, a01, a02,
///        a11, a12, a22) by Cramer's rule; x = 0 when A is (numerically) singular.
__hostdev__ inline void solveSymmetric3x3(const double A[6], const double b[3], double x[3])
{
    const double a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3], a4 = A[4], a5 = A[5];
    const double cof00 = a3 * a5 - a4 * a4, cof01 = a4 * a2 - a1 * a5, cof02 = a1 * a4 - a3 * a2;
    const double determinant = a0 * cof00 + a1 * cof01 + a2 * cof02;
    if (fabs(determinant) < 1e-30) {
        x[0] = x[1] = x[2] = 0.0;
        return;
    }
    const double invDet = 1.0 / determinant;
    const double cof11 = a0 * a5 - a2 * a2, cof12 = a1 * a2 - a0 * a4, cof22 = a0 * a3 - a1 * a1;
    x[0] = invDet * (cof00 * b[0] + cof01 * b[1] + cof02 * b[2]);
    x[1] = invDet * (cof01 * b[0] + cof11 * b[1] + cof12 * b[2]);
    x[2] = invDet * (cof02 * b[0] + cof12 * b[1] + cof22 * b[2]);
}

/// @brief One cube edge's iso crossing: the interpolated crossing point in the cell's [0, 1]^3
///        frame and the unit field gradient there.
struct EdgeCrossing
{
    float point[3];
    float normal[3];
};

/// @brief The iso crossing of cube edge @a edge from the cell's 8 corner values and gradients.
///        Returns false, leaving @a out untouched, when the edge does not cross the iso surface.
__device__ inline bool computeEdgeCrossing(int edge, const float cornerValue[8], const float cornerGradient[8][3],
                                           const BoxStencilGeometry& geometry, float iso, EdgeCrossing& out)
{
    const int   cornerA = geometry.edgeCornerA[edge], cornerB = geometry.edgeCornerB[edge];
    const float valueA = cornerValue[cornerA], valueB = cornerValue[cornerB];
    if ((valueA < iso) == (valueB < iso)) return false;
    // the fraction along A->B where the linearly interpolated value reaches iso (the midpoint
    // when the endpoint values are numerically equal)
    const float delta = valueB - valueA;
    float       t     = (fabsf(delta) > 1e-12f) ? (iso - valueA) / delta : 0.5f;
    t                 = fminf(fmaxf(t, 0.f), 1.f);
    float edgeNormal[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        out.point[axis] = geometry.cornerOffset[cornerA][axis] +
                          t * (geometry.cornerOffset[cornerB][axis] - geometry.cornerOffset[cornerA][axis]);
        edgeNormal[axis] = cornerGradient[cornerA][axis] + t * (cornerGradient[cornerB][axis] - cornerGradient[cornerA][axis]);
    }
    // the epsilon keeps a (near-)zero gradient finite
    const float invLen = rsqrtf(edgeNormal[0] * edgeNormal[0] + edgeNormal[1] * edgeNormal[1] +
                                edgeNormal[2] * edgeNormal[2] + 1e-24f);
    out.normal[0] = edgeNormal[0] * invLen;
    out.normal[1] = edgeNormal[1] * invLen;
    out.normal[2] = edgeNormal[2] * invLen;
    return true;
}

/// @brief Gather a cell's 8 corner values and gradients through the neighbor table.
__device__ inline void loadCellCorners(int32_t cell, const int32_t* neighborTable, const float* value,
                                       const float* gradient, const BoxStencilGeometry& geometry,
                                       float cornerValue[8], float cornerGradient[8][3])
{
    const int numColumns = geometry.numColumns;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        const int32_t slot = neighborTable[int64_t(cell) * numColumns + geometry.cornerColumn[corner]];
        cornerValue[corner]       = value[slot];
        cornerGradient[corner][0] = gradient[int64_t(slot) * 3 + 0];
        cornerGradient[corner][1] = gradient[int64_t(slot) * 3 + 1];
        cornerGradient[corner][2] = gradient[int64_t(slot) * 3 + 2];
    }
}

/// @brief The quad dual to the minimal (+x, +y or +z, by @a dir) edge of voxel @a slot: when that
///        edge is bipolar and the 4 cells around it all carry a vertex, fills @a quad with their
///        vertices and returns true. Each voxel owns only its 3 forward edges, so every quad is
///        produced once.
__device__ inline bool tryBuildQuad(int64_t slot, int dir, const int32_t* neighborTable, const float* value,
                                    const int32_t* cellVertex, const BoxStencilGeometry& geometry, float iso,
                                    int32_t quad[4])
{
    if (slot == 0) return false;
    const int     numColumns = geometry.numColumns;
    const int32_t forward    = neighborTable[slot * numColumns + geometry.fanForwardColumn[dir]];
    if (forward <= 0) return false;
    if ((value[slot] < iso) == (value[forward] < iso)) return false; // not bipolar
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int     column = geometry.fanColumn[dir][i];
        const int32_t cell   = column == geometry.centerColumn ? int32_t(slot) : neighborTable[slot * numColumns + column];
        const int32_t vertex = cellVertex[cell]; // cellVertex[0] == -1 (the background)
        if (vertex < 0) return false;
        quad[i] = vertex;
    }
    return true;
}

/// @brief Triangle @a which (0 or 1) of the split of @a quad; false if it is degenerate, which
///        happens only when cells of the quad were collapsed onto one vertex by decimation.
__device__ inline bool splitQuadTriangle(const int32_t quad[4], int which, int32_t tri[3])
{
    const bool useDiagonal13 = (quad[0] == quad[2]) && (quad[1] != quad[3]);
    if (which == 0) {
        tri[0] = quad[0];
        tri[1] = quad[1];
        tri[2] = useDiagonal13 ? quad[3] : quad[2];
    } else {
        tri[0] = useDiagonal13 ? quad[1] : quad[0];
        tri[1] = quad[2];
        tri[2] = quad[3];
    }
    return tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2];
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Kernels. Every per-voxel array of a chunk is indexed by its chunk-wide slot: the voxel with
// batch-wide voxel index f (the exclusive sum of the preceding grids' voxel counts plus its value
// index minus 1) sits at slot 1 + f - chunkVoxelBase, and slot 0 is a background shared by the
// inactive neighbors of every grid of the chunk.

/// @brief Per-voxel gather, one CUDA block per VBM block: decodes the voxel's 3x3x3 box stencil
///        once and writes its neighbor-table row, its value, its central-difference gradient,
///        whether its cube is a surface cell, and its index-space coordinate. CUDA block b handles
///        batch-wide VBM block vbBegin + b; gap blocks between grids, and the slots past a grid's
///        last voxel, exit at once. The stencil reaches only voxels of the same grid, so the
///        neighbor table, and all connectivity built from it, never crosses grids.
template<int Log2BlockWidth, typename EntrySource>
struct GatherFunctor
{
    static constexpr int MaxThreadsPerBlock = 1 << Log2BlockWidth;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    using ViewT = VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>;
    using VBM   = VoxelBlockManager<Log2BlockWidth>;

    __device__ void operator()(const ViewT view, uint64_t vbBegin, uint64_t chunkVoxelBase,
                               const uint64_t* voxelBase, const float* field, const int64_t* fieldOffsets,
                               BoxStencilGeometry geometry, float iso, int32_t* neighborTable, float* value,
                               float* gradient, uint8_t* surfaceFlag, int32_t* voxelCoord) const
    {
        if (blockIdx.x == 0 && threadIdx.x == 0) { // the background slot
            value[0] = iso + 1.f;
            gradient[0] = gradient[1] = gradient[2] = 0.f;
            surfaceFlag[0] = 0;
        }
        const uint64_t vb = vbBegin + blockIdx.x; // this CUDA block's batch-wide VBM block
        // a gap block maps to the grid before it and falls past that grid's voxels below
        const uint32_t g = view.lastGridAtOrBefore(vb);
        if (g == ViewT::UnusedGridIndex) return;
        const uint64_t localBlock = vb - view.blockBase(g);
        if (localBlock * MaxThreadsPerBlock + threadIdx.x >= view.voxelCount(g)) return;
        const NanoGrid<ValueOnIndex>* grid = view.entry(g).grid;
        uint32_t leafIndex;
        uint16_t voxelOffset;
        VBM::template decodeInverseMap<ValueOnIndex>(grid, view.firstLeafID()[vb],
                                                     view.jumpMap() + vb * VBM::JumpMapLength,
                                                     1 + localBlock * MaxThreadsPerBlock, int(threadIdx.x),
                                                     leafIndex, voxelOffset);
        if (leafIndex == VBM::UnusedLeafIndex) return; // unreachable after the voxelCount test
        uint64_t stencil[27]; // value indices of the grid, 0 for inactive neighbors
        VBM::template computeBoxStencil<ValueOnIndex>(grid, leafIndex, voxelOffset, stencil);

        const int64_t slotBase  = int64_t(voxelBase[g] - chunkVoxelBase);
        const float*  gridField = field + fieldOffsets[g];
        auto slotOf = [&](int spoke) -> int64_t {
            const uint64_t v = stencil[spoke];
            return v ? int64_t(v) + slotBase : 0;
        };
        const int64_t center     = slotOf(13); // >= 1
        const float   centerValue = gridField[stencil[13]];
        const int     numColumns = geometry.numColumns;
        for (int column = 0; column < numColumns; ++column)
            neighborTable[center * numColumns + column] = int32_t(slotOf(geometry.usedSpokes[column]));
        value[center] = centerValue;

        // central-difference gradient in index space; an inactive face neighbor takes the
        // center's value
        auto faceValue = [&](int spoke) {
            const uint64_t v = stencil[spoke];
            return v ? gridField[v] : centerValue;
        };
        gradient[center * 3 + 0] = 0.5f * (faceValue(geometry.faceSpoke[1]) - faceValue(geometry.faceSpoke[0]));
        gradient[center * 3 + 1] = 0.5f * (faceValue(geometry.faceSpoke[3]) - faceValue(geometry.faceSpoke[2]));
        gradient[center * 3 + 2] = 0.5f * (faceValue(geometry.faceSpoke[5]) - faceValue(geometry.faceSpoke[4]));

        // the cube anchored at this voxel is a surface cell iff its 8 corners are all active and
        // straddle iso; inactive corners read as outside (iso + 1)
        int  insideCount = 0;
        bool allActive   = true;
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const uint64_t v = stencil[geometry.cornerSpoke[corner]];
            insideCount += ((v ? gridField[v] : iso + 1.f) < iso) ? 1 : 0;
            if (corner) allActive &= (v != 0); // corner 0 is this voxel, active by construction
        }
        surfaceFlag[center] = (allActive && insideCount > 0 && insideCount < 8) ? 1u : 0u;

        const Coord ijk = grid->tree().template getFirstNode<0>()[leafIndex].offsetToGlobalCoord(voxelOffset);
        voxelCoord[center * 3 + 0] = ijk[0];
        voxelCoord[center * 3 + 1] = ijk[1];
        voxelCoord[center * 3 + 2] = ijk[2];
    }
}; // GatherFunctor

/// @brief The grid of each surface cell: the last grid of the chunk whose first batch-wide voxel
///        index is at or below the cell's (grids without voxels share the next grid's base and are
///        passed over).
struct SurfaceGridFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const uint64_t* voxelBase, uint32_t gridBegin,
                               uint32_t gridEnd, uint64_t chunkVoxelBase, int32_t* surfaceGrid) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const uint64_t f = chunkVoxelBase + uint64_t(surfaceCells[i]) - 1;
        surfaceGrid[i] = int32_t(gridBegin - 1 +
            nanovdb::tools::cuda::detail::countAtOrBelow(gridEnd - gridBegin, f,
                [&](uint32_t k) { return voxelBase[gridBegin + k]; }));
    }
};

/// @brief cellVertex[surfaceCells[i]] = i (every other slot holds -1); cellVertex[i] = i when
///        surfaceCells is null.
struct CellVertexFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, int32_t* cellVertex) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) cellVertex[surfaceCells ? surfaceCells[i] : int32_t(i)] = int32_t(i);
    }
};

/// @brief One vertex per surface cell: the minimiser of E(x) = sum_i [n_i . (x - p_i)]^2 over the
///        cell's edge crossings p_i with unit normals n_i, i.e. the solution of the normal
///        equations A x = b with A = sum n n^T and b = sum n (n . p), re-centred on the crossing
///        centroid, Tikhonov-regularised and clamped to the cell. The cell's gradient is kept as
///        the vertex's reference normal (it orients the triangles and becomes the output normal).
template<typename ViewT, typename XformT>
struct PlaceVerticesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid,
                               const int32_t* neighborTable, const float* value, const float* gradient,
                               const int32_t* voxelCoord, BoxStencilGeometry geometry, float iso,
                               ViewT view, XformT xform, Vec3f* points, Vec3f* refNormals) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int32_t cell = surfaceCells[i];
        float cornerValue[8], cornerGradient[8][3];
        loadCellCorners(cell, neighborTable, value, gradient, geometry, cornerValue, cornerGradient);
        double A[6] = {0, 0, 0, 0, 0, 0}, b[3] = {0, 0, 0}, crossingSum[3] = {0, 0, 0};
        int    crossingCount = 0;
#pragma unroll
        for (int edge = 0; edge < 12; ++edge) {
            EdgeCrossing ec;
            if (!computeEdgeCrossing(edge, cornerValue, cornerGradient, geometry, iso, ec)) continue;
            const double nx = ec.normal[0], ny = ec.normal[1], nz = ec.normal[2];
            const double np = nx * ec.point[0] + ny * ec.point[1] + nz * ec.point[2];
            A[0] += nx * nx; A[1] += nx * ny; A[2] += nx * nz;
            A[3] += ny * ny; A[4] += ny * nz; A[5] += nz * nz;
            b[0] += nx * np; b[1] += ny * np; b[2] += nz * np;
            crossingSum[0] += ec.point[0]; crossingSum[1] += ec.point[1]; crossingSum[2] += ec.point[2];
            ++crossingCount;
        }
        const double invCount = crossingCount > 0 ? 1.0 / crossingCount : 0.0;
        const double c[3] = {crossingSum[0] * invCount, crossingSum[1] * invCount, crossingSum[2] * invCount};
        double local[3] = {c[0], c[1], c[2]};
        if (crossingCount > 0) {
            // with y = x - c: (A + 0.05 I) y = b - A c, then x = c + y clamped to the cell. The
            // regularisation keeps rank-deficient systems (flat cells) solvable and pulls their
            // free directions toward the centroid.
            const double Ac[3] = {A[0] * c[0] + A[1] * c[1] + A[2] * c[2],
                                  A[1] * c[0] + A[3] * c[1] + A[4] * c[2],
                                  A[2] * c[0] + A[4] * c[1] + A[5] * c[2]};
            const double rhs[3] = {b[0] - Ac[0], b[1] - Ac[1], b[2] - Ac[2]};
            const double regularized[6] = {A[0] + 0.05, A[1], A[2], A[3] + 0.05, A[4], A[5] + 0.05};
            double y[3];
            solveSymmetric3x3(regularized, rhs, y);
            for (int axis = 0; axis < 3; ++axis) local[axis] = fmin(fmax(c[axis] + y[axis], 0.0), 1.0);
        }
        const uint32_t g = uint32_t(surfaceGrid[i]);
        const Vec3f ijk(float(voxelCoord[int64_t(cell) * 3 + 0]) + float(local[0]),
                        float(voxelCoord[int64_t(cell) * 3 + 1]) + float(local[1]),
                        float(voxelCoord[int64_t(cell) * 3 + 2]) + float(local[2]));
        points[i] = xform(view.entry(g).grid, g, ijk);
        refNormals[i] = Vec3f(gradient[int64_t(cell) * 3 + 0], gradient[int64_t(cell) * 3 + 1],
                              gradient[int64_t(cell) * 3 + 2]);
    }
}; // PlaceVerticesFunctor

/// @brief Connectivity, pass 1: the number of triangles each surface cell emits. (Every quad has
///        its owning voxel among its 4 cells, so iterating the surface cells finds them all.)
struct CountTrianglesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* neighborTable,
                               const float* value, const int32_t* cellVertex, BoxStencilGeometry geometry,
                               float iso, uint32_t* triangleCounts) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int32_t  quad[4], tri[3];
        uint32_t count = 0;
#pragma unroll
        for (int dir = 0; dir < 3; ++dir) {
            if (!tryBuildQuad(surfaceCells[i], dir, neighborTable, value, cellVertex, geometry, iso, quad)) continue;
            for (int which = 0; which < 2; ++which) count += splitQuadTriangle(quad, which, tri) ? 1 : 0;
        }
        triangleCounts[i] = count;
    }
};

/// @brief Connectivity, pass 2: each surface cell writes its triangles at its exclusive prefix
///        sum, so the triangle order is deterministic.
struct WriteTrianglesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* neighborTable,
                               const float* value, const int32_t* cellVertex, BoxStencilGeometry geometry,
                               float iso, const uint64_t* triangleOffsets, Vec3i* triangles) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int32_t  quad[4], tri[3];
        uint64_t t = triangleOffsets[i];
#pragma unroll
        for (int dir = 0; dir < 3; ++dir) {
            if (!tryBuildQuad(surfaceCells[i], dir, neighborTable, value, cellVertex, geometry, iso, quad)) continue;
            for (int which = 0; which < 2; ++which)
                if (splitQuadTriangle(quad, which, tri)) triangles[t++] = Vec3i(tri[0], tri[1], tri[2]);
        }
    }
};

/// @brief Flip each triangle whose geometric normal disagrees with the sum of its vertices'
///        reference normals.
struct OrientTrianglesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const Vec3f* points, const Vec3f* refNormals, Vec3i* triangles) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const Vec3i t = triangles[i];
        const Vec3f p0 = points[t[0]];
        const Vec3f faceNormal = (points[t[1]] - p0).cross(points[t[2]] - p0);
        const Vec3f refSum = refNormals[t[0]] + refNormals[t[1]] + refNormals[t[2]];
        if (faceNormal.dot(refSum) < 0.f) triangles[i] = Vec3i(t[0], t[2], t[1]);
    }
};

/// @brief Mark the vertices some triangle references.
struct MarkReferencedFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const Vec3i* triangles, uint32_t* referenced) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const Vec3i t = triangles[i];
        referenced[t[0]] = referenced[t[1]] = referenced[t[2]] = 1u;
    }
};

/// @brief Compact the referenced vertices (keeping their order) with unit normals, and their grids.
struct CompactVerticesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const uint32_t* referenced, const uint32_t* newIndex,
                               const Vec3f* points, const Vec3f* refNormals, const int32_t* vertexGrid,
                               Vec3f* outPoints, Vec3f* outNormals, int32_t* outGrid) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n || !referenced[i]) return;
        const uint32_t k = newIndex[i];
        outPoints[k] = points[i];
        const Vec3f m = refNormals[i];
        outNormals[k] = m / fmaxf(sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]), 1e-12f);
        if (outGrid) outGrid[k] = vertexGrid[i];
    }
};

/// @brief Renumber the triangles onto the compacted vertices and, for a chunk of several grids,
///        onto the vertices of their own grid (whose first vertex is gridPointOffset[g] - chunkPointBase),
///        recording each triangle's grid.
struct RenumberTrianglesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const uint32_t* newIndex, const int32_t* keptGrid,
                               const uint64_t* gridPointOffset, uint64_t chunkPointBase, Vec3i* triangles,
                               int32_t* triangleGrid) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const Vec3i t = triangles[i];
        Vec3i r = Vec3i(int32_t(newIndex[t[0]]), int32_t(newIndex[t[1]]), int32_t(newIndex[t[2]]));
        if (keptGrid) {
            const int32_t g = keptGrid[r[0]];
            const int32_t first = int32_t(gridPointOffset[g] - chunkPointBase);
            r = Vec3i(r[0] - first, r[1] - first, r[2] - first);
            triangleGrid[i] = g;
        }
        triangles[i] = r;
    }
};

/// @brief Per-grid offsets of items sorted by grid: offsets[g] = base + (index of the first item of
///        grid g, or of the first item of a later grid when g has none) for g in [gridBegin, gridEnd).
///        Thread i in [0, n] writes the offsets of the grids from just after item i-1's grid up to
///        item i's grid (thread n: up to gridEnd - 1), so every grid is written exactly once.
struct SegmentOffsetsFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* itemGrid, int32_t gridBegin, int32_t gridEnd,
                               uint64_t base, uint64_t* offsets) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i > n) return;
        const int32_t first = i ? itemGrid[i - 1] + 1 : gridBegin;
        const int32_t last  = i < n ? itemGrid[i] : gridEnd - 1;
        for (int32_t g = first; g <= last; ++g) offsets[g] = base + i;
    }
};

/// @brief out[i] = value for i in [0, n).
template<typename T>
struct FillFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, T value, T* out) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) out[i] = value;
    }
};


//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Decimation (setReduce / setAdaptivity). Instead of one vertex per surface cell, cells are grouped into
// clusters with one vertex each: uniform F x F x F index-space blocks, or, adaptively, coarse blocks whose
// cells' normals agree (flat regions) while the cells of the other blocks stay apart. Each cell's QEF is
// summed into its cluster and one vertex is solved per cluster; quads whose cells collapsed onto one vertex
// then degenerate and are dropped. Every grid has its own range of cluster keys, so clusters never span
// grids. The cluster sums use double atomics, so the summation order, and with it the decimated vertex
// positions to the last bits, can vary from run to run. This is a single-level form of the octree
// simplification of Ju et al. 2002.

/// @brief Per-grid parameters of the cluster keys, one row per grid of the chunk: the grid's key origin
///        (its minimum surface-cell coordinate), its key spans, and the base of its range of
///        2 * fineCount keys (fine per-voxel keys, then the adaptive pass's coarse keys).
enum GridKeyParam : int { KeyOriginX = 0, KeyOriginY, KeyOriginZ, KeySpanY, KeySpanZ, KeyFineCount, KeyBase, GridKeyParamCount };

__device__ inline long long packKey(long long x, long long y, long long z, long long spanY, long long spanZ)
{
    return (x * spanY + y) * spanZ + z;
}

/// @brief A surface cell's coordinate relative to its grid's key origin, and the grid's key parameters.
struct CellKeyFrame
{
    const int64_t* params;
    long long      x, y, z;

    __device__ CellKeyFrame(uint64_t i, const int32_t* surfaceCells, const int32_t* surfaceGrid, int32_t gridBegin,
                            const int32_t* voxelCoord, const int64_t* gridKeyParams)
        : params(gridKeyParams + int64_t(surfaceGrid[i] - gridBegin) * GridKeyParamCount)
    {
        const int64_t cell = surfaceCells[i];
        x = voxelCoord[cell * 3 + 0] - params[KeyOriginX];
        y = voxelCoord[cell * 3 + 1] - params[KeyOriginY];
        z = voxelCoord[cell * 3 + 2] - params[KeyOriginZ];
    }
};

/// @brief Per-grid bounds [min x, y, z, max x, y, z] of the surface-cell coordinates (initialised to
///        +/- INT32_MAX). Surface cells are grouped by grid, so almost every warp holds one grid: it then
///        reduces in registers and issues one atomic per bound rather than 32 contending ones.
struct SurfaceBoundsFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static_assert(MaxThreadsPerBlock % 32 == 0, "the reduction works on whole warps");

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid, int32_t gridBegin,
                               const int32_t* voxelCoord, int32_t* bounds) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int32_t g = surfaceGrid[i];
        const int64_t cell = surfaceCells[i];
        int32_t lo[3], hi[3];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) lo[axis] = hi[axis] = voxelCoord[cell * 3 + axis];
        // every lane of a warp whose last index is in range got past the return above, so the warp
        // can synchronize as a whole
        constexpr unsigned fullMask = 0xffffffffu;
        const bool fullWarp = (i - (threadIdx.x & 31)) + 32 <= n;
        if (fullWarp && __all_sync(fullMask, g == __shfl_sync(fullMask, g, 0))) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    lo[axis] = min(lo[axis], __shfl_xor_sync(fullMask, lo[axis], offset));
                    hi[axis] = max(hi[axis], __shfl_xor_sync(fullMask, hi[axis], offset));
                }
            }
            if ((threadIdx.x & 31) != 0) return;
        }
        int32_t* gridBounds = bounds + int64_t(g - gridBegin) * 6;
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            atomicMin(&gridBounds[axis], lo[axis]);
            atomicMax(&gridBounds[3 + axis], hi[axis]);
        }
    }
};

/// @brief The key parameters of each grid of the chunk from its bounds; also writes the size of its
///        key range (2 * fineCount, 0 for a grid without surface cells) for the scan into key bases.
struct GridKeyParamsFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* bounds, int64_t* gridKeyParams, int64_t* keyRange) const
    {
        const uint64_t g = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (g >= n) return;
        const int32_t* b = bounds + g * 6;
        int64_t* params = gridKeyParams + g * GridKeyParamCount;
        int64_t span[3];
        for (int axis = 0; axis < 3; ++axis) {
            params[KeyOriginX + axis] = b[axis];
            span[axis] = int64_t(b[3 + axis]) - int64_t(b[axis]) + 1;
            if (span[axis] < 0) span[axis] = 0; // no surface cells
        }
        params[KeySpanY] = span[1];
        params[KeySpanZ] = span[2];
        params[KeyFineCount] = span[0] * span[1] * span[2];
        keyRange[g] = 2 * params[KeyFineCount];
    }
};

/// @brief Initialise the per-grid bounds to [+INT32_MAX x 3, -INT32_MAX - 1 x 3].
struct InitBoundsFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, int32_t* bounds) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) bounds[i] = (i % 6) < 3 ? 2147483647 : -2147483647 - 1;
    }
};

/// @brief Copy the scanned key bases into the key parameters.
struct SetKeyBaseFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int64_t* keyBase, int64_t* gridKeyParams) const
    {
        const uint64_t g = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (g < n) gridKeyParams[g * GridKeyParamCount + KeyBase] = keyBase[g];
    }
};

/// @brief The uniform-block cluster key of each surface cell (also the coarse-block key of the
///        adaptive pass), in its grid's key range.
struct ClusterKeyFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid, int32_t gridBegin,
                               const int32_t* voxelCoord, const int64_t* gridKeyParams, int blockSize, int64_t* keys) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const CellKeyFrame f(i, surfaceCells, surfaceGrid, gridBegin, voxelCoord, gridKeyParams);
        keys[i] = f.params[KeyBase] + packKey(f.x / blockSize, f.y / blockSize, f.z / blockSize,
                                              f.params[KeySpanY], f.params[KeySpanZ]);
    }
};

/// @brief The adaptive cluster key: a cell of a flat block takes the block's coarse key (offset past
///        the fine keys, so the two never collide) and collapses with it; any other cell keeps its own
///        fine key and stays at full resolution.
struct ClusterKeyAdaptiveFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid, int32_t gridBegin,
                               const int32_t* voxelCoord, const int64_t* gridKeyParams, int blockSize,
                               const int32_t* blockOfCell, const uint8_t* blockIsFlat, int64_t* keys) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const CellKeyFrame f(i, surfaceCells, surfaceGrid, gridBegin, voxelCoord, gridKeyParams);
        const long long spanY = f.params[KeySpanY], spanZ = f.params[KeySpanZ];
        if (blockIsFlat[blockOfCell[i]])
            keys[i] = f.params[KeyBase] + f.params[KeyFineCount] +
                      packKey(f.x / blockSize, f.y / blockSize, f.z / blockSize, spanY, spanZ);
        else
            keys[i] = f.params[KeyBase] + packKey(f.x, f.y, f.z, spanY, spanZ);
    }
};

/// @brief Head flags of a sorted key array: 1 where a key differs from its predecessor.
struct HeadFlagsFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int64_t* sortedKeys, int32_t* flags) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) flags[i] = (i == 0 || sortedKeys[i] != sortedKeys[i - 1]) ? 1 : 0;
    }
};

/// @brief label[sortedIndex[i]] = rank[i] - 1: each element's dense rank among the distinct keys.
struct ScatterRanksFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* sortedIndex, const int32_t* rank, int32_t* label) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) label[sortedIndex[i]] = rank[i] - 1;
    }
};

/// @brief Each surface cell's flatness signal: the normalized sum of its unit edge-crossing normals,
///        near unit length on a flat patch and shorter where the normals disagree.
struct CellNormalFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* neighborTable, const float* value,
                               const float* gradient, BoxStencilGeometry geometry, float iso, Vec3f* cellNormals) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float cornerValue[8], cornerGradient[8][3];
        loadCellCorners(surfaceCells[i], neighborTable, value, gradient, geometry, cornerValue, cornerGradient);
        float sum[3] = {0, 0, 0};
#pragma unroll
        for (int edge = 0; edge < 12; ++edge) {
            EdgeCrossing ec;
            if (!computeEdgeCrossing(edge, cornerValue, cornerGradient, geometry, iso, ec)) continue;
            sum[0] += ec.normal[0];
            sum[1] += ec.normal[1];
            sum[2] += ec.normal[2];
        }
        const float invLen = rsqrtf(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2] + 1e-24f);
        cellNormals[i] = Vec3f(sum[0] * invLen, sum[1] * invLen, sum[2] * invLen);
    }
};

/// @brief Sum the cell normals and count the cells of each coarse block.
struct BlockNormalSumFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* blockOfCell, const Vec3f* cellNormals, float* blockNormalSum,
                               uint32_t* blockCellCount) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int64_t b = blockOfCell[i];
        atomicAdd(&blockNormalSum[b * 3 + 0], cellNormals[i][0]);
        atomicAdd(&blockNormalSum[b * 3 + 1], cellNormals[i][1]);
        atomicAdd(&blockNormalSum[b * 3 + 2], cellNormals[i][2]);
        atomicAdd(&blockCellCount[b], 1u);
    }
};

/// @brief A coarse block is flat when the mean of its cells' normals is at least cosThreshold long.
struct BlockFlatFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const float* blockNormalSum, const uint32_t* blockCellCount,
                               float cosThreshold, uint8_t* blockIsFlat) const
    {
        const uint64_t b = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (b >= n) return;
        const float x = blockNormalSum[b * 3 + 0], y = blockNormalSum[b * 3 + 1], z = blockNormalSum[b * 3 + 2];
        const float count = float(blockCellCount[b] ? blockCellCount[b] : 1u);
        blockIsFlat[b] = (sqrtf(x * x + y * y + z * z) / count >= cosThreshold) ? 1u : 0u;
    }
};

/// @brief cellVertex[surfaceCells[i]] = clusterId[i], and the grid of each cluster (all cells of a
///        cluster are in one grid, so the concurrent writes agree).
struct ClusterVertexFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid,
                               const int32_t* clusterId, int32_t* cellVertex, int32_t* clusterGrid) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        cellVertex[surfaceCells[i]] = clusterId[i];
        clusterGrid[clusterId[i]] = surfaceGrid[i];
    }
};

/// @brief The per-cluster QEF: each surface cell adds the terms of its edge crossings, expressed in
///        its grid's key frame (shared by the cells of a cluster), to its cluster's sums.
/// @note  sums holds 16 doubles per cluster: A (6), b (3), crossing sum (3), crossing count (1),
///        normal sum (3).
struct ClusterQefFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* surfaceCells, const int32_t* surfaceGrid, int32_t gridBegin,
                               const int32_t* clusterId, const int32_t* neighborTable, const float* value,
                               const float* gradient, const int32_t* voxelCoord, BoxStencilGeometry geometry,
                               float iso, const int64_t* gridKeyParams, double* sums) const
    {
        const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;
        float cornerValue[8], cornerGradient[8][3];
        loadCellCorners(surfaceCells[i], neighborTable, value, gradient, geometry, cornerValue, cornerGradient);
        const CellKeyFrame f(i, surfaceCells, surfaceGrid, gridBegin, voxelCoord, gridKeyParams);
        const double ox = double(f.x), oy = double(f.y), oz = double(f.z);
        double local[16] = {};
        int crossingCount = 0;
#pragma unroll
        for (int edge = 0; edge < 12; ++edge) {
            EdgeCrossing ec;
            if (!computeEdgeCrossing(edge, cornerValue, cornerGradient, geometry, iso, ec)) continue;
            const float  nx = ec.normal[0], ny = ec.normal[1], nz = ec.normal[2];
            const double px = ox + ec.point[0], py = oy + ec.point[1], pz = oz + ec.point[2];
            const double np = nx * px + ny * py + nz * pz;
            local[0] += nx * nx; local[1] += nx * ny; local[2] += nx * nz;
            local[3] += ny * ny; local[4] += ny * nz; local[5] += nz * nz;
            local[6] += nx * np; local[7] += ny * np; local[8] += nz * np;
            local[9] += px; local[10] += py; local[11] += pz;
            local[13] += nx; local[14] += ny; local[15] += nz;
            ++crossingCount;
        }
        local[12] = double(crossingCount);
        double* clusterSums = sums + int64_t(clusterId[i]) * 16;
        for (int j = 0; j < 16; ++j) atomicAdd(&clusterSums[j], local[j]);
    }
};

/// @brief One vertex per cluster from its summed QEF (centroid-regularised as for single cells, then
///        clamped to within blockSize of the crossing centroid), with the summed normal as its
///        reference normal.
template<typename ViewT, typename XformT>
struct ClusterSolveFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(uint64_t n, const int32_t* clusterGrid, int32_t gridBegin, const double* sums,
                               const int64_t* gridKeyParams, int blockSize, ViewT view, XformT xform, Vec3f* points,
                               Vec3f* refNormals) const
    {
        const uint64_t c = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (c >= n) return;
        const double* s = sums + c * 16;
        const double invCount = s[12] > 0 ? 1.0 / s[12] : 0.0;
        const double centroid[3] = {s[9] * invCount, s[10] * invCount, s[11] * invCount};
        const double regularized[6] = {s[0] + 0.05, s[1], s[2], s[3] + 0.05, s[4], s[5] + 0.05};
        const double rhs[3] = {s[6] + 0.05 * centroid[0], s[7] + 0.05 * centroid[1], s[8] + 0.05 * centroid[2]};
        double x[3];
        solveSymmetric3x3(regularized, rhs, x);
        for (int axis = 0; axis < 3; ++axis)
            x[axis] = fmin(fmax(x[axis], centroid[axis] - blockSize), centroid[axis] + blockSize);
        // from the grid's key frame back to index space
        const uint32_t g = uint32_t(clusterGrid[c]);
        const int64_t* params = gridKeyParams + int64_t(g - gridBegin) * GridKeyParamCount;
        const Vec3f ijk(float(x[0] + params[KeyOriginX]), float(x[1] + params[KeyOriginY]), float(x[2] + params[KeyOriginZ]));
        points[c] = xform(view.entry(g).grid, g, ijk);
        refNormals[c] = Vec3f(float(s[13]), float(s[14]), float(s[15]));
    }
};

} // namespace dualcontour::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// ndef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

/// @brief Dual-contouring mesh extraction from a scalar field on the active voxels of one
///        ValueOnIndex grid, or of a batch of them.
/// @tparam BuildT    NanoVDB build type; only ValueOnIndex is supported.
/// @tparam ResourceT Stream-ordered resource all device scratch and the output are allocated from.
template<typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class DualContour
{
    static_assert(std::is_same<BuildT, ValueOnIndex>::value, "DualContour requires a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "DualContour allocates stream-ordered scratch and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "DualContour stores scratch up to uint64_t width and requires word-aligned allocations");

    template<typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
    using GridT = NanoGrid<BuildT>;

public:
    using MeshT = DualContourMesh<ResourceT>;

    /// @brief Log2 of the VoxelBlockManager block width the single-grid getMesh uses
    static constexpr int DefaultLog2BlockWidth = 7;

    /// @param stream   CUDA stream all work is ordered on
    /// @param resource resource all device scratch and the output are allocated from; it must
    ///                 outlive this operator and the meshes it returns
    explicit DualContour(cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream), mResource(&resource), mTempDevicePool(resource) {}

    /// @brief Set the iso value of the extracted surface (default 0)
    void setIsoValue(float iso) { mIso = iso; }

    /// @brief Set uniform cluster-collapse decimation: cells are grouped into reduce^3 index-space blocks
    ///        with one vertex each (default 1, i.e. no decimation unless adaptivity > 0). With
    ///        adaptivity > 0 it sets the coarse-block size instead (8 when reduce <= 1).
    void setReduce(int reduce) { mReduce = reduce > 1 ? reduce : 1; }

    /// @brief Set curvature-adaptive decimation in [0, 1.5] (default 0, off): coarse blocks whose cells'
    ///        normals agree within adaptivity * 60 degrees collapse to one vertex, while the cells of the
    ///        other blocks keep full resolution. Decimated vertex positions are summed with double
    ///        atomics, so their last bits can vary from run to run.
    void setAdaptivity(float adaptivity) { mAdaptivity = adaptivity < 0.f ? 0.f : (adaptivity > 1.5f ? 1.5f : adaptivity); }

    /// @brief Set the most active voxels meshed together (default 2^22); consecutive grids are
    ///        grouped into chunks of at most this many voxels, a larger grid forming its own chunk.
    ///        The per-voxel scratch (~90 bytes per voxel) is sized by the largest chunk.
    void setMaxChunkVoxels(uint64_t maxChunkVoxels) { mMaxChunkVoxels = maxChunkVoxels ? maxChunkVoxels : 1; }

    /// @brief Mesh a single grid.
    /// @param d_grid   Device ValueOnIndex grid without active tiles that indexes exactly its
    ///                 active voxels (as NanoVDB's CUDA builders produce)
    /// @param d_field  Device sidecar indexed by value index: d_field[v] is the value of the
    ///                 voxel with value index v >= 1 (d_field[0] is not read)
    /// @param xform    Vertex transform from index space (default: the grid's map)
    template<typename XformT = DualContourIndexToWorld>
    MeshT getMesh(const GridT* d_grid, const float* d_field, const XformT& xform = XformT());

    /// @brief Mesh a batch of grids through their batched VoxelBlockManager.
    /// @param vbm           View of the batch's VoxelBlockManager build, usable on the device
    /// @param hostEntries   The same entry source on the host (or any host entry source giving
    ///                      the same voxelCount(g) and blockBase(g)); used to plan the chunks
    /// @param d_field       Device array of the grids' values
    /// @param d_fieldOffsets Optional device array of gridCount offsets: the voxel with value index
    ///                      v >= 1 of grid g has value d_field[d_fieldOffsets[g] + v]. If null, the
    ///                      grids' values are packed back to back without background entries:
    ///                      grid g's start at the sum of the preceding grids' voxel counts.
    /// @param xform         Vertex transform from index space (default: each grid's map)
    template<int Log2BlockWidth, typename EntrySource, typename HostEntrySource,
             typename XformT = DualContourIndexToWorld>
    MeshT getMesh(const VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>& vbm,
                  const HostEntrySource& hostEntries, const float* d_field,
                  const int64_t* d_fieldOffsets = nullptr, const XformT& xform = XformT());

private:
    /// @brief One chunk's meshes, grouped by grid; triangles index their own grid's vertices.
    struct ChunkMesh
    {
        BufT<Vec3f> points, normals;
        BufT<Vec3i> triangles;
    };

    template<int Log2BlockWidth, typename EntrySource, typename XformT>
    void meshChunk(const VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>& vbm, uint32_t gridBegin,
                   uint32_t gridEnd, uint64_t vbBegin, uint64_t vbEnd, uint64_t chunkVoxelBase, uint64_t chunkVoxelCount,
                   const uint64_t* d_voxelBase, const float* d_field, const int64_t* d_fieldOffsets,
                   const XformT& xform, uint64_t& pointBase, uint64_t& triangleBase, uint64_t* d_pointOffsets,
                   uint64_t* d_triangleOffsets, std::vector<ChunkMesh>& chunks);

    /// @brief label[i] = the dense rank of keys[i] among the distinct keys; returns their count
    uint64_t denseRelabel(const int64_t* keys, uint64_t n, int32_t* label);

    /// @brief Launch functor Op over @a threads threads (none when 0) with the given arguments
    template<typename Op, typename... Args>
    void launch(uint64_t threads, Args... args)
    {
        if (threads == 0) return;
        util::cuda::operatorKernel<Op>
            <<<util::cuda::blocksPerGrid(threads, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(args...);
        cudaCheckError();
    }

    template<typename T>
    BufT<T> alloc(uint64_t count) { return BufT<T>(mStream, nanovdb::cuda::ResourceRef<ResourceT>(*mResource), count, nanovdb::cuda::noInit); }

    template<typename T>
    T readBack(const T* d_value)
    {
        T value;
        cudaCheck(cudaMemcpyAsync(&value, d_value, sizeof(T), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        return value;
    }

    cudaStream_t                       mStream{0};
    ResourceT*                         mResource; // non-owning
    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;
    float                              mIso{0.f};
    int                                mReduce{1};
    float                              mAdaptivity{0.f};
    uint64_t                           mMaxChunkVoxels{uint64_t(1) << 22};
}; // tools::cuda::DualContour<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename XformT>
DualContourMesh<ResourceT> DualContour<BuildT, ResourceT>::getMesh(const GridT* d_grid, const float* d_field,
                                                                   const XformT& xform)
{
    constexpr int L = DefaultLog2BlockWidth;
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid, mStream);
    if (treeData.mTileCount[0] || treeData.mTileCount[1] || treeData.mTileCount[2])
        throw std::runtime_error("DualContour: grids with active tiles are not supported");

    // a batch of one grid
    const uint64_t voxelCount = treeData.mVoxelCount;
    const uint64_t blockCount = (voxelCount + (uint64_t(1) << L) - 1) >> L;
    const VoxelBlockManagerBatchEntry hostEntry{d_grid, voxelCount, 0, 0};
    auto entries = this->template alloc<VoxelBlockManagerBatchEntry>(1);
    auto firstLeafID = this->template alloc<uint32_t>(blockCount);
    auto jumpMap = this->template alloc<uint64_t>(blockCount * VoxelBlockManager<L>::JumpMapLength);
    auto fieldOffset = this->template alloc<int64_t>(1);
    const int64_t zero = 0;
    cudaCheck(cudaMemcpyAsync(entries.data(), &hostEntry, sizeof(hostEntry), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaMemcpyAsync(fieldOffset.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice, mStream));
    buildVoxelBlockManagerBatch<L>(entries.data(), 1u, treeData.mNodeCount[0], blockCount, firstLeafID.data(),
                                   jumpMap.data(), mStream);
    const VoxelBlockManagerBatchView<L> view(VoxelBlockManagerBatchEntryArray{entries.data()}, 1u, blockCount,
                                             firstLeafID.data(), jumpMap.data());
    return this->getMesh(view, VoxelBlockManagerBatchEntryArray{&hostEntry}, d_field, fieldOffset.data(), xform);
}

template<typename BuildT, typename ResourceT>
template<int Log2BlockWidth, typename EntrySource, typename HostEntrySource, typename XformT>
DualContourMesh<ResourceT> DualContour<BuildT, ResourceT>::getMesh(
    const VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>& vbm, const HostEntrySource& hostEntries,
    const float* d_field, const int64_t* d_fieldOffsets, const XformT& xform)
{
    using nanovdb::tools::cuda::detail::entryBlockBase;
    using nanovdb::tools::cuda::detail::entryVoxelCount;
    const uint32_t gridCount = vbm.gridCount();
    if (gridCount >= uint32_t(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("DualContour: too many grids");

    // the grids' first batch-wide voxel indices (and the total), and the packed field offsets
    std::vector<uint64_t> voxelBase(gridCount + 1, 0), blockBase(gridCount + 1, 0);
    for (uint32_t g = 0; g < gridCount; ++g) {
        voxelBase[g + 1] = voxelBase[g] + entryVoxelCount(hostEntries, g);
        blockBase[g] = entryBlockBase(hostEntries, g);
    }
    blockBase[gridCount] = vbm.blockCount();
    auto d_voxelBase = this->template alloc<uint64_t>(gridCount + 1);
    cudaCheck(cudaMemcpyAsync(d_voxelBase.data(), voxelBase.data(), (gridCount + 1) * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, mStream));
    auto packedFieldOffsets = this->template alloc<int64_t>(0);
    if (!d_fieldOffsets && gridCount) {
        std::vector<int64_t> offsets(gridCount);
        for (uint32_t g = 0; g < gridCount; ++g) offsets[g] = int64_t(voxelBase[g]) - 1;
        packedFieldOffsets = this->template alloc<int64_t>(gridCount);
        cudaCheck(cudaMemcpyAsync(packedFieldOffsets.data(), offsets.data(), gridCount * sizeof(int64_t),
                                  cudaMemcpyHostToDevice, mStream));
        d_fieldOffsets = packedFieldOffsets.data();
    }

    MeshT mesh(mStream, nanovdb::cuda::ResourceRef<ResourceT>(*mResource));
    mesh.pointOffsets = this->template alloc<uint64_t>(gridCount + 1);
    mesh.triangleOffsets = this->template alloc<uint64_t>(gridCount + 1);

    // greedily group consecutive grids into chunks of at most mMaxChunkVoxels voxels
    std::vector<ChunkMesh> chunks;
    uint64_t pointBase = 0, triangleBase = 0;
    for (uint32_t gridBegin = 0; gridBegin < gridCount;) {
        uint32_t gridEnd = gridBegin + 1;
        while (gridEnd < gridCount && voxelBase[gridEnd + 1] - voxelBase[gridBegin] <= mMaxChunkVoxels) ++gridEnd;
        const uint64_t chunkVoxelCount = voxelBase[gridEnd] - voxelBase[gridBegin];
        if (chunkVoxelCount + 1 > uint64_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("DualContour: a grid has too many active voxels for 32-bit chunk indexing");
        this->meshChunk(vbm, gridBegin, gridEnd, blockBase[gridBegin], blockBase[gridEnd], voxelBase[gridBegin],
                        chunkVoxelCount, d_voxelBase.data(), d_field, d_fieldOffsets, xform, pointBase, triangleBase,
                        mesh.pointOffsets.data(), mesh.triangleOffsets.data(), chunks);
        gridBegin = gridEnd;
    }
    this->template launch<dualcontour::detail::FillFunctor<uint64_t>>(1, uint64_t(1), pointBase, mesh.pointOffsets.data() + gridCount);
    this->template launch<dualcontour::detail::FillFunctor<uint64_t>>(1, uint64_t(1), triangleBase, mesh.triangleOffsets.data() + gridCount);

    // the chunks' meshes back to back (each is already grouped by grid, in grid order)
    if (chunks.size() == 1) {
        mesh.points = std::move(chunks[0].points);
        mesh.normals = std::move(chunks[0].normals);
        mesh.triangles = std::move(chunks[0].triangles);
    } else {
        mesh.points = this->template alloc<Vec3f>(pointBase);
        mesh.normals = this->template alloc<Vec3f>(pointBase);
        mesh.triangles = this->template alloc<Vec3i>(triangleBase);
        uint64_t p = 0, t = 0;
        for (ChunkMesh& chunk : chunks) {
            const uint64_t np = chunk.points.size(), nt = chunk.triangles.size();
            if (np) {
                cudaCheck(cudaMemcpyAsync(mesh.points.data() + p, chunk.points.data(), np * sizeof(Vec3f), cudaMemcpyDeviceToDevice, mStream));
                cudaCheck(cudaMemcpyAsync(mesh.normals.data() + p, chunk.normals.data(), np * sizeof(Vec3f), cudaMemcpyDeviceToDevice, mStream));
            }
            if (nt) cudaCheck(cudaMemcpyAsync(mesh.triangles.data() + t, chunk.triangles.data(), nt * sizeof(Vec3i), cudaMemcpyDeviceToDevice, mStream));
            p += np;
            t += nt;
        }
    }
    mesh.hostPointOffsets.resize(gridCount + 1);
    mesh.hostTriangleOffsets.resize(gridCount + 1);
    cudaCheck(cudaMemcpyAsync(mesh.hostPointOffsets.data(), mesh.pointOffsets.data(), (gridCount + 1) * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaMemcpyAsync(mesh.hostTriangleOffsets.data(), mesh.triangleOffsets.data(),
                              (gridCount + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    return mesh;
}

template<typename BuildT, typename ResourceT>
uint64_t DualContour<BuildT, ResourceT>::denseRelabel(const int64_t* keys, uint64_t n, int32_t* label)
{
    namespace dc = dualcontour::detail;
    auto sortedKeys = this->template alloc<int64_t>(n);
    auto index = this->template alloc<int32_t>(n);
    auto sortedIndex = this->template alloc<int32_t>(n);
    auto rank = this->template alloc<int32_t>(n);
    this->template launch<dc::CellVertexFunctor>(n, uint64_t(n), static_cast<const int32_t*>(nullptr), index.data());
    CALL_CUBS(DeviceRadixSort::SortPairs, keys, sortedKeys.data(), index.data(), sortedIndex.data(), int64_t(n), 0,
              int(sizeof(int64_t) * 8));
    this->template launch<dc::HeadFlagsFunctor>(n, uint64_t(n), sortedKeys.data(), rank.data());
    CALL_CUBS(DeviceScan::InclusiveSum, rank.data(), rank.data(), int64_t(n));
    this->template launch<dc::ScatterRanksFunctor>(n, uint64_t(n), sortedIndex.data(), rank.data(), label);
    return uint64_t(this->readBack(rank.data() + n - 1)); // host sync
}

/// @brief Mesh the consecutive grids [gridBegin, gridEnd). Pipeline:
///   1. gather: one VBM decode per voxel -> neighbor table, value, gradient, surface flag, coordinate
///   2. compact the surface cells and find each one's grid
///   3. place one vertex per surface cell (QEF), or when decimating group the cells into
///      per-grid clusters and solve one vertex per cluster
///   4. connectivity: one quad per bipolar minimal edge, split into triangles (count -> scan ->
///      write), oriented by the gradient
///   5. drop unreferenced vertices, normalize the normals, renumber the triangles onto their
///      grid's vertices and record the per-grid offsets
/// Surface cells, and hence vertices and triangles, are ordered by slot and so grouped by grid,
/// and a grid's part is the same as when it is meshed alone. Host syncs: steps 2, 4 and 5 once each,
/// plus one or two cluster relabels when decimating.
template<typename BuildT, typename ResourceT>
template<int Log2BlockWidth, typename EntrySource, typename XformT>
void DualContour<BuildT, ResourceT>::meshChunk(
    const VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>& vbm, uint32_t gridBegin, uint32_t gridEnd,
    uint64_t vbBegin, uint64_t vbEnd, uint64_t chunkVoxelBase, uint64_t chunkVoxelCount, const uint64_t* d_voxelBase,
    const float* d_field, const int64_t* d_fieldOffsets, const XformT& xform, uint64_t& pointBase,
    uint64_t& triangleBase, uint64_t* d_pointOffsets, uint64_t* d_triangleOffsets, std::vector<ChunkMesh>& chunks)
{
    namespace dc = dualcontour::detail;
    constexpr dc::BoxStencilGeometry geometry = dc::makeBoxStencilGeometry();
    const uint32_t chunkGridCount = gridEnd - gridBegin;
    auto emptyChunk = [&] { // every grid of the chunk gets an empty mesh
        this->template launch<dc::FillFunctor<uint64_t>>(chunkGridCount, uint64_t(chunkGridCount), pointBase, d_pointOffsets + gridBegin);
        this->template launch<dc::FillFunctor<uint64_t>>(chunkGridCount, uint64_t(chunkGridCount), triangleBase, d_triangleOffsets + gridBegin);
    };
    if (chunkVoxelCount == 0 || vbEnd <= vbBegin) return emptyChunk();
    if (vbEnd - vbBegin > uint64_t(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("DualContour: too many VoxelBlockManager blocks in one chunk");

    // 1. the gather, one CUDA block per VBM block of the chunk
    const uint64_t valueCount = chunkVoxelCount + 1; // slot 0 is the background
    const int numColumns = geometry.numColumns;
    auto neighborTable = this->template alloc<int32_t>(valueCount * numColumns);
    auto value = this->template alloc<float>(valueCount);
    auto gradient = this->template alloc<float>(valueCount * 3);
    auto surfaceFlag = this->template alloc<uint8_t>(valueCount);
    auto voxelCoord = this->template alloc<int32_t>(valueCount * 3);
    using Gather = dc::GatherFunctor<Log2BlockWidth, EntrySource>;
    util::cuda::operatorKernel<Gather><<<uint32_t(vbEnd - vbBegin), Gather::MaxThreadsPerBlock, 0, mStream>>>(
        vbm, vbBegin, chunkVoxelBase, d_voxelBase, d_field, d_fieldOffsets, geometry, mIso, neighborTable.data(),
        value.data(), gradient.data(), surfaceFlag.data(), voxelCoord.data());
    cudaCheckError();

    // 2. the surface cells, in ascending slot order (grouped by grid), and their grids
    auto surfaceCells = this->template alloc<int32_t>(valueCount);
    auto count = this->template alloc<uint64_t>(1);
    CALL_CUBS(DeviceSelect::Flagged, ::cuda::counting_iterator<int32_t>(0), surfaceFlag.data(), surfaceCells.data(),
              count.data(), int64_t(valueCount));
    const uint64_t surfaceCellCount = this->readBack(count.data()); // host sync
    if (surfaceCellCount == 0) return emptyChunk();
    const bool singleGrid = chunkGridCount == 1;
    BufT<int32_t> surfaceGrid = this->template alloc<int32_t>(surfaceCellCount);
    if (singleGrid) {
        this->template launch<dc::FillFunctor<int32_t>>(surfaceCellCount, uint64_t(surfaceCellCount),
                                                        int32_t(gridBegin), surfaceGrid.data());
    } else {
        this->template launch<dc::SurfaceGridFunctor>(surfaceCellCount, uint64_t(surfaceCellCount), surfaceCells.data(), d_voxelBase, gridBegin,
                                                      gridEnd, chunkVoxelBase, surfaceGrid.data());
    }

    // 3. one vertex per surface cell, or per cluster of surface cells when decimating
    auto cellVertex = this->template alloc<int32_t>(valueCount);
    cudaCheck(cudaMemsetAsync(cellVertex.data(), 0xff, valueCount * sizeof(int32_t), mStream)); // -1
    using ViewT = VoxelBlockManagerBatchView<Log2BlockWidth, EntrySource>;
    const bool decimate = mReduce != 1 || mAdaptivity > 0.f;
    uint64_t vertexCount = surfaceCellCount;
    auto points = this->template alloc<Vec3f>(0);
    auto refNormals = this->template alloc<Vec3f>(0);
    auto clusterGrid = this->template alloc<int32_t>(0);
    if (!decimate) {
        this->template launch<dc::CellVertexFunctor>(surfaceCellCount, uint64_t(surfaceCellCount), surfaceCells.data(),
                                                     cellVertex.data());
        points = this->template alloc<Vec3f>(vertexCount);
        refNormals = this->template alloc<Vec3f>(vertexCount);
        this->template launch<dc::PlaceVerticesFunctor<ViewT, XformT>>(
            surfaceCellCount, uint64_t(surfaceCellCount), surfaceCells.data(), surfaceGrid.data(), neighborTable.data(),
            value.data(), gradient.data(), voxelCoord.data(), geometry, mIso, vbm, xform, points.data(), refNormals.data());
    } else {
        // cluster blocks: uniform reduce^3 blocks, or 8^3 coarse blocks for the adaptive collapse
        const int blockSize = (mAdaptivity > 0.f && mReduce <= 1) ? 8 : mReduce;
        const int32_t gb = int32_t(gridBegin);
        const uint64_t n = surfaceCellCount;
        // per-grid key parameters, from the bounds of each grid's surface cells
        auto bounds = this->template alloc<int32_t>(uint64_t(chunkGridCount) * 6);
        this->template launch<dc::InitBoundsFunctor>(uint64_t(chunkGridCount) * 6, uint64_t(chunkGridCount) * 6, bounds.data());
        this->template launch<dc::SurfaceBoundsFunctor>(n, n, surfaceCells.data(), surfaceGrid.data(), gb,
                                                        voxelCoord.data(), bounds.data());
        auto gridKeyParams = this->template alloc<int64_t>(uint64_t(chunkGridCount) * dc::GridKeyParamCount);
        auto keyRange = this->template alloc<int64_t>(chunkGridCount);
        auto keyBase = this->template alloc<int64_t>(chunkGridCount);
        this->template launch<dc::GridKeyParamsFunctor>(chunkGridCount, uint64_t(chunkGridCount), bounds.data(),
                                                        gridKeyParams.data(), keyRange.data());
        CALL_CUBS(DeviceScan::ExclusiveSum, keyRange.data(), keyBase.data(), int64_t(chunkGridCount));
        this->template launch<dc::SetKeyBaseFunctor>(chunkGridCount, uint64_t(chunkGridCount), keyBase.data(),
                                                     gridKeyParams.data());

        auto keys = this->template alloc<int64_t>(n);
        auto clusterId = this->template alloc<int32_t>(n);
        this->template launch<dc::ClusterKeyFunctor>(n, n, surfaceCells.data(), surfaceGrid.data(), gb, voxelCoord.data(),
                                                     gridKeyParams.data(), blockSize, keys.data());
        if (mAdaptivity > 0.f) { // flat coarse blocks collapse, the cells of the others stay apart
            auto cellNormals = this->template alloc<Vec3f>(n);
            this->template launch<dc::CellNormalFunctor>(n, n, surfaceCells.data(), neighborTable.data(), value.data(),
                                                         gradient.data(), geometry, mIso, cellNormals.data());
            auto blockOfCell = this->template alloc<int32_t>(n);
            const uint64_t blockCount = this->denseRelabel(keys.data(), n, blockOfCell.data()); // host sync
            auto blockNormalSum = this->template alloc<float>(blockCount * 3);
            auto blockCellCount = this->template alloc<uint32_t>(blockCount);
            auto blockIsFlat = this->template alloc<uint8_t>(blockCount);
            cudaCheck(cudaMemsetAsync(blockNormalSum.data(), 0, blockCount * 3 * sizeof(float), mStream));
            cudaCheck(cudaMemsetAsync(blockCellCount.data(), 0, blockCount * sizeof(uint32_t), mStream));
            this->template launch<dc::BlockNormalSumFunctor>(n, n, blockOfCell.data(), cellNormals.data(),
                                                             blockNormalSum.data(), blockCellCount.data());
            const float cosThreshold = float(std::cos(double(mAdaptivity) * 60.0 * 3.14159265358979323846 / 180.0));
            this->template launch<dc::BlockFlatFunctor>(blockCount, blockCount, blockNormalSum.data(),
                                                        blockCellCount.data(), cosThreshold, blockIsFlat.data());
            this->template launch<dc::ClusterKeyAdaptiveFunctor>(n, n, surfaceCells.data(), surfaceGrid.data(), gb,
                                                                 voxelCoord.data(), gridKeyParams.data(), blockSize,
                                                                 blockOfCell.data(), blockIsFlat.data(), keys.data());
        }
        vertexCount = this->denseRelabel(keys.data(), n, clusterId.data()); // host sync
        // clusters are numbered in key order, and so grouped by grid like the surface cells
        clusterGrid = this->template alloc<int32_t>(vertexCount);
        this->template launch<dc::ClusterVertexFunctor>(n, n, surfaceCells.data(), surfaceGrid.data(), clusterId.data(),
                                                        cellVertex.data(), clusterGrid.data());
        auto sums = this->template alloc<double>(vertexCount * 16);
        cudaCheck(cudaMemsetAsync(sums.data(), 0, vertexCount * 16 * sizeof(double), mStream));
        this->template launch<dc::ClusterQefFunctor>(n, n, surfaceCells.data(), surfaceGrid.data(), gb, clusterId.data(),
                                                     neighborTable.data(), value.data(), gradient.data(),
                                                     voxelCoord.data(), geometry, mIso, gridKeyParams.data(), sums.data());
        points = this->template alloc<Vec3f>(vertexCount);
        refNormals = this->template alloc<Vec3f>(vertexCount);
        this->template launch<dc::ClusterSolveFunctor<ViewT, XformT>>(vertexCount, vertexCount, clusterGrid.data(), gb,
                                                                      sums.data(), gridKeyParams.data(), blockSize, vbm,
                                                                      xform, points.data(), refNormals.data());
    }
    const int32_t* vertexGrid = decimate ? clusterGrid.data() : surfaceGrid.data();

    // 4. connectivity: count -> exclusive scan -> write, then orient
    auto triangleCounts = this->template alloc<uint32_t>(surfaceCellCount);
    auto triangleOffsets = this->template alloc<uint64_t>(surfaceCellCount + 1);
    this->template launch<dc::CountTrianglesFunctor>(surfaceCellCount, uint64_t(surfaceCellCount), surfaceCells.data(), neighborTable.data(),
                                                     value.data(), cellVertex.data(), geometry, mIso,
                                                     triangleCounts.data());
    cudaCheck(cudaMemsetAsync(triangleOffsets.data(), 0, sizeof(uint64_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum, triangleCounts.data(), triangleOffsets.data() + 1, int64_t(surfaceCellCount));
    const uint64_t triangleCount = this->readBack(triangleOffsets.data() + surfaceCellCount); // host sync
    if (triangleCount == 0) return emptyChunk();
    auto triangles = this->template alloc<Vec3i>(triangleCount);
    this->template launch<dc::WriteTrianglesFunctor>(surfaceCellCount, uint64_t(surfaceCellCount), surfaceCells.data(), neighborTable.data(),
                                                     value.data(), cellVertex.data(), geometry, mIso,
                                                     triangleOffsets.data(), triangles.data());
    this->template launch<dc::OrientTrianglesFunctor>(triangleCount, uint64_t(triangleCount), points.data(), refNormals.data(), triangles.data());

    // 5. drop the vertices no triangle references, keeping the order
    auto referenced = this->template alloc<uint32_t>(vertexCount);
    auto newIndex = this->template alloc<uint32_t>(vertexCount + 1);
    cudaCheck(cudaMemsetAsync(referenced.data(), 0, vertexCount * sizeof(uint32_t), mStream));
    this->template launch<dc::MarkReferencedFunctor>(triangleCount, uint64_t(triangleCount), triangles.data(), referenced.data());
    cudaCheck(cudaMemsetAsync(newIndex.data(), 0, sizeof(uint32_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum, referenced.data(), newIndex.data() + 1, int64_t(vertexCount));
    const uint64_t keptCount = this->readBack(newIndex.data() + vertexCount); // host sync

    ChunkMesh chunk{this->template alloc<Vec3f>(keptCount), this->template alloc<Vec3f>(keptCount),
                    this->template alloc<Vec3i>(0)};
    auto keptGrid = this->template alloc<int32_t>(singleGrid ? 0 : keptCount);
    auto triangleGrid = this->template alloc<int32_t>(singleGrid ? 0 : triangleCount);
    this->template launch<dc::CompactVerticesFunctor>(vertexCount, uint64_t(vertexCount), referenced.data(), newIndex.data(), points.data(),
                                                      refNormals.data(), vertexGrid, chunk.points.data(),
                                                      chunk.normals.data(), keptGrid.data());
    if (singleGrid) {
        this->template launch<dc::FillFunctor<uint64_t>>(1, uint64_t(1), pointBase, d_pointOffsets + gridBegin);
        this->template launch<dc::FillFunctor<uint64_t>>(1, uint64_t(1), triangleBase, d_triangleOffsets + gridBegin);
    } else {
        this->template launch<dc::SegmentOffsetsFunctor>(keptCount + 1, uint64_t(keptCount), keptGrid.data(), int32_t(gridBegin),
                                                         int32_t(gridEnd), pointBase, d_pointOffsets);
    }
    this->template launch<dc::RenumberTrianglesFunctor>(triangleCount, uint64_t(triangleCount), newIndex.data(), keptGrid.data(),
                                                        d_pointOffsets, pointBase, triangles.data(),
                                                        triangleGrid.data());
    if (!singleGrid)
        this->template launch<dc::SegmentOffsetsFunctor>(triangleCount + 1, uint64_t(triangleCount), triangleGrid.data(), int32_t(gridBegin),
                                                         int32_t(gridEnd), triangleBase, d_triangleOffsets);
    chunk.triangles = std::move(triangles);
    pointBase += keptCount;
    triangleBase += triangleCount;
    chunks.push_back(std::move(chunk));
}

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_DUALCONTOUR_CUH_HAS_BEEN_INCLUDED
