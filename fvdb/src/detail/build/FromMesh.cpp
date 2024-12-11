// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Build.h"

#include <detail/ops/Ops.h>
#include <detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

namespace fvdb {
namespace detail {
namespace build {

template <typename GridType, typename ScalarType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromMeshCPU(const JaggedTensor &vertices, const JaggedTensor &triangles,
                     const std::vector<VoxelCoordTransform> &tx) {
    using Vec3T      = nanovdb::math::Vec3<ScalarType>;
    using ProxyGridT = nanovdb::tools::build::Grid<float>;

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(vertices.num_outer_lists());

    for (int64_t bidx = 0; bidx < vertices.num_outer_lists(); bidx += 1) {
        const torch::Tensor        ti  = triangles.index({ bidx }).jdata();
        const torch::Tensor        vi  = vertices.index({ bidx }).jdata();
        const VoxelCoordTransform &txi = tx[bidx];

        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        // int64_t numSearched = 0;
        // int64_t numFound = 0;
        // For eacjh face, compute thee min max voxels
        for (int faceId = 0; faceId < ti.size(0); faceId += 1) {
            const torch::Tensor                  face         = ti.index({ faceId }); // 3
            const torch::Tensor                  faceVertices = vi.index({ face });   // [3, 3]
            torch::TensorAccessor<ScalarType, 2> faceVerticesAcc =
                faceVertices.accessor<ScalarType, 2>();
            const Vec3T v1 = txi.apply(
                Vec3T(faceVerticesAcc[0][0], faceVerticesAcc[0][1], faceVerticesAcc[0][2]));
            const Vec3T v2 = txi.apply(
                Vec3T(faceVerticesAcc[1][0], faceVerticesAcc[1][1], faceVerticesAcc[1][2]));
            const Vec3T v3 = txi.apply(
                Vec3T(faceVerticesAcc[2][0], faceVerticesAcc[2][1], faceVerticesAcc[2][2]));

            const Vec3T      e1 = v2 - v1;
            const Vec3T      e2 = v3 - v1;
            const ScalarType spacing =
                sqrt(3.0) / 3.0; // This is very conservative spacing but fine for now
            const int32_t numU = ceil((e1.length() + spacing) / spacing);
            const int32_t numV = ceil((e2.length() + spacing) / spacing);

            // numSearched += (numU * numV);
            for (int i = 0; i < numU; i += 1) {
                for (int j = 0; j < numV; j += 1) {
                    ScalarType u = ScalarType(i) / (ScalarType(std::max(numU - 1, 1)));
                    ScalarType v = ScalarType(j) / (ScalarType(std::max(numV - 1, 1)));
                    if (u + v >= 1.0) {
                        u = 1.0 - u;
                        v = 1.0 - v;
                    }
                    const Vec3T          p   = v1 + e1 * u + e2 * v;
                    const nanovdb::Coord ijk = p.round();

                    proxyGridAccessor.setValue(ijk, 1.0f);
                    // numFound += 1;
                }
            }
        }

        // std::cerr << "I searched over " << numSearched << " voxels" << std::endl;
        // std::cerr << "I found " << numFound << " voxels" << std::endl;
        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().setDevice(torch::kCPU, true);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromMesh(bool isMutable, const JaggedTensor meshVertices, const JaggedTensor meshFaces,
                  const std::vector<VoxelCoordTransform> &tx) {
    if (meshVertices.device().is_cuda()) {
        JaggedTensor coords = ops::dispatchIJKForMesh<torch::kCUDA>(meshVertices, meshFaces, tx);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return AT_DISPATCH_FLOATING_TYPES(
                meshVertices.scalar_type(), "buildGridFromMeshCPU", [&]() {
                    return buildGridFromMeshCPU<GridType, scalar_t>(meshVertices, meshFaces, tx);
                });
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
