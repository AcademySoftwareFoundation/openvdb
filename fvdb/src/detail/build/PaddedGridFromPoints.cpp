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

template <typename GridType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromPointsCPU(const JaggedTensor                     &pointsJagged,
                             const std::vector<VoxelCoordTransform> &txs,
                             const nanovdb::Coord &bmin, const nanovdb::Coord &bmax) {
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        pointsJagged.scalar_type(), "buildPaddedGridFromPoints", [&]() {
            using ScalarT = scalar_t;
            static_assert(is_floating_point_or_half<ScalarT>::value,
                          "Invalid type for points, must be floating point");
            using MathT      = typename at::opmath_type<ScalarT>;
            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            pointsJagged.check_valid();

            const torch::TensorAccessor<ScalarT, 2> &pointsAcc =
                pointsJagged.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &pointsBOffsetsAcc =
                pointsJagged.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(pointsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (pointsBOffsetsAcc.size(0) - 1); bi += 1) {
                VoxelCoordTransform tx = txs[bi];

                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = pointsBOffsetsAcc[bi];
                const int64_t end   = pointsBOffsetsAcc[bi + 1];

                for (int64_t pi = start; pi < end; pi += 1) {
                    nanovdb::Coord ijk0 = tx.apply(static_cast<MathT>(pointsAcc[pi][0]),
                                                   static_cast<MathT>(pointsAcc[pi][1]),
                                                   static_cast<MathT>(pointsAcc[pi][2]))
                                              .round();

                    // Splat the normal to the 8 neighboring voxels
                    for (int di = bmin[0]; di <= bmax[0]; di += 1) {
                        for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
                            for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                                const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                                proxyGridAccessor.setValue(ijk, 1.0f);
                            }
                        }
                    }
                }

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
        });
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromPoints(bool isMutable, const JaggedTensor &points,
                          const std::vector<VoxelCoordTransform> &txs, const nanovdb::Coord &bmin,
                          const nanovdb::Coord &bmax) {
    if (points.device().is_cuda()) {
        JaggedTensor coords =
            ops::dispatchPaddedIJKForPoints<torch::kCUDA>(points, bmin, bmax, txs);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);

    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildPaddedGridFromPointsCPU<GridType>(points, txs, bmin, bmax);
        });
    }
}

} // namespace build
} // namespace detail
} // namespace fvdb
