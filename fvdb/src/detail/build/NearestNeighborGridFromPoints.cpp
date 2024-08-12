// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "Build.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include "detail/utils/Utils.h"
#include "detail/ops/Ops.h"


namespace fvdb {
namespace detail {
namespace build {


template <typename GridType>
nanovdb::GridHandle<TorchDeviceBuffer> buildNearestNeighborGridFromPointsCPU(const JaggedTensor& jaggedPoints,
                                                                             const std::vector<VoxelCoordTransform>& txs) {

    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(jaggedPoints.scalar_type(), "buildNearestNeighborGridFromPoints", [&]() {
        using ScalarT = scalar_t;
        using MathT = typename at::opmath_type<ScalarT>;
        using Vec3T = typename nanovdb::math::Vec3<MathT>;
        using ProxyGridT = nanovdb::tools::build::Grid<float>;

        static_assert(is_floating_point_or_half<ScalarT>::value, "Invalid type for points, must be floating point");

        jaggedPoints.check_valid();

        const torch::TensorAccessor<ScalarT, 2>& pointsAcc = jaggedPoints.jdata().accessor<ScalarT, 2>();
        const torch::TensorAccessor<fvdb::JOffsetsType, 1>& pointsBOffsetsAcc = jaggedPoints.joffsets().accessor<fvdb::JOffsetsType, 1>();

        std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
        batchHandles.reserve(pointsBOffsetsAcc.size(0) - 1);
        for (int bi = 0; bi < (pointsBOffsetsAcc.size(0) - 1); bi += 1) {

            const VoxelCoordTransform& tx = txs[bi];

            auto proxyGrid = std::make_shared<ProxyGridT>(-1.0f);
            auto proxyGridAccessor = proxyGrid->getWriteAccessor();

            const int64_t start = pointsBOffsetsAcc[bi];
            const int64_t end = pointsBOffsetsAcc[bi+1];

            for (int64_t pi = start; pi < end; pi += 1) {
                Vec3T ijk0 = tx.apply(static_cast<MathT>(pointsAcc[pi][0]),
                                    static_cast<MathT>(pointsAcc[pi][1]),
                                    static_cast<MathT>(pointsAcc[pi][2]));
                nanovdb::Coord ijk000 = ijk0.floor();
                nanovdb::Coord ijk001 = ijk000 + nanovdb::Coord(0, 0, 1);
                nanovdb::Coord ijk010 = ijk000 + nanovdb::Coord(0, 1, 0);
                nanovdb::Coord ijk011 = ijk000 + nanovdb::Coord(0, 1, 1);
                nanovdb::Coord ijk100 = ijk000 + nanovdb::Coord(1, 0, 0);
                nanovdb::Coord ijk101 = ijk000 + nanovdb::Coord(1, 0, 1);
                nanovdb::Coord ijk110 = ijk000 + nanovdb::Coord(1, 1, 0);
                nanovdb::Coord ijk111 = ijk000 + nanovdb::Coord(1, 1, 1);

                proxyGridAccessor.setValue(ijk000, 11.0f);
                proxyGridAccessor.setValue(ijk001, 11.0f);
                proxyGridAccessor.setValue(ijk010, 11.0f);
                proxyGridAccessor.setValue(ijk011, 11.0f);
                proxyGridAccessor.setValue(ijk100, 11.0f);
                proxyGridAccessor.setValue(ijk101, 11.0f);
                proxyGridAccessor.setValue(ijk110, 11.0f);
                proxyGridAccessor.setValue(ijk111, 11.0f);
            }

            proxyGridAccessor.merge();
            auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(*proxyGrid, 0u, false, false);
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


nanovdb::GridHandle<TorchDeviceBuffer> buildNearestNeighborGridFromPoints(bool isMutable,
                                                                          const JaggedTensor& points,
                                                                          const std::vector<VoxelCoordTransform>& txs) {
    if (points.device().is_cuda()) {
        JaggedTensor coords = ops::dispatchNearestNeighborIJKForPoints<torch::kCUDA>(points, txs);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildNearestNeighborGridFromPointsCPU<GridType>(points, txs);
        });
    }
}



} // namespace build
} // namespace detail
} // namespace fvdb
