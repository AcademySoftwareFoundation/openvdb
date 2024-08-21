// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "Build.h"

#include <detail/ops/Ops.h>
#include <detail/utils/Utils.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <ATen/OpMathType.h>

namespace fvdb {
namespace detail {
namespace build {

template <typename GridType>
nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromCoordsCPU(const JaggedTensor &jaggedCoords, const nanovdb::Coord &bmin,
                             const nanovdb::Coord &bmax) {
    return AT_DISPATCH_INTEGRAL_TYPES(
        jaggedCoords.scalar_type(), "buildPaddedGridFromCoords", [&]() {
            using ScalarT = scalar_t;
            jaggedCoords.check_valid();

            static_assert(std::is_integral<ScalarT>::value,
                          "Invalid type for coords, must be integral");

            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            const torch::TensorAccessor<ScalarT, 2> &coordsAcc =
                jaggedCoords.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &coordsBOffsetsAcc =
                jaggedCoords.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(coordsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (coordsBOffsetsAcc.size(0) - 1); bi += 1) {
                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = coordsBOffsetsAcc[bi];
                const int64_t end   = coordsBOffsetsAcc[bi + 1];

                for (unsigned ci = start; ci < end; ci += 1) {
                    nanovdb::Coord ijk0(coordsAcc[ci][0], coordsAcc[ci][1], coordsAcc[ci][2]);

                    // Splat the normal to the 8 neighboring voxels
                    for (int di = bmin[0]; di <= bmax[0]; di += 1) {
                        for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
                            for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                                const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                                proxyGridAccessor.setValue(ijk, 11);
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
buildPaddedGridFromCoords(bool isMutable, const JaggedTensor &coords, const nanovdb::Coord &bmin,
                          const nanovdb::Coord &bmax) {
    if (coords.device().is_cuda()) {
        JaggedTensor buildCoords =
            ops::dispatchPaddedIJKForCoords<torch::kCUDA>(coords, bmin, bmax);
        return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(buildCoords, isMutable);
    } else {
        return FVDB_DISPATCH_GRID_TYPES_MUTABLE(isMutable, [&]() {
            return buildPaddedGridFromCoordsCPU<GridType>(coords, bmin, bmax);
        });
    }
}
} // namespace build
} // namespace detail
} // namespace fvdb
