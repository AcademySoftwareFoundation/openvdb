#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/MarchingCubesData.h"

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarT>
__hostdev__ static inline int getCubeType(const ScalarT* sdfValues) {
    int cubeType = 0;
    if (sdfValues[0] < 0) cubeType |= 1;
    if (sdfValues[1] < 0) cubeType |= 2;
    if (sdfValues[2] < 0) cubeType |= 4;
    if (sdfValues[3] < 0) cubeType |= 8;
    if (sdfValues[4] < 0) cubeType |= 16;
    if (sdfValues[5] < 0) cubeType |= 32;
    if (sdfValues[6] < 0) cubeType |= 64;
    if (sdfValues[7] < 0) cubeType |= 128;
    return cubeType;
}

template<typename ScalarT>
__hostdev__ static inline nanovdb::math::Vec4<ScalarT> sdfInterp(
        const nanovdb::math::Vec3<ScalarT> p1, const nanovdb::math::Vec3<ScalarT> p2, ScalarT valp1, ScalarT valp2) {

    if (std::abs(0.0f - valp1) < 1.0e-5f)
        return nanovdb::math::Vec4<ScalarT>(p1[0], p1[1], p1[2], 1.0);

    if (std::abs(0.0f - valp2) < 1.0e-5f)
        return nanovdb::math::Vec4<ScalarT>(p2[0], p2[1], p2[2], 0.0);

    if (std::abs(valp1 - valp2) < 1.0e-5f)
        return nanovdb::math::Vec4<ScalarT>(p1[0], p1[1], p1[2], 1.0);

    ScalarT w2 = (0.0 - valp1) / (valp2 - valp1);
    ScalarT w1 = 1 - w2;

    return nanovdb::math::Vec4<ScalarT>(
            p1[0] * w1 + p2[0] * w2,
            p1[1] * w1 + p2[1] * w2,
            p1[2] * w1 + p2[2] * w2, w1);
}

template <typename ScalarT>
__hostdev__ static inline void fillVertList(nanovdb::math::Vec4<ScalarT>* vert_list, int edge_config,
                                            nanovdb::math::Vec3<ScalarT>* points, ScalarT* sdf_vals) {
    if (edge_config & 1) vert_list[0] = sdfInterp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
    if (edge_config & 2) vert_list[1] = sdfInterp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
    if (edge_config & 4) vert_list[2] = sdfInterp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
    if (edge_config & 8) vert_list[3] = sdfInterp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
    if (edge_config & 16) vert_list[4] = sdfInterp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
    if (edge_config & 32) vert_list[5] = sdfInterp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
    if (edge_config & 64) vert_list[6] = sdfInterp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
    if (edge_config & 128) vert_list[7] = sdfInterp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
    if (edge_config & 256) vert_list[8] = sdfInterp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
    if (edge_config & 512) vert_list[9] = sdfInterp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
    if (edge_config & 1024) vert_list[10] = sdfInterp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
    if (edge_config & 2048) vert_list[11] = sdfInterp(points[3], points[7], sdf_vals[3], sdf_vals[7]);
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void countVerticesCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                       BatchGridAccessor<GridType> batchAccessor,
                                       TensorAccessor<ScalarType, 1> sdf, ScalarType level,
                                       TensorAccessor<int64_t, 1> nVertices) {
    const nanovdb::NanoGrid<GridType>* grid = batchAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

    auto base = leaf.getValue(voxelIdx) - 1 + batchAccessor.voxelOffset(batchIdx);
    auto gridAcc = grid->getAccessor();

    ScalarType sdfValues[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        nanovdb::Coord vCoord = ijk + nanovdb::Coord(marchingCubesCubeRelTable[i][0],
                                                     marchingCubesCubeRelTable[i][1],
                                                     marchingCubesCubeRelTable[i][2]);
        if (gridAcc.template get<ActiveOrUnmasked<GridType>>(vCoord)) {
            sdfValues[i] = sdf[batchAccessor.voxelOffset(batchIdx) + gridAcc.getValue(vCoord) - 1] - level;
        } else {
            // Incomplete cube, return
            return;
        }
    }

    int cubeType = getCubeType(sdfValues);
    nVertices[base] = marchingCubesNumVertsTable[cubeType];
}

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void meshingCubeCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                     BatchGridAccessor<GridType> batchAccessor,
                                     TensorAccessor<ScalarType, 1> sdf, ScalarType level,
                                     TensorAccessor<int64_t, 1> countCsum,
                                     TensorAccessor<ScalarType, 3> triangles,
                                     TensorAccessor<int64_t, 3> vertIds) {
    const nanovdb::NanoGrid<GridType>* grid = batchAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
    VoxelCoordTransform transform = batchAccessor.primalTransform(batchIdx);

    auto base = leaf.getValue(voxelIdx) - 1 + batchAccessor.voxelOffset(batchIdx);
    auto gridAcc = grid->getAccessor();

    ScalarType sdfValues[8];
    int64_t pointIds[8];
    nanovdb::math::Vec3<ScalarType> points[8];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        nanovdb::Coord vCoord = ijk + nanovdb::Coord(marchingCubesCubeRelTable[i][0],
                                                     marchingCubesCubeRelTable[i][1],
                                                     marchingCubesCubeRelTable[i][2]);
        if (gridAcc.template get<ActiveOrUnmasked<GridType>>(vCoord)) {
            pointIds[i] = batchAccessor.voxelOffset(batchIdx) + gridAcc.getValue(vCoord) - 1;
            sdfValues[i] = sdf[pointIds[i]] - level;
            points[i] = transform.applyInv(static_cast<ScalarType>(vCoord[0]),
                                           static_cast<ScalarType>(vCoord[1]),
                                           static_cast<ScalarType>(vCoord[2]));
        } else {
            // Incomplete cube, return
            return;
        }
    }

    int cubeType = getCubeType(sdfValues);
    int edgeConfig = marchingCubesEdgeTable[cubeType];
    if (edgeConfig == 0) return;

    nanovdb::math::Vec4<ScalarType> vertList[12];
    fillVertList(vertList, edgeConfig, points, sdfValues);

    // Write triangles to array.
    for (int i = 0; marchingCubesTriTable[cubeType][i] != -1; i += 3) {
        int64_t triangleIdx = countCsum[base] / 3 + i / 3;
        #pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            int64_t vlid = marchingCubesTriTable[cubeType][i + vi];
            for (int d = 0; d < 3; ++d) {
                triangles[triangleIdx][vi][d] = vertList[vlid][d];
            }
            int64_t vid0 = pointIds[marchingCubesE2iTable[vlid][0]];
            int64_t vid1 = pointIds[marchingCubesE2iTable[vlid][1]];
            if (vid0 < vid1) {
                int64_t t = vid1; vid1 = vid0; vid0 = t;
            }
            vertIds[triangleIdx][vi][0] = batchIdx;
            vertIds[triangleIdx][vi][1] = vid0;
            vertIds[triangleIdx][vi][2] = vid1;
        }
    }

}

template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor> MarchingCubes(const GridBatchImpl& batchHdl,
                                              const torch::Tensor& sdf,
                                              double level) {

    batchHdl.checkDevice(sdf);
    TORCH_CHECK_TYPE(sdf.is_floating_point(), "field must have a floating point type");
    TORCH_CHECK(sdf.dim() == 1, std::string("Expected field to have 1 dimension (shape (n,)) but got ") +
                                std::to_string(sdf.dim()) + " dimensions");

    auto longOpts = torch::TensorOptions().dtype(torch::kLong).device(sdf.device());
    auto scalarOpts = torch::TensorOptions().dtype(sdf.dtype()).device(sdf.device());
    torch::Tensor nVertices = torch::zeros({sdf.size(0)}, longOpts);

    // Count the number of vertices
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(sdf.scalar_type(), "countVertices", ([&] {
            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto sdfAcc = tensorAccessor<DeviceTag, scalar_t, 1>(sdf);
            auto nVerticesAcc = tensorAccessor<DeviceTag, int64_t, 1>(nVertices);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, BatchGridAccessor<GridType> batchAcc) {
                    countVerticesCallback<scalar_t, GridType, TorchRAcc32>(
                            bidx, lidx, vidx, cidx, batchAcc, sdfAcc, static_cast<scalar_t>(level),
                            nVerticesAcc);
                };
                forEachVoxelCUDA<GridType>(128, 1, batchHdl, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, BatchGridAccessor<GridType> batchAcc) {
                    countVerticesCallback<scalar_t, GridType, TorchAcc>(
                            bidx, lidx, vidx, cidx, batchAcc, sdfAcc, static_cast<scalar_t>(level),
                            nVerticesAcc);
                };
                forEachVoxelCPU<GridType>(1, batchHdl, cb);
            }
        }));
    });

    // cumsum to determine starting position.
    torch::Tensor countCsum = torch::cumsum(nVertices, 0);
    int64_t nTriangles = countCsum[-1].item<int64_t>() / 3;
    countCsum = torch::roll(countCsum, torch::IntList(1));
    countCsum[0] = 0;

    // Generate triangles
    torch::Tensor triangles = torch::empty({nTriangles, 3, 3}, scalarOpts);
    torch::Tensor vertIds = torch::empty({nTriangles, 3, 3}, longOpts);

    if (nTriangles > 0) {
        FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(sdf.scalar_type(), "meshingCubes", ([&] {
                auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
                auto sdfAcc = tensorAccessor<DeviceTag, scalar_t, 1>(sdf);
                auto countCsumAcc = tensorAccessor<DeviceTag, int64_t, 1>(countCsum);
                auto trianglesAcc = tensorAccessor<DeviceTag, scalar_t, 3>(triangles);
                auto vertIdsAcc = tensorAccessor<DeviceTag, int64_t, 3>(vertIds);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                            BatchGridAccessor<GridType> batchAcc) {
                        meshingCubeCallback<scalar_t, GridType, TorchRAcc32>(
                                bidx, lidx, vidx, cidx, batchAcc, sdfAcc, static_cast<scalar_t>(level),
                                countCsumAcc, trianglesAcc, vertIdsAcc);
                    };
                    forEachVoxelCUDA<GridType>(128, 1, batchHdl, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                            BatchGridAccessor<GridType> batchAcc) {
                        meshingCubeCallback<scalar_t, GridType, TorchAcc>(
                                bidx, lidx, vidx, cidx, batchAcc, sdfAcc, static_cast<scalar_t>(level),
                                countCsumAcc, trianglesAcc, vertIdsAcc);
                    };
                    forEachVoxelCPU<GridType>(1, batchHdl, cb);
                }
            }));
        });
    }

    // Flatten
    triangles = triangles.view({-1, 3});
    vertIds = vertIds.view({-1, 3});

    // Merge triangles by detecting the same vertex position.
    //  (sort to keep lexicographical order with batch-dim first)
    auto unqRet = torch::unique_dim(vertIds, 0, true, true);
    torch::Tensor unqVertIdx = std::get<0>(unqRet);
    torch::Tensor unqTriangles = std::get<1>(unqRet);

    torch::Tensor vertices = torch::zeros({unqVertIdx.size(0), 3}, scalarOpts);
    vertices.index_put_({unqTriangles}, triangles);

    // Compute batch index for vertices and triangles
    unqTriangles = unqTriangles.view({-1, 3});
    torch::Tensor vBatchIdx = unqVertIdx.index({torch::indexing::Slice(), 0});
    torch::Tensor tBatchIdx = vBatchIdx.index({unqTriangles.index({torch::indexing::Slice(), 0})});

    JaggedTensor retVertices = JaggedTensor::from_data_and_jidx(vertices, vBatchIdx, batchHdl.batchSize());
    JaggedTensor retTriangles = JaggedTensor::from_data_and_jidx(unqTriangles, tBatchIdx, batchHdl.batchSize());
    JaggedTensor retUniqueVertices = JaggedTensor::from_data_and_jidx(unqVertIdx, vBatchIdx, batchHdl.batchSize());

    // Fix triangle indices per mesh
    int64_t cumNumVerts = 0;
    for (int i = 1; i < batchHdl.batchSize(); i += 1) {
        cumNumVerts += retVertices.index({i - 1}).jdata().size(0);
        retTriangles.index({i}).jdata().sub_(cumNumVerts);
    }

    return {retVertices, retTriangles, retUniqueVertices};
}


template <>
std::vector<JaggedTensor> dispatchMarchingCubes<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                            const torch::Tensor& sdf,
                                                            double level) {
    return MarchingCubes<torch::kCUDA>(batchHdl, sdf, level);
}

template <>
std::vector<JaggedTensor> dispatchMarchingCubes<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                           const torch::Tensor& sdf,
                                                           double level) {
    return MarchingCubes<torch::kCPU>(batchHdl, sdf, level);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
