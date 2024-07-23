// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "FVDB.h"

#include "detail/autograd/Autograd.h"
#include "detail/io/IO.h"
#include "detail/ops/jagged/JaggedOps.h"

#include <ATen/cuda/CUDAContext.h>

#include <execution>

namespace fvdb {

std::vector<torch::Tensor> volumeRender(const torch::Tensor& sigmas, const torch::Tensor& rgbs,
                                        const torch::Tensor& deltaTs, const torch::Tensor& ts,
                                        const torch::Tensor& jOffsets, double transmittanceThresh) {
    return detail::autograd::VolumeRender::apply(sigmas, rgbs, deltaTs, ts, jOffsets, transmittanceThresh);
}

JaggedTensor scaledDotProductAttention(const JaggedTensor& query,
                                       const JaggedTensor& key,
                                       const JaggedTensor& value,
                                       float scale) {

    cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
    const int computeCapability = p->major * 10 + p->minor;

    if (computeCapability < 90) {
        // https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        // - query: (N, ..., L, E)
        // - key: (N, ..., S, E)
        // - value: (N, ..., S, V)
        std::vector<torch::Tensor> outList;
        torch::Tensor qOffsets = query.joffsets().cpu();
        torch::Tensor kvOffsets = key.joffsets().cpu();

        for (int64_t b = 0; b < query.num_tensors(); ++b) {
            int64_t qStart = qOffsets[b].item<int64_t>();
            int64_t qEnd = qOffsets[b+1].item<int64_t>();
            int64_t kvStart = kvOffsets[b].item<int64_t>();
            int64_t kvEnd = kvOffsets[b+1].item<int64_t>();

            torch::Tensor q = query.jdata().index({torch::indexing::Slice(qStart, qEnd)}).permute({1, 0, 2});
            torch::Tensor k = key.jdata().index({torch::indexing::Slice(kvStart, kvEnd)}).permute({1, 0, 2});
            torch::Tensor v = value.jdata().index({torch::indexing::Slice(kvStart, kvEnd)}).permute({1, 0, 2});

            torch::Tensor out = at::native::scaled_dot_product_attention(q, k, v, {}, 0.0, false, scale);
            outList.push_back(out.permute({1, 0, 2}));
        }

        return JaggedTensor(outList);
    }

    // Custom implementation with CUDNN is only available for Hopper.
    torch::Tensor qLengths = query.joffsets().index({torch::indexing::Slice(1, query.num_tensors())});
    torch::Tensor kvLengths = key.joffsets().index({torch::indexing::Slice(1, query.num_tensors())});
    torch::Tensor res = detail::autograd::Attention::apply(
        query.jdata(), key.jdata(), value.jdata(), qLengths, kvLengths, scale)[0];
    return query.jagged_like(res);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
from_nanovdb(nanovdb::GridHandle<nanovdb::HostBuffer>& handle){
    return detail::io::fromNVDB(handle);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
to_nanovdb(const GridBatch& gridBatch,
           const torch::optional<JaggedTensor> maybeData,
           const torch::optional<StringOrListOfStrings> maybeNames){
    return detail::io::toNVDB(gridBatch, maybeData, maybeNames);
}


GridBatch jcat(const std::vector<GridBatch>& vec) {
     std::vector<c10::intrusive_ptr<detail::GridBatchImpl>> vecHdls;
     std::transform(vec.begin(), vec.end(), std::back_inserter(vecHdls),
                    [](const GridBatch& grid) { return grid.impl(); });
     return GridBatch(detail::GridBatchImpl::concatenate(vecHdls));
}

JaggedTensor jcat(const std::vector<JaggedTensor>& vec, torch::optional<int64_t> dim) {
    return JaggedTensor::jcat(vec, dim);
}

void save(const std::string& path,
          const GridBatch& gridBatch,
          const torch::optional<JaggedTensor> maybeData,
          const torch::optional<StringOrListOfStrings> maybeNames,
          bool compressed,
          bool verbose) {
    detail::io::saveNVDB(path, gridBatch, maybeData, maybeNames, compressed, verbose);
}


std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string& path,
     NanoVDBFileGridIdentifier gridIdentifier,
     TorchDeviceOrString device,
     bool verbose) {
    return detail::io::loadNVDB(path, gridIdentifier, device, verbose);
}

GridBatch sparse_grid_from_points(const JaggedTensor& points,
                                  const Vec3i& pad_min,
                                  const Vec3i& pad_max,
                                  const Vec3dBatchOrScalar& voxel_sizes,
                                  const Vec3dBatch& origins,
                                  bool is_mutable) {
    auto ret = GridBatch(points.device(), is_mutable);
    ret.set_from_points(points, pad_min, pad_max, voxel_sizes, origins);
    return ret;
}


GridBatch sparse_grid_from_ijk(const JaggedTensor& ijk,
                               const Vec3i& pad_min,
                               const Vec3i& pad_max,
                               const Vec3dBatchOrScalar& voxel_sizes,
                               const Vec3dBatch& origins,
                               bool is_mutable) {
    auto ret = GridBatch(ijk.device(), is_mutable);
    ret.set_from_ijk(ijk, pad_min, pad_max, voxel_sizes, origins);
    return ret;
}


GridBatch sparse_grid_from_nearest_voxels_to_points(const JaggedTensor& points,
                                                    const Vec3dBatchOrScalar& voxel_sizes,
                                                    const Vec3dBatch& origins,
                                                    bool is_mutable) {
    auto ret = GridBatch(points.device(), is_mutable);
    ret.set_from_nearest_voxels_to_points(points, voxel_sizes, origins);
    return ret;
}


GridBatch sparse_grid_from_dense(const int64_t numGrids,
                                 const Vec3i& denseDims,
                                 const Vec3i& ijkMin,
                                 const Vec3dBatchOrScalar& voxel_sizes,
                                 const Vec3dBatch& origins,
                                 torch::optional<torch::Tensor> mask,
                                 TorchDeviceOrString device, bool is_mutable) {
    auto ret = GridBatch(device, is_mutable);
    ret.set_from_dense_grid(numGrids, denseDims, ijkMin, voxel_sizes, origins, mask);
    return ret;
}

GridBatch sparse_grid_from_mesh(const JaggedTensor& vertices,
                                const JaggedTensor& faces,
                                const Vec3dBatchOrScalar& voxel_sizes,
                                const Vec3dBatch& origins,
                                bool is_mutable) {
    auto ret = GridBatch(vertices.device(), is_mutable);
    ret.set_from_mesh(vertices, faces, voxel_sizes, origins);
    return ret;
}


std::vector<int64_t> jdataShape1(const std::vector<int64_t>& lsizes, const std::vector<int64_t>& rsizes) {
    const int64_t totalElements = std::reduce(std::execution::par, lsizes.begin(), lsizes.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalElements);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());
    return shape;
}

std::tuple<int64_t, std::vector<int64_t>> jdataShape2(const std::vector<std::vector<int64_t>>& lsizes, const std::vector<int64_t>& rsizes) {
    std::vector<int64_t> elementCountsPerList;
    std::vector<int64_t> tensorCountsPerList;
    elementCountsPerList.reserve(lsizes.size());
    tensorCountsPerList.reserve(lsizes.size());
    for (const auto& l : lsizes) {
        elementCountsPerList.push_back(std::reduce(std::execution::par, l.begin(), l.end()));
        tensorCountsPerList.push_back(l.size());
    }
    const int64_t totalSize = std::reduce(std::execution::par, elementCountsPerList.begin(), elementCountsPerList.end());
    const int64_t totalTensors = std::reduce(std::execution::par, tensorCountsPerList.begin(), tensorCountsPerList.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalSize);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());

    return std::make_tuple(totalTensors, shape);
}

#define __FVDB__BUILDER(FNAME, JFNAME) \
    JaggedTensor JFNAME(const std::vector<int64_t>& lsizes, \
                        const std::vector<int64_t> rsizes, \
                        at::TensorOptions options) { \
        auto shape = jdataShape1(lsizes, rsizes); \
        return JaggedTensor(lsizes, FNAME(shape, options)); \
    } \
    \
    JaggedTensor JFNAME(const std::vector<std::vector<int64_t>>& lsizes, \
                        const std::vector<int64_t> rsizes, \
                        at::TensorOptions options) { \
        auto shape = jdataShape2(lsizes, rsizes); \
        return JaggedTensor(lsizes, std::get<0>(shape), FNAME(std::get<1>(shape), options)); \
    }

__FVDB__BUILDER(torch::rand, jrand)
__FVDB__BUILDER(torch::randn, jrandn)
__FVDB__BUILDER(torch::zeros, jzeros)
__FVDB__BUILDER(torch::ones, jones)
__FVDB__BUILDER(torch::empty, jempty)

#undef __FVDB__BUILDER

} // namespace fvdb