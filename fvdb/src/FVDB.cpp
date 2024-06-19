#include "FVDB.h"

#include "detail/autograd/Autograd.h"
#include "detail/io/IO.h"
#include "detail/ops/jagged/JaggedOps.h"

#include <ATen/cuda/CUDAContext.h>


namespace fvdb {

std::vector<torch::Tensor> volumeRender(const torch::Tensor& sigmas, const torch::Tensor& rgbs,
                                        const torch::Tensor& deltaTs, const torch::Tensor& ts,
                                        const torch::Tensor& packInfo, double transmittanceThresh) {
    return detail::autograd::VolumeRender::apply(sigmas, rgbs, deltaTs, ts, packInfo, transmittanceThresh);
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

        for (int64_t b = 0; b < query.batch_size(); ++b) {
            int64_t qStart = qOffsets[b][0].item<int64_t>();
            int64_t qEnd = qOffsets[b][1].item<int64_t>();
            int64_t kvStart = kvOffsets[b][0].item<int64_t>();
            int64_t kvEnd = kvOffsets[b][1].item<int64_t>();

            torch::Tensor q = query.jdata().index({torch::indexing::Slice(qStart, qEnd)}).permute({1, 0, 2});
            torch::Tensor k = key.jdata().index({torch::indexing::Slice(kvStart, kvEnd)}).permute({1, 0, 2});
            torch::Tensor v = value.jdata().index({torch::indexing::Slice(kvStart, kvEnd)}).permute({1, 0, 2});

            torch::Tensor out = at::native::scaled_dot_product_attention(q, k, v, {}, 0.0, false, scale);
            outList.push_back(out.permute({1, 0, 2}));
        }

        return JaggedTensor(outList);
    }

    // Custom implementation with CUDNN is only available for Hopper.
    torch::Tensor qLengths = query.joffsets().index({torch::indexing::Slice(), 1});
    torch::Tensor kvLengths = key.joffsets().index({torch::indexing::Slice(), 1});
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


GridBatch cat(const std::vector<GridBatch>& vec) {
     std::vector<c10::intrusive_ptr<detail::GridBatchImpl>> vecHdls;
     std::transform(vec.begin(), vec.end(), std::back_inserter(vecHdls),
                    [](const GridBatch& grid) { return grid.impl(); });
     return GridBatch(detail::GridBatchImpl::concatenate(vecHdls));
}

JaggedTensor cat(const std::vector<JaggedTensor>& vec, int dim) {
    return JaggedTensor::concatenate(vec, dim);
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


} // namespace fvdb