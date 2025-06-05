// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/GridBatch.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/autograd/Autograd.h>
#include <fvdb/detail/io/IO.h>
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {

GridBatch::GridBatch(const torch::Device &device) {
    mImpl = c10::make_intrusive<detail::GridBatchImpl>(device);
}

GridBatch::GridBatch(const std::string &device_string) {
    torch::Device device(device_string);
    if (device.is_cuda() && !device.has_index()) {
        device.set_index(c10::cuda::current_device());
    }
    detail::RAIIDeviceGuard guard(device);
    mImpl = c10::make_intrusive<detail::GridBatchImpl>(device);
}

GridBatch::GridBatch() {
    mImpl = detail::GridBatchImpl::createFromEmpty(
        torch::kCPU, nanovdb::Vec3d(1.0), nanovdb::Vec3d(0.0));
}

std::pair<JaggedTensor, GridBatch>
GridBatch::max_pool(Vec3iOrScalar pool_factor,
                    const JaggedTensor &data,
                    Vec3iOrScalar stride,
                    std::optional<GridBatch> coarse_grid) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        data.ldim() == 1,
        "Expected data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        data.ldim(),
        "list dimensions");

    nanovdb::Coord pool_factor_coord = pool_factor.value();
    nanovdb::Coord stride_coord      = stride.value();

    for (int i = 0; i < 3; i += 1) {
        if (stride_coord[i] == 0) {
            stride_coord[i] = pool_factor_coord[i];
        }
    }

    c10::intrusive_ptr<detail::GridBatchImpl> coarse_grid_impl;
    if (coarse_grid.has_value()) {
        coarse_grid_impl = coarse_grid.value().impl();
    } else {
        coarse_grid_impl = coarsened_grid(stride_coord).impl();
    }

    torch::Tensor pool_data = detail::autograd::MaxPoolGrid::apply(
        impl(), coarse_grid_impl, pool_factor_coord, stride_coord, data.jdata())[0];

    return std::make_pair(coarse_grid_impl->jaggedTensor(pool_data), GridBatch(coarse_grid_impl));
}

std::pair<JaggedTensor, GridBatch>
GridBatch::avg_pool(Vec3iOrScalar pool_factor,
                    const JaggedTensor &data,
                    Vec3iOrScalar stride,
                    std::optional<GridBatch> coarse_grid) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        data.ldim() == 1,
        "Expected data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        data.ldim(),
        "list dimensions");

    nanovdb::Coord pool_factor_coord = pool_factor.value();
    nanovdb::Coord stride_coord      = stride.value();

    for (int i = 0; i < 3; i += 1) {
        if (stride_coord[i] == 0) {
            stride_coord[i] = pool_factor_coord[i];
        }
    }

    c10::intrusive_ptr<detail::GridBatchImpl> coarse_grid_impl;
    if (coarse_grid.has_value()) {
        coarse_grid_impl = coarse_grid.value().impl();
    } else {
        coarse_grid_impl = coarsened_grid(stride_coord).impl();
    }

    torch::Tensor pool_data = detail::autograd::AvgPoolGrid::apply(
        impl(), coarse_grid_impl, pool_factor_coord, stride_coord, data.jdata())[0];

    return std::make_pair(coarse_grid_impl->jaggedTensor(pool_data), GridBatch(coarse_grid_impl));
}

std::pair<JaggedTensor, GridBatch>
GridBatch::subdivide(Vec3iOrScalar subdiv_factor,
                     const JaggedTensor &data,
                     const std::optional<JaggedTensor> mask,
                     std::optional<GridBatch> fine_grid) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        data.ldim() == 1,
        "Expected data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        data.ldim(),
        "list dimensions");
    if (mask.has_value()) {
        TORCH_CHECK_VALUE(
            mask.value().ldim() == 1,
            "Expected mask to have 1 list dimension, i.e. be a single list of coordinate values, but got",
            mask.value().ldim(),
            "list dimensions");
    }
    const nanovdb::Coord upsampleFactorCoord = subdiv_factor.value();

    c10::intrusive_ptr<detail::GridBatchImpl> fineGrid;
    if (fine_grid.has_value()) {
        fineGrid = fine_grid.value().impl();
    } else {
        fineGrid = subdivided_grid(subdiv_factor, mask).impl();
    }

    torch::Tensor subdivData = detail::autograd::UpsampleGrid::apply(
        impl(), fineGrid, upsampleFactorCoord, data.jdata())[0];

    return std::make_pair(fineGrid->jaggedTensor(subdivData), GridBatch(fineGrid));
}

JaggedTensor
GridBatch::read_from_dense(const torch::Tensor &dense_data, const Vec3iBatch &dense_origins) const {
    detail::RAIIDeviceGuard guard(device());
    torch::Tensor retData =
        detail::autograd::ReadFromDense::apply(impl(), dense_data, dense_origins)[0];
    return impl()->jaggedTensor(retData);
}

torch::Tensor
GridBatch::write_to_dense(const JaggedTensor &sparse_data,
                          const std::optional<Vec3iBatch> &min_coord,
                          const std::optional<Vec3i> &grid_size) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        sparse_data.ldim() == 1,
        "Expected sparse_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        sparse_data.ldim(),
        "list dimensions");
    return detail::autograd::ReadIntoDense::apply(
        impl(), sparse_data.jdata(), min_coord, grid_size)[0];
}

JaggedTensor
GridBatch::fill_from_grid(const JaggedTensor &other_features,
                          const GridBatch &other_grid,
                          float default_value) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        other_features.ldim() == 1,
        "Expected features to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        other_features.ldim(),
        "list dimensions");
    torch::Tensor retData = detail::autograd::FillFromGrid::apply(
        other_grid.impl(), impl(), other_features.jdata(), default_value)[0];

    return impl()->jaggedTensor(retData);
}

JaggedTensor
GridBatch::grid_to_world(const JaggedTensor &ijk) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    torch::Tensor ret = detail::autograd::TransformPoints::apply(
        impl(), ijk, ijk.jdata(), true /*isInverse*/, false /*isDual*/)[0];

    return ijk.jagged_like(ret);
}

JaggedTensor
GridBatch::world_to_grid(const JaggedTensor &points) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    torch::Tensor ret = detail::autograd::TransformPoints::apply(
        impl(), points, points.jdata(), false /* isInverse*/, false /*isDual*/)[0];

    return points.jagged_like(ret);
}

JaggedTensor
GridBatch::sample_trilinear(const JaggedTensor &points, const JaggedTensor &voxel_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        voxel_data.ldim() == 1,
        "Expected voxel_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        voxel_data.ldim(),
        "list dimensions");
    torch::Tensor ret = detail::autograd::SampleGridTrilinear::apply(
        impl(), points, voxel_data.jdata(), false /*returnGrad*/)[0];
    return points.jagged_like(ret);
}

std::vector<JaggedTensor>
GridBatch::sample_trilinear_with_grad(const JaggedTensor &points,
                                      const JaggedTensor &voxel_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        voxel_data.ldim() == 1,
        "Expected voxel_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        voxel_data.ldim(),
        "list dimensions");
    std::vector<torch::Tensor> ret = detail::autograd::SampleGridTrilinear::apply(
        impl(), points, voxel_data.jdata(), true /*returnGrad*/);

    return {points.jagged_like(ret[0]), points.jagged_like(ret[1])};
}

JaggedTensor
GridBatch::sample_bezier(const JaggedTensor &points, const JaggedTensor &voxel_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        voxel_data.ldim() == 1,
        "Expected voxel_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        voxel_data.ldim(),
        "list dimensions");
    torch::Tensor ret = detail::autograd::SampleGridBezier::apply(
        impl(), points, voxel_data.jdata(), false /*returnGrad*/)[0];
    return points.jagged_like(ret);
}

std::vector<JaggedTensor>
GridBatch::sample_bezier_with_grad(const JaggedTensor &points,
                                   const JaggedTensor &voxel_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        voxel_data.ldim() == 1,
        "Expected voxel_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        voxel_data.ldim(),
        "list dimensions");
    auto ret = detail::autograd::SampleGridBezier::apply(
        impl(), points, voxel_data.jdata(), true /*returnGrad*/);
    return {points.jagged_like(ret[0]), points.jagged_like(ret[1])};
}

JaggedTensor
GridBatch::splat_trilinear(const JaggedTensor &points, const JaggedTensor &points_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        points_data.ldim() == 1,
        "Expected points_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points_data.ldim(),
        "list dimensions");
    torch::Tensor ret =
        detail::autograd::SplatIntoGridTrilinear::apply(impl(), points, points_data.jdata())[0];
    if (grid_count() == 1) {
        return JaggedTensor(ret);
    } else {
        return impl()->jaggedTensor(ret);
    }
}

JaggedTensor
GridBatch::splat_bezier(const JaggedTensor &points, const JaggedTensor &points_data) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        points_data.ldim() == 1,
        "Expected points_data to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points_data.ldim(),
        "list dimensions");
    torch::Tensor ret =
        detail::autograd::SplatIntoGridBezier::apply(impl(), points, points_data.jdata())[0];
    if (grid_count() == 1) {
        return JaggedTensor(ret);
    } else {
        return impl()->jaggedTensor(ret);
    }
}

void
GridBatch::set_from_mesh(const JaggedTensor &mesh_vertices,
                         const JaggedTensor &mesh_faces,
                         const Vec3dBatchOrScalar &voxel_sizes,
                         const Vec3dBatch &origins) {
    detail::RAIIDeviceGuard guard(device());
    impl()->checkDevice(mesh_vertices);
    impl()->checkDevice(mesh_faces);
    const int64_t numGrids = mesh_vertices.joffsets().size(0) - 1;
    const std::vector<nanovdb::Vec3d> voxSizesVec =
        voxel_sizes.value(numGrids, true /* onlyPositive */, "voxel_sizes");
    const std::vector<nanovdb::Vec3d> voxOriginsVec =
        origins.value(numGrids, false /* onlyPositive */, "voxel_origins");
    mImpl = detail::GridBatchImpl::createFromMesh(
        mesh_vertices, mesh_faces, voxSizesVec, voxOriginsVec);
}

void
GridBatch::set_from_points(const JaggedTensor &points,
                           const Vec3dBatchOrScalar &voxel_sizes,
                           const Vec3dBatch &origins) {
    detail::RAIIDeviceGuard guard(device());
    impl()->checkDevice(points);
    const int64_t numGrids = points.joffsets().size(0) - 1;
    const std::vector<nanovdb::Vec3d> voxSizesVec =
        voxel_sizes.value(numGrids, true /* onlyPositive */, "voxel_sizes");
    const std::vector<nanovdb::Vec3d> voxOriginsVec =
        origins.value(numGrids, false /* onlyPositive */, "voxel_origins");
    mImpl = detail::GridBatchImpl::createFromPoints(points, voxSizesVec, voxOriginsVec);
}

void
GridBatch::set_from_nearest_voxels_to_points(const JaggedTensor &points,
                                             const Vec3dBatchOrScalar &voxel_sizes,
                                             const Vec3dBatch &origins) {
    detail::RAIIDeviceGuard guard(device());
    impl()->checkDevice(points);
    const int64_t numGrids = points.joffsets().size(0) - 1;
    const std::vector<nanovdb::Vec3d> voxSizesVec =
        voxel_sizes.value(numGrids, true /* onlyPositive */, "voxel_sizes");
    const std::vector<nanovdb::Vec3d> voxOriginsVec =
        origins.value(numGrids, false /* onlyPositive */, "voxel_origins");
    mImpl =
        detail::GridBatchImpl::createFromNeighborVoxelsToPoints(points, voxSizesVec, voxOriginsVec);
}

void
GridBatch::set_from_ijk(const JaggedTensor &coords,
                        const Vec3dBatchOrScalar &voxel_sizes,
                        const Vec3dBatch &origins) {
    detail::RAIIDeviceGuard guard(device());
    impl()->checkDevice(coords);
    const int64_t numGrids = coords.joffsets().size(0) - 1;
    const std::vector<nanovdb::Vec3d> voxSizesVec =
        voxel_sizes.value(numGrids, true /* onlyPositive */, "voxel_sizes");
    const std::vector<nanovdb::Vec3d> voxOriginsVec =
        origins.value(numGrids, false /* onlyPositive */, "voxel_origins");
    mImpl = detail::GridBatchImpl::createFromIjk(coords, voxSizesVec, voxOriginsVec);
}

void
GridBatch::set_from_dense_grid(const int64_t num_grids,
                               const Vec3i &dense_dims,
                               const Vec3i &ijk_min,
                               const Vec3dBatchOrScalar &voxel_sizes,
                               const Vec3dBatch &origins,
                               std::optional<torch::Tensor> mask) {
    detail::RAIIDeviceGuard guard(device());
    const nanovdb::Coord &denseDims = dense_dims.value();
    const nanovdb::Coord &ijkMin    = ijk_min.value();
    impl()->checkDevice(mask);
    std::vector<nanovdb::Vec3d> voxSizesVec =
        voxel_sizes.value(num_grids, true /* onlyPositive */, "voxel_sizes");
    std::vector<nanovdb::Vec3d> voxOriginsVec =
        origins.value(num_grids, false /* onlyPositive */, "voxel_origins");
    mImpl = detail::GridBatchImpl::dense(
        num_grids, device(), denseDims, ijkMin, voxSizesVec, voxOriginsVec, mask);
}

GridBatch
GridBatch::dual_grid(bool exclude_border) const {
    return GridBatch(impl()->dual(exclude_border));
}

GridBatch
GridBatch::coarsened_grid(Vec3iOrScalar branch_factor) const {
    const nanovdb::Coord branchFactorCoord = branch_factor.value();
    return GridBatch(impl()->coarsen(branchFactorCoord));
}

GridBatch
GridBatch::subdivided_grid(Vec3iOrScalar subdiv_factor,
                           const std::optional<JaggedTensor> mask) const {
    const nanovdb::Coord subdivFactorCoord = subdiv_factor.value();
    return GridBatch(impl()->upsample(subdivFactorCoord, mask));
}

GridBatch
GridBatch::clipped_grid(const Vec3iBatch &ijk_min, const Vec3iBatch &ijk_max) const {
    const std::vector<nanovdb::Coord> &bboxMins =
        ijk_min.value(impl()->batchSize(), false, "ijk_min");
    const std::vector<nanovdb::Coord> &bboxMaxs =
        ijk_max.value(impl()->batchSize(), false, "ijk_max");
    return GridBatch(impl()->clip(bboxMins, bboxMaxs));
}

std::pair<JaggedTensor, GridBatch>
GridBatch::clip(const JaggedTensor &features,
                const Vec3iBatch &ijk_min,
                const Vec3iBatch &ijk_max) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        features.ldim() == 1,
        "Expected features to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        features.ldim(),
        "list dimensions");

    impl()->checkDevice(features);
    TORCH_CHECK(features.rsize(0) == total_voxels(), "Value count of features does not match grid");
    TORCH_CHECK(features.num_outer_lists() == grid_count(),
                "Batch size of features does not match grid.");
    TORCH_CHECK(torch::equal(features.joffsets(), impl()->voxelOffsets()),
                "Offsets of features does not match grid.");

    const std::vector<nanovdb::Coord> &bboxMins =
        ijk_min.value(impl()->batchSize(), false, "ijk_min");
    const std::vector<nanovdb::Coord> &bboxMaxs =
        ijk_max.value(impl()->batchSize(), false, "ijk_max");

    auto [clippedGridPtr, activeVoxelMask] = impl()->clipWithMask(bboxMins, bboxMaxs);

    // features clipped to voxels in bounds
    JaggedTensor clippedFeatures = features.rmask(activeVoxelMask.jdata());
    GridBatch clippedGrid(clippedGridPtr);
    return std::make_pair(clippedFeatures, clippedGrid);
}

std::vector<JaggedTensor>
GridBatch::marching_cubes(const JaggedTensor &field, double level) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        field.ldim() == 1,
        "Expected field to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        field.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(field.is_floating_point(), "field must have a floating point type");
    TORCH_CHECK_VALUE(field.numel() == total_voxels(), "Value count not match!");
    TORCH_CHECK_VALUE(field.num_outer_lists() == grid_count(), "Batch size not match!");
    torch::Tensor fieldJdata = field.jdata();
    if (fieldJdata.dim() == 0) {
        fieldJdata = fieldJdata.unsqueeze(0);
    }
    if (fieldJdata.dim() != 1) {
        fieldJdata = fieldJdata.squeeze();
    }
    TORCH_CHECK(fieldJdata.dim() == 1,
                std::string("Expected field to have 1 effective dimension but got ") +
                    std::to_string(field.rdim()) + " dimensions");
    impl()->checkDevice(field);
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchMarchingCubes<DeviceTag>(*impl(), fieldJdata, level);
    });
}

JaggedTensor
GridBatch::sparse_conv_halo(const JaggedTensor &input,
                            const torch::Tensor &weight,
                            int variant) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        input.ldim() == 1,
        "Expected input to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        input.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(input.is_floating_point(), "input must have a floating point type");
    TORCH_CHECK_VALUE(input.rsize(0) == total_voxels(), "Value count not match!");
    TORCH_CHECK_VALUE(input.num_outer_lists() == grid_count(), "Batch size not match!");
    impl()->checkDevice(input);
    torch::Tensor ret =
        detail::autograd::SparseConvolutionHalo::apply(impl(), input.jdata(), weight, variant)[0];
    return input.jagged_like(ret);
}

GridBatch
GridBatch::conv_grid(Vec3iOrScalar kernel_size, Vec3iOrScalar stride) const {
    return GridBatch(impl()->convolutionOutput(kernel_size.value(), stride.value()));
}

GridBatch
GridBatch::dilated_grid(const int dilation) const {
    return GridBatch(impl()->dilate(dilation));
}

GridBatch
GridBatch::merged_grid(const GridBatch &other) const {
    return GridBatch(impl()->merge(other.impl()));
}

std::vector<JaggedTensor>
GridBatch::voxels_along_rays(const JaggedTensor &ray_origins,
                             const JaggedTensor &ray_directions,
                             int64_t max_vox,
                             double eps,
                             bool return_ijk,
                             bool cumulative) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ray_origins.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_origins.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        ray_directions.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_directions.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchVoxelsAlongRays<DeviceTag>(
            *impl(), ray_origins, ray_directions, max_vox, eps, return_ijk, cumulative);
    });
}

JaggedTensor
GridBatch::segments_along_rays(const JaggedTensor &ray_origins,
                               const JaggedTensor &ray_directions,
                               int64_t max_segments,
                               double eps) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ray_origins.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_origins.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        ray_directions.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_directions.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchSegmentsAlongRays<DeviceTag>(
            *impl(), ray_origins, ray_directions, max_segments, eps);
    });
}

JaggedTensor
GridBatch::ray_implicit_intersection(const JaggedTensor &ray_origins,
                                     const JaggedTensor &ray_directions,
                                     const JaggedTensor &gridScalars,
                                     double eps) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ray_origins.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_origins.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        ray_directions.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_directions.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        gridScalars.ldim() == 1,
        "Expected grid_scalars to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        gridScalars.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchRayImplicitIntersection<DeviceTag>(
            *impl(), ray_origins, ray_directions, gridScalars, eps);
    });
}

JaggedTensor
GridBatch::uniform_ray_samples(const JaggedTensor &ray_origins,
                               const JaggedTensor &ray_directions,
                               const JaggedTensor &t_min,
                               const JaggedTensor &t_max,
                               double step_size,
                               double cone_angle,
                               bool include_end_segments,
                               bool return_midpoint,
                               double eps) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ray_origins.ldim() == 1,
        "Expected ray_origins to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_origins.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        ray_directions.ldim() == 1,
        "Expected ray_directions to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ray_directions.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        t_min.ldim() == 1,
        "Expected t_min to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        t_min.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(
        t_max.ldim() == 1,
        "Expected t_max to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        t_max.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchUniformRaySamples<DeviceTag>(*impl(),
                                                                       ray_origins,
                                                                       ray_directions,
                                                                       t_min,
                                                                       t_max,
                                                                       step_size,
                                                                       cone_angle,
                                                                       include_end_segments,
                                                                       return_midpoint,
                                                                       eps);
    });
}

JaggedTensor
GridBatch::neighbor_indexes(const JaggedTensor &ijk, int32_t extent, int32_t bitshift) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    TORCH_CHECK_VALUE(extent >= 0, "extent must be >= 0");
    nanovdb::Coord extentMin(-extent, -extent, -extent);
    nanovdb::Coord extentMax(extent, extent, extent);
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchVoxelNeighborhood<DeviceTag>(
            *impl(), ijk, extentMin, extentMax, bitshift);
    });
}

JaggedTensor
GridBatch::points_in_active_voxel(const JaggedTensor &points) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL(device(), [&]() {
        return fvdb::detail::ops::dispatchPointsInGrid<DeviceTag>(*impl(), points);
    });
}

JaggedTensor
GridBatch::cubes_intersect_grid(const JaggedTensor &cube_centers,
                                const Vec3dOrScalar &cube_min,
                                const Vec3dOrScalar &cube_max) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        cube_centers.ldim() == 1,
        "Expected cube_centers to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        cube_centers.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchCubesIntersectGrid<DeviceTag>(
            *impl(), cube_centers, cube_min, cube_max);
    });
}

JaggedTensor
GridBatch::cubes_in_grid(const JaggedTensor &cube_centers,
                         const Vec3dOrScalar &cube_min,
                         const Vec3dOrScalar &cube_max) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        cube_centers.ldim() == 1,
        "Expected cube_centers to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        cube_centers.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchCubesInGrid<DeviceTag>(
            *impl(), cube_centers, cube_min, cube_max);
    });
}

JaggedTensor
GridBatch::coords_in_active_voxel(const JaggedTensor &ijk) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchCoordsInGrid<DeviceTag>(*impl(), ijk);
    });
}

JaggedTensor
GridBatch::ijk_to_index(const JaggedTensor &ijk, bool cumulative) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchIjkToIndex<DeviceTag>(*impl(), ijk, cumulative);
    });
}

JaggedTensor
GridBatch::ijk_to_inv_index(const JaggedTensor &ijk, bool cumulative) const {
    detail::RAIIDeviceGuard guard(device());
    TORCH_CHECK_VALUE(
        ijk.ldim() == 1,
        "Expected ijk to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        ijk.ldim(),
        "list dimensions");
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchIjkToInvIndex<DeviceTag>(*impl(), ijk, cumulative);
    });
}

JaggedTensor
GridBatch::ijk() const {
    detail::RAIIDeviceGuard guard(device());
    return FVDB_DISPATCH_KERNEL(this->device(), [&]() {
        return fvdb::detail::ops::dispatchActiveGridCoords<DeviceTag>(*impl());
    });
}

std::vector<JaggedTensor>
GridBatch::viz_edge_network(bool returnVoxelCoordinates) const {
    detail::RAIIDeviceGuard guard(device());
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return fvdb::detail::ops::dispatchGridEdgeNetwork<DeviceTag>(*impl(),
                                                                     returnVoxelCoordinates);
    });
}

} // namespace fvdb
