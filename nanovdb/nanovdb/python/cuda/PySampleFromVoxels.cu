// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PySampleFromVoxels.h"

#include <nanobind/ndarray.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/SampleFromVoxels.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace {

template<typename BuildT> __global__ void sampleFromVoxels(unsigned int numPoints, const BuildT* points, const NanoGrid<BuildT>* d_grid, BuildT* values)
{
    using TreeT = NanoTree<BuildT>;
    using Vec3T = math::Vec3<BuildT>;

    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < numPoints; i += blockDim.x * gridDim.x) {
        Vec3T worldPos(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        Vec3T indexPos = d_grid->worldToIndex(worldPos);

        math::SampleFromVoxels<TreeT, 1, false> sampler(d_grid->tree());
        values[i] = sampler(indexPos);
    }
}

template<typename BuildT>
__global__ void sampleFromVoxels(unsigned int numPoints, const BuildT* points, const NanoGrid<BuildT>* d_grid, BuildT* values, BuildT* gradients)
{
    using TreeT = NanoTree<BuildT>;
    using Vec3T = math::Vec3<BuildT>;

    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < numPoints; i += blockDim.x * gridDim.x) {
        Vec3T worldPos(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        Vec3T indexPos = d_grid->worldToIndex(worldPos);

        math::SampleFromVoxels<TreeT, 1, false> sampler(d_grid->tree());
        values[i] = sampler(indexPos);

        Vec3T inv2Dx = (BuildT).5 / d_grid->voxelSize();
        Vec3T gradient = Vec3T(sampler(indexPos + Vec3T(1, 0, 0)) - sampler(indexPos - Vec3T(1, 0, 0)),
                               sampler(indexPos + Vec3T(0, 1, 0)) - sampler(indexPos - Vec3T(0, 1, 0)),
                               sampler(indexPos + Vec3T(0, 0, 1)) - sampler(indexPos - Vec3T(0, 0, 1))) *
                         inv2Dx;
        gradients[3 * i] = gradient[0];
        gradients[3 * i + 1] = gradient[1];
        gradients[3 * i + 2] = gradient[2];
    }
}

} // namespace

namespace pynanovdb {

template<typename BuildT> void defineSampleFromVoxels(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::ndarray<BuildT, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> points,
           NanoGrid<BuildT>*                                                          d_grid,
           nb::ndarray<BuildT, nb::shape<-1>, nb::device::cuda>                  values) {
            constexpr unsigned int numThreads = 128;
            unsigned int           numBlocks = (points.shape(0) + numThreads - 1) / numThreads;
            sampleFromVoxels<<<numBlocks, numThreads>>>(points.shape(0), points.data(), d_grid, values.data());
        },
        "points"_a,
        "d_grid"_a,
        "values"_a);
    m.def(
        name,
        [](nb::ndarray<BuildT, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> points,
           NanoGrid<BuildT>*                                                          d_grid,
           nb::ndarray<BuildT, nb::shape<-1>, nb::device::cuda>                  values,
           nb::ndarray<BuildT, nb::shape<-1, 3>, nb::device::cuda>               gradients) {
            constexpr unsigned int numThreads = 128;
            unsigned int           numBlocks = (points.shape(0) + numThreads - 1) / numThreads;
            sampleFromVoxels<<<numBlocks, numThreads>>>(points.shape(0), points.data(), d_grid, values.data(), gradients.data());
        },
        "points"_a,
        "d_grid"_a,
        "values"_a,
        "gradients"_a);
}

template void defineSampleFromVoxels<float>(nb::module_&, const char*);
template void defineSampleFromVoxels<double>(nb::module_&, const char*);

} // namespace pynanovdb
