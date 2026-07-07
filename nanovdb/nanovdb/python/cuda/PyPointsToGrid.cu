// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyPointsToGrid.h"

#include <nanobind/ndarray.h>

#include <cstdint>

#include <nanovdb/tools/cuda/PointsToGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

// Fancy pointer adapter over an (N, 3) int32 device tensor that dereferences to
// a nanovdb::Coord, i.e. INDEX-space voxel coordinates. Used by the
// voxelsToGrid path (BuildT != Point, plus the Point/index build types).
class NdArrayCoordPtr
{
    const int32_t* data;
    int64_t        stride0, stride1;

public:
    __hostdev__ NdArrayCoordPtr(const int32_t* data, int64_t stride0, int64_t stride1)
        : data(data)
        , stride0(stride0)
        , stride1(stride1)
    {
    }
    __hostdev__ inline nanovdb::Coord operator[](size_t i) const
    {
        nanovdb::Coord::ValueType x = data[i * stride0 + 0 * stride1];
        nanovdb::Coord::ValueType y = data[i * stride0 + 1 * stride1];
        nanovdb::Coord::ValueType z = data[i * stride0 + 2 * stride1];
        return nanovdb::Coord(x, y, z);
    }
    __hostdev__ inline nanovdb::Coord operator*() const
    {
        size_t                    i = 0;
        nanovdb::Coord::ValueType x = data[i * stride0 + 0 * stride1];
        nanovdb::Coord::ValueType y = data[i * stride0 + 1 * stride1];
        nanovdb::Coord::ValueType z = data[i * stride0 + 2 * stride1];
        return nanovdb::Coord(x, y, z);
    }
};

// Fancy pointer adapter over an (N, 3) float/double device tensor that
// dereferences to a nanovdb::math::Vec3<T>, i.e. WORLD-space point positions.
// Used by the pointsToGrid path which builds NanoGrid<Point>. The world
// pointsToGrid requires the dereferenced element to be Vec3f or Vec3d.
template<typename ScalarT> class NdArrayVec3Ptr
{
    const ScalarT* data;
    int64_t        stride0, stride1;

public:
    using element_type = nanovdb::math::Vec3<ScalarT>;

    __hostdev__ NdArrayVec3Ptr(const ScalarT* data, int64_t stride0, int64_t stride1)
        : data(data)
        , stride0(stride0)
        , stride1(stride1)
    {
    }
    __hostdev__ inline element_type operator[](size_t i) const
    {
        return element_type(data[i * stride0 + 0 * stride1],
                            data[i * stride0 + 1 * stride1],
                            data[i * stride0 + 2 * stride1]);
    }
    __hostdev__ inline element_type operator*() const
    {
        return element_type(data[0 * stride1], data[1 * stride1], data[2 * stride1]);
    }
};

template<typename BuildT> void defineVoxelsToGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::ndarray<int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> tensor,
           double                                                                voxelSize,
           uintptr_t                                                             stream) {
            cudaStream_t   s = reinterpret_cast<cudaStream_t>(stream);
            NdArrayCoordPtr points(tensor.data(), tensor.stride(0), tensor.stride(1));
            const size_t    count = tensor.shape(0);
            // voxelsToGrid only builds the grid topology (no blind data); pure
            // CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::voxelsToGrid<BuildT>(
                points, count, voxelSize, nanovdb::cuda::DeviceBuffer(), s);
        },
        "tensor"_a,
        "voxelSize"_a = 1.0,
        "stream"_a = 0,
        "Rasterize the given (N, 3) int32 device tensor of index-space voxel "
        "coordinates into a fresh device GridHandle. voxelSize is the world "
        "size of a voxel; stream is a raw CUDA stream handle (Python int; "
        "0 = default stream).");
}

template<typename BuildT> void definePointsToGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::ndarray<BuildT, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> tensor,
           double                                                               voxelSize,
           uintptr_t                                                            stream) {
            cudaStream_t            s = reinterpret_cast<cudaStream_t>(stream);
            NdArrayVec3Ptr<BuildT>  points(tensor.data(), tensor.stride(0), tensor.stride(1));
            const size_t            count = tensor.shape(0);
            // Build a NanoGrid<Point> at the requested fixed voxel size,
            // encoding the world points as blind data. Mirror the converter
            // usage of the legacy Rgba8 path (the fixed-voxelSize free function
            // has no out-of-line definition in the header), but set the Point
            // build type and the world->index map scale.
            nb::gil_scoped_release                            release;
            nanovdb::tools::cuda::PointsToGrid<nanovdb::Point> converter(
                voxelSize, nanovdb::Vec3d(0.0), s);
            converter.setPointType(nanovdb::PointType::Default);
            return converter.getHandle(points, count);
        },
        "tensor"_a,
        "voxelSize"_a = 1.0,
        "stream"_a = 0,
        "Rasterize the given (N, 3) float or double device tensor of "
        "WORLD-space point positions into a fresh device GridHandle of type "
        "NanoGrid<Point>, encoding the point coordinates as blind data. "
        "voxelSize is the world size of a voxel; stream is a raw CUDA stream "
        "handle (Python int; 0 = default stream).");
}

// voxelsToGrid (index-space int32 Coord input). The legacy Rgba8 binding maps
// onto defineVoxelsToGrid<Rgba8>; additional build types build matching grids.
//
// NOTE: voxelsToGrid<Point> is intentionally NOT instantiated. The C++
// PointsToGrid<Point>::countNodes static_asserts that Point coordinates be
// Vec3f or Vec3d (PointsToGrid.cuh:590), so the int32-Coord voxelsToGrid input
// is rejected at compile time for BuildT == Point. Build Point grids from
// world-space float/double positions via definePointsToGrid<float|double>.
template void defineVoxelsToGrid<nanovdb::math::Rgba8>(nb::module_&, const char*);
template void defineVoxelsToGrid<nanovdb::ValueOnIndex>(nb::module_&, const char*);
template void defineVoxelsToGrid<nanovdb::ValueIndex>(nb::module_&, const char*);

// pointsToGrid (world-space float/double input -> NanoGrid<Point>).
template void definePointsToGrid<float>(nb::module_&, const char*);
template void definePointsToGrid<double>(nb::module_&, const char*);

} // namespace pynanovdb
