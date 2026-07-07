// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyMeshToGrid.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

void defineMeshToGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::ndarray<float, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda>   points,
           nb::ndarray<int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> triangles,
           double voxelSize, float halfWidth, const std::string& gridName,
           uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // Vec3f / Vec3i are three contiguous scalars, so the c_contig
            // (N, 3) tensors reinterpret element-for-element.
            const nanovdb::Vec3f* d_points =
                reinterpret_cast<const nanovdb::Vec3f*>(points.data());
            const nanovdb::Vec3i* d_triangles =
                reinterpret_cast<const nanovdb::Vec3i*>(triangles.data());
            const uint32_t pointCount = static_cast<uint32_t>(points.shape(0));
            const uint32_t triangleCount = static_cast<uint32_t>(triangles.shape(0));
            nanovdb::Map map;
            map.set(voxelSize, nanovdb::Vec3d(0.0, 0.0, 0.0), 1.0);
            // MeshToGrid launches kernels and synchronizes the stream; pure
            // CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::MeshToGrid<nanovdb::ValueOnIndex> converter(
                d_points, pointCount, d_triangles, triangleCount, map, s);
            converter.setNarrowBandWidth(halfWidth);
            if (!gridName.empty())
                converter.setGridName(gridName);
            // Compute a checksum during the build so the result validates.
            converter.setChecksum(nanovdb::CheckMode::Full);
            return converter.getHandleAndUDF();
        },
        "points"_a,
        "triangles"_a,
        "voxelSize"_a = 1.0,
        "halfWidth"_a = 3.0,
        "gridName"_a = "",
        "stream"_a = 0,
        "Rasterize a triangle mesh into a narrow-band unsigned distance field on "
        "the device. points is an (N, 3) float32 CUDA array of vertex world "
        "positions; triangles is an (M, 3) int32 CUDA array of vertex indices. "
        "Returns a tuple (handle, udf): handle is a device ValueOnIndex "
        "GridHandle holding the narrow-band topology, and udf is a "
        "nanovdb.cuda.DeviceBuffer of (valueCount) float32 unsigned distances "
        "(in voxel units) indexed by the grid's per-voxel value index -- feed it "
        "straight to indexToGrid to bake a Float distance grid. voxelSize is the "
        "world size of a voxel; halfWidth is the narrow-band half-width in voxel "
        "units; stream is a raw CUDA stream handle (Python int; 0 = default "
        "stream).");
}

} // namespace pynanovdb
