// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDistributedPointsToGrid.h"

#include <nanobind/ndarray.h>

#include <cstdint>

#include <nanovdb/cuda/DeviceMesh.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

template<typename BuildT> void defineDistributedPointsToGrid(nb::module_& m, const char* name)
{
    using ConverterT = nanovdb::tools::cuda::DistributedPointsToGrid<BuildT>;
    using BufferT = nanovdb::cuda::UnifiedBuffer;

    nb::class_<ConverterT>(m, name,
        "Multi-GPU builder of a NanoVDB grid from an array of index-space voxel "
        "coordinates, distributed over a DeviceMesh. Construct with a DeviceMesh "
        "(which must outlive this object) plus a voxel scale and translation, "
        "then call getHandle(voxels, count). On a single-GPU mesh this still "
        "runs the trivial single-device path.")
        .def(
            "__init__",
            [](ConverterT* self, const nanovdb::cuda::DeviceMesh& mesh,
               double scale, nb::tuple trans) {
                nanovdb::Vec3d t(0.0);
                if (trans.size() == 3)
                    t = nanovdb::Vec3d(nb::cast<double>(trans[0]),
                                       nb::cast<double>(trans[1]),
                                       nb::cast<double>(trans[2]));
                else if (trans.size() != 0)
                    throw nb::value_error("translation must be a 3-tuple or empty.");
                nb::gil_scoped_release release;
                new (self) ConverterT(mesh, scale, t);
            },
            "mesh"_a, "scale"_a = 1.0, "translation"_a = nb::make_tuple(0.0, 0.0, 0.0),
            nb::keep_alive<1, 2>(),  // keep the DeviceMesh alive for our lifetime
            "Construct a converter over the given DeviceMesh. scale is the "
            "uniform voxel size and translation is a 3-tuple world offset used "
            "to build the output grid's index-to-world map. The DeviceMesh is "
            "held by reference and MUST outlive this converter.")
        .def(
            "getHandle",
            [](ConverterT& self,
               nb::ndarray<int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda_managed> voxels) {
                // The (N, 3) c-contiguous int32 array is bit-compatible with a
                // dense nanovdb::Coord[N] (Coord == 3 x int32). The pipeline
                // issues cudaMemAdvise / cudaMemPrefetchAsync on this pointer,
                // so the memory MUST be CUDA managed (unified) memory — hence the
                // nb::device::cuda_managed constraint (DLPack kDLCUDAManaged=13).
                // Plain device memory (cuda=2) is rejected: prefetch to a device
                // ordinal is invalid on non-managed memory.
                auto* coords = reinterpret_cast<nanovdb::Coord*>(voxels.data());
                const size_t count = voxels.shape(0);
                nb::gil_scoped_release release;
                return self.template getHandle<nanovdb::Coord*, BufferT>(
                    coords, count, BufferT());
            },
            "voxels"_a,
            "Rasterize the given (N, 3) int32 array of index-space voxel "
            "coordinates into a fresh UnifiedGridHandle of type "
            "NanoGrid<BuildT>. The array MUST be backed by CUDA managed "
            "(unified) memory — the multi-GPU pipeline applies memory advise "
            "and prefetch directly to its pointer.");
}

template void defineDistributedPointsToGrid<nanovdb::ValueOnIndex>(nb::module_&, const char*);
template void defineDistributedPointsToGrid<nanovdb::ValueIndex>(nb::module_&, const char*);
template void defineDistributedPointsToGrid<nanovdb::math::Rgba8>(nb::module_&, const char*);

} // namespace pynanovdb

#endif
