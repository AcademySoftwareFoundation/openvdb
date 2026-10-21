// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <nanovdb/cuda/GridHandle.cuh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// The Python DeviceGridHandle keeps its upload/download API, which has no
// single-space Python replacement yet, so these use the deprecated dual-space
// buffer transfers until it moves to single-space storage.
NANOVDB_SUPPRESS_DEPRECATION_BEGIN
static void deviceUpload(GridHandle<nanovdb::cuda::DualDeviceBuffer>& handle, bool sync) { handle.buffer().deviceUpload(cudaStream_t(0), sync); }
static void deviceDownload(GridHandle<nanovdb::cuda::DualDeviceBuffer>& handle, bool sync) { handle.buffer().deviceDownload(cudaStream_t(0), sync); }
NANOVDB_SUPPRESS_DEPRECATION_END

void defineDeviceGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DualDeviceBuffer;
    defineGridHandle<BufferT>(m, "DeviceGridHandle")
        .def(
            "__init__",
            [](GridHandle<BufferT>&                                 handle,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cpu>  cpu_t,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cuda> cuda_t) {
                assert(cpu_t.size() == cuda_t.size());
                BufferT buffer(cpu_t.size() * sizeof(uint32_t), cpu_t.data(), cuda_t.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "cpu_t"_a.noconvert(),
            "cuda_t"_a.noconvert())
        .def("deviceFloatGrid", nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<float>), "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceDoubleGrid", nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<double>), "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceInt32Grid", nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<int32_t>), "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceVec3fGrid", nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<Vec3f>), "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceRGBA8Grid",
             nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<math::Rgba8>),
             "n"_a = 0,
             nb::rv_policy::reference_internal)
        .def("deviceUpload", &deviceUpload, "sync"_a = true)
        .def("deviceDownload", &deviceDownload, "sync"_a = true);
}

} // namespace pynanovdb

#endif
