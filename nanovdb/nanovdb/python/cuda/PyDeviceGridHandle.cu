// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <nanovdb/cuda/DeviceBuffer.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineDeviceGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto cls = defineGridHandle<BufferT>(m, "DeviceGridHandle")
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
            "cuda_t"_a.noconvert());

#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, HandleMethod, DeviceMethod) \
    cls.def(DeviceMethod,                                        \
            nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<T>), \
            "n"_a = 0,                                                 \
            nb::rv_policy::reference_internal);
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, HandleMethod, DeviceMethod) \
    cls.def(DeviceMethod,                                        \
            nb::overload_cast<uint32_t>(&GridHandle<BufferT>::template deviceGrid<T>), \
            "n"_a = 0,                                                 \
            nb::rv_policy::reference_internal);
#include "../BuildTypes.def"

    cls.def(
            "deviceUpload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceUpload(nullptr, sync); }, "sync"_a = true)
        .def(
            "deviceDownload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceDownload(nullptr, sync); }, "sync"_a = true);
}

} // namespace pynanovdb

#endif
