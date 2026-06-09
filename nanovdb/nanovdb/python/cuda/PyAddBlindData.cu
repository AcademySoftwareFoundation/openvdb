// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyAddBlindData.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

#include <nanovdb/tools/cuda/AddBlindData.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT, typename BlindDataT>
void defineAddBlindData(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>*                                   d_grid,
           nb::ndarray<BlindDataT, nb::ndim<1>, nb::c_contig, nb::device::cuda> blindData,
           nanovdb::GridBlindDataClass                                  blindClass,
           nanovdb::GridBlindDataSemantic                               semantics,
           const std::string&                                          dataName,
           uintptr_t                                                   stream) {
            cudaStream_t        s = reinterpret_cast<cudaStream_t>(stream);
            const BlindDataT*   d_blindData = blindData.data();
            const uint64_t      valueCount = static_cast<uint64_t>(blindData.size());
            // addBlindData copies the grid into a fresh device buffer with the
            // blind data appended and launches kernels on the stream; pure
            // CUDA touching no Python objects, so release the GIL. dataName is
            // already a C++-owned std::string (nanobind's stl/string caster
            // materialized it from the Python str), so its c_str() stays valid
            // across the GIL release for the duration of this call frame.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::addBlindData<BuildT, BlindDataT>(
                d_grid, d_blindData, valueCount, blindClass, semantics,
                dataName.c_str(), nanovdb::cuda::DeviceBuffer(), s);
        },
        "d_grid"_a,
        "blindData"_a,
        "blindClass"_a = nanovdb::GridBlindDataClass::Unknown,
        "semantics"_a = nanovdb::GridBlindDataSemantic::Unknown,
        "name"_a = "",
        "stream"_a = 0,
        "Append a flat 1-D device array of blind data to a copy of a device "
        "grid and return a fresh device GridHandle with the blind data "
        "attached. valueCount is taken from the array length; blindClass and "
        "semantics tag the new GridBlindMetaData entry (a RuntimeError is "
        "raised on an invalid combination). stream is a raw CUDA stream "
        "handle (Python int; 0 = default stream).");
}

// Grid BuildTs x blind-data element types. addBlindData itself has no BuildT
// restriction beyond BufferTraits<BufferT>::hasDeviceDual (satisfied by
// DeviceBuffer); we expose the common combinations. The same overload name is
// reused so nanobind dispatches on the grid class plus the array dtype.
template void defineAddBlindData<float, float>(nb::module_&, const char*);
template void defineAddBlindData<float, double>(nb::module_&, const char*);
template void defineAddBlindData<float, uint32_t>(nb::module_&, const char*);
template void defineAddBlindData<double, float>(nb::module_&, const char*);
template void defineAddBlindData<double, double>(nb::module_&, const char*);
template void defineAddBlindData<double, uint32_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueOnIndex, float>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueOnIndex, double>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueOnIndex, uint32_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueIndex, float>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueIndex, double>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueIndex, uint32_t>(nb::module_&, const char*);
// Signed-integer blind payloads (e.g. integer labels / ids), for the same grid
// BuildTs.
template void defineAddBlindData<float, int32_t>(nb::module_&, const char*);
template void defineAddBlindData<float, int64_t>(nb::module_&, const char*);
template void defineAddBlindData<double, int32_t>(nb::module_&, const char*);
template void defineAddBlindData<double, int64_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueOnIndex, int32_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueOnIndex, int64_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueIndex, int32_t>(nb::module_&, const char*);
template void defineAddBlindData<nanovdb::ValueIndex, int64_t>(nb::module_&, const char*);

} // namespace pynanovdb
