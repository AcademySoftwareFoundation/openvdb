// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyDeviceGridChecksum.h"

#include <cstdint>

// Pull in cuda/GridHandle.cuh first: it includes GridChecksum.cuh and then
// uses nanovdb::tools::cuda::updateChecksum, so it must be parsed before any
// translation unit sets GridChecksum.cuh's include guard (otherwise the guard
// would skip GridChecksum.cuh's body and leave updateChecksum undeclared when
// GridHandle.cuh is parsed). Matches the unittest's include ordering.
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/cuda/GridChecksum.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

// All three entries take the device NanoGrid<BuildT>* and use the GridData*
// device overloads in GridChecksum.cuh, which copy just the header host-side
// and run CRC32 on device — they never dereference the device pointer on the
// host, so reinterpreting the typed device grid pointer to GridData* is safe.
template<typename BuildT>
void defineDeviceGridChecksum(nb::module_& m)
{
    m.def(
        "evalChecksum",
        [](const nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::CheckMode mode,
           uintptr_t stream) -> nanovdb::Checksum {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            const nanovdb::GridData* d_gridData =
                reinterpret_cast<const nanovdb::GridData*>(d_grid);
            // Pure CUDA (CRC32 on device, header copied D2H); release the GIL.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::evalChecksum(d_gridData, mode, s);
        },
        "d_grid"_a,
        "mode"_a = nanovdb::CheckMode::Default,
        "stream"_a = 0,
        "Compute and return the Checksum of the device grid for the given "
        "CheckMode without modifying it. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");

    m.def(
        "validateChecksum",
        [](const nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::CheckMode mode,
           uintptr_t stream) -> bool {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            const nanovdb::GridData* d_gridData =
                reinterpret_cast<const nanovdb::GridData*>(d_grid);
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::validateChecksum(d_gridData, mode, s);
        },
        "d_grid"_a,
        "mode"_a = nanovdb::CheckMode::Default,
        "stream"_a = 0,
        "Return True iff the device grid's stored checksum matches a freshly "
        "computed one for the given CheckMode. A grid with an empty stored "
        "checksum is considered valid. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");

    m.def(
        "updateChecksum",
        [](nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::CheckMode mode,
           uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            nanovdb::GridData* d_gridData =
                reinterpret_cast<nanovdb::GridData*>(d_grid);
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::updateChecksum(d_gridData, mode, s);
        },
        "d_grid"_a,
        "mode"_a = nanovdb::CheckMode::Default,
        "stream"_a = 0,
        "Recompute and write the checksum of the device grid in place using "
        "the given CheckMode (returns None). stream is a raw CUDA stream "
        "handle (Python int; 0 = default stream).");
}

// No BuildT restriction on the GridChecksum device entries; instantiate for
// the same set the host checksum dispatch covers via callNanoGrid.
template void defineDeviceGridChecksum<float>(nb::module_&);
template void defineDeviceGridChecksum<double>(nb::module_&);
template void defineDeviceGridChecksum<int16_t>(nb::module_&);
template void defineDeviceGridChecksum<int32_t>(nb::module_&);
template void defineDeviceGridChecksum<int64_t>(nb::module_&);
template void defineDeviceGridChecksum<uint8_t>(nb::module_&);
template void defineDeviceGridChecksum<uint32_t>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Vec3f>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Vec3d>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Vec4f>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Vec4d>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::math::Rgba8>(nb::module_&);
template void defineDeviceGridChecksum<bool>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Fp4>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Fp8>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::Fp16>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::FpN>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::ValueIndex>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::ValueOnIndex>(nb::module_&);
template void defineDeviceGridChecksum<nanovdb::ValueMask>(nb::module_&);

} // namespace pynanovdb
