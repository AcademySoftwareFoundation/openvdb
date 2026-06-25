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

// One-thread kernel that overwrites the grid's GridClass field in place. Grid
// publicly derives GridData, so mGridClass is reachable on the device pointer.
template<typename BuildT>
__global__ void setGridClassKernel(nanovdb::NanoGrid<BuildT>* d_grid,
                                   nanovdb::GridClass gridClass)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) d_grid->mGridClass = gridClass;
}

// Device-side mutable grid-header metadata (the read-only counterparts already
// exist as getters on the grid objects)
template<typename BuildT>
void defineDeviceGridMetadata(nb::module_& m)
{
    m.def(
        "setGridClass",
        [](nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::GridClass gridClass,
           uintptr_t stream) {
            if (!d_grid) throw nb::value_error("setGridClass: d_grid is None.");
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            nb::gil_scoped_release release;
            setGridClassKernel<BuildT><<<1, 1, 0, s>>>(d_grid, gridClass);
            // Refresh the checksum (preserving its existing mode) so that
            // checksum-validating readers still accept the grid after the
            // class field changed; a no-op for grids with checksum disabled.
            nanovdb::tools::cuda::updateChecksum(
                reinterpret_cast<nanovdb::GridData*>(d_grid), s);
            cudaStreamSynchronize(s);
        },
        "d_grid"_a,
        "gridClass"_a,
        "stream"_a = 0,
        "Set the device grid's GridClass (e.g. GridClass.LevelSet) in place and "
        "refresh its checksum, preserving the checksum mode (returns None). "
        "stream is a raw CUDA stream handle (Python int; 0 = default stream).");
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

// Same BuildT set as the checksum entries above.
template void defineDeviceGridMetadata<float>(nb::module_&);
template void defineDeviceGridMetadata<double>(nb::module_&);
template void defineDeviceGridMetadata<int16_t>(nb::module_&);
template void defineDeviceGridMetadata<int32_t>(nb::module_&);
template void defineDeviceGridMetadata<int64_t>(nb::module_&);
template void defineDeviceGridMetadata<uint8_t>(nb::module_&);
template void defineDeviceGridMetadata<uint32_t>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Vec3f>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Vec3d>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Vec4f>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Vec4d>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::math::Rgba8>(nb::module_&);
template void defineDeviceGridMetadata<bool>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Fp4>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Fp8>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::Fp16>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::FpN>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::ValueIndex>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::ValueOnIndex>(nb::module_&);
template void defineDeviceGridMetadata<nanovdb::ValueMask>(nb::module_&);

} // namespace pynanovdb
