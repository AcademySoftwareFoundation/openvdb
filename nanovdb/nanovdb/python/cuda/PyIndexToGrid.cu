// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyIndexToGrid.h"

#include <nanobind/ndarray.h>

#include <cstdint>

#include <nanovdb/tools/cuda/IndexToGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

// Scalar destination value type (e.g. float, double): d_srcValues is a flat,
// 1-D device array indexed by the IndexGrid's per-element uint64 indices.
template<typename DstBuildT, typename SrcBuildT>
void defineIndexToGridScalar(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<SrcBuildT>*                              d_srcGrid,
           nb::ndarray<DstBuildT, nb::ndim<1>, nb::c_contig, nb::device::cuda> values,
           uintptr_t                                                  stream) {
            cudaStream_t      s = reinterpret_cast<cudaStream_t>(stream);
            const DstBuildT*  d_values = values.data();
            // indexToGrid launches kernels and synchronizes the stream; pure
            // CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::indexToGrid<DstBuildT, SrcBuildT>(
                d_srcGrid, d_values, nanovdb::cuda::DeviceBuffer(), s);
        },
        "d_srcGrid"_a,
        "values"_a,
        "stream"_a = 0,
        "Combine a device IndexGrid (ValueIndex / ValueOnIndex) with a flat "
        "1-D device array of destination values into a fresh device "
        "GridHandle of the destination value type. values is indexed by the "
        "IndexGrid's per-voxel / per-node uint64 indices, so it must be sized "
        "to cover the grid's value count. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");
}

// Vec3 destination value type (Vec3f / Vec3d): d_srcValues is an (N, 3) device
// array of the matching scalar; Vec3<T> is exactly three contiguous scalars so
// the c_contig tensor reinterprets element-for-element as Vec3<T>*.
template<typename DstBuildT, typename SrcBuildT>
void defineIndexToGridVec3(nb::module_& m, const char* name)
{
    using ScalarT = typename DstBuildT::ValueType;
    m.def(
        name,
        [](nanovdb::NanoGrid<SrcBuildT>* d_srcGrid,
           nb::ndarray<ScalarT, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> values,
           uintptr_t                     stream) {
            cudaStream_t     s = reinterpret_cast<cudaStream_t>(stream);
            const DstBuildT* d_values = reinterpret_cast<const DstBuildT*>(values.data());
            // indexToGrid launches kernels and synchronizes the stream; pure
            // CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::indexToGrid<DstBuildT, SrcBuildT>(
                d_srcGrid, d_values, nanovdb::cuda::DeviceBuffer(), s);
        },
        "d_srcGrid"_a,
        "values"_a,
        "stream"_a = 0,
        "Combine a device IndexGrid (ValueIndex / ValueOnIndex) with an (N, 3) "
        "device array of destination vector values into a fresh device "
        "GridHandle of the destination Vec3 value type. The array is indexed "
        "by the IndexGrid's per-voxel / per-node uint64 indices, so N must "
        "cover the grid's value count. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");
}

// Destination types: float / double (scalar) and Vec3f / Vec3d (vector), each
// for both index source types (ValueIndex and ValueOnIndex). DstBuildT is
// restricted by IndexToGrid's processLeafsKernel static_assert to non-special
// types, so the quantized / index / mask BuildTs are intentionally NOT
// instantiated.
template void defineIndexToGridScalar<float, nanovdb::ValueIndex>(nb::module_&, const char*);
template void defineIndexToGridScalar<float, nanovdb::ValueOnIndex>(nb::module_&, const char*);
template void defineIndexToGridScalar<double, nanovdb::ValueIndex>(nb::module_&, const char*);
template void defineIndexToGridScalar<double, nanovdb::ValueOnIndex>(nb::module_&, const char*);

template void
defineIndexToGridVec3<nanovdb::Vec3f, nanovdb::ValueIndex>(nb::module_&, const char*);
template void
defineIndexToGridVec3<nanovdb::Vec3f, nanovdb::ValueOnIndex>(nb::module_&, const char*);
template void
defineIndexToGridVec3<nanovdb::Vec3d, nanovdb::ValueIndex>(nb::module_&, const char*);
template void
defineIndexToGridVec3<nanovdb::Vec3d, nanovdb::ValueOnIndex>(nb::module_&, const char*);

} // namespace pynanovdb
