// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyPointsToGrid.h"

#include <nanobind/ndarray.h>

#include <nanovdb/tools/cuda/PointsToGrid.cuh>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

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

template<typename BuildT> void definePointsToGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::ndarray<int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> tensor) {
            NdArrayCoordPtr                            points(tensor.data(), tensor.stride(0), tensor.stride(1));
            nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
            auto                                       handle = converter.getHandle(points, tensor.shape(0));
            return handle;
        },
        "tensor"_a);
}

template void definePointsToGrid<nanovdb::math::Rgba8>(nb::module_&, const char*);

} // namespace pynanovdb
