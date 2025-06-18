// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef PYTHON_TYPECASTERS_H
#define PYTHON_TYPECASTERS_H

#include <fvdb/Types.h>

#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace pybind11 {
namespace detail {

const static inline pybind11::module TORCH_MODULE = py::module_::import("torch");

// Already defined in upstream pytorch: https://github.com/pytorch/pytorch/pull/126865
// (starting from version 2.4)
#if (!defined(TORCH_VERSION_MAJOR) || (TORCH_VERSION_MAJOR < 2) || \
     (TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR < 4))
template <> struct type_caster<torch::ScalarType> : public type_caster_base<torch::ScalarType> {
    using base = type_caster_base<torch::ScalarType>;

  public:
    torch::ScalarType st_value;

    bool
    load(handle src, bool convert) {
        if (THPDtype_Check(src.ptr())) {
            st_value = reinterpret_cast<THPDtype *>(src.ptr())->scalar_type;
            value    = &st_value;
            return true;
        } else {
            return base::load(src, convert);
        }
    }

    static handle
    cast(const at::ScalarType &src, return_value_policy policy, handle parent) {
        auto result = TORCH_MODULE.attr(fvdb::detail::TorchScalarTypeToStr(src).c_str());
        Py_INCREF(result.ptr());
        return result;
    }
};
#endif

template <typename CoordRetT>
bool
loadCoordType(handle src, bool convert, CoordRetT &outValue, void **outPtr) {
    std::vector<typename CoordRetT::ValueType> coordVec;
    try {
        coordVec = src.cast<std::vector<typename CoordRetT::ValueType>>();
    } catch (pybind11::cast_error &e) {
        typename CoordRetT::ValueType coordScalar;
        try {
            if constexpr (CoordRetT::SupportsScalarCast) {
                coordScalar = src.cast<typename CoordRetT::ValueType>();
            } else {
                throw pybind11::cast_error();
            }
        } catch (pybind11::cast_error &e) {
            if (THPVariable_Check(src.ptr())) {
                torch::Tensor data = THPVariable_Unpack(src.ptr());
                outValue           = CoordRetT(data);
                *outPtr            = &outValue;
                return true;
            } else {
                return false;
            }
        }
        if constexpr (CoordRetT::SupportsScalarCast) {
            outValue = CoordRetT(coordScalar);
            *outPtr  = &outValue;
            return true;
        }
    }
    outValue = CoordRetT(coordVec);
    *outPtr  = &outValue;
    return true;
}

template <typename CoordBatchRetT>
bool
loadVecBatch(handle src, bool convert, CoordBatchRetT &outValue, void **outPtr) {
    using ValType = typename CoordBatchRetT::ValueType;

    std::vector<ValType> coordVec;
    try {
        if constexpr (CoordBatchRetT::SupportsBroadcast) {
            coordVec = src.cast<std::vector<ValType>>();
        } else {
            throw pybind11::cast_error();
        }
    } catch (pybind11::cast_error &e) {
        std::vector<std::vector<ValType>> coordVecVec;
        try {
            coordVecVec = src.cast<std::vector<std::vector<ValType>>>();
        } catch (pybind11::cast_error &e) {
            ValType val;
            try {
                if constexpr (CoordBatchRetT::SupportsScalarCast) {
                    val = src.cast<ValType>();
                } else {
                    throw pybind11::cast_error();
                }
            } catch (pybind11::cast_error &e) {
                if (THPVariable_Check(src.ptr())) {
                    torch::Tensor data = THPVariable_Unpack(src.ptr());
                    outValue           = CoordBatchRetT(data);
                    *outPtr            = &outValue;
                    return true;
                } else {
                    return false;
                }
            }
            if constexpr (CoordBatchRetT::SupportsScalarCast) {
                outValue = CoordBatchRetT(val);
                *outPtr  = &outValue;
                return true;
            }
        }

        outValue = CoordBatchRetT(coordVecVec);
        *outPtr  = &outValue;
        return true;
    }
    if constexpr (CoordBatchRetT::SupportsBroadcast) {
        outValue = CoordBatchRetT(coordVec);
        *outPtr  = &outValue;
        return true;
    }
}

#define INBOUND_TYPE_CASTER(DataType, loadFunction)                                \
    template <> struct type_caster<DataType> : public type_caster_base<DataType> { \
        using base = type_caster_base<DataType>;                                   \
                                                                                   \
      public:                                                                      \
        DataType coord_value;                                                      \
        bool                                                                       \
        load(handle src, bool convert) {                                           \
            if (!loadFunction<DataType>(src, convert, coord_value, &value)) {      \
                return base::load(src, convert);                                   \
            }                                                                      \
            return true;                                                           \
        }                                                                          \
    };

INBOUND_TYPE_CASTER(fvdb::Vec3d, loadCoordType)
INBOUND_TYPE_CASTER(fvdb::Vec3dOrScalar, loadCoordType)
INBOUND_TYPE_CASTER(fvdb::Vec3i, loadCoordType)
INBOUND_TYPE_CASTER(fvdb::Vec3iOrScalar, loadCoordType)
INBOUND_TYPE_CASTER(fvdb::Vec4i, loadCoordType)
INBOUND_TYPE_CASTER(fvdb::Vec3dBatchOrScalar, loadVecBatch)
INBOUND_TYPE_CASTER(fvdb::Vec3dBatch, loadVecBatch)
INBOUND_TYPE_CASTER(fvdb::Vec3iBatch, loadVecBatch)

} // namespace detail
} // namespace pybind11

#endif // PYTHON_TYPECASTERS_H
