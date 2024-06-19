#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Types.h"
#include "JaggedTensor.h"


namespace pybind11 {
namespace detail {

const static inline pybind11::module TORCH_MODULE = py::module_::import("torch");


template <> struct type_caster<fvdb::JaggedTensor> : public type_caster_base<fvdb::JaggedTensor> {
    using base = type_caster_base<fvdb::JaggedTensor>;
public:
    fvdb::JaggedTensor jag_value;

    bool load(handle src, bool convert) {
        if (THPVariable_Check(src.ptr())) {
            // TODO: (@fwilliams) Might need to reinterpret steal here?
            torch::Tensor data = THPVariable_Unpack(src.ptr());
            jag_value = fvdb::JaggedTensor({data});
            value = &jag_value;
            return true;
        } else {
            return base::load(src, convert);
        }
    }

    static handle cast(const fvdb::JaggedTensor& src, return_value_policy policy, handle parent) {
        return base::cast(src, policy, parent);
    }
};


template <> struct type_caster<torch::ScalarType> : public type_caster_base<torch::ScalarType> {
    using base = type_caster_base<torch::ScalarType>;
public:
    torch::ScalarType st_value;

    bool load(handle src, bool convert) {
        if (THPDtype_Check(src.ptr())) {
            st_value = reinterpret_cast<THPDtype*>(src.ptr())->scalar_type;
            value = &st_value;
            return true;
        } else {
            return base::load(src, convert);
        }
    }

    static handle cast(const at::ScalarType& src, return_value_policy policy, handle parent) {
        auto result = TORCH_MODULE.attr(fvdb::detail::TorchScalarTypeToStr(src).c_str());
        Py_INCREF(result.ptr());
        return result;
    }
};


template <> struct type_caster<fvdb::TorchDeviceOrString> : public type_caster_base<fvdb::TorchDeviceOrString> {
    using base = type_caster_base<fvdb::TorchDeviceOrString>;
public:
    fvdb::TorchDeviceOrString dev_value;

    bool load(handle src, bool convert) {

        std::string deviceString;
        try {
            deviceString = src.cast<std::string>();
        } catch (pybind11::cast_error& e) {
            if (THPDevice_Check(src.ptr())) {
                dev_value = reinterpret_cast<THPDevice*>(src.ptr())->device;
                value = &dev_value;
                return true;
            } else {
                return false;
            }
        }
        dev_value = deviceString;
        value = &dev_value;
        return true;
    }
};


template <> struct type_caster<at::indexing::TensorIndex> : public type_caster_base<at::indexing::TensorIndex> {
    at::indexing::TensorIndex idx_value = at::indexing::None;

    bool load(handle src, bool convert) {

        if (py::isinstance<py::ellipsis>(src)) {
            idx_value = at::indexing::Ellipsis;
            value = &idx_value;
            return true;
        }
        if (py::isinstance<py::slice>(src)) {
            py::ssize_t start, stop, step;
            py::slice slice = src.cast<py::slice>();
            PySlice_Unpack(slice.ptr(), &start, &stop, &step);

            // FIXME: (@fwilliams) -- This is a bit weird. Ideally we want the same behavior as Pyslice_ComputeEx
            //                        but we don't know the size apriori here. In any case, we only use this for
            //                        JaggedTensor which doesn't support negative step sizes anyway.
            if (step < 0) {
                if (start >= PY_SSIZE_T_MAX) {
                    start = -1;
                }

                if (stop <= PY_SSIZE_T_MIN) {
                    stop = 0;
                }
            }
            idx_value = at::indexing::Slice(start, stop, step);
            value = &idx_value;
            return true;
        }
        if (py::isinstance<py::int_>(src)) {
            idx_value = src.cast<int64_t>();
            value = &idx_value;
            return true;
        }
        if (py::isinstance<py::none>(src)) {
            idx_value = at::indexing::None;
            value = &idx_value;
            return true;
        }
        return false;
    }
};


template <> struct type_caster<fvdb::StringOrListOfStrings> : public type_caster_base<fvdb::StringOrListOfStrings> {
    using base = type_caster_base<fvdb::StringOrListOfStrings>;
public:
    fvdb::StringOrListOfStrings strlist_value;

    bool load(handle src, bool convert) {
        try {
            std::string deviceString = src.cast<std::string>();
            strlist_value = fvdb::StringOrListOfStrings(deviceString);
            value = &strlist_value;
            return true;
        } catch (pybind11::cast_error& e) {
            std::vector<std::string> stringList = src.cast<std::vector<std::string>>();
            strlist_value = fvdb::StringOrListOfStrings(stringList);
            value = &strlist_value;
            return true;
        }
        return false;
    }
};


template <> struct type_caster<fvdb::NanoVDBFileGridIdentifier> : public type_caster_base<fvdb::NanoVDBFileGridIdentifier> {
    using base = type_caster_base<fvdb::NanoVDBFileGridIdentifier>;
public:
    fvdb::NanoVDBFileGridIdentifier id_value;

    bool load(handle src, bool convert) {

        if (src.is_none()) {
            value = &id_value;
            return true;
        }

        if (base::load(src, convert)) {
            return true;
        }
        else if (py::isinstance<py::int_>(src)) {
            id_value = src.cast<uint64_t>();
            value = &id_value;
            return true;
        }
        else if (py::isinstance<py::str>(src)) {
            id_value = src.cast<std::string>();
            value = &id_value;
            return true;
        }
        else if (py::isinstance<py::list>(src)) {
            try{
                id_value = src.cast<std::vector<uint64_t>>();
            }
            catch (pybind11::cast_error& e) {
                try{
                    id_value = src.cast<std::vector<std::string>>();
                } catch (pybind11::cast_error& e) {
                    return false;
                }
            }
            value = &id_value;
            return true;
        }

        return false;
    }
};


template <typename CoordRetT>
bool loadCoordType(handle src, bool convert, CoordRetT& outValue, void** outPtr) {

    std::vector<typename CoordRetT::ValueType> coordVec;
    try {
        coordVec = src.cast<std::vector<typename CoordRetT::ValueType>>();
    } catch (pybind11::cast_error& e) {
        typename CoordRetT::ValueType coordScalar;
        try {
            if constexpr (CoordRetT::SupportsScalarCast) {
                coordScalar = src.cast<typename CoordRetT::ValueType>();
            } else {
                throw pybind11::cast_error();
            }
        } catch (pybind11::cast_error& e) {
            if (THPVariable_Check(src.ptr())) {
                torch::Tensor data = THPVariable_Unpack(src.ptr());
                outValue = CoordRetT(data);
                *outPtr = &outValue;
                return true;
            } else {
                return false;
            }
        }
        if constexpr (CoordRetT::SupportsScalarCast) {
            outValue = CoordRetT(coordScalar);
            *outPtr = &outValue;
            return true;
        }
    }
    outValue = CoordRetT(coordVec);
    *outPtr = &outValue;
    return true;
}


template <typename CoordBatchRetT>
bool loadVecBatch(handle src, bool convert, CoordBatchRetT& outValue, void** outPtr) {
    using ValType = typename CoordBatchRetT::ValueType;

    std::vector<ValType> coordVec;
    try {
        if constexpr (CoordBatchRetT::SupportsBroadcast) {
            coordVec = src.cast<std::vector<ValType>>();
        } else {
            throw pybind11::cast_error();
        }
    } catch (pybind11::cast_error& e) {
        std::vector<std::vector<ValType>> coordVecVec;
        try {
            coordVecVec = src.cast<std::vector<std::vector<ValType>>>();
        } catch (pybind11::cast_error& e) {
            ValType val;
            try {
                if constexpr (CoordBatchRetT::SupportsScalarCast) {
                    val = src.cast<ValType>();
                } else {
                    throw pybind11::cast_error();
                }
            } catch (pybind11::cast_error& e) {
                if (THPVariable_Check(src.ptr())) {
                    torch::Tensor data = THPVariable_Unpack(src.ptr());
                    outValue = CoordBatchRetT(data);
                    *outPtr = &outValue;
                    return true;
                } else {
                    return false;
                }
            }
            if constexpr (CoordBatchRetT::SupportsScalarCast) {
                outValue = CoordBatchRetT(val);
                *outPtr = &outValue;
                return true;
            }
        }

        outValue = CoordBatchRetT(coordVecVec);
        *outPtr = &outValue;
        return true;
    }
    if constexpr (CoordBatchRetT::SupportsBroadcast) {
        outValue = CoordBatchRetT(coordVec);
        *outPtr = &outValue;
        return true;
    }
}


#define INBOUND_TYPE_CASTER(DataType, loadFunction) \
    template <> struct type_caster<DataType> : public type_caster_base<DataType> { \
        using base = type_caster_base<DataType>;                                   \
    public:                                                                        \
        DataType coord_value;                                                      \
        bool load(handle src, bool convert) {                                      \
            if (!loadFunction<DataType>(src, convert, coord_value, &value)){       \
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
