// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include <torch/extension.h>

#include "FVDB.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"


void bind_jagged_tensor(py::module& m) {

    py::class_<fvdb::JaggedTensor>(m, "JaggedTensor")
        .def(py::init<std::vector<std::vector<torch::Tensor>>&>(), py::arg("tensor_list"))
        .def(py::init<std::vector<torch::Tensor>&>(), py::arg("tensor_list"), R"_FVDB_(
            Initialize JaggedTensor from a list of tensors.

            Args:
                tensor_list (list of torch.Tensor): Tensors in this list must have the same shape except for the first (jagged) dimension.

            Returns:
                jt (JaggedTensor): The concatenated JaggedTensor with the same order as the list.)_FVDB_")
        .def(py::init<torch::Tensor>(), py::arg("tensor"), R"_FVDB_(
            Initialize JaggedTensor from one single tensor.

            Args:
                tensor (torch.Tensor): Tensor to be converted to a JaggedTensor.

            Returns:
                jt (JaggedTensor): The JaggedTensor of batch size 1 containing this single tensor.)_FVDB_")
        .def_property("jdata", &fvdb::JaggedTensor::jdata, &fvdb::JaggedTensor::set_data, "The data of the JaggedTensor.")
        .def_property_readonly("jidx", [](const fvdb::JaggedTensor& self) {
            // FIXME: (@fwilliams) This is a bit ugly and the abstraction leaks
            if (self.jidx().numel() == 0 && self.jdata().numel() > 0) {
                return torch::zeros({self.jdata().size(0)}, torch::TensorOptions().device(self.device()).dtype(torch::kInt16));
            } else {
                return self.jidx();
            }
        }, "The indices indicating the batch index where the element belong to.")
        .def_property_readonly("joffsets", &fvdb::JaggedTensor::joffsets, "A [batch_size, 2] array where each row contains the start and end row index in jdata.")

        .def_property_readonly("num_tensors", &fvdb::JaggedTensor::num_tensors, "The number of tensors in the JaggedTensor.")

        .def_property_readonly("is_cuda", &fvdb::JaggedTensor::is_cuda, "Whether the JaggedTensor is on a CUDA device.")
        .def_property_readonly("is_cpu", &fvdb::JaggedTensor::is_cpu, "Whether the JaggedTensor is on a CPU device.")

        .def("__getitem__", &fvdb::JaggedTensor::index)
        .def("__len__", &fvdb::JaggedTensor::num_outer_lists)

        .def("__add__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"), py::is_operator())
        .def("__add__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"), py::is_operator())
        .def("__add__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"), py::is_operator())
        .def("__add__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"), py::is_operator())

        .def("__iadd__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator+=), py::arg("other"), py::is_operator())
        .def("__iadd__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator+=), py::arg("other"), py::is_operator())
        .def("__iadd__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator+=), py::arg("other"), py::is_operator())
        .def("__iadd__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator+=), py::arg("other"), py::is_operator())

        .def("__sub__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"), py::is_operator())
        .def("__sub__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"), py::is_operator())
        .def("__sub__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"), py::is_operator())
        .def("__sub__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"), py::is_operator())

        .def("__isub__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator-=), py::arg("other"), py::is_operator())
        .def("__isub__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator-=), py::arg("other"), py::is_operator())
        .def("__isub__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator-=), py::arg("other"), py::is_operator())
        .def("__isub__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator-=), py::arg("other"), py::is_operator())

        .def("__neg__", py::overload_cast<>(&fvdb::JaggedTensor::operator-, py::const_), py::is_operator())

        .def("__mul__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"), py::is_operator())
        .def("__mul__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"), py::is_operator())
        .def("__mul__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"), py::is_operator())
        .def("__mul__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"), py::is_operator())

        .def("__imul__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator*=), py::arg("other"), py::is_operator())
        .def("__imul__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator*=), py::arg("other"), py::is_operator())
        .def("__imul__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator*=), py::arg("other"), py::is_operator())
        .def("__imul__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator*=), py::arg("other"), py::is_operator())

        .def("__gt__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator>, py::const_), py::arg("other"), py::is_operator())
        .def("__gt__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator>, py::const_), py::arg("other"), py::is_operator())
        .def("__gt__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator>, py::const_), py::arg("other"), py::is_operator())
        .def("__gt__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator>, py::const_), py::arg("other"), py::is_operator())

        .def("__ge__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator>=, py::const_), py::arg("other"), py::is_operator())
        .def("__ge__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator>=, py::const_), py::arg("other"), py::is_operator())
        .def("__ge__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator>=, py::const_), py::arg("other"), py::is_operator())
        .def("__ge__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator>=, py::const_), py::arg("other"), py::is_operator())

        .def("__lt__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator<, py::const_), py::arg("other"), py::is_operator())
        .def("__lt__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator<, py::const_), py::arg("other"), py::is_operator())
        .def("__lt__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator<, py::const_), py::arg("other"), py::is_operator())
        .def("__lt__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator<, py::const_), py::arg("other"), py::is_operator())

        .def("__le__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator<=, py::const_), py::arg("other"), py::is_operator())
        .def("__le__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator<=, py::const_), py::arg("other"), py::is_operator())
        .def("__le__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator<=, py::const_), py::arg("other"), py::is_operator())
        .def("__le__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator<=, py::const_), py::arg("other"), py::is_operator())

        .def("__eq__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator==, py::const_), py::arg("other"), py::is_operator())
        .def("__eq__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator==, py::const_), py::arg("other"), py::is_operator())
        .def("__eq__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator==, py::const_), py::arg("other"), py::is_operator())
        .def("__eq__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator==, py::const_), py::arg("other"), py::is_operator())

        .def("__ne__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator!=, py::const_), py::arg("other"), py::is_operator())
        .def("__ne__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator!=, py::const_), py::arg("other"), py::is_operator())
        .def("__ne__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator!=, py::const_), py::arg("other"), py::is_operator())
        .def("__ne__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator!=, py::const_), py::arg("other"), py::is_operator())

        .def("__truediv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"), py::is_operator())
        .def("__truediv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"), py::is_operator())
        .def("__truediv__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"), py::is_operator())
        .def("__truediv__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"), py::is_operator())

        .def("__itruediv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator/=), py::arg("other"), py::is_operator())
        .def("__itruediv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator/=), py::arg("other"), py::is_operator())
        .def("__itruediv__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator/=), py::arg("other"), py::is_operator())
        .def("__itruediv__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator/=), py::arg("other"), py::is_operator())

        .def("__floordiv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"), py::is_operator())
        .def("__floordiv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"), py::is_operator())
        .def("__floordiv__", py::overload_cast<const int>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"), py::is_operator())
        .def("__floordiv__", py::overload_cast<const float>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"), py::is_operator())

        .def("__ifloordiv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::floordiveq), py::arg("other"), py::is_operator())
        .def("__ifloordiv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::floordiveq), py::arg("other"), py::is_operator())
        .def("__ifloordiv__", py::overload_cast<const int>(&fvdb::JaggedTensor::floordiveq), py::arg("other"), py::is_operator())
        .def("__ifloordiv__", py::overload_cast<const float>(&fvdb::JaggedTensor::floordiveq), py::arg("other"), py::is_operator())

        .def("__pow__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::pow, py::const_), py::arg("other"), py::is_operator())
        .def("__pow__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::pow, py::const_), py::arg("other"), py::is_operator())
        .def("__pow__", py::overload_cast<const int>(&fvdb::JaggedTensor::pow, py::const_), py::arg("other"), py::is_operator())
        .def("__pow__", py::overload_cast<const float>(&fvdb::JaggedTensor::pow, py::const_), py::arg("other"), py::is_operator())

        .def("__ipow__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::poweq), py::arg("other"), py::is_operator())
        .def("__ipow__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::poweq), py::arg("other"), py::is_operator())
        .def("__ipow__", py::overload_cast<const int>(&fvdb::JaggedTensor::poweq), py::arg("other"), py::is_operator())
        .def("__ipow__", py::overload_cast<const float>(&fvdb::JaggedTensor::poweq), py::arg("other"), py::is_operator())

        .def("__mod__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"), py::is_operator())
        .def("__mod__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"), py::is_operator())
        .def("__mod__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"), py::is_operator())
        .def("__mod__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"), py::is_operator())

        .def("__imod__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator%=), py::arg("other"), py::is_operator())
        .def("__imod__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator%=), py::arg("other"), py::is_operator())
        .def("__imod__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator%=), py::arg("other"), py::is_operator())
        .def("__imod__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator%=), py::arg("other"), py::is_operator())

        .def("float", [](const fvdb::JaggedTensor& self) { return self.to(torch::kFloat32); })
        .def("double", [](const fvdb::JaggedTensor& self) { return self.to(torch::kFloat64); })
        .def("int", [](const fvdb::JaggedTensor& self) { return self.to(torch::kInt32); })
        .def("long", [](const fvdb::JaggedTensor& self) { return self.to(torch::kInt64); })

        .def("cpu", [](const fvdb::JaggedTensor& self) { return self.to(torch::kCPU); })
        .def("cuda", [](const fvdb::JaggedTensor& self) { return self.to(torch::kCUDA); })

        .def("to", [](fvdb::JaggedTensor& self, fvdb::TorchDeviceOrString device) { return self.to(device.value(), self.scalar_type()); })
        .def("to", [](fvdb::JaggedTensor& self, torch::ScalarType dtype) { return self.to(self.device(), dtype); })
        .def("to", [](const fvdb::JaggedTensor& self, fvdb::TorchDeviceOrString device) { return self.to(device.value()); }, py::arg("device"))

        .def("type", [](const fvdb::JaggedTensor& self, torch::ScalarType dtype) { return self.to(dtype); })
        .def("type_as", [](const fvdb::JaggedTensor& self, const fvdb::JaggedTensor& jagged_tensor) { return self.to(jagged_tensor.dtype()); })
        .def("rmask", &fvdb::JaggedTensor::rmask, py::arg("mask"))

        .def("jagged_like", &fvdb::JaggedTensor::jagged_like, py::arg("data"))

        .def("requires_grad_", [](const fvdb::JaggedTensor& self, bool requires_grad) { return self.set_requires_grad(requires_grad); })
        .def("detach", [](const fvdb::JaggedTensor& self) { return self.detach(); })
        .def("clone", &fvdb::JaggedTensor::clone)

        .def("jreshape", py::overload_cast<const std::vector<int64_t>&>(&fvdb::JaggedTensor::jreshape, py::const_), py::arg("lshape"))
        .def("jreshape", py::overload_cast<const std::vector<std::vector<int64_t>>&>(&fvdb::JaggedTensor::jreshape, py::const_), py::arg("lshape"))
        .def("jreshape_as", &fvdb::JaggedTensor::jreshape_as, py::arg("other"))

        .def("jflatten", &fvdb::JaggedTensor::jflatten, py::arg("dim") = 0)

        .def("unbind", [](const fvdb::JaggedTensor& self) {
            if (self.ldim() == 1) {
                return py::cast(self.unbind1());
            } else if (self.ldim() == 2) {
                return py::cast(self.unbind2());
            } else {
                TORCH_CHECK(false, "fVDB does not currently support JaggedTensors with list dimension > 2 (i.e. a list of lists)");
            }
        })

        .def("sqrt", &fvdb::JaggedTensor::sqrt)
        .def("abs", &fvdb::JaggedTensor::abs)
        .def("round", &fvdb::JaggedTensor::round, py::arg("decimals") = 0)
        .def("floor", &fvdb::JaggedTensor::floor)
        .def("ceil", &fvdb::JaggedTensor::ceil)

        .def("sqrt_", &fvdb::JaggedTensor::sqrt_)
        .def("abs_", &fvdb::JaggedTensor::abs_)
        .def("round_", &fvdb::JaggedTensor::round_, py::arg("decimals") = 0)
        .def("floor_", &fvdb::JaggedTensor::floor_)
        .def("ceil_", &fvdb::JaggedTensor::ceil_)

        // .def("jagged_argsort", &fvdb::JaggedTensor::jagged_argsort, R"_FVDB_(
        //     Returns the indices that would sort each batch element in ascending order, note that jdata has to be 1-dimensional.
        //
        //     Returns:
        //         indices (torch.Tensor): An indexing tensor with the same size as jdata, that permutes the elements of data to be in sorted order.)_FVDB_")
        .def("jsum", &fvdb::JaggedTensor::jsum, py::arg("dim") = 0, py::arg("keepdim") = false, R"_FVDB_(
            Returns the sum of each batch element.

            Returns:
                sum (torch.Tensor): A tensor of size (batch_size, *) containing the sum of each batch element, feature dimensions are preserved.)_FVDB_")
        .def("jmin", &fvdb::JaggedTensor::jmin, py::arg("dim") = 0, py::arg("keepdim") = false, R"_FVDB_(
            Returns the minimum of each batch element.

            Returns:
                values (torch.Tensor): A tensor of size (batch_size, *) containing the minimum value.
                indices (torch.Tensor): A tensor of size (batch_size, *) containing the argmin.)_FVDB_")
        .def("jmax", &fvdb::JaggedTensor::jmax, py::arg("dim") = 0, py::arg("keepdim") = false, R"_FVDB_(
            Returns the maximum of each batch element.

            Returns:
                values (torch.Tensor): A tensor of size (batch_size, *) containing the maximum value.
                indices (torch.Tensor): A tensor of size (batch_size, *) containing the argmax.)_FVDB_")
        .def_property_readonly("rshape", [](const fvdb::JaggedTensor& self) { return self.jdata().sizes(); }, "The shape of the raw data tensor.")
        .def_property_readonly("lshape", [](const fvdb::JaggedTensor& self) {
            if (self.ldim() == 1) {
                return py::cast(self.lsizes1());
            } else if (self.ldim() == 2) {
                return py::cast(self.lsizes2());
            } else {
                TORCH_CHECK(false, "fVDB does not currently support JaggedTensors with list dimension > 2 (i.e. a list of lists)");
            }
         }, "The shape of jdata.")
        .def_property_readonly("ldim", &fvdb::JaggedTensor::ldim,
                               "The list dimension of this JaggedTensor. E.g. a list has ldim 1, a list-of-lists has ldim2 and so on.")
        .def_property_readonly("eshape", &fvdb::JaggedTensor::esizes, "The shape each element indexed by this JaggedTensor. Same as jdata.shape[1:].")
        .def_property_readonly("edim", &fvdb::JaggedTensor::edim, "The number of dimensions of each element indexed by this JaggedTensor. Same as len(jdata.shape[1:]).")
        .def_property_readonly("dtype", [](const fvdb::JaggedTensor& self) { return self.scalar_type(); })
        .def_property_readonly("device", &fvdb::JaggedTensor::device)
        .def_property("requires_grad", &fvdb::JaggedTensor::requires_grad, &fvdb::JaggedTensor::set_requires_grad)
        .def(py::pickle(
            [](const fvdb::JaggedTensor& t) {
                int version = 2; // 2 is the current version
                return py::make_tuple(version, t.jdata(), t.joffsets(), t.jlidx(), t.num_outer_lists());
            },
            [](py::tuple t) {
                // Francis: I forgot to version the pickle format, so we'll treat a length 3 tuple as version 0
                //          and you always need to have more or fewer than 3 things. There will be a version from
                //          now on
                if (t.size() == 3) {
                    const torch::Tensor data = THPVariable_Unpack(t[0].ptr());
                    const torch::Tensor jidx = THPVariable_Unpack(t[1].ptr()).to(fvdb::JIdxScalarType);
                    const torch::Tensor jlidx = torch::empty({0, 1}, torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(data.device()));
		            const int64_t batchSize = py::cast<int>(t[2]);
                    return fvdb::JaggedTensor::from_data_indices_and_list_ids(data, jidx, jlidx, batchSize);
                }

                if (t.size() > 0) {
                    TORCH_CHECK(false, "Invalid pickle format");
                }
                int version = t[0].cast<int>();
                if (version == 1 || version == 2) {
                    const torch::Tensor jdata = THPVariable_Unpack(t[1].ptr());

                    const torch::Tensor joffsetsLoaded = THPVariable_Unpack(t[2].ptr());
                    torch::Tensor joffsets;
                    if (version == 1) {
                        TORCH_CHECK(joffsetsLoaded.dim() == 2, "Invalid pickle format: joffsets must have shape (num_tensors, 2)");
                        using SliceT = torch::indexing::Slice;
                        joffsets = torch::empty({joffsetsLoaded.size(0) + 1}, torch::TensorOptions().dtype(fvdb::JOffsetsScalarType).device(joffsetsLoaded.device()));
                        joffsets.index_put_({SliceT(0, joffsetsLoaded.size(0))}, joffsetsLoaded.index({SliceT(), 0}));
                        joffsets.index_put_({-1}, joffsetsLoaded.index({-1, 1}));
                        joffsets = joffsets.to(fvdb::JOffsetsScalarType);
                    } else {
                        TORCH_CHECK(version == 2, "Invalid pickle format: version must be 1 or 2");
                        TORCH_CHECK(joffsetsLoaded.dim() == 1, "Invalid pickle format: joffsets must have shape (num_tensors+1)");
                        joffsets = joffsetsLoaded.to(fvdb::JOffsetsScalarType);
                    }

                    const torch::Tensor jlidx = THPVariable_Unpack(t[3].ptr());
                    const int64_t numOuterLists = t[4].cast<int64_t>();
                    TORCH_CHECK(numOuterLists == joffsets.size(0), "Invalid pickle format: numOuterLists does not match joffsets size");
                    TORCH_CHECK(jlidx.size(0) == 0 || jlidx.size(0) == joffsets.size(0), "Invalid pickle format: jlidx size does not match joffsets size");
                    return fvdb::JaggedTensor::from_data_offsets_and_list_ids(jdata, joffsets, jlidx);
                } else {
                    TORCH_CHECK(false, "Invalid JaggedTensor pickle version (got version = ", version, ")");
                }
            }
        ));

}