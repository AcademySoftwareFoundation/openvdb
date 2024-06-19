#include <torch/extension.h>

#include "FVDB.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "TypeCasters.h"


void bind_jagged_tensor(py::module& m) {

    py::class_<fvdb::JaggedTensor>(m, "JaggedTensor")
        .def(py::init<std::vector<torch::Tensor>&>(), py::arg("tensor_list"), R"_FVDB_(
            Initialize jagged tensor from a list of tensors.

            Args:
                tensor_list (list of torch.Tensor): Tensors in this list must have the same shape except for the first (jagged) dimension.

            Returns:
                jt (JaggedTensor): The concatenated JaggedTensor with the same order as the list.)_FVDB_")
        .def(py::init<torch::Tensor>(), py::arg("tensor"), R"_FVDB_(
            Initialize jagged tensor from one single tensor.

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
        .def_static("from_data_and_jidx", &fvdb::JaggedTensor::from_data_and_jidx, py::arg("data"), py::arg("jidx"), py::arg("batch_size"), R"_FVDB_(
            Initialize jagged tensor from data and jidx.

            Args:
                data (torch.Tensor): The data of the JaggedTensor.
                jidx (torch.Tensor): The indices indicating the batch index where the element belong to.
                batch_size (int): The batch size of the JaggedTensor.

            Returns:
                jt (JaggedTensor): The JaggedTensor with the given data and jidx.)_FVDB_")
        .def_static("from_data_and_offsets", &fvdb::JaggedTensor::from_data_and_offsets, py::arg("data"), py::arg("offsets"), R"_FVDB_(
            Initialize jagged tensor from data and offsets.

            Args:
                data (torch.Tensor): The data of the JaggedTensor.
                offsets (torch.Tensor): The offsets indicating the start and end row index in data.

            Returns:
                jt (JaggedTensor): The JaggedTensor with the given data and offsets.)_FVDB_")
        .def_property_readonly("joffsets", &fvdb::JaggedTensor::joffsets, "A [batch_size, 2] array where each row contains the start and end row index in jdata.")
        .def("__getitem__", &fvdb::JaggedTensor::index)
        .def("__add__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"))
        .def("__add__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"))
        .def("__add__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"))
        .def("__add__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator+, py::const_), py::arg("other"))
        .def("__sub__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"))
        .def("__sub__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"))
        .def("__sub__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"))
        .def("__sub__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator-, py::const_), py::arg("other"))
        .def("__neg__", py::overload_cast<>(&fvdb::JaggedTensor::operator-, py::const_))
        .def("__mul__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"))
        .def("__mul__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"))
        .def("__mul__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"))
        .def("__mul__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator*, py::const_), py::arg("other"))
        .def("__truediv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"))
        .def("__truediv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"))
        .def("__truediv__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"))
        .def("__truediv__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator/, py::const_), py::arg("other"))
        .def("__floordiv__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"))
        .def("__floordiv__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"))
        .def("__floordiv__", py::overload_cast<const int>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"))
        .def("__floordiv__", py::overload_cast<const float>(&fvdb::JaggedTensor::floordiv, py::const_), py::arg("other"))
        .def("__mod__", py::overload_cast<const torch::Tensor&>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"))
        .def("__mod__", py::overload_cast<const fvdb::JaggedTensor&>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"))
        .def("__mod__", py::overload_cast<const int>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"))
        .def("__mod__", py::overload_cast<const float>(&fvdb::JaggedTensor::operator%, py::const_), py::arg("other"))
        .def("round", &fvdb::JaggedTensor::round, py::arg("decimals") = 0)
        .def("float", [](const fvdb::JaggedTensor& self) { return self.to(torch::kFloat32); })
        .def("double", [](const fvdb::JaggedTensor& self) { return self.to(torch::kFloat64); })
        .def("int", [](const fvdb::JaggedTensor& self) { return self.to(torch::kInt32); })
        .def("long", [](const fvdb::JaggedTensor& self) { return self.to(torch::kInt64); })
        .def("cpu", [](const fvdb::JaggedTensor& self) { return self.to(torch::kCPU); })
        .def("cuda", [](const fvdb::JaggedTensor& self) { return self.to(torch::kCUDA); })
        .def("to", [](fvdb::JaggedTensor& self, fvdb::TorchDeviceOrString device) { return self.to(device.value(), self.scalar_type()); })
        .def("type", [](const fvdb::JaggedTensor& self, torch::ScalarType dtype) { return self.to(dtype); })
        .def("type_as", [](const fvdb::JaggedTensor& self, const fvdb::JaggedTensor& jagged_tensor) { return self.to(jagged_tensor.dtype()); })
        .def("to", [](const fvdb::JaggedTensor& self, fvdb::TorchDeviceOrString device) { return self.to(device.value()); }, py::arg("device"))
        .def("r_masked_select", &fvdb::JaggedTensor::r_masked_select, py::arg("mask"))
        .def("jagged_like", &fvdb::JaggedTensor::jagged_like, py::arg("data"))
        .def("requires_grad_", [](const fvdb::JaggedTensor& self, bool requires_grad) { return self.set_requires_grad(requires_grad); })
        .def("detach", [](const fvdb::JaggedTensor& self) { return self.detach(); })
        .def("clone", &fvdb::JaggedTensor::clone)
        .def("jagged_argsort", &fvdb::JaggedTensor::jagged_argsort, R"_FVDB_(
            Returns the indices that would sort each batch element in ascending order, note that jdata has to be 1-dimensional.

            Returns:
                indices (torch.Tensor): An indexing tensor with the same size as jdata, that permutes the elements of data to be in sorted order.)_FVDB_")
        .def("jagged_sum", &fvdb::JaggedTensor::jagged_sum, R"_FVDB_(
            Returns the sum of each batch element.

            Returns:
                sum (torch.Tensor): A tensor of size (batch_size, *) containing the sum of each batch element, feature dimensions are preserved.)_FVDB_")
        .def("jagged_min", &fvdb::JaggedTensor::jagged_min, R"_FVDB_(
            Returns the minimum of each batch element.

            Returns:
                values (torch.Tensor): A tensor of size (batch_size, *) containing the minimum value.
                indices (torch.Tensor): A tensor of size (batch_size, *) containing the argmin.)_FVDB_")
        .def("jagged_max", &fvdb::JaggedTensor::jagged_max, R"_FVDB_(
            Returns the maximum of each batch element.

            Returns:
                values (torch.Tensor): A tensor of size (batch_size, *) containing the maximum value.
                indices (torch.Tensor): A tensor of size (batch_size, *) containing the argmax.)_FVDB_")
        .def_property_readonly("r_shape", [](const fvdb::JaggedTensor& self) { return self.jdata().sizes(); }, "The shape of jdata.")
        .def_property_readonly("dtype", [](const fvdb::JaggedTensor& self) { return self.scalar_type(); })
        .def_property_readonly("device", &fvdb::JaggedTensor::device)
        .def_property("requires_grad", &fvdb::JaggedTensor::requires_grad, &fvdb::JaggedTensor::set_requires_grad)
        .def(py::pickle(
            [](const fvdb::JaggedTensor& t) {
                return py::make_tuple(t.jdata(), t.jidx(), t.batch_size());
            },
            [](py::tuple t) {
                // FIXME: Store mOffset to get rid of ambiguity. (this will be a breaking change)
                torch::Tensor data = THPVariable_Unpack(t[0].ptr());
                torch::Tensor jidx = THPVariable_Unpack(t[1].ptr());
		int batchSize = py::cast<int>(t[2]);
                // FIXME: This is a super slow way to build the grid
                return fvdb::JaggedTensor::from_data_and_jidx(data, jidx, batchSize);
            }
        ));

}