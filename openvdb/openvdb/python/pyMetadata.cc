// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <pybind11/pybind11.h>
#include <openvdb/openvdb.h>
#include "pyTypeCasters.h"

namespace py = pybind11;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace {

class MetadataWrap: public Metadata
{
private:
    MetadataWrap();
public:
    MetadataWrap(const MetadataWrap&) = delete;
    MetadataWrap& operator=(const MetadataWrap&) = delete;
    Name typeName() const override { PYBIND11_OVERRIDE_PURE(Name, Metadata, typeName, ); }
    Metadata::Ptr copy() const override { PYBIND11_OVERRIDE_PURE(Metadata::Ptr, Metadata, copy, ); }
    void copy(const Metadata& other) override { PYBIND11_OVERRIDE_PURE(void, Metadata, copy, other); }
    std::string str() const override { PYBIND11_OVERRIDE_PURE(std::string, Metadata, str, ); }
    bool asBool() const override { PYBIND11_OVERRIDE_PURE(bool, Metadata, asBool, ); }
    Index32 size() const override { PYBIND11_OVERRIDE_PURE(Index32, Metadata, size, ); }

protected:
    void readValue(std::istream& is, Index32 numBytes) override {
        PYBIND11_OVERRIDE_PURE(void, Metadata, readValue, is, numBytes);
    }
    void writeValue(std::ostream& os) const override {
        PYBIND11_OVERRIDE_PURE(void, Metadata, writeValue, os);
    }
};

} // end anonymous namespace


void
exportMetadata(py::module_ m)
{
    py::class_<Metadata, MetadataWrap, Metadata::Ptr>(m,
        /*classname=*/"Metadata",
        /*docstring=*/
            "Class that holds the value of a single item of metadata of a type\n"
            "for which no Python equivalent exists (typically a custom type)")
        .def("copy", static_cast<Metadata::Ptr (Metadata::*)() const>(&Metadata::copy),
            "copy() -> Metadata\n\nReturn a copy of this value.")
        .def("copy", static_cast<void (Metadata::*)(const Metadata&)>(&Metadata::copy),
            "copy() -> Metadata\n\nReturn a copy of this value.")
        .def("type", &Metadata::typeName,
            "type() -> str\n\nReturn the name of this value's type.")
        .def("size", &Metadata::size,
            "size() -> int\n\nReturn the size of this value in bytes.")
        .def("__nonzero__", &Metadata::asBool)
        .def("__str__", &Metadata::str);
}
