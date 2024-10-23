// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/shared_ptr.h>
#include <openvdb/openvdb.h>
#include "pyTypeCasters.h"

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace {

class MetadataWrap: public Metadata
{
    NB_TRAMPOLINE(Metadata, 8);
private:
    MetadataWrap();
public:
    MetadataWrap(const MetadataWrap&) = delete;
    MetadataWrap& operator=(const MetadataWrap&) = delete;
    Name typeName() const override { NB_OVERRIDE_PURE(typeName); }
    Metadata::Ptr copy() const override { NB_OVERRIDE_PURE(copy); }
    void copy(const Metadata& other) override { NB_OVERRIDE_PURE(copy, other); }
    std::string str() const override { NB_OVERRIDE_PURE(str); }
    bool asBool() const override { NB_OVERRIDE_PURE(asBool); }
    Index32 size() const override { NB_OVERRIDE_PURE(size); }

protected:
    void readValue(std::istream& is, Index32 numBytes) override {
        NB_OVERRIDE_PURE(readValue, is, numBytes);
    }
    void writeValue(std::ostream& os) const override {
        NB_OVERRIDE_PURE(writeValue, os);
    }
};

} // end anonymous namespace


void
exportMetadata(nb::module_ m)
{
    nb::class_<Metadata, MetadataWrap>(m,
        /*classname=*/"Metadata",
        /*docstring=*/
            "Class that holds the value of a single item of metadata of a type\n"
            "for which no Python equivalent exists (typically a custom type)")
        .def("copy", static_cast<Metadata::Ptr (Metadata::*)() const>(&Metadata::copy),
            "Return a copy of this value.")
        .def("copy", static_cast<void (Metadata::*)(const Metadata&)>(&Metadata::copy),
            "Return a copy of this value.")
        .def("type", &Metadata::typeName,
            "Return the name of this value's type.")
        .def("size", &Metadata::size,
            "Return the size of this value in bytes.")
        .def("__nonzero__", &Metadata::asBool)
        .def("__str__", &Metadata::str);
}
