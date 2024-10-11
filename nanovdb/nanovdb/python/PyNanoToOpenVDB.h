#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void defineNanoToOpenVDB(nb::module_& m);

} // namespace pynanovdb
