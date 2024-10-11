#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineSampleFromVoxels(nb::module_& m, const char* name);

} // namespace pynanovdb
