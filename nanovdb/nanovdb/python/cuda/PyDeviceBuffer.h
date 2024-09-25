#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
void defineDeviceBuffer(nb::module_& m);
#endif

} // namespace pynanovdb
