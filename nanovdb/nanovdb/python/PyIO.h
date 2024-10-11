#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineIOModule(nb::module_& m);

} // namespace pynanovdb
