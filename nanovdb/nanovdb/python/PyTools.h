#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineToolsModule(nb::module_& m);

} // namespace pynanovdb
