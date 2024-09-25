#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineCheckMode(nb::module_& m);
void defineChecksum(nb::module_& m);
void defineUpdateChecksum(nb::module_& m);

} // namespace pynanovdb
