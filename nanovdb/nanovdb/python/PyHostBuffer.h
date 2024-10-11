#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineHostBuffer(nb::module_& m);

}
