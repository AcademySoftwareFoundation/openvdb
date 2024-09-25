#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineNearestNeighborSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTrilinearSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTriquadraticSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTricubicSampler(nb::module_& m, const char* name);

} // namespace pynanovdb
