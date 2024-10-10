#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineCreateNanoGrid(nb::module_& m, const char* name);

#ifdef NANOVDB_USE_OPENVDB
template<typename BufferT> void defineOpenToNanoVDB(nb::module_& m);
#endif

} // namespace pynanovdb
