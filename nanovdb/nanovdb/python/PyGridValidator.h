#ifndef NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void defineValidateGrids(nb::module_& m);

} // namespace pynanovdb

#endif
