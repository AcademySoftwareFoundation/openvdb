#ifndef NANOVDB_PYIO_HAS_BEEN_INCLUDED
#define NANOVDB_PYIO_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineIOModule(nb::module_& m);

} // namespace pynanovdb

#endif
