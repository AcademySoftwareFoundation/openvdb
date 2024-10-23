#ifndef NANOVDB_PYTOOLS_HAS_BEEN_INCLUDED
#define NANOVDB_PYTOOLS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineToolsModule(nb::module_& m);

} // namespace pynanovdb

#endif
