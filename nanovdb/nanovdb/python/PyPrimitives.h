#ifndef NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED
#define NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void definePrimitives(nb::module_& m);

} // namespace pynanovdb

#endif
