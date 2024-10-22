#include "PyHostBuffer.h"

#include <nanovdb/HostBuffer.h>

namespace nb = nanobind;
using namespace nanovdb;

namespace pynanovdb {

void defineHostBuffer(nb::module_& m)
{
    nb::class_<HostBuffer>(m, "HostBuffer");
}

} // namespace pynanovdb
