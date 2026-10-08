// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYBUILDTYPES_HAS_BEEN_INCLUDED
#define NANOVDB_PYBUILDTYPES_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace pynanovdb {

// Python-side counterpart of nanovdb::callNanoGrid. Routes a runtime GridType
// to OpT::known<BuildT>(args...) for every BuildT listed in BuildTypes.def and
// to OpT::unknown(args...) for any other GridType. callNanoGrid itself cannot
// drive the bindings because its case list differs from the bound set (it has
// no PointIndex arm, for one).
template<typename OpT, typename... ArgsT>
auto callPyBuildT(nanovdb::GridType gridType, ArgsT&&... args)
{
    switch (gridType) {
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
    case nanovdb::GridType::GridTypeEnum: return OpT::template known<T>(args...);
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    case nanovdb::GridType::GridTypeEnum: return OpT::template known<T>(args...);
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
    case nanovdb::GridType::GridTypeEnum: return OpT::template known<T>(args...);
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
    case nanovdb::GridType::GridTypeEnum: return OpT::template known<T>(args...);
#include "BuildTypes.def"
    default:
        return OpT::unknown(args...);
    }
}

} // namespace pynanovdb

#endif
