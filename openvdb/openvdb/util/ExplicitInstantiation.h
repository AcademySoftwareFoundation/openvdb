// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_UTIL_EXPLICIT_INSTANTIATION_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_EXPLICIT_INSTANTIATION_HAS_BEEN_INCLUDED

#undef OPENVDB_INSTANTIATE
#undef OPENVDB_INSTANTIATE_CLASS
#undef OPENVDB_INSTANTIATE_STRUCT

#ifdef _WIN32
    #define OPENVDB_INSTANTIATE template __declspec(dllexport)
    #define OPENVDB_INSTANTIATE_CLASS template class __declspec(dllexport)
    #define OPENVDB_INSTANTIATE_STRUCT template struct __declspec(dllexport)
#else
    #define OPENVDB_INSTANTIATE template
    #define OPENVDB_INSTANTIATE_CLASS template class
    #define OPENVDB_INSTANTIATE_STRUCT template struct
#endif

#endif // OPENVDB_UTIL_EXPLICIT_INSTANTIATION_HAS_BEEN_INCLUDED
