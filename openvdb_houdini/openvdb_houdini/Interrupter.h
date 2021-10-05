// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Interrupter.h
/// @brief Houdini Interrupter

#ifndef OPENVDB_HOUDINI_INTERRUPTER_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_INTERRUPTER_HAS_BEEN_INCLUDED

#include <openvdb/util/NullInterrupter.h> // for CustomInterrupter

namespace openvdb_houdini {

using Interrupter = openvdb::util::CustomInterrupter;

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_INTERRUPTER_HAS_BEEN_INCLUDED
