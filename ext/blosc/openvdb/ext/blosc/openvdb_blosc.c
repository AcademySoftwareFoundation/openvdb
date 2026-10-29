// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// Private compiled copy of vdb_blosc.c for OpenVDB. Symbols are prefixed so
// they cannot clash with any other copy of blosc in the same binary.
#define VDB_BLOSC_SYMBOL_PREFIX openvdb
#include "openvdb_blosc.h"
#include "vdb_blosc.c"
