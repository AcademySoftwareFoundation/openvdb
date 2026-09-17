// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// NanoVDB's private compiled copy of vdb_blosc.c, renamed so its symbols
// cannot clash with any other copy of blosc linked into the same binary
// (for example OpenVDB's own copy, or a system blosc library).
#define VDB_BLOSC_SYMBOL_PREFIX nanovdb
#include "vdb_blosc.c"
