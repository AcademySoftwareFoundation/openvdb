// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_BLOSC_H
#define OPENVDB_BLOSC_H

// OpenVDB's private copy of vdb_blosc.h, renamed so its symbols cannot
// clash with any other copy of blosc linked into the same binary (for
// example NanoVDB's own copy, or a system blosc library).
#define VDB_BLOSC_SYMBOL_PREFIX openvdb
#include "vdb_blosc.h"

#endif
