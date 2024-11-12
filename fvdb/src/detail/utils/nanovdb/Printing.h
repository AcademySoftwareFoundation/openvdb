// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_PRINTING_H
#define FVDB_DETAIL_UTILS_NANOVDB_PRINTING_H

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>

#include <iostream>

// Make std::cout/std::cerr work with nanovdb types
template <typename T>
inline std::ostream &
operator<<(std::ostream &os, const nanovdb::math::Ray<T> &r) {
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir=" << r.invDir() << " t0=" << r.t0()
       << " t1=" << r.t1();
    return os;
}

#endif // FVDB_DETAIL_UTILS_NANOVDB_PRINTING_H