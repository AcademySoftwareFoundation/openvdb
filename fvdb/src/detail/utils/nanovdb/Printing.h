#include <iostream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/io/IO.h>


// Make std::cout/std::cerr work with nanovdb types
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const nanovdb::math::Ray<T>& r) {
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir="<<r.invDir()
       << " t0=" << r.t0()  << " t1="  << r.t1();
    return os;
}

