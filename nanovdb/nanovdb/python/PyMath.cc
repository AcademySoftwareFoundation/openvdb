// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyMath.h"

#include <sstream>

#include <nanovdb/math/Math.h>
#include <nanovdb/io/IO.h> // for __repr__

#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>

#include "PySampleFromVoxels.h"
#include "cuda/PySampleFromVoxels.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

void defineCoord(nb::module_& m)
{
    using ValueType = math::Coord::ValueType;

    nb::class_<math::Coord>(m, "Coord", "Signed (i, j, k) 32-bit integer coordinate class, similar to openvdb::math::Coord")
        .def(nb::init<>())
        .def(nb::init<ValueType>(), "n"_a)
        .def(nb::init<ValueType, ValueType, ValueType>(), "i"_a, "j"_a, "k"_a)
        .def_prop_rw(
            "x", [](const math::Coord& ijk) { return ijk.x(); }, [](math::Coord& ijk, int32_t i) { ijk.x() = i; })
        .def_prop_rw(
            "y", [](const math::Coord& ijk) { return ijk.y(); }, [](math::Coord& ijk, int32_t j) { ijk.y() = j; })
        .def_prop_rw(
            "z", [](const math::Coord& ijk) { return ijk.z(); }, [](math::Coord& ijk, int32_t k) { ijk.z() = k; })
        .def_static("max", &math::Coord::max)
        .def_static("min", &math::Coord::min)
        .def_static("memUsage", &math::Coord::memUsage)
        .def(
            "__getitem__",
            [](const math::Coord& ijk, size_t i) {
                if (i >= 3) {
                    throw nb::index_error();
                }
                return ijk[static_cast<math::Coord::IndexType>(i)];
            },
            "i"_a)
        .def(
            "__setitem__",
            [](math::Coord& ijk, size_t i, ValueType value) {
                if (i >= 3) {
                    throw nb::index_error();
                }
                ijk[static_cast<math::Coord::IndexType>(i)] = value;
            },
            "i"_a,
            "value"_a)
        .def(
            "__and__", [](const math::Coord& a, math::Coord::IndexType b) { return a & b; }, nb::is_operator(), "n"_a)
        .def(
            "__lshift__", [](const math::Coord& a, math::Coord::IndexType b) { return a << b; }, nb::is_operator(), "n"_a)
        .def(
            "__rshift__", [](const math::Coord& a, math::Coord::IndexType b) { return a >> b; }, nb::is_operator(), "n"_a)
        .def(nb::self < nb::self, "rhs"_a)
        .def(nb::self == nb::self, "rhs"_a)
        .def(nb::self != nb::self, "rhs"_a)
        .def(
            "__iand__", [](math::Coord& a, int b) { return a &= b; }, nb::is_operator(), "n"_a)
        .def(
            "__ilshift__", [](math::Coord& a, uint32_t b) { return a <<= b; }, nb::is_operator(), "n"_a)
        .def(
            "__irshift__", [](math::Coord& a, uint32_t b) { return a >>= b; }, nb::is_operator(), "n"_a)
        .def(
            "__iadd__", [](math::Coord& a, int b) { return a += b; }, nb::is_operator(), "n"_a)
        .def(nb::self + nb::self, "rhs"_a)
        .def(nb::self - nb::self, "rhs"_a)
        .def(-nb::self)
        .def(nb::self += nb::self, "rhs"_a)
        .def(nb::self -= nb::self, "rhs"_a)
        .def("minComponent", &math::Coord::minComponent, "other"_a)
        .def("maxComponent", &math::Coord::maxComponent, "other"_a)
        .def("offsetBy", nb::overload_cast<ValueType, ValueType, ValueType>(&math::Coord::offsetBy, nb::const_), "dx"_a, "dy"_a, "dz"_a)
        .def("offsetBy", nb::overload_cast<ValueType>(&math::Coord::offsetBy, nb::const_), "n"_a)
        .def_static("lessThan", &math::Coord::lessThan, "a"_a, "b"_a)
        .def_static("Floor", &math::Coord::template Floor<math::Vec3<float>>, "xyz"_a)
        .def_static("Floor", &math::Coord::template Floor<math::Vec3<double>>, "xyz"_a)
        .def("hash", &math::Coord::template hash<12>)
        .def("octant", &math::Coord::octant)
        .def("asVec3s", &math::Coord::asVec3s)
        .def("asVec3d", &math::Coord::asVec3d)
        .def("round", &math::Coord::round)
        .def("__repr__", [](const math::Coord& ijk) {
            std::stringstream ostr;
            ostr << ijk;
            return ostr.str();
        });
}

template<typename T> void defineVec3(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::Vec3<T>>(m, name, doc)
        .def(nb::init<>())
        .def(nb::init<T>(), "x"_a)
        .def(nb::init<T, T, T>(), "x"_a, "y"_a, "z"_a)
        .def(nb::init<math::Vec3<T>>(), "v"_a)
        .def(nb::init<math::Coord>(), "ijk"_a)
        .def(nb::self == nb::self, "rhs"_a)
        .def(nb::self != nb::self, "rhs"_a)
        .def(
            "__getitem__",
            [](const math::Vec3<T>& v, size_t i) {
                if (i >= math::Vec3<T>::SIZE) {
                    throw nb::index_error();
                }
                return v[static_cast<int>(i)];
            },
            "i"_a)
        .def(
            "__setitem__",
            [](math::Vec3<T>& v, size_t i, T value) {
                if (i >= math::Vec3<T>::SIZE) {
                    throw nb::index_error();
                }
                v[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a)
        .def("dot", &math::Vec3<T>::template dot<math::Vec3<T>>, "v"_a)
        .def("cross", &math::Vec3<T>::template cross<math::Vec3<T>>, "v"_a)
        .def("lengthSqr", &math::Vec3<T>::lengthSqr)
        .def("length", &math::Vec3<T>::length)
        .def(-nb::self)
        .def(nb::self * nb::self, "v"_a)
        .def(nb::self / nb::self, "v"_a)
        .def(nb::self + nb::self, "v"_a)
        .def(nb::self - nb::self, "v"_a)
        .def(nb::self + math::Coord(), "ijk"_a)
        .def(nb::self - math::Coord(), "ijk"_a)
        .def(nb::self * T(), "s"_a)
        .def(nb::self / T(), "s"_a)
        .def(nb::self += nb::self, "v"_a)
        .def(nb::self += math::Coord(), "ijk"_a)
        .def(nb::self -= nb::self, "v"_a)
        .def(nb::self -= math::Coord(), "ijk"_a)
        .def(nb::self *= T(), "s"_a)
        .def(nb::self /= T(), "s"_a)
        .def("normalize", &math::Vec3<T>::normalize)
        .def("minComponent", &math::Vec3<T>::minComponent, "other"_a)
        .def("maxComponent", &math::Vec3<T>::maxComponent, "other"_a)
        .def("min", &math::Vec3<T>::min)
        .def("max", &math::Vec3<T>::max)
        .def("floor", &math::Vec3<T>::floor)
        .def("ceil", &math::Vec3<T>::ceil)
        .def("round", &math::Vec3<T>::round)
        .def(
            "__mul__", [](const T& a, math::Vec3<T> b) { return a * b; }, nb::is_operator(), "b"_a)
        .def(
            "__truediv__", [](const T& a, math::Vec3<T> b) { return a / b; }, nb::is_operator(), "b"_a)
        .def("__repr__", [](const math::Vec3<T>& v) {
            std::stringstream ostr;
            ostr << v;
            return ostr.str();
        });
}

template<typename T> void defineVec4(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::Vec4<T>>(m, name, doc)
        .def(nb::init<>())
        .def(nb::init<T>(), "x"_a)
        .def(nb::init<T, T, T, T>(), "x"_a, "y"_a, "z"_a, "w"_a)
        .def(nb::init<math::Vec4<T>>(), "v"_a)
        .def(nb::self == nb::self, "rhs"_a)
        .def(nb::self != nb::self, "rhs"_a)
        .def(
            "__getitem__",
            [](const math::Vec4<T>& v, size_t i) {
                if (i >= math::Vec4<T>::SIZE) {
                    throw nb::index_error();
                }
                return v[static_cast<int>(i)];
            },
            "i"_a)
        .def(
            "__setitem__",
            [](math::Vec4<T>& v, size_t i, T value) {
                if (i >= math::Vec4<T>::SIZE) {
                    throw nb::index_error();
                }
                v[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a)
        .def("dot", &math::Vec4<T>::template dot<math::Vec4<T>>, "v"_a)
        .def("lengthSqr", &math::Vec4<T>::lengthSqr)
        .def("length", &math::Vec4<T>::length)
        .def(-nb::self)
        .def(nb::self * nb::self, "v"_a)
        .def(nb::self / nb::self, "v"_a)
        .def(nb::self + nb::self, "v"_a)
        .def(nb::self - nb::self, "v"_a)
        .def(nb::self * T(), "s"_a)
        .def(nb::self / T(), "s"_a)
        .def(nb::self += nb::self, "v"_a)
        .def(nb::self -= nb::self, "v"_a)
        .def(nb::self *= T(), "s"_a)
        .def(nb::self /= T(), "s"_a)
        .def("normalize", &math::Vec4<T>::normalize)
        .def("minComponent", &math::Vec4<T>::minComponent, "other"_a)
        .def("maxComponent", &math::Vec4<T>::maxComponent, "other"_a)
        .def(
            "__mul__", [](const T& a, math::Vec4<T> b) { return a * b; }, nb::is_operator(), "b"_a)
        .def(
            "__truediv__", [](const T& a, math::Vec4<T> b) { return a / b; }, nb::is_operator(), "b"_a)
        .def("__repr__", [](const math::Vec4<T>& v) {
            std::stringstream ostr;
            ostr << v;
            return ostr.str();
        });
}

void defineRgba8(nb::module_& m)
{
    using ValueType = math::Rgba8::ValueType;

    nb::class_<math::Rgba8>(m, "Rgba8", "8-bit red, green, blue, alpha packed into 32 bit unsigned int")
        .def(nb::init<>())
        .def(nb::init<const math::Rgba8&>(), "other"_a)
        .def(nb::init<uint8_t, uint8_t, uint8_t, uint8_t>(), "r"_a, "g"_a, "b"_a, "a"_a = 255)
        .def(nb::init<uint8_t>(), "v"_a)
        .def(nb::init<float, float, float, float>(), "r"_a, "g"_a, "b"_a, "a"_a = 1.0)
        .def(nb::init<Vec3f>(), "rgb"_a)
        .def(nb::init<Vec4f>(), "rgba"_a)
        .def(nb::self < nb::self, "rhs"_a)
        .def(nb::self == nb::self, "rhs"_a)
        .def("lengthSqr", &math::Rgba8::lengthSqr)
        .def("length", &math::Rgba8::length)
        .def("asFloat", &math::Rgba8::asFloat, "n"_a)
        .def(
            "__getitem__",
            [](const math::Rgba8& rgba, size_t i) {
                if (i >= 4) {
                    throw nb::index_error();
                }
                return rgba[static_cast<int>(i)];
            },
            "i"_a)
        .def(
            "__setitem__",
            [](math::Rgba8& rgba, size_t i, ValueType value) {
                if (i >= 4) {
                    throw nb::index_error();
                }
                rgba[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a)
        .def_prop_rw(
            "packed", [](const math::Rgba8& rgba) { return rgba.packed(); }, [](math::Rgba8& rgba, uint32_t packed) { rgba.packed() = packed; })
        .def_prop_rw(
            "r", [](const math::Rgba8& rgba) { return rgba.r(); }, [](math::Rgba8& rgba, uint8_t r) { rgba.r() = r; })
        .def_prop_rw(
            "g", [](const math::Rgba8& rgba) { return rgba.g(); }, [](math::Rgba8& rgba, uint8_t g) { rgba.g() = g; })
        .def_prop_rw(
            "b", [](const math::Rgba8& rgba) { return rgba.b(); }, [](math::Rgba8& rgba, uint8_t b) { rgba.b() = b; })
        .def_prop_rw(
            "a", [](const math::Rgba8& rgba) { return rgba.a(); }, [](math::Rgba8& rgba, uint8_t a) { rgba.a() = a; })
        .def("asVec3f", [](const math::Rgba8& rgba) { return Vec3f(rgba); })
        .def("asVec4f", [](const math::Rgba8& rgba) { return Vec4f(rgba); });
}

template<typename Vec3T> void defineBaseBBox(nb::module_& m, const char* name)
{
    nb::class_<math::BaseBBox<Vec3T>>(m, name)
        .def(nb::self == nb::self, "rhs"_a)
        .def(nb::self != nb::self, "rhs"_a)
        .def(
            "__getitem__",
            [](const math::BaseBBox<Vec3T>& bbox, size_t i) {
                if (i >= 2) {
                    throw nb::index_error();
                }
                return bbox[static_cast<int>(i)];
            },
            "i"_a)
        .def(
            "__setitem__",
            [](math::BaseBBox<Vec3T>& bbox, size_t i, const Vec3T& value) {
                if (i >= 2) {
                    throw nb::index_error();
                }
                bbox[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a)
        .def_prop_rw(
            "min", [](const math::BaseBBox<Vec3T>& bbox) { return bbox.min(); }, [](math::BaseBBox<Vec3T>& bbox, const Vec3T& min) { bbox.min() = min; })
        .def_prop_rw(
            "max", [](const math::BaseBBox<Vec3T>& bbox) { return bbox.max(); }, [](math::BaseBBox<Vec3T>& bbox, const Vec3T& max) { bbox.max() = max; })
        .def("translate", &math::BaseBBox<Vec3T>::translate, "xyz"_a)
        .def("expand", nb::overload_cast<const Vec3T&>(&math::BaseBBox<Vec3T>::expand), "xyz"_a)
        .def("expand", nb::overload_cast<const math::BaseBBox<Vec3T>&>(&math::BaseBBox<Vec3T>::expand), "bbox"_a)
        .def("intersect", &math::BaseBBox<Vec3T>::intersect, "bbox"_a)
        .def("isInside", &math::BaseBBox<Vec3T>::isInside, "xyz"_a);
}

template<typename Vec3T> void defineBBoxFloatingPoint(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::BBox<Vec3T, true>, math::BaseBBox<Vec3T>>(m, name, doc)
        .def(nb::init<>())
        .def(nb::init<const Vec3T&, const Vec3T&>(), "min"_a, "max"_a)
        .def(nb::init<const math::Coord&, const math::Coord&>(), "min"_a, "max"_a)
        .def_static("createCube", &math::BBox<Vec3T>::createCube, "min"_a, "dim"_a)
        .def(nb::init<const math::BaseBBox<math::Coord>&>(), "bbox"_a)
        .def("empty", &math::BBox<Vec3T>::empty)
        .def("dim", &math::BBox<Vec3T>::dim)
        .def("isInside", &math::BBox<Vec3T>::isInside, "p"_a)
        .def("__repr__", [](const math::BBox<Vec3T>& b) {
            std::stringstream ostr;
            ostr << b;
            return ostr.str();
        });
}

template<typename CoordT> void defineBBoxInteger(nb::module_& m, const char* name, const char* doc)
{
    using ValueType = typename CoordT::ValueType;

    nb::class_<math::BBox<CoordT, false>, math::BaseBBox<CoordT>>(m, name, doc)
        .def(nb::init<>())
        .def(nb::init<const CoordT&, const CoordT&>(), "min"_a, "max"_a)
        .def(
            "__iter__",
            [](const math::BBox<CoordT>& b) { return nb::make_iterator(nb::type<math::BBox<CoordT>>(), "CoordBBoxIterator", b.begin(), b.end()); },
            nb::keep_alive<0, 1>())
        .def_static("createCube", nb::overload_cast<const CoordT&, ValueType>(&math::BBox<CoordT>::createCube), "min"_a, "dim"_a)
        .def_static("createCube", nb::overload_cast<ValueType, ValueType>(&math::BBox<CoordT>::createCube), "min"_a, "max"_a)
        .def("is_divisible", &math::BBox<CoordT>::is_divisible)
        .def("empty", &math::BBox<CoordT>::empty)
        .def("dim", &math::BBox<CoordT>::dim)
        .def("volume", &math::BBox<CoordT>::volume)
        .def("isInside", nb::overload_cast<const CoordT&>(&math::BBox<CoordT>::isInside, nb::const_), "p"_a)
        .def("isInside", nb::overload_cast<const math::BBox<CoordT>&>(&math::BBox<CoordT>::isInside, nb::const_), "b"_a)
        .def("asFloat", &math::BBox<CoordT>::template asReal<float>)
        .def("asDouble", &math::BBox<CoordT>::template asReal<double>)
        .def("hasOverlap", &math::BBox<CoordT>::hasOverlap, "b"_a)
        .def("expandBy", &math::BBox<CoordT>::expandBy, "padding"_a)
        .def("__repr__", [](const CoordBBox& b) {
            std::stringstream ostr;
            ostr << b;
            return ostr.str();
        });
}

} // namespace

void defineMathModule(nb::module_& m)
{
    defineCoord(m);

    defineVec3<float>(m, "Vec3f", "Vector class with three float components, similar to openvdb::math::Vec3f");
    defineVec3<double>(m, "Vec3d", "Vector class with three double components, similar to openvdb::math::Vec3d");

    defineVec4<float>(m, "Vec4f", "Vector class with four float components, similar to openvdb::math::Vec4f");
    defineVec4<double>(m, "Vec4d", "Vector class with four double components, similar to openvdb::math::Vec4f");

    defineRgba8(m);

    defineBaseBBox<Vec3f>(m, "Vec3fBaseBBox");
    defineBBoxFloatingPoint<Vec3f>(m, "Vec3fBBox", "Bounding box for Vec3f minimum and maximum");

    defineBaseBBox<Vec3d>(m, "Vec3dBaseBBox");
    defineBBoxFloatingPoint<Vec3d>(m, "Vec3dBBox", "Bounding box for Vec3d minimum and maximum");

    defineBaseBBox<math::Coord>(m, "CoordBaseBBox");
    defineBBoxInteger<math::Coord>(m, "CoordBBox", "Bounding box for Coord minimum and maximum");

    defineNearestNeighborSampler<float>(m, "FloatNearestNeighborSampler");
    defineTrilinearSampler<float>(m, "FloatTrilinearSampler");
    defineTriquadraticSampler<float>(m, "FloatTriquadraticSampler");
    defineTricubicSampler<float>(m, "FloatTricubicSampler");

    defineNearestNeighborSampler<double>(m, "DoubleNearestNeighborSampler");
    defineTrilinearSampler<double>(m, "DoubleTrilinearSampler");
    defineTriquadraticSampler<double>(m, "DoubleTriquadraticSampler");
    defineTricubicSampler<double>(m, "DoubleTricubicSampler");

    defineNearestNeighborSampler<int32_t>(m, "Int32NearestNeighborSampler");
    defineTrilinearSampler<int32_t>(m, "Int32TrilinearSampler");
    defineTriquadraticSampler<int32_t>(m, "Int32TriquadraticSampler");
    defineTricubicSampler<int32_t>(m, "Int32TricubicSampler");

    defineNearestNeighborSampler<Vec3f>(m, "Vec3fNearestNeighborSampler");
    defineTrilinearSampler<Vec3f>(m, "Vec3fTrilinearSampler");
    defineTriquadraticSampler<Vec3f>(m, "Vec3fTriquadraticSampler");
    defineTricubicSampler<Vec3f>(m, "Vec3fTricubicSampler");

#ifdef NANOVDB_USE_CUDA
    nb::module_ cudaModule = m.def_submodule("cuda");
    cudaModule.doc() = "A submodule that implements CUDA-accelerated math functions";

    defineSampleFromVoxels<float>(cudaModule, "sampleFromVoxels");
    defineSampleFromVoxels<double>(cudaModule, "sampleFromVoxels");
#endif
}

} // namespace pynanovdb
