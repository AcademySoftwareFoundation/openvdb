// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyMath.h"

#include <sstream>

#include <nanovdb/math/Math.h>
#include <nanovdb/io/IO.h> // for __repr__

#include <nanobind/stl/string.h>
#include <nanobind/stl/bind_vector.h>

#include "PySampleFromVoxels.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

void defineCoord(nb::module_& m)
{
    using ValueType = math::Coord::ValueType;

    nb::class_<math::Coord>(m, "Coord", "Signed (i, j, k) 32-bit integer coordinate class, similar to openvdb::math::Coord")
        .def(nb::init<>(),
             "Construct (0, 0, 0).")
        .def(nb::init<ValueType>(), "n"_a,
             "Construct (n, n, n).")
        .def(nb::init<ValueType, ValueType, ValueType>(), "i"_a, "j"_a, "k"_a,
             "Construct (i, j, k).")
        .def_prop_rw(
            "x", [](const math::Coord& ijk) { return ijk.x(); }, [](math::Coord& ijk, int32_t i) { ijk.x() = i; },
            "First component of the (i, j, k) triple.")
        .def_prop_rw(
            "y", [](const math::Coord& ijk) { return ijk.y(); }, [](math::Coord& ijk, int32_t j) { ijk.y() = j; },
            "Second component of the (i, j, k) triple.")
        .def_prop_rw(
            "z", [](const math::Coord& ijk) { return ijk.z(); }, [](math::Coord& ijk, int32_t k) { ijk.z() = k; },
            "Third component of the (i, j, k) triple.")
        .def_static("max", &math::Coord::max,
                    "Largest representable Coord (INT32_MAX in every component).")
        .def_static("min", &math::Coord::min,
                    "Smallest representable Coord (INT32_MIN in every component).")
        .def_static("memUsage", &math::Coord::memUsage,
                    "Byte size of a Coord instance.")
        .def(
            "__getitem__",
            [](const math::Coord& ijk, size_t i) {
                if (i >= 3) {
                    throw nb::index_error();
                }
                return ijk[static_cast<math::Coord::IndexType>(i)];
            },
            "i"_a,
            "Read the i-th component (0=x, 1=y, 2=z).")
        .def(
            "__setitem__",
            [](math::Coord& ijk, size_t i, ValueType value) {
                if (i >= 3) {
                    throw nb::index_error();
                }
                ijk[static_cast<math::Coord::IndexType>(i)] = value;
            },
            "i"_a,
            "value"_a,
            "Write the i-th component (0=x, 1=y, 2=z).")
        .def(
            "__and__", [](const math::Coord& a, math::Coord::IndexType b) { return a & b; }, nb::is_operator(), "n"_a,
            "Component-wise bitwise AND with the scalar n.")
        .def(
            "__lshift__", [](const math::Coord& a, math::Coord::IndexType b) { return a << b; }, nb::is_operator(), "n"_a,
            "Component-wise left shift by n bits.")
        .def(
            "__rshift__", [](const math::Coord& a, math::Coord::IndexType b) { return a >> b; }, nb::is_operator(), "n"_a,
            "Component-wise right shift by n bits.")
        .def(nb::self < nb::self, "rhs"_a,
             "Lexicographic less-than comparison.")
        .def(nb::self == nb::self, "rhs"_a,
             "Equality of all three components.")
        .def(nb::self != nb::self, "rhs"_a,
             "Inequality of any one component.")
        .def(
            "__iand__", [](math::Coord& a, int b) { return a &= b; }, nb::is_operator(), "n"_a,
            "In-place component-wise bitwise AND with the scalar n.")
        .def(
            "__ilshift__", [](math::Coord& a, uint32_t b) { return a <<= b; }, nb::is_operator(), "n"_a,
            "In-place component-wise left shift by n bits.")
        .def(
            "__irshift__", [](math::Coord& a, uint32_t b) { return a >>= b; }, nb::is_operator(), "n"_a,
            "In-place component-wise right shift by n bits.")
        .def(
            "__iadd__", [](math::Coord& a, int b) { return a += b; }, nb::is_operator(), "n"_a,
            "In-place add the scalar n to every component.")
        .def(nb::self + nb::self, "rhs"_a,
             "Component-wise addition.")
        .def(nb::self - nb::self, "rhs"_a,
             "Component-wise subtraction.")
        .def(-nb::self,
             "Negate every component.")
        .def(nb::self += nb::self, "rhs"_a,
             "In-place component-wise addition.")
        .def(nb::self -= nb::self, "rhs"_a,
             "In-place component-wise subtraction.")
        .def("minComponent", &math::Coord::minComponent, "other"_a,
             "Component-wise minimum with other. See nanovdb::math::Coord::minComponent in NanoVDB.h.")
        .def("maxComponent", &math::Coord::maxComponent, "other"_a,
             "Component-wise maximum with other. See nanovdb::math::Coord::maxComponent in NanoVDB.h.")
        .def("offsetBy", nb::overload_cast<ValueType, ValueType, ValueType>(&math::Coord::offsetBy, nb::const_), "dx"_a, "dy"_a, "dz"_a,
             "Return a new Coord offset by (dx, dy, dz). See nanovdb::math::Coord::offsetBy in NanoVDB.h.")
        .def("offsetBy", nb::overload_cast<ValueType>(&math::Coord::offsetBy, nb::const_), "n"_a,
             "Return a new Coord offset by n in every component.")
        .def_static("lessThan", &math::Coord::lessThan, "a"_a, "b"_a,
                    "Component-wise a < b returning a Coord of 0 / 1 flags.")
        .def_static("Floor", &math::Coord::template Floor<math::Vec3<float>>, "xyz"_a,
                    "Floor each component of a Vec3f to produce an integer Coord.")
        .def_static("Floor", &math::Coord::template Floor<math::Vec3<double>>, "xyz"_a,
                    "Floor each component of a Vec3d to produce an integer Coord.")
        .def("hash", &math::Coord::template hash<12>,
             "Spatial hash of this coordinate suited for hashed root-table lookups.")
        .def("octant", &math::Coord::octant,
             "Return the 0..7 octant index of this coordinate's sign bits.")
        .def("asVec3s", &math::Coord::asVec3s,
             "Convert to a Vec3f (float) with no scaling.")
        .def("asVec3d", &math::Coord::asVec3d,
             "Convert to a Vec3d (double) with no scaling.")
        .def("round", &math::Coord::round,
             "Component-wise round; for an integer Coord this is the identity.")
        .def("__repr__", [](const math::Coord& ijk) {
            std::stringstream ostr;
            ostr << ijk;
            return ostr.str();
        });
}

template<typename T> void defineVec3(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::Vec3<T>>(m, name, doc)
        .def(nb::init<>(),
             "Construct a zero-initialized vector.")
        .def(nb::init<T>(), "x"_a,
             "Construct (x, x, x).")
        .def(nb::init<T, T, T>(), "x"_a, "y"_a, "z"_a,
             "Construct (x, y, z).")
        .def(nb::init<math::Vec3<T>>(), "v"_a,
             "Copy-construct from another Vec3.")
        .def(nb::init<math::Coord>(), "ijk"_a,
             "Construct from an integer Coord, casting each component.")
        .def(nb::self == nb::self, "rhs"_a,
             "Component-wise equality.")
        .def(nb::self != nb::self, "rhs"_a,
             "Component-wise inequality.")
        .def(
            "__getitem__",
            [](const math::Vec3<T>& v, size_t i) {
                if (i >= math::Vec3<T>::SIZE) {
                    throw nb::index_error();
                }
                return v[static_cast<int>(i)];
            },
            "i"_a,
            "Read the i-th component (0=x, 1=y, 2=z).")
        .def(
            "__setitem__",
            [](math::Vec3<T>& v, size_t i, T value) {
                if (i >= math::Vec3<T>::SIZE) {
                    throw nb::index_error();
                }
                v[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a,
            "Write the i-th component (0=x, 1=y, 2=z).")
        .def("dot", &math::Vec3<T>::template dot<math::Vec3<T>>, "v"_a,
             "Dot product with another vector.")
        .def("cross", &math::Vec3<T>::template cross<math::Vec3<T>>, "v"_a,
             "Cross product with another vector.")
        .def("lengthSqr", &math::Vec3<T>::lengthSqr,
             "Squared Euclidean length (cheaper than length()).")
        .def("length", &math::Vec3<T>::length,
             "Euclidean length of this vector.")
        .def(-nb::self,
             "Negate every component.")
        .def(nb::self * nb::self, "v"_a,
             "Component-wise multiplication.")
        .def(nb::self / nb::self, "v"_a,
             "Component-wise division.")
        .def(nb::self + nb::self, "v"_a,
             "Component-wise addition.")
        .def(nb::self - nb::self, "v"_a,
             "Component-wise subtraction.")
        .def(nb::self + math::Coord(), "ijk"_a,
             "Add an integer Coord component-wise.")
        .def(nb::self - math::Coord(), "ijk"_a,
             "Subtract an integer Coord component-wise.")
        .def(nb::self * T(), "s"_a,
             "Multiply every component by the scalar s.")
        .def(nb::self / T(), "s"_a,
             "Divide every component by the scalar s.")
        .def(nb::self += nb::self, "v"_a,
             "In-place component-wise addition.")
        .def(nb::self += math::Coord(), "ijk"_a,
             "In-place add an integer Coord.")
        .def(nb::self -= nb::self, "v"_a,
             "In-place component-wise subtraction.")
        .def(nb::self -= math::Coord(), "ijk"_a,
             "In-place subtract an integer Coord.")
        .def(nb::self *= T(), "s"_a,
             "In-place scalar multiply.")
        .def(nb::self /= T(), "s"_a,
             "In-place scalar divide.")
        .def("normalize", &math::Vec3<T>::normalize,
             "Scale this vector to unit length in place.")
        .def("minComponent", &math::Vec3<T>::minComponent, "other"_a,
             "Component-wise minimum with other.")
        .def("maxComponent", &math::Vec3<T>::maxComponent, "other"_a,
             "Component-wise maximum with other.")
        .def("min", &math::Vec3<T>::min,
             "Smallest single component of this vector.")
        .def("max", &math::Vec3<T>::max,
             "Largest single component of this vector.")
        .def("floor", &math::Vec3<T>::floor,
             "Component-wise floor.")
        .def("ceil", &math::Vec3<T>::ceil,
             "Component-wise ceiling.")
        .def("round", &math::Vec3<T>::round,
             "Component-wise round.")
        .def(
            "__mul__", [](const T& a, math::Vec3<T> b) { return a * b; }, nb::is_operator(), "b"_a,
            "Right-multiply: scalar * Vec3.")
        .def(
            "__truediv__", [](const T& a, math::Vec3<T> b) { return a / b; }, nb::is_operator(), "b"_a,
            "Right-divide: scalar / Vec3, component-wise.")
        .def("__repr__", [](const math::Vec3<T>& v) {
            std::stringstream ostr;
            ostr << v;
            return ostr.str();
        });
}

template<typename T> void defineVec4(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::Vec4<T>>(m, name, doc)
        .def(nb::init<>(),
             "Construct a zero-initialized vector.")
        .def(nb::init<T>(), "x"_a,
             "Construct (x, x, x, x).")
        .def(nb::init<T, T, T, T>(), "x"_a, "y"_a, "z"_a, "w"_a,
             "Construct (x, y, z, w).")
        .def(nb::init<math::Vec4<T>>(), "v"_a,
             "Copy-construct from another Vec4.")
        .def(nb::self == nb::self, "rhs"_a,
             "Component-wise equality.")
        .def(nb::self != nb::self, "rhs"_a,
             "Component-wise inequality.")
        .def(
            "__getitem__",
            [](const math::Vec4<T>& v, size_t i) {
                if (i >= math::Vec4<T>::SIZE) {
                    throw nb::index_error();
                }
                return v[static_cast<int>(i)];
            },
            "i"_a,
            "Read the i-th component (0=x, 1=y, 2=z, 3=w).")
        .def(
            "__setitem__",
            [](math::Vec4<T>& v, size_t i, T value) {
                if (i >= math::Vec4<T>::SIZE) {
                    throw nb::index_error();
                }
                v[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a,
            "Write the i-th component (0=x, 1=y, 2=z, 3=w).")
        .def("dot", &math::Vec4<T>::template dot<math::Vec4<T>>, "v"_a,
             "Dot product with another vector.")
        .def("lengthSqr", &math::Vec4<T>::lengthSqr,
             "Squared Euclidean length (cheaper than length()).")
        .def("length", &math::Vec4<T>::length,
             "Euclidean length of this vector.")
        .def(-nb::self,
             "Negate every component.")
        .def(nb::self * nb::self, "v"_a,
             "Component-wise multiplication.")
        .def(nb::self / nb::self, "v"_a,
             "Component-wise division.")
        .def(nb::self + nb::self, "v"_a,
             "Component-wise addition.")
        .def(nb::self - nb::self, "v"_a,
             "Component-wise subtraction.")
        .def(nb::self * T(), "s"_a,
             "Multiply every component by the scalar s.")
        .def(nb::self / T(), "s"_a,
             "Divide every component by the scalar s.")
        .def(nb::self += nb::self, "v"_a,
             "In-place component-wise addition.")
        .def(nb::self -= nb::self, "v"_a,
             "In-place component-wise subtraction.")
        .def(nb::self *= T(), "s"_a,
             "In-place scalar multiply.")
        .def(nb::self /= T(), "s"_a,
             "In-place scalar divide.")
        .def("normalize", &math::Vec4<T>::normalize,
             "Scale this vector to unit length in place.")
        .def("minComponent", &math::Vec4<T>::minComponent, "other"_a,
             "Component-wise minimum with other.")
        .def("maxComponent", &math::Vec4<T>::maxComponent, "other"_a,
             "Component-wise maximum with other.")
        .def(
            "__mul__", [](const T& a, math::Vec4<T> b) { return a * b; }, nb::is_operator(), "b"_a,
            "Right-multiply: scalar * Vec4.")
        .def(
            "__truediv__", [](const T& a, math::Vec4<T> b) { return a / b; }, nb::is_operator(), "b"_a,
            "Right-divide: scalar / Vec4, component-wise.")
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
        .def(nb::init<>(),
             "Construct a fully transparent black Rgba8 (all components 0).")
        .def(nb::init<const math::Rgba8&>(), "other"_a,
             "Copy-construct from another Rgba8.")
        .def(nb::init<uint8_t, uint8_t, uint8_t, uint8_t>(), "r"_a, "g"_a, "b"_a, "a"_a = 255,
             "Construct from four 0..255 uint8 channels; a defaults to fully opaque.")
        .def(nb::init<uint8_t>(), "v"_a,
             "Construct a gray Rgba8 with every channel set to v.")
        .def(nb::init<float, float, float, float>(), "r"_a, "g"_a, "b"_a, "a"_a = 1.0,
             "Construct from four 0..1 floats, clamped and quantized to uint8.")
        .def(nb::init<Vec3f>(), "rgb"_a,
             "Construct from an RGB float triple; alpha defaults to opaque.")
        .def(nb::init<Vec4f>(), "rgba"_a,
             "Construct from an RGBA float quadruple.")
        .def(nb::self < nb::self, "rhs"_a,
             "Less-than comparison on the packed uint32 representation.")
        .def(nb::self == nb::self, "rhs"_a,
             "Equality on the packed uint32 representation.")
        .def("lengthSqr", &math::Rgba8::lengthSqr,
             "Squared length over (r, g, b, a) as integers.")
        .def("length", &math::Rgba8::length,
             "Euclidean length over (r, g, b, a) as floats.")
        .def("asFloat", &math::Rgba8::asFloat, "n"_a,
             "Return the n-th channel as a 0..1 float.")
        .def(
            "__getitem__",
            [](const math::Rgba8& rgba, size_t i) {
                if (i >= 4) {
                    throw nb::index_error();
                }
                return rgba[static_cast<int>(i)];
            },
            "i"_a,
            "Read the i-th channel as a uint8 (0=r, 1=g, 2=b, 3=a).")
        .def(
            "__setitem__",
            [](math::Rgba8& rgba, size_t i, ValueType value) {
                if (i >= 4) {
                    throw nb::index_error();
                }
                rgba[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a,
            "Write the i-th channel as a uint8 (0=r, 1=g, 2=b, 3=a).")
        .def_prop_rw(
            "packed", [](const math::Rgba8& rgba) { return rgba.packed(); }, [](math::Rgba8& rgba, uint32_t packed) { rgba.packed() = packed; },
            "The raw 32-bit packed RGBA representation.")
        .def_prop_rw(
            "r", [](const math::Rgba8& rgba) { return rgba.r(); }, [](math::Rgba8& rgba, uint8_t r) { rgba.r() = r; },
            "Red channel as a uint8 0..255.")
        .def_prop_rw(
            "g", [](const math::Rgba8& rgba) { return rgba.g(); }, [](math::Rgba8& rgba, uint8_t g) { rgba.g() = g; },
            "Green channel as a uint8 0..255.")
        .def_prop_rw(
            "b", [](const math::Rgba8& rgba) { return rgba.b(); }, [](math::Rgba8& rgba, uint8_t b) { rgba.b() = b; },
            "Blue channel as a uint8 0..255.")
        .def_prop_rw(
            "a", [](const math::Rgba8& rgba) { return rgba.a(); }, [](math::Rgba8& rgba, uint8_t a) { rgba.a() = a; },
            "Alpha channel as a uint8 0..255.")
        .def("asVec3f", [](const math::Rgba8& rgba) { return Vec3f(rgba); },
             "Convert RGB channels to a Vec3f of 0..1 floats (alpha dropped).")
        .def("asVec4f", [](const math::Rgba8& rgba) { return Vec4f(rgba); },
             "Convert RGBA channels to a Vec4f of 0..1 floats.");
}

template<typename Vec3T> void defineBaseBBox(nb::module_& m, const char* name)
{
    nb::class_<math::BaseBBox<Vec3T>>(m, name,
        "Axis-aligned bounding-box base class. Stores a min / max corner; "
        "concrete subclasses add the open-interval vs closed-interval semantics.")
        .def(nb::self == nb::self, "rhs"_a,
             "Equality of both min and max corners.")
        .def(nb::self != nb::self, "rhs"_a,
             "Inequality of either min or max corner.")
        .def(
            "__getitem__",
            [](const math::BaseBBox<Vec3T>& bbox, size_t i) {
                if (i >= 2) {
                    throw nb::index_error();
                }
                return bbox[static_cast<int>(i)];
            },
            "i"_a,
            "Read corner 0 (min) or corner 1 (max).")
        .def(
            "__setitem__",
            [](math::BaseBBox<Vec3T>& bbox, size_t i, const Vec3T& value) {
                if (i >= 2) {
                    throw nb::index_error();
                }
                bbox[static_cast<int>(i)] = value;
            },
            "i"_a,
            "value"_a,
            "Write corner 0 (min) or corner 1 (max).")
        .def_prop_rw(
            "min", [](const math::BaseBBox<Vec3T>& bbox) { return bbox.min(); }, [](math::BaseBBox<Vec3T>& bbox, const Vec3T& min) { bbox.min() = min; },
            "Minimum corner of the bounding box.")
        .def_prop_rw(
            "max", [](const math::BaseBBox<Vec3T>& bbox) { return bbox.max(); }, [](math::BaseBBox<Vec3T>& bbox, const Vec3T& max) { bbox.max() = max; },
            "Maximum corner of the bounding box.")
        .def("translate", &math::BaseBBox<Vec3T>::translate, "xyz"_a,
             "Translate this bounding box by xyz in place.")
        .def("expand", nb::overload_cast<const Vec3T&>(&math::BaseBBox<Vec3T>::expand), "xyz"_a,
             "Grow this bounding box to include the point xyz.")
        .def("expand", nb::overload_cast<const math::BaseBBox<Vec3T>&>(&math::BaseBBox<Vec3T>::expand), "bbox"_a,
             "Grow this bounding box to include another bbox in its entirety.")
        .def("intersect", &math::BaseBBox<Vec3T>::intersect, "bbox"_a,
             "Shrink this bounding box to the intersection with bbox.")
        .def("isInside", &math::BaseBBox<Vec3T>::isInside, "xyz"_a,
             "True iff xyz lies inside this bounding box.");
}

template<typename Vec3T> void defineBBoxFloatingPoint(nb::module_& m, const char* name, const char* doc)
{
    nb::class_<math::BBox<Vec3T, true>, math::BaseBBox<Vec3T>>(m, name, doc)
        .def(nb::init<>(),
             "Construct an empty bounding box (min > max sentinel).")
        .def(nb::init<const Vec3T&, const Vec3T&>(), "min"_a, "max"_a,
             "Construct from explicit min and max corners.")
        .def(nb::init<const math::Coord&, const math::Coord&>(), "min"_a, "max"_a,
             "Construct from integer Coord corners, cast to floating-point.")
        .def_static("createCube", &math::BBox<Vec3T>::createCube, "min"_a, "dim"_a,
                    "Construct an axis-aligned cube of side dim anchored at min.")
        .def(nb::init<const math::BaseBBox<math::Coord>&>(), "bbox"_a,
             "Construct from an integer CoordBBox, cast to floating-point.")
        .def("empty", &math::BBox<Vec3T>::empty,
             "True iff this bbox is empty (any min component > the matching max).")
        .def("dim", &math::BBox<Vec3T>::dim,
             "Return max - min as a Vec3 of side lengths.")
        .def("isInside", &math::BBox<Vec3T>::isInside, "p"_a,
             "True iff p lies inside this bounding box (half-open interval).")
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
        .def(nb::init<>(),
             "Construct an empty CoordBBox (min > max sentinel).")
        .def(nb::init<const CoordT&, const CoordT&>(), "min"_a, "max"_a,
             "Construct from explicit min and max Coord corners (inclusive).")
        .def(
            "__iter__",
            [](const math::BBox<CoordT>& b) { return nb::make_iterator(nb::type<math::BBox<CoordT>>(), "CoordBBoxIterator", b.begin(), b.end()); },
            nb::keep_alive<0, 1>(),
            "Iterate over every Coord in this CoordBBox in row-major order.")
        .def_static("createCube", nb::overload_cast<const CoordT&, ValueType>(&math::BBox<CoordT>::createCube), "min"_a, "dim"_a,
                    "Construct a cube of side dim voxels anchored at the min Coord.")
        .def_static("createCube", nb::overload_cast<ValueType, ValueType>(&math::BBox<CoordT>::createCube), "min"_a, "max"_a,
                    "Construct a cube spanning [min, max] in every axis.")
        .def("is_divisible", &math::BBox<CoordT>::is_divisible,
             "True iff this CoordBBox has more than one voxel in every axis.")
        .def("empty", &math::BBox<CoordT>::empty,
             "True iff this CoordBBox is empty (any min component > the matching max).")
        .def("dim", &math::BBox<CoordT>::dim,
             "Return max - min + 1 as a Coord of side lengths (inclusive).")
        .def("volume", &math::BBox<CoordT>::volume,
             "Total number of voxels enclosed by this CoordBBox.")
        .def("isInside", nb::overload_cast<const CoordT&>(&math::BBox<CoordT>::isInside, nb::const_), "p"_a,
             "True iff the integer point p lies inside this CoordBBox.")
        .def("isInside", nb::overload_cast<const math::BBox<CoordT>&>(&math::BBox<CoordT>::isInside, nb::const_), "b"_a,
             "True iff b lies entirely inside this CoordBBox.")
        .def("asFloat", &math::BBox<CoordT>::template asReal<float>,
             "Convert this CoordBBox to a floating-point BBox of Vec3f.")
        .def("asDouble", &math::BBox<CoordT>::template asReal<double>,
             "Convert this CoordBBox to a floating-point BBox of Vec3d.")
        .def("hasOverlap", &math::BBox<CoordT>::hasOverlap, "b"_a,
             "True iff this CoordBBox shares any voxel with b.")
        .def("expandBy", &math::BBox<CoordT>::expandBy, "padding"_a,
             "Grow this CoordBBox by the given padding in every direction.")
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

#define NANOVDB_PY_FOR_EACH_SAMPLEABLE_BUILDT(T, Suffix)               \
    defineNearestNeighborSampler<T>(m, #Suffix "NearestNeighborSampler"); \
    defineTrilinearSampler<T>(m, #Suffix "TrilinearSampler");       \
    defineTriquadraticSampler<T>(m, #Suffix "TriquadraticSampler"); \
    defineTricubicSampler<T>(m, #Suffix "TricubicSampler");
#include "BuildTypes.def"
}

} // namespace pynanovdb
