// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <openvdb/openvdb.h>
#include "pyTypeCasters.h"

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace pyTransform {

inline Vec3d indexToWorld(math::Transform& t, const Vec3d& p) { return t.indexToWorld(p); }
inline Vec3d worldToIndex(math::Transform& t, const Vec3d& p) { return t.worldToIndex(p); }

inline Coord worldToIndexCellCentered(math::Transform& t, const Vec3d& p) {
    return t.worldToIndexCellCentered(p);
}
inline Coord worldToIndexNodeCentered(math::Transform& t, const Vec3d& p) {
    return t.worldToIndexNodeCentered(p);
}


inline std::string
info(math::Transform& t)
{
    std::ostringstream ostr;
    t.print(ostr);
    return ostr.str();
}


inline math::Transform::Ptr
createLinearTransform(double dim)
{
    return math::Transform::createLinearTransform(dim);
}


inline math::Transform::Ptr
createLinearTransform(const std::vector<std::vector<double> >& sequence)
{
    Mat4R m;

    // // Verify that obj is a four-element sequence.
    bool is4x4Seq = sequence.size() ==4;
    for (size_t i = 0; i < sequence.size(); ++i)
        is4x4Seq &= sequence[i].size() == 4;

    if (is4x4Seq) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                m[row][col] = sequence[row][col];
            }
        }
    }
    if (!is4x4Seq) {
        throw nb::value_error("expected a 4 x 4 sequence of numeric values");
    }

    return math::Transform::createLinearTransform(m);
}


inline math::Transform::Ptr
createFrustum(const Coord& xyzMin, const Coord& xyzMax,
    double taper, double depth, double voxelDim = 1.0)
{
    return math::Transform::createFrustumTransform(
        BBoxd(xyzMin.asVec3d(), xyzMax.asVec3d()), taper, depth, voxelDim);
}


////////////////////////////////////////


struct PickleSuite
{
    enum { STATE_MAJOR = 0, STATE_MINOR, STATE_FORMAT, STATE_XFORM };

    /// Return a tuple representing the state of the given Transform.
    static std::tuple<uint32_t, uint32_t, uint32_t, nb::bytes> getState(const math::Transform& xform)
    {
        std::ostringstream ostr(std::ios_base::binary);
        // Serialize the Transform to a string.
        xform.write(ostr);

        nb::bytes bytesObj(ostr.str().c_str(), ostr.str().length());

        // Construct a state tuple comprising the version numbers of
        // the serialization format and the serialized Transform.
        // Convert the byte string to a "bytes" sequence.
        return std::make_tuple(
            uint32_t(OPENVDB_LIBRARY_MAJOR_VERSION),
            uint32_t(OPENVDB_LIBRARY_MINOR_VERSION),
            uint32_t(OPENVDB_FILE_VERSION),
            bytesObj);
    }

    /// Restore the given Transform to a saved state.
    static void setState(math::Transform& xform, const std::tuple<uint32_t, uint32_t, uint32_t, nb::bytes>& state)
    {
        openvdb::VersionId libVersion;
        uint32_t formatVersion = 0;

        libVersion.first = std::get<0>(state);
        libVersion.second = std::get<1>(state);
        formatVersion = std::get<2>(state);

        nb::bytes bytesObj = std::get<3>(state);
        std::string serialized(bytesObj.c_str(), bytesObj.c_str() + bytesObj.size());

        // Restore the internal state of the C++ object.
        std::istringstream istr(serialized, std::ios_base::binary);
        io::setVersion(istr, libVersion, formatVersion);
        new (&xform) math::Transform;
        xform.read(istr);
    }
}; // struct PickleSuite

} // namespace pyTransform


void
exportTransform(nb::module_ m)
{
    nb::enum_<math::Axis>(m, "Axis")
        .value("X", math::X_AXIS)
        .value("Y", math::Y_AXIS)
        .value("Z", math::Z_AXIS)
        .export_values();

    nb::class_<math::Transform>(m, "Transform")
        .def(nb::init<>())

        .def("deepCopy", &math::Transform::copy,
            "Return a copy of this transform.")

        /// @todo Should this also be __str__()?
        .def("info", &pyTransform::info,
            "Return a string containing a description of this transform.")

        .def("__getstate__", &pyTransform::PickleSuite::getState)
        .def("__setstate__", &pyTransform::PickleSuite::setState)

        .def_prop_ro("typeName", &math::Transform::mapType,
            "name of this transform's type")
        .def_prop_ro("isLinear", &math::Transform::isLinear,
            "True if this transform is linear")

        .def("preRotate", &math::Transform::preRotate,
            nb::arg("radians"), nb::arg("axis") = math::X_AXIS,
            "Prepend a rotation about either Axis.X, Axis.Y or Axis.Z.")
        .def("preTranslate", &math::Transform::preTranslate, nb::arg("xyz"),
            "Prepend a translation.")
        .def("preScale", nb::overload_cast<double>(&math::Transform::preScale), nb::arg("s"),
            "Prepend a uniform scale.")
        .def("preScale", nb::overload_cast<const Vec3d&>(&math::Transform::preScale), nb::arg("sxyz"),
            "Prepend a nonuniform scale.")
        .def("preShear", &math::Transform::preShear,
            nb::arg("s"), nb::arg("axis0"), nb::arg("axis1"),
            "Prepend a shear (axis0 and axis1 are either Axis.X, Axis.Y or Axis.Z).")

        .def("postRotate", &math::Transform::postRotate,
            nb::arg("radians"), nb::arg("axis") = math::X_AXIS,
            "Postfix a rotation about either Axis.X, Axis.Y or Axis.Z.")
        .def("postTranslate", &math::Transform::postTranslate, nb::arg("xyz"),
            "Postfix a translation.")
        .def("postScale", nb::overload_cast<double>(&math::Transform::postScale), nb::arg("s"),
            "Postfix a uniform scale.")
        .def("postScale", nb::overload_cast<const Vec3d&>(&math::Transform::postScale), nb::arg("sxyz"),
            "Postfix a nonuniform scale.")
        .def("postShear", &math::Transform::postShear,
            nb::arg("s"), nb::arg("axis0"), nb::arg("axis1"),
            "Postfix a shear (axis0 and axis1 are either Axis.X, Axis.Y or Axis.Z).")

        .def("voxelSize", nb::overload_cast<>(&math::Transform::voxelSize, nb::const_),
            "Return the size of voxels of the linear component of this transform.")
        .def("voxelSize", nb::overload_cast<const Vec3d&>(&math::Transform::voxelSize, nb::const_), nb::arg("xyz"),
            "Return the size of the voxel at position (x, y, z).")

        .def("voxelVolume", nb::overload_cast<>(&math::Transform::voxelVolume, nb::const_),
            "Return the voxel volume of the linear component of this transform.")
        .def("voxelVolume", nb::overload_cast<const Vec3d&>(&math::Transform::voxelVolume, nb::const_), nb::arg("xyz"),
            "Return the voxel volume at position (x, y, z).")

        .def("indexToWorld", &pyTransform::indexToWorld, nb::arg("xyz"),
            "Apply this transformation to the given coordinates.")
        .def("worldToIndex", &pyTransform::worldToIndex, nb::arg("xyz"),
            "Apply the inverse of this transformation to the given coordinates.")
        .def("worldToIndexCellCentered", &pyTransform::worldToIndexCellCentered,
            nb::arg("xyz"),
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result to the nearest integer coordinates.")
        .def("worldToIndexNodeCentered", &pyTransform::worldToIndexNodeCentered,
            nb::arg("xyz"),
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result down to the nearest integer coordinates.")

        // Allow Transforms to be compared for equality and inequality.
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    m.def("createLinearTransform", nb::overload_cast<double>(&pyTransform::createLinearTransform),
        nb::arg("voxelSize") = 1.0,
        "Create a new linear transform with the given uniform voxel size.");

    m.def("createLinearTransform", nb::overload_cast<const std::vector<std::vector<double> >&>(&pyTransform::createLinearTransform), nb::arg("matrix"),
        "Create a new linear transform from a 4 x 4 matrix given as a sequence\n"
        "of the form [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]],\n"
        "where [m, n, o, p] is the translation component.");

    m.def("createFrustumTransform", &pyTransform::createFrustum,
        nb::arg("xyzMin"), nb::arg("xyzMax"),
         nb::arg("taper"), nb::arg("depth"), nb::arg("voxelSize") = 1.0,
        "Create a new frustum transform with unit bounding box (xyzMin, xyzMax)\n"
        "and the given taper, depth and uniform voxel size.");
}
