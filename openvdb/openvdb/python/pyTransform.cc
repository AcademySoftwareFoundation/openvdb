// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <openvdb/openvdb.h>
#include "pyTypeCasters.h"

namespace py = pybind11;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace pyTransform {

inline void scale1(math::Transform& t, double s) { t.preScale(s); }
inline void scale3(math::Transform& t, const Vec3d& xyz) { t.preScale(xyz); }

inline Vec3d voxelDim0(math::Transform& t) { return t.voxelSize(); }
inline Vec3d voxelDim1(math::Transform& t, const Vec3d& p) { return t.voxelSize(p); }

inline double voxelVolume0(math::Transform& t) { return t.voxelVolume(); }
inline double voxelVolume1(math::Transform& t, const Vec3d& p) { return t.voxelVolume(p); }

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
        throw py::value_error("expected a 4 x 4 sequence of numeric values");
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
    static py::tuple getState(const math::Transform& xform)
    {
        std::ostringstream ostr(std::ios_base::binary);
        // Serialize the Transform to a string.
        xform.write(ostr);

        // Construct a state tuple comprising the version numbers of
        // the serialization format and the serialized Transform.
        // Convert the byte string to a "bytes" sequence.
        py::bytes bytesObj(ostr.str());
        return py::make_tuple(
            uint32_t(OPENVDB_LIBRARY_MAJOR_VERSION),
            uint32_t(OPENVDB_LIBRARY_MINOR_VERSION),
            uint32_t(OPENVDB_FILE_VERSION),
            bytesObj);
    }

    /// Restore the given Transform to a saved state.
    static math::Transform setState(py::tuple state)
    {
        bool badState = (py::len(state) != 4);

        openvdb::VersionId libVersion;
        uint32_t formatVersion = 0;
        if (!badState) {
            // Extract the serialization format version numbers.
            const int idx[3] = { STATE_MAJOR, STATE_MINOR, STATE_FORMAT };
            uint32_t version[3] = { 0, 0, 0 };
            for (int i = 0; i < 3 && !badState; ++i) {
                if (py::isinstance<py::int_>(state[idx[i]]))
                    version[i] = py::cast<uint32_t>(state[idx[i]]);
                else badState = true;
            }
            libVersion.first = version[0];
            libVersion.second = version[1];
            formatVersion = version[2];
        }

        std::string serialized;
        if (!badState) {
            // Extract the sequence containing the serialized Transform.
            py::object bytesObj = state[int(STATE_XFORM)];
            if (py::isinstance<py::bytes>(bytesObj))
                serialized = py::cast<py::bytes>(bytesObj);
            else badState = true;
        }

        if (badState) {
            std::ostringstream os;
            os << "expected (int, int, int, bytes) tuple in call to __setstate__; found ";
            os << py::cast<std::string>(state.attr("__repr__")());
            throw py::value_error(os.str());
        }

        // Restore the internal state of the C++ object.
        std::istringstream istr(serialized, std::ios_base::binary);
        io::setVersion(istr, libVersion, formatVersion);
        math::Transform xform;
        xform.read(istr);
        return xform;
    }
}; // struct PickleSuite

} // namespace pyTransform


void
exportTransform(py::module_ m)
{
    py::enum_<math::Axis>(m, "Axis")
        .value("X", math::X_AXIS)
        .value("Y", math::Y_AXIS)
        .value("Z", math::Z_AXIS)
        .export_values();

    py::class_<math::Transform, math::Transform::Ptr>(m, "Transform")
        .def(py::init<>())

        .def("deepCopy", &math::Transform::copy,
            "deepCopy() -> Transform\n\n"
            "Return a copy of this transform.")

        /// @todo Should this also be __str__()?
        .def("info", &pyTransform::info,
            "info() -> str\n\n"
            "Return a string containing a description of this transform.\n")

        .def(py::pickle(&pyTransform::PickleSuite::getState, &pyTransform::PickleSuite::setState))

        .def_property_readonly("typeName", &math::Transform::mapType,
            "name of this transform's type")
        .def_property_readonly("isLinear", &math::Transform::isLinear,
            "True if this transform is linear")

        .def("rotate", &math::Transform::preRotate,
            py::arg("radians"), py::arg("axis") = math::X_AXIS,
            "rotate(radians, axis)\n\n"
            "Accumulate a rotation about either Axis.X, Axis.Y or Axis.Z.")
        .def("translate", &math::Transform::postTranslate, py::arg("xyz"),
            "translate((x, y, z))\n\n"
            "Accumulate a translation.")
        .def("scale", &pyTransform::scale1, py::arg("s"),
            "scale(s)\n\n"
            "Accumulate a uniform scale.")
        .def("scale", &pyTransform::scale3, py::arg("sxyz"),
            "scale((sx, sy, sz))\n\n"
            "Accumulate a nonuniform scale.")
        .def("shear", &math::Transform::preShear,
            py::arg("s"), py::arg("axis0"), py::arg("axis1"),
            "shear(s, axis0, axis1)\n\n"
            "Accumulate a shear (axis0 and axis1 are either\n"
            "Axis.X, Axis.Y or Axis.Z).")

        .def("voxelSize", &pyTransform::voxelDim0,
            "voxelSize() -> (dx, dy, dz)\n\n"
            "Return the size of voxels of the linear component of this transform.")
        .def("voxelSize", &pyTransform::voxelDim1, py::arg("xyz"),
            "voxelSize((x, y, z)) -> (dx, dy, dz)\n\n"
            "Return the size of the voxel at position (x, y, z).")

        .def("voxelVolume", &pyTransform::voxelVolume0,
            "voxelVolume() -> float\n\n"
            "Return the voxel volume of the linear component of this transform.")
        .def("voxelVolume", &pyTransform::voxelVolume1, py::arg("xyz"),
            "voxelVolume((x, y, z)) -> float\n\n"
            "Return the voxel volume at position (x, y, z).")

        .def("indexToWorld", &pyTransform::indexToWorld, py::arg("xyz"),
            "indexToWorld((x, y, z)) -> (x', y', z')\n\n"
            "Apply this transformation to the given coordinates.")
        .def("worldToIndex", &pyTransform::worldToIndex, py::arg("xyz"),
            "worldToIndex((x, y, z)) -> (x', y', z')\n\n"
            "Apply the inverse of this transformation to the given coordinates.")
        .def("worldToIndexCellCentered", &pyTransform::worldToIndexCellCentered,
            py::arg("xyz"),
            "worldToIndexCellCentered((x, y, z)) -> (i, j, k)\n\n"
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result to the nearest integer coordinates.")
        .def("worldToIndexNodeCentered", &pyTransform::worldToIndexNodeCentered,
            py::arg("xyz"),
            "worldToIndexNodeCentered((x, y, z)) -> (i, j, k)\n\n"
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result down to the nearest integer coordinates.")

        // Allow Transforms to be compared for equality and inequality.
        .def(py::self == py::self)
        .def(py::self != py::self);

    m.def("createLinearTransform", py::overload_cast<double>(&pyTransform::createLinearTransform),
        py::arg("voxelSize") = 1.0,
        "createLinearTransform(voxelSize) -> Transform\n\n"
        "Create a new linear transform with the given uniform voxel size.");

    m.def("createLinearTransform", py::overload_cast<const std::vector<std::vector<double> >&>(&pyTransform::createLinearTransform), py::arg("matrix"),
        "createLinearTransform(matrix) -> Transform\n\n"
        "Create a new linear transform from a 4 x 4 matrix given as a sequence\n"
        "of the form [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]],\n"
        "where [m, n, o, p] is the translation component.");

    m.def("createFrustumTransform", &pyTransform::createFrustum,
        py::arg("xyzMin"), py::arg("xyzMax"),
         py::arg("taper"), py::arg("depth"), py::arg("voxelSize") = 1.0,
        "createFrustumTransform(xyzMin, xyzMax, taper, depth, voxelSize) -> Transform\n\n"
        "Create a new frustum transform with unit bounding box (xyzMin, xyzMax)\n"
        "and the given taper, depth and uniform voxel size.");
}
