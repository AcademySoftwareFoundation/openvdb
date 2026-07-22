// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyRayIntersector.cc
/// @brief nanobind wrappers for ray intersectors

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/RayIntersector.h>
#include <limits>
#include <memory>
#include <vector>
#include "pyTypeCasters.h"

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace {

using RayT = math::Ray<double>;

inline double
sequenceItemAsDouble(PyObject* sequence, Py_ssize_t index)
{
    PyObject* item = PySequence_GetItem(sequence, index);
    if (!item) {
        throw nb::python_error();
    }
    PyObject* number = PyNumber_Float(item);
    Py_DECREF(item);
    if (!number) {
        throw nb::python_error();
    }
    const double value = PyFloat_AsDouble(number);
    Py_DECREF(number);
    if (PyErr_Occurred()) {
        throw nb::python_error();
    }
    return value;
}

inline Vec3d
sequenceAsVec3d(PyObject* object)
{
    if (!PySequence_Check(object) || PySequence_Length(object) != 3) {
        throw nb::type_error("expected a 3-item point or vector");
    }
    return Vec3d(
        sequenceItemAsDouble(object, 0),
        sequenceItemAsDouble(object, 1),
        sequenceItemAsDouble(object, 2));
}

inline RayT
rayFromObject(nb::object rayObj)
{
    PyObject* ray = rayObj.ptr();
    if (!PySequence_Check(ray)) {
        throw nb::type_error("expected ray as ((origin), (direction)) or a flat 6/8-item sequence");
    }

    const Py_ssize_t length = PySequence_Length(ray);
    if (length == 2 || length == 4) {
        PyObject* originObj = PySequence_GetItem(ray, 0);
        if (!originObj) {
            throw nb::python_error();
        }
        PyObject* directionObj = PySequence_GetItem(ray, 1);
        if (!directionObj) {
            Py_DECREF(originObj);
            throw nb::python_error();
        }
        const Vec3d origin = sequenceAsVec3d(originObj);
        const Vec3d direction = sequenceAsVec3d(directionObj);
        Py_DECREF(originObj);
        Py_DECREF(directionObj);

        if (length == 4) {
            return RayT(origin, direction, sequenceItemAsDouble(ray, 2), sequenceItemAsDouble(ray, 3));
        }
        return RayT(origin, direction);
    }

    if (length == 6 || length == 8) {
        const Vec3d origin(
            sequenceItemAsDouble(ray, 0),
            sequenceItemAsDouble(ray, 1),
            sequenceItemAsDouble(ray, 2));
        const Vec3d direction(
            sequenceItemAsDouble(ray, 3),
            sequenceItemAsDouble(ray, 4),
            sequenceItemAsDouble(ray, 5));
        if (length == 8) {
            return RayT(origin, direction, sequenceItemAsDouble(ray, 6), sequenceItemAsDouble(ray, 7));
        }
        return RayT(origin, direction);
    }

    throw nb::type_error("expected ray as ((origin), (direction)) or a flat 6/8-item sequence");
}

template<typename CoordT>
inline RayT
rayFromArrayRow(nb::ndarray<CoordT, nb::ndim<2>, nb::device::cpu> rays, size_t n)
{
    const Vec3d origin(rays(n, 0), rays(n, 1), rays(n, 2));
    const Vec3d direction(rays(n, 3), rays(n, 4), rays(n, 5));
    if (rays.shape(1) == 8) {
        return RayT(origin, direction, rays(n, 6), rays(n, 7));
    }
    return RayT(origin, direction);
}

class PyLevelSetRayIntersector
{
public:
    explicit PyLevelSetRayIntersector(FloatGrid::ConstPtr grid, float isoValue = 0.0f)
        : mImpl(new Impl<FloatGrid>(grid, isoValue))
    {
    }

#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    explicit PyLevelSetRayIntersector(DoubleGrid::ConstPtr grid, double isoValue = 0.0)
        : mImpl(new Impl<DoubleGrid>(grid, isoValue))
    {
    }
#endif

    double isoValue() const
    {
        return mImpl->isoValue();
    }

    nb::object intersectWS(nb::object ray) const
    {
        return mImpl->intersectWS(rayFromObject(ray));
    }

    nb::tuple intersectBatchWS(nb::ndarray<float, nb::ndim<2>, nb::device::cpu> rays) const
    {
        return mImpl->intersectBatchWS(rays);
    }

    nb::tuple intersectBatchWS(nb::ndarray<double, nb::ndim<2>, nb::device::cpu> rays) const
    {
        return mImpl->intersectBatchWS(rays);
    }

private:
    struct BaseImpl
    {
        virtual ~BaseImpl() = default;
        virtual double isoValue() const = 0;
        virtual nb::object intersectWS(const RayT& ray) const = 0;
        virtual nb::tuple intersectBatchWS(
            nb::ndarray<float, nb::ndim<2>, nb::device::cpu> rays) const = 0;
        virtual nb::tuple intersectBatchWS(
            nb::ndarray<double, nb::ndim<2>, nb::device::cpu> rays) const = 0;
    };

    template<typename GridType>
    struct Impl: public BaseImpl
    {
        using ValueT = typename GridType::ValueType;
        using IntersectorT = tools::LevelSetRayIntersector<GridType>;

        explicit Impl(typename GridType::ConstPtr grid, ValueT isoValue)
            : mGrid(grid)
            , mIntersector()
        {
            if (!mGrid) {
                OPENVDB_THROW(ValueError, "LevelSetRayIntersector requires a non-null grid");
            }
            mIntersector.reset(new IntersectorT(*mGrid, isoValue));
        }

        double isoValue() const override
        {
            return static_cast<double>(mIntersector->getIsoValue());
        }

        nb::object intersectWS(const RayT& ray) const override
        {
            Vec3d position;
            Vec3d normal;
            double time = 0.0;
            if (!mIntersector->intersectsWS(ray, position, normal, time)) {
                return nb::none();
            }
            return nb::cast(nb::make_tuple(position, normal, time));
        }

        nb::tuple intersectBatchWS(
            nb::ndarray<float, nb::ndim<2>, nb::device::cpu> rays) const override
        {
            return this->intersectBatch(rays);
        }

        nb::tuple intersectBatchWS(
            nb::ndarray<double, nb::ndim<2>, nb::device::cpu> rays) const override
        {
            return this->intersectBatch(rays);
        }

        template<typename CoordT>
        nb::tuple intersectBatch(nb::ndarray<CoordT, nb::ndim<2>, nb::device::cpu> rays) const
        {
            if (rays.shape(1) != 6 && rays.shape(1) != 8) {
                throw nb::value_error("expected an Nx6 or Nx8 ray array");
            }

            const size_t size = rays.shape(0);
            auto hits = new bool[size];
            auto positions = new std::vector<Vec3d>(size);
            auto normals = new std::vector<Vec3d>(size);
            auto times = new std::vector<double>(size);
            const double nan = std::numeric_limits<double>::quiet_NaN();

            for (size_t n = 0; n < size; ++n) {
                Vec3d position(nan);
                Vec3d normal(nan);
                double time = nan;
                hits[n] = mIntersector->intersectsWS(rayFromArrayRow(rays, n), position, normal, time);
                (*positions)[n] = position;
                (*normals)[n] = normal;
                (*times)[n] = time;
            }

            nb::capsule hitsDeleter(hits, [](void* p) noexcept {
                delete[] static_cast<bool*>(p);
            });
            nb::ndarray<nb::numpy, bool> hitsArray(hits, {size}, hitsDeleter);

            nb::capsule positionsDeleter(positions, [](void* p) noexcept {
                delete static_cast<std::vector<Vec3d>*>(p);
            });
            nb::ndarray<nb::numpy, double> positionsArray(
                size ? positions->data()->asPointer() : nullptr,
                {size, size_t(3)}, positionsDeleter, {3, 1});

            nb::capsule normalsDeleter(normals, [](void* p) noexcept {
                delete static_cast<std::vector<Vec3d>*>(p);
            });
            nb::ndarray<nb::numpy, double> normalsArray(
                size ? normals->data()->asPointer() : nullptr,
                {size, size_t(3)}, normalsDeleter, {3, 1});

            nb::capsule timesDeleter(times, [](void* p) noexcept {
                delete static_cast<std::vector<double>*>(p);
            });
            nb::ndarray<nb::numpy, double> timesArray(times->data(), {size}, timesDeleter);

            return nb::make_tuple(hitsArray, positionsArray, normalsArray, timesArray);
        }

        typename GridType::ConstPtr mGrid;
        std::unique_ptr<IntersectorT> mIntersector;
    };

    std::unique_ptr<BaseImpl> mImpl;
};

} // namespace

void
exportLevelSetRayIntersector(nb::module_ m)
{
    auto cls = nb::class_<PyLevelSetRayIntersector>(m, "LevelSetRayIntersector",
        "Reusable world-space ray intersector for level set grids.");

    cls.def(nb::init<FloatGrid::ConstPtr, float>(),
            nb::arg("grid"), nb::arg("isoValue") = 0.0f,
            "Construct a ray intersector for a FloatGrid level set.");

#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    cls.def(nb::init<DoubleGrid::ConstPtr, double>(),
            nb::arg("grid"), nb::arg("isoValue") = 0.0,
            "Construct a ray intersector for a DoubleGrid level set.");
#endif

    cls.def_prop_ro("isoValue", &PyLevelSetRayIntersector::isoValue,
            "Iso-value used for ray intersections.");

    cls.def("intersectWS", &PyLevelSetRayIntersector::intersectWS, nb::arg("ray"),
            "Intersect one world-space ray. Return None on miss or (position, normal, time) on hit.");

    cls.def("intersectBatchWS",
            [](const PyLevelSetRayIntersector& intersector,
               nb::ndarray<float, nb::ndim<2>, nb::device::cpu> rays) {
                return intersector.intersectBatchWS(rays);
            },
            nb::arg("rays"),
            "Intersect an Nx6 or Nx8 array of world-space rays.");

    cls.def("intersectBatchWS",
            [](const PyLevelSetRayIntersector& intersector,
               nb::ndarray<double, nb::ndim<2>, nb::device::cpu> rays) {
                return intersector.intersectBatchWS(rays);
            },
            nb::arg("rays"),
            "Intersect an Nx6 or Nx8 array of world-space rays.");
}
