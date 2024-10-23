#ifndef OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED
#define OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <openvdb/openvdb.h>
#include <openvdb/MetaMap.h>

// Custom type casters must be defined consistently in EVERY translation unit of
// nanobind code. Thus, we define them in this header file and include it in every
// implementation file.

namespace nb = nanobind;

namespace nanobind { namespace detail {
    template <> struct type_caster<openvdb::Coord> {
    public:
        NB_TYPE_CASTER(openvdb::Coord, const_name("openvdb::Coord"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length == 1) {
                openvdb::Coord::Int32 items[1] = { 0 }; // init required to workaround gcc warning
                for (Py_ssize_t i = 0; i < length; ++i) {
                    PyObject* item = PySequence_GetItem(source, i);
                    if (item) {
                        PyObject* number = PyNumber_Long(item);
                        if (number) {
                            items[i] = static_cast<openvdb::Coord::Int32>(PyLong_AsLong(number));
                        }
                        Py_XDECREF(number);
                    }
                    Py_XDECREF(item);

                    if (PyErr_Occurred())
                        return false;
                }
                value.reset(items[0]);
                return true;
            }
            else if (length == 3)
            {
                openvdb::Coord::Int32 items[3] = { 0, 0, 0 }; // init required to workaround gcc warning
                for (Py_ssize_t i = 0; i < length; ++i) {
                    PyObject* item = PySequence_GetItem(source, i);
                    if (item) {
                        PyObject* number = PyNumber_Long(item);
                        if (number) {
                            items[i] = static_cast<openvdb::Coord::Int32>(PyLong_AsLong(number));
                        }
                        Py_XDECREF(number);
                    }
                    Py_XDECREF(item);

                    if (PyErr_Occurred())
                        return false;
                }
                value.reset(items[0], items[1], items[2]);
                return true;
            }
            else
            {
                return false;
            }
        }

        static handle from_cpp(openvdb::Coord src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2i> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2i, const_name("openvdb::Vec2i"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec2i::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec2i::value_type>(PyLong_AsLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec2i src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2I> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2I, const_name("openvdb::Vec2I"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec2I::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec2I::value_type>(PyLong_AsUnsignedLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec2I src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2s> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2s, const_name("openvdb::Vec2s"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec2s::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec2s::value_type>(PyFloat_AsDouble(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec2s src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2d> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2d, const_name("openvdb::Vec2d"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec2d::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = PyFloat_AsDouble(number);
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec2d src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3i> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3i, const_name("openvdb::Vec3i"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec3i::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec3i::value_type>(PyLong_AsLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec3i src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3I> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3I, const_name("openvdb::Vec3I"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec3I::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec3I::value_type>(PyLong_AsUnsignedLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec3I src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3s> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3s, const_name("openvdb::Vec3s"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec3s::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec3s::value_type>(PyFloat_AsDouble(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec3s src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3d> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3d, const_name("openvdb::Vec3d"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec3d::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = PyFloat_AsDouble(number);
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec3d src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4i> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4i, const_name("openvdb::Vec4i"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec4i::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec4i::value_type>(PyLong_AsLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec4i src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4I> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4I, const_name("openvdb::Vec4I"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec4I::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Long(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec4I::value_type>(PyLong_AsUnsignedLong(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec4I src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4s> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4s, const_name("openvdb::Vec4s"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec4s::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = static_cast<openvdb::Vec4s::value_type>(PyFloat_AsDouble(number));
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec4s src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4d> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4d, const_name("openvdb::Vec4d"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length != openvdb::Vec4d::size)
                return false;

            for (Py_ssize_t i = 0; i < length; ++i) {
                PyObject* item = PySequence_GetItem(source, i);
                if (item) {
                    PyObject* number = PyNumber_Float(item);
                    if (number) {
                        value(static_cast<int>(i)) = PyFloat_AsDouble(number);
                    }
                    Py_XDECREF(number);
                }
                Py_XDECREF(item);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Vec4d src, rv_policy, cleanup_list*) noexcept {
            nb::tuple tuple = nb::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Mat4s> {
    public:
        NB_TYPE_CASTER(openvdb::Mat4s, const_name("openvdb::Mat4s"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t rows = PySequence_Length(source);
            if (rows != 4)
                return false;

            for (Py_ssize_t i = 0; i < rows; ++i) {
                PyObject* row = PySequence_GetItem(source, i);
                if (row) {
                    if (!PySequence_Check(row)) {
                        Py_DECREF(row);
                        return false;
                    }

                    Py_ssize_t cols = PySequence_Length(row);
                    if (cols != 4) {
                        Py_DECREF(row);
                        return false;
                    }

                    for (Py_ssize_t j = 0; j < cols; ++j) {
                        PyObject* item = PySequence_GetItem(row, j);
                        if (item) {
                            PyObject* number = PyNumber_Float(item);
                            if (number) {
                                value(static_cast<int>(i), static_cast<int>(j)) = static_cast<openvdb::Mat4s::value_type>(PyFloat_AsDouble(number));
                            }
                            Py_XDECREF(number);
                        }
                        Py_XDECREF(item);
                    }
                }
                Py_XDECREF(row);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Mat4s src, rv_policy, cleanup_list*) noexcept {
            nb::list list;
            for (int i = 0; i < 4; ++i) {
                nb::list sublist;
                for (int j = 0; j < 4; ++j) {
                    sublist.append(src(i, j));
                }
                list.append(sublist);
            }
            return list.release();
        }
    };

    template <> struct type_caster<openvdb::Mat4d> {
    public:
        NB_TYPE_CASTER(openvdb::Mat4d, const_name("openvdb::Mat4d"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t rows = PySequence_Length(source);
            if (rows != 4)
                return false;

            for (Py_ssize_t i = 0; i < rows; ++i) {
                PyObject* row = PySequence_GetItem(source, i);
                if (row) {
                    if (!PySequence_Check(row)) {
                        Py_DECREF(row);
                        return false;
                    }

                    Py_ssize_t cols = PySequence_Length(row);
                    if (cols != 4) {
                        Py_DECREF(row);
                        return false;
                    }

                    for (Py_ssize_t j = 0; j < cols; ++j) {
                        PyObject* item = PySequence_GetItem(row, j);
                        if (item) {
                            PyObject* number = PyNumber_Float(item);
                            if (number) {
                                value(static_cast<int>(i), static_cast<int>(j)) = PyFloat_AsDouble(number);
                            }
                            Py_XDECREF(number);
                        }
                        Py_XDECREF(item);
                    }
                }
                Py_XDECREF(row);

                if (PyErr_Occurred())
                    return false;
            }
            return true;
        }

        static handle from_cpp(openvdb::Mat4d src, rv_policy, cleanup_list*) noexcept {
            nb::list list;
            for (int i = 0; i < 4; ++i) {
                nb::list sublist;
                for (int j = 0; j < 4; ++j) {
                    sublist.append(src(i, j));
                }
                list.append(sublist);
            }
            return list.release();
        }
    };

    template <> struct type_caster<openvdb::PointIndex32> {
    public:
        NB_TYPE_CASTER(openvdb::PointIndex32, const_name("openvdb::PointIndex32"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            PyObject* number = PyNumber_Long(source);
            if (number) {
                value = static_cast<typename openvdb::PointIndex32::IntType>(PyLong_AsLong(number));
            }
            Py_XDECREF(number);

            if (PyErr_Occurred())
                return false;

            return true;
        }

        static handle from_cpp(openvdb::PointIndex32 src, rv_policy, cleanup_list*) noexcept {
            nb::int_ integer(static_cast<openvdb::PointIndex32::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointIndex64> {
    public:
        NB_TYPE_CASTER(openvdb::PointIndex64, const_name("openvdb::PointIndex64"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            PyObject* number = PyNumber_Long(source);
            if (number) {
                value = static_cast<typename openvdb::PointIndex64::IntType>(PyLong_AsLong(number));
            }
            Py_XDECREF(number);

            if (PyErr_Occurred())
                return false;

            return true;
        }

        static handle from_cpp(openvdb::PointIndex64 src, rv_policy, cleanup_list*) noexcept {
            nb::int_ integer(static_cast<openvdb::PointIndex64::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointDataIndex32> {
    public:
        NB_TYPE_CASTER(openvdb::PointDataIndex32, const_name("openvdb::PointDataIndex32"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            PyObject* number = PyNumber_Long(source);
            if (number) {
                value = static_cast<typename openvdb::PointDataIndex32::IntType>(PyLong_AsLong(number));
            }
            Py_XDECREF(number);

            if (PyErr_Occurred())
                return false;

            return true;
        }

        static handle from_cpp(openvdb::PointDataIndex32 src, rv_policy, cleanup_list*) noexcept {
            nb::int_ integer(static_cast<openvdb::PointDataIndex32::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointDataIndex64> {
    public:
        NB_TYPE_CASTER(openvdb::PointDataIndex64, const_name("openvdb::PointDataIndex64"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            PyObject* source = src.ptr();
            PyObject* number = PyNumber_Long(source);
            if (number) {
                value = static_cast<typename openvdb::PointDataIndex64::IntType>(PyLong_AsLong(number));
            }
            Py_XDECREF(number);

            if (PyErr_Occurred())
                return false;

            return true;
        }

        static handle from_cpp(openvdb::PointDataIndex64 src, rv_policy, cleanup_list*) noexcept {
            nb::int_ integer(static_cast<openvdb::PointDataIndex64::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::StringMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::StringMetadata, const_name("openvdb::StringMetadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::str>(src)) {
                return false;
            }

            value.setValue(nb::cast<std::string>(src));
            return true;
        }

        static handle from_cpp(openvdb::StringMetadata src, rv_policy, cleanup_list*) noexcept {
            return nb::str(src.value().c_str()).release();
        }
    };

    template <> struct type_caster<openvdb::BoolMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::BoolMetadata, const_name("openvdb::BoolMetadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::bool_>(src)) {
                return false;
            }

            value.setValue(nb::cast<bool>(src));
            return true;
        }

        static handle from_cpp(openvdb::BoolMetadata src, rv_policy, cleanup_list*) noexcept {
            return nb::bool_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Int32Metadata> {
    public:
        NB_TYPE_CASTER(openvdb::Int32Metadata, const_name("openvdb::Int32Metadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::int_>(src)) {
                return false;
            }

            value.setValue(nb::cast<openvdb::Int32>(src));
            return true;
        }

        static handle from_cpp(openvdb::Int32Metadata src, rv_policy, cleanup_list*) noexcept {
            return nb::int_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Int64Metadata> {
    public:
        NB_TYPE_CASTER(openvdb::Int64Metadata, const_name("openvdb::Int64Metadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::int_>(src)) {
                return false;
            }

            value.setValue(nb::cast<openvdb::Int64>(src));
            return true;
        }

        static handle from_cpp(openvdb::Int64Metadata src, rv_policy, cleanup_list*) noexcept {
            return nb::int_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::FloatMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::FloatMetadata, const_name("openvdb::FloatMetadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::float_>(src)) {
                return false;
            }

            value.setValue(nb::cast<float>(src));
            return true;
        }

        static handle from_cpp(openvdb::FloatMetadata src, rv_policy, cleanup_list*) noexcept {
            return nb::float_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::DoubleMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::DoubleMetadata, const_name("openvdb::DoubleMetadata"))

        bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
            if (!nb::isinstance<nb::float_>(src)) {
                return false;
            }

            value.setValue(nb::cast<double>(src));
            return true;
        }

        static handle from_cpp(openvdb::DoubleMetadata src, rv_policy, cleanup_list*) noexcept {
            return nb::float_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Vec2IMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2IMetadata, const_name("openvdb::Vec2IMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec2i> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec2i>());
            return true;
        }

        static handle from_cpp(openvdb::Vec2IMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec2i>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Vec3IMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3IMetadata, const_name("openvdb::Vec3IMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec3i> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec3i>());
            return true;
        }

        static handle from_cpp(openvdb::Vec3IMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec3i>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Vec4IMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4IMetadata, const_name("openvdb::Vec4IMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec4i> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec4i>());
            return true;
        }

        static handle from_cpp(openvdb::Vec4IMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec4i>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Vec2DMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec2DMetadata, const_name("openvdb::Vec2DMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec2d> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec2d>());
            return true;
        }

        static handle from_cpp(openvdb::Vec2DMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec2d>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Vec3DMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec3DMetadata, const_name("openvdb::Vec3DMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec3d> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec3d>());
            return true;
        }

        static handle from_cpp(openvdb::Vec3DMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec3d>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Vec4DMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Vec4DMetadata, const_name("openvdb::Vec4DMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec4d> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Vec4d>());
            return true;
        }

        static handle from_cpp(openvdb::Vec4DMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Vec4d>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Mat4SMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Mat4SMetadata, const_name("openvdb::Mat4SMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::list>(src)) {
                return false;
            }

            make_caster<openvdb::Mat4s> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Mat4s>());
            return true;
        }

        static handle from_cpp(openvdb::Mat4SMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Mat4s>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::Mat4DMetadata> {
    public:
        NB_TYPE_CASTER(openvdb::Mat4DMetadata, const_name("openvdb::Mat4DMetadata"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            if (!nb::isinstance<nb::list>(src)) {
                return false;
            }

            make_caster<openvdb::Mat4d> caster;
            if (!caster.from_python(src, flags, cleanup))
                return false;

            value.setValue(caster.operator cast_t<openvdb::Mat4d>());
            return true;
        }

        static handle from_cpp(openvdb::Mat4DMetadata src, rv_policy policy, cleanup_list* cleanup) {
            return make_caster<openvdb::Mat4d>::from_cpp(src.value(), policy, cleanup);
        }
    };

    template <> struct type_caster<openvdb::MetaMap> {
    public:
        NB_TYPE_CASTER(openvdb::MetaMap, const_name("openvdb::MetaMap"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
            nb::dict dictionary = nb::borrow<nb::dict>(src);
            for (auto [key, val] : dictionary) {
                std::string name;
                if (nb::isinstance<nb::str>(key)) {
                    name = nb::cast<std::string>(key);
                } else {
                    throw nb::type_error("Expected string as metadata name");
                }

                // Note: the order of the following tests is significant, as it
                // avoids unnecessary type promotion (e.g., of ints to doubles).
                // Python does not natively support single precision and therefore
                // all floating point metadata is promoted to doubles.

                bool success = false;
                if (nb::isinstance<nb::str>(val)) {
                    make_caster<openvdb::StringMetadata> caster;
                    if(caster.from_python(val, flags, cleanup)) {
                        value.insertMeta(name, caster.operator cast_t<openvdb::StringMetadata>());
                        success = true;
                    }
                } else if (nb::isinstance<nb::bool_>(val)) {
                    make_caster<openvdb::BoolMetadata> caster;
                    if(caster.from_python(val, flags, cleanup)) {
                        value.insertMeta(name, caster.operator cast_t<openvdb::BoolMetadata>());
                        success = true;
                    }
                } else if (nb::isinstance<nb::int_>(val)) {
                    const openvdb::Int64 n = nb::cast<openvdb::Int64>(val);
                    if (n <= std::numeric_limits<openvdb::Int32>::max() && n >= std::numeric_limits<openvdb::Int32>::min()) {
                        make_caster<openvdb::Int32Metadata> caster;
                        if(caster.from_python(val, flags, cleanup)) {
                            value.insertMeta(name, caster.operator cast_t<openvdb::Int32Metadata>());
                            success = true;
                        }
                    } else {
                        make_caster<openvdb::Int64Metadata> caster;
                        if(caster.from_python(val, flags, cleanup)) {
                            value.insertMeta(name, caster.operator cast_t<openvdb::Int64Metadata>());
                            success = true;
                        }
                    }
                } else if (nb::isinstance<nb::float_>(val)) {
                    make_caster<openvdb::DoubleMetadata> caster;
                    if(caster.from_python(val, flags, cleanup)) {
                        value.insertMeta(name, caster.operator cast_t<openvdb::DoubleMetadata>());
                        success = true;
                    }
                } else if (nb::isinstance<nb::tuple>(val)) {
                    nb::tuple t = nb::cast<nb::tuple>(val);
                    size_t size = t.size();
                    bool isIntegerTuple = true;
                    for (size_t i = 0; i < size; ++i) {
                        isIntegerTuple &= nb::isinstance<nb::int_>(t[i]);
                    }

                    if (isIntegerTuple) {
                        switch(size) {
                            case 2: {
                                make_caster<openvdb::Vec2IMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec2IMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            case 3: {
                                make_caster<openvdb::Vec3IMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec3IMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            case 4: {
                                make_caster<openvdb::Vec4IMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec4IMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            default:
                                break;
                        }
                    } else {
                        switch(size) {
                            case 2: {
                                make_caster<openvdb::Vec2DMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec2DMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            case 3: {
                                make_caster<openvdb::Vec3DMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec3DMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            case 4: {
                                make_caster<openvdb::Vec4DMetadata> caster;
                                if (caster.from_python(val, flags, cleanup)) {
                                    value.insertMeta(name, caster.operator cast_t<openvdb::Vec4DMetadata>());
                                    success = true;
                                }
                                break;
                                }
                            default:
                                break;
                        }
                    }
                } else if (nb::isinstance<nb::list>(val)) {
                    make_caster<openvdb::Mat4DMetadata> caster;
                    if (caster.from_python(val, flags, cleanup)) {
                        value.insertMeta(name, caster.operator cast_t<openvdb::Mat4DMetadata>());
                        success = true;
                    }
                } else if (nb::isinstance<openvdb::Metadata::Ptr>(val)) {
                    openvdb::Metadata::Ptr metadata = nb::cast<openvdb::Metadata::Ptr>(val);
                    if (metadata) {
                        value.insertMeta(name, *metadata);
                    }

                }

                if (!success) {
                    const std::string valAsStr = nb::cast<std::string>(val.attr("__str__")());
                    const std::string valType = nb::cast<std::string>(val.attr("__class__").attr("__name__"));
                    throw nb::type_error(std::string("metadata value " + valAsStr + " of type " + valType + " is not allowed").c_str());
                }
            }

            return true;
        }

        static handle from_cpp(openvdb::MetaMap src, rv_policy policy, cleanup_list* cleanup) {
            nb::dict dict;
            for (openvdb::MetaMap::ConstMetaIterator it = src.beginMeta(); it != src.endMeta(); ++it) {
                if (openvdb::Metadata::Ptr meta = it->second) {
                    nb::object obj(nb::cast(meta));
                    const std::string typeName = meta->typeName();
                    if (typeName == openvdb::StringMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::StringMetadata>::from_cpp(static_cast<openvdb::StringMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::DoubleMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::DoubleMetadata>::from_cpp(static_cast<openvdb::DoubleMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::FloatMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::FloatMetadata>::from_cpp(static_cast<openvdb::FloatMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Int32Metadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Int32Metadata>::from_cpp(static_cast<openvdb::Int32Metadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Int64Metadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Int64Metadata>::from_cpp(static_cast<openvdb::Int64Metadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::BoolMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::BoolMetadata>::from_cpp(static_cast<openvdb::BoolMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec2DMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec2DMetadata>::from_cpp(static_cast<openvdb::Vec2DMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec2IMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec2IMetadata>::from_cpp(static_cast<openvdb::Vec2IMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec2SMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec2SMetadata>::from_cpp(static_cast<openvdb::Vec2SMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec3DMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec3DMetadata>::from_cpp(static_cast<openvdb::Vec3DMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec3IMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec3IMetadata>::from_cpp(static_cast<openvdb::Vec3IMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec3SMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec3SMetadata>::from_cpp(static_cast<openvdb::Vec3SMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec4DMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec4DMetadata>::from_cpp(static_cast<openvdb::Vec4DMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec4IMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec4IMetadata>::from_cpp(static_cast<openvdb::Vec4IMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Vec4SMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Vec4SMetadata>::from_cpp(static_cast<openvdb::Vec4SMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Mat4SMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Mat4SMetadata>::from_cpp(static_cast<openvdb::Mat4SMetadata&>(*meta), policy, cleanup));
                    } else if (typeName == openvdb::Mat4DMetadata::staticTypeName()) {
                        obj = steal<nb::object>(make_caster<openvdb::Mat4DMetadata>::from_cpp(static_cast<openvdb::Mat4DMetadata&>(*meta), policy, cleanup));
                    }
                    dict[nb::str(it->first.c_str())] = obj;
                }
            }
            return dict.release();
        }
    };
}}

#endif // OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED
