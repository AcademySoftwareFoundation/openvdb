#ifndef OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED
#define OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED

#include <pybind11/pybind11.h>
#include <openvdb/openvdb.h>
#include <openvdb/MetaMap.h>

// Custom type casters must be defined consistently in EVERY translation unit of
// pybind11 code. Thus, we define them in this header file and include it in every
// implementation file.

namespace py = pybind11;

namespace pybind11 { namespace detail {
    template <> struct type_caster<openvdb::Coord> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Coord, const_name("openvdb::Coord"));

        bool load(handle src, bool) {
            PyObject* source = src.ptr();
            if (!PySequence_Check(source))
                return false;

            Py_ssize_t length = PySequence_Length(source);
            if (length == 1) {
                openvdb::Coord::Int32 items[1];
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
                openvdb::Coord::Int32 items[3];
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

        static handle cast(openvdb::Coord src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2i> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2i, const_name("openvdb::Vec2i"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec2i src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2I> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2I, const_name("openvdb::Vec2I"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec2I src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2s> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2s, const_name("openvdb::Vec2s"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec2s src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec2d> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2d, const_name("openvdb::Vec2d"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec2d src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3i> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3i, const_name("openvdb::Vec3i"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec3i src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3I> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3I, const_name("openvdb::Vec3I"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec3I src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3s> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3s, const_name("openvdb::Vec3s"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec3s src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec3d> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3d, const_name("openvdb::Vec3d"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec3d src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4i> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4i, const_name("openvdb::Vec4i"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec4i src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4I> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4I, const_name("openvdb::Vec4I"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec4I src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4s> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4s, const_name("openvdb::Vec4s"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec4s src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Vec4d> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4d, const_name("openvdb::Vec4d"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Vec4d src, return_value_policy, handle) {
            py::tuple tuple = py::make_tuple(src[0], src[1], src[2], src[3]);
            return tuple.release();
        }
    };

    template <> struct type_caster<openvdb::Mat4s> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Mat4s, const_name("openvdb::Mat4s"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Mat4s src, return_value_policy, handle) {
            py::list list;
            for (int i = 0; i < 4; ++i) {
                py::list sublist;
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
        PYBIND11_TYPE_CASTER(openvdb::Mat4d, const_name("openvdb::Mat4d"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::Mat4d src, return_value_policy, handle) {
            py::list list;
            for (int i = 0; i < 4; ++i) {
                py::list sublist;
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
        PYBIND11_TYPE_CASTER(openvdb::PointIndex32, const_name("openvdb::PointIndex32"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::PointIndex32 src, return_value_policy, handle) {
            py::int_ integer(static_cast<openvdb::PointIndex32::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointIndex64> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::PointIndex64, const_name("openvdb::PointIndex64"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::PointIndex64 src, return_value_policy, handle) {
            py::int_ integer(static_cast<openvdb::PointIndex64::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointDataIndex32> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::PointDataIndex32, const_name("openvdb::PointDataIndex32"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::PointDataIndex32 src, return_value_policy, handle) {
            py::int_ integer(static_cast<openvdb::PointDataIndex32::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::PointDataIndex64> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::PointDataIndex64, const_name("openvdb::PointDataIndex64"));

        bool load(handle src, bool) {
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

        static handle cast(openvdb::PointDataIndex64 src, return_value_policy, handle) {
            py::int_ integer(static_cast<openvdb::PointDataIndex64::IntType>(src));
            return integer.release();
        }
    };

    template <> struct type_caster<openvdb::StringMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::StringMetadata, _("openvdb::StringMetadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::str>(src)) {
                return false;
            }

            value.setValue(py::cast<std::string>(src));
            return true;
        }

        static handle cast(openvdb::StringMetadata src, return_value_policy, handle) {
            return py::str(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::BoolMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::BoolMetadata, _("openvdb::BoolMetadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::bool_>(src)) {
                return false;
            }

            value.setValue(py::cast<bool>(src));
            return true;
        }

        static handle cast(openvdb::BoolMetadata src, return_value_policy, handle) {
            return py::bool_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Int32Metadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Int32Metadata, _("openvdb::Int32Metadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::int_>(src)) {
                return false;
            }

            value.setValue(py::cast<openvdb::Int32>(src));
            return true;
        }

        static handle cast(openvdb::Int32Metadata src, return_value_policy, handle) {
            return py::int_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Int64Metadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Int64Metadata, _("openvdb::Int64Metadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::int_>(src)) {
                return false;
            }

            value.setValue(py::cast<openvdb::Int64>(src));
            return true;
        }

        static handle cast(openvdb::Int64Metadata src, return_value_policy, handle) {
            return py::int_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::FloatMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::FloatMetadata, _("openvdb::FloatMetadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::float_>(src)) {
                return false;
            }

            value.setValue(py::cast<float>(src));
            return true;
        }

        static handle cast(openvdb::FloatMetadata src, return_value_policy, handle) {
            return py::float_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::DoubleMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::DoubleMetadata, _("openvdb::DoubleMetadata"));

        bool load(handle src, bool) {
            if (!py::isinstance<py::float_>(src)) {
                return false;
            }

            value.setValue(py::cast<double>(src));
            return true;
        }

        static handle cast(openvdb::DoubleMetadata src, return_value_policy, handle) {
            return py::float_(src.value()).release();
        }
    };

    template <> struct type_caster<openvdb::Vec2IMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2IMetadata, _("openvdb::Vec2IMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec2i> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec2i&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec2IMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec2i>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Vec3IMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3IMetadata, _("openvdb::Vec3IMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec3i> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec3i&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec3IMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec3i>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Vec4IMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4IMetadata, _("openvdb::Vec4IMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec4i> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec4i&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec4IMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec4i>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Vec2DMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec2DMetadata, _("openvdb::Vec2DMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec2d> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec2d&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec2DMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec2d>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Vec3DMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec3DMetadata, _("openvdb::Vec3DMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec3d> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec3d&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec3DMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec3d>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Vec4DMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Vec4DMetadata, _("openvdb::Vec4DMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) {
                return false;
            }

            make_caster<openvdb::Vec4d> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Vec4d&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Vec4DMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Vec4d>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Mat4SMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Mat4SMetadata, _("openvdb::Mat4SMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::list>(src)) {
                return false;
            }

            make_caster<openvdb::Mat4s> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Mat4s&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Mat4SMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Mat4s>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::Mat4DMetadata> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::Mat4DMetadata, _("openvdb::Mat4DMetadata"));

        bool load(handle src, bool convert) {
            if (!py::isinstance<py::list>(src)) {
                return false;
            }

            make_caster<openvdb::Mat4d> conv;
            if (!conv.load(src, convert))
                return false;

            value.setValue(cast_op<openvdb::Mat4d&&>(std::move(conv)));
            return true;
        }

        static handle cast(openvdb::Mat4DMetadata src, return_value_policy policy, handle parent) {
            return make_caster<openvdb::Mat4d>::cast(src.value(), policy, parent);
        }
    };

    template <> struct type_caster<openvdb::MetaMap> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::MetaMap, _("openvdb::MetaMap"));

        bool load(handle src, bool convert) {
            py::dict dictionary = py::reinterpret_borrow<py::dict>(src);
            for (auto [key, val] : dictionary) {
                std::string name;
                if (py::isinstance<py::str>(key)) {
                    name = py::cast<std::string>(key);
                } else {
                    throw py::type_error("Expected string as metadata name");
                }

                // Note: the order of the following tests is significant, as it
                // avoids unnecessary type promotion (e.g., of ints to doubles).
                // Python does not natively support single precision and therefore
                // all floating point metadata is promoted to doubles.

                bool success = false;
                if (py::isinstance<py::str>(val)) {
                    make_caster<openvdb::StringMetadata> conv;
                    if(conv.load(val, convert)) {
                        value.insertMeta(name, cast_op<openvdb::StringMetadata&&>(std::move(conv)));
                        success = true;
                    }
                } else if (py::isinstance<py::bool_>(val)) {
                    make_caster<openvdb::BoolMetadata> conv;
                    if(conv.load(val, convert)) {
                        value.insertMeta(name, cast_op<openvdb::BoolMetadata&&>(std::move(conv)));
                        success = true;
                    }
                } else if (py::isinstance<py::int_>(val)) {
                    const openvdb::Int64 n = py::cast<openvdb::Int64>(val);
                    if (n <= std::numeric_limits<openvdb::Int32>::max() && n >= std::numeric_limits<openvdb::Int32>::min()) {
                        make_caster<openvdb::Int32Metadata> conv;
                        if(conv.load(val, convert)) {
                            value.insertMeta(name, cast_op<openvdb::Int32Metadata&&>(std::move(conv)));
                            success = true;
                        }
                    } else {
                        make_caster<openvdb::Int64Metadata> conv;
                        if(conv.load(val, convert)) {
                            value.insertMeta(name, cast_op<openvdb::Int64Metadata&&>(std::move(conv)));
                            success = true;
                        }
                    }
                } else if (py::isinstance<py::float_>(val)) {
                    make_caster<openvdb::DoubleMetadata> conv;
                    if(conv.load(val, convert)) {
                        value.insertMeta(name, cast_op<openvdb::DoubleMetadata&&>(std::move(conv)));
                        success = true;
                    }
                } else if (py::isinstance<py::tuple>(val)) {
                    py::tuple t = py::cast<py::tuple>(val);
                    size_t size = t.size();
                    bool isIntegerTuple = true;
                    for (size_t i = 0; i < size; ++i) {
                        isIntegerTuple &= py::isinstance<py::int_>(t[i]);
                    }

                    if (isIntegerTuple) {
                        switch(size) {
                            case 2: {
                                make_caster<openvdb::Vec2IMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec2IMetadata&&>(std::move(conv)));
                                    success = true;
                                }
                                break;
                                }
                            case 3: {
                                make_caster<openvdb::Vec3IMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec3IMetadata&&>(std::move(conv)));
                                    success = true;
                                }
                                break;
                                }
                            case 4: {
                                make_caster<openvdb::Vec4IMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec4IMetadata&&>(std::move(conv)));
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
                                make_caster<openvdb::Vec2DMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec2DMetadata&&>(std::move(conv)));
                                    success = true;
                                }
                                break;
                                }
                            case 3: {
                                make_caster<openvdb::Vec3DMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec3DMetadata&&>(std::move(conv)));
                                    success = true;
                                }
                                break;
                                }
                            case 4: {
                                make_caster<openvdb::Vec4DMetadata> conv;
                                if (conv.load(val, convert)) {
                                    value.insertMeta(name, cast_op<openvdb::Vec4DMetadata&&>(std::move(conv)));
                                    success = true;
                                }
                                break;
                                }
                            default:
                                break;
                        }
                    }
                } else if (py::isinstance<py::list>(val)) {
                    make_caster<openvdb::Mat4DMetadata> conv;
                    if (conv.load(val, convert)) {
                        value.insertMeta(name, cast_op<openvdb::Mat4DMetadata&&>(std::move(conv)));
                        success = true;
                    }
                } else if (py::isinstance<openvdb::Metadata::Ptr>(val)) {
                    openvdb::Metadata::Ptr metadata = py::cast<openvdb::Metadata::Ptr>(val);
                    if (metadata) {
                        value.insertMeta(name, *metadata);
                    }

                }

                if (!success) {
                    const std::string valAsStr = py::cast<std::string>(val.attr("__str__")());
                    const std::string valType = py::cast<std::string>(val.attr("__class__").attr("__name__"));
                    throw py::type_error(std::string("metadata value " + valAsStr + " of type " + valType + " is not allowed"));
                }
            }

            return true;
        }

        static handle cast(openvdb::MetaMap src, return_value_policy policy, handle parent) {
            py::dict dict;
            for (openvdb::MetaMap::ConstMetaIterator it = src.beginMeta(); it != src.endMeta(); ++it) {
                if (openvdb::Metadata::Ptr meta = it->second) {
                    py::object obj(py::cast(meta));
                    const std::string typeName = meta->typeName();
                    if (typeName == openvdb::StringMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::StringMetadata>::cast(static_cast<openvdb::StringMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::DoubleMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::DoubleMetadata>::cast(static_cast<openvdb::DoubleMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::FloatMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::FloatMetadata>::cast(static_cast<openvdb::FloatMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Int32Metadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Int32Metadata>::cast(static_cast<openvdb::Int32Metadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Int64Metadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Int64Metadata>::cast(static_cast<openvdb::Int64Metadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::BoolMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::BoolMetadata>::cast(static_cast<openvdb::BoolMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec2DMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec2DMetadata>::cast(static_cast<openvdb::Vec2DMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec2IMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec2IMetadata>::cast(static_cast<openvdb::Vec2IMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec2SMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec2SMetadata>::cast(static_cast<openvdb::Vec2SMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec3DMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec3DMetadata>::cast(static_cast<openvdb::Vec3DMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec3IMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec3IMetadata>::cast(static_cast<openvdb::Vec3IMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec3SMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec3SMetadata>::cast(static_cast<openvdb::Vec3SMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec4DMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec4DMetadata>::cast(static_cast<openvdb::Vec4DMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec4IMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec4IMetadata>::cast(static_cast<openvdb::Vec4IMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Vec4SMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Vec4SMetadata>::cast(static_cast<openvdb::Vec4SMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Mat4SMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Mat4SMetadata>::cast(static_cast<openvdb::Mat4SMetadata&>(*meta), policy, parent));
                    } else if (typeName == openvdb::Mat4DMetadata::staticTypeName()) {
                        obj = reinterpret_steal<py::object>(make_caster<openvdb::Mat4DMetadata>::cast(static_cast<openvdb::Mat4DMetadata&>(*meta), policy, parent));
                    }
                    dict[py::str(it->first)] = obj;
                }
            }
            return dict.release();
        }
    };
}}

#endif // OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED
