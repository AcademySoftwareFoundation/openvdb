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
#if PY_MAJOR_VERSION >= 3
                value = static_cast<typename openvdb::PointIndex32::IntType>(PyLong_AsLong(number));
#else
                value = static_cast<typename openvdb::PointIndex32::IntType>(PyInt_AsLong(number));
#endif
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
#if PY_MAJOR_VERSION >= 3
                value = static_cast<typename openvdb::PointIndex64::IntType>(PyLong_AsLong(number));
#else
                value = static_cast<typename openvdb::PointIndex64::IntType>(PyInt_AsLong(number));
#endif
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

    template <> struct type_caster<openvdb::MetaMap> {
    public:
        PYBIND11_TYPE_CASTER(openvdb::MetaMap, _("openvdb::Metamap"));

        bool load(handle src, bool) {
            py::dict dictionary = py::reinterpret_borrow<py::dict>(src);
            for (auto item : dictionary) {
                std::string name;
                py::object key = py::reinterpret_borrow<py::object>(item.first);
                if (py::isinstance<py::str>(key)) {
                    name = key.cast<std::string>();
                } else {
                    throw py::type_error("Expected string as metadata name");
                }

                openvdb::Metadata::Ptr metadata;
                // Note: the order of the following tests is significant, as it
                // avoids unnecessary type promotion (e.g., of ints to doubles).
                // Python does not natively support single precision and therefore
                // all floating point metadata is promoted to doubles.
                py::object val = py::reinterpret_borrow<py::object>(item.second);

                if (py::isinstance<py::str>(val)) {
                    metadata.reset(new openvdb::StringMetadata(val.cast<std::string>()));
                } else if (py::isinstance<py::bool_>(val)) {
                    metadata.reset(new openvdb::BoolMetadata(val.cast<bool>()));
                } else if (py::isinstance<py::int_>(val)) {
                    const openvdb::Int64 n = val.cast<openvdb::Int64>();
                    if (n <= std::numeric_limits<openvdb::Int32>::max()
                        && n >= std::numeric_limits<openvdb::Int32>::min()) {
                        metadata.reset(new openvdb::Int32Metadata(static_cast<openvdb::Int32>(n)));
                    } else {
                        metadata.reset(new openvdb::Int64Metadata(n));
                    }
                } else if (py::isinstance<py::float_>(val)) {
                    metadata.reset(new openvdb::DoubleMetadata(val.cast<double>()));
                } else if (py::isinstance<py::tuple>(val)) {
                    py::tuple t = val.cast<py::tuple>();
                    size_t size = t.size();
                    bool isIntegerTuple = true;
                    for (size_t i = 0; i < size; ++i) {
                        isIntegerTuple &= py::isinstance<py::int_>(t[i]);
                    }

                    if (isIntegerTuple) {
                        switch(size) {
                            case 2:
                                metadata.reset(new openvdb::Vec2IMetadata(t.cast<openvdb::Vec2i>()));
                                break;
                            case 3:
                                metadata.reset(new openvdb::Vec3IMetadata(t.cast<openvdb::Vec3i>()));
                                break;
                            case 4:
                                metadata.reset(new openvdb::Vec4IMetadata(t.cast<openvdb::Vec4i>()));
                                break;
                            default:
                                break;
                        }
                    } else {
                        switch(size) {
                            case 2:
                                metadata.reset(new openvdb::Vec2DMetadata(t.cast<openvdb::Vec2d>()));
                                break;
                            case 3:
                                metadata.reset(new openvdb::Vec3DMetadata(t.cast<openvdb::Vec3d>()));
                                break;
                            case 4:
                                metadata.reset(new openvdb::Vec4DMetadata(t.cast<openvdb::Vec4d>()));
                                break;
                            default:
                                break;
                        }
                    }
                } else if (py::isinstance<py::list>(val)) {
                    py::list list = val.cast<py::list>();
                    bool valid = (list.size() == 4);
                    if (valid) {
                        for (size_t i = 0; i < list.size(); ++i) {
                            valid &= py::isinstance<py::list>(list[i]);
                            if (valid) {
                                py::list sublist = list[i].cast<py::list>();
                                valid &= (sublist.size() == 4);
                                if (valid) {
                                    for (size_t j = 0; j < sublist.size(); ++j) {
                                        valid &= (py::isinstance<py::float_>(sublist[j]) || py::isinstance<py::int_>(sublist[j]));
                                    }
                                }
                            }
                        }
                    }
                    if (valid) {
                        metadata.reset(new openvdb::Mat4DMetadata(val.cast<openvdb::Mat4d>()));
                    }
                } else if (py::isinstance<openvdb::Metadata::Ptr>(val)) {
                    metadata = val.cast<openvdb::Metadata::Ptr>();
                }

                if (metadata) {
                    value.insertMeta(name, *metadata);
                } else {
                    const std::string valAsStr = (val.attr("__str__")()).cast<std::string>();
                    const std::string valType = val.attr("__class__").attr("__name__").cast<std::string>();
                    throw py::type_error(std::string("metadata value " + valAsStr + " of type " + valType + " is not allowed"));
                }

            }

            return true;
        }

        static handle cast(openvdb::MetaMap src, return_value_policy, handle) {
            py::dict dict;
            for (openvdb::MetaMap::ConstMetaIterator it = src.beginMeta(); it != src.endMeta(); ++it) {
                if (openvdb::Metadata::Ptr meta = it->second) {
                    py::object obj(py::cast(meta));
                    const std::string typeName = meta->typeName();
                    if (typeName == openvdb::StringMetadata::staticTypeName()) {
                        obj = py::str(static_cast<openvdb::StringMetadata&>(*meta).value());
                    } else if (typeName == openvdb::DoubleMetadata::staticTypeName()) {
                        obj = py::float_(static_cast<openvdb::DoubleMetadata&>(*meta).value());
                    } else if (typeName == openvdb::FloatMetadata::staticTypeName()) {
                        obj = py::float_(static_cast<openvdb::FloatMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Int32Metadata::staticTypeName()) {
                        obj = py::int_(static_cast<openvdb::Int32Metadata&>(*meta).value());
                    } else if (typeName == openvdb::Int64Metadata::staticTypeName()) {
                        obj = py::int_(static_cast<openvdb::Int64Metadata&>(*meta).value());
                    } else if (typeName == openvdb::BoolMetadata::staticTypeName()) {
                        obj = py::bool_(static_cast<openvdb::BoolMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec2DMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec2DMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec2IMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec2IMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec2SMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec2SMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec3DMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec3DMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec3IMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec3IMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec3SMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec3SMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec4DMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec4DMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec4IMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec4IMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Vec4SMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Vec4SMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Mat4SMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Mat4SMetadata&>(*meta).value());
                    } else if (typeName == openvdb::Mat4DMetadata::staticTypeName()) {
                        obj = py::cast(static_cast<openvdb::Mat4DMetadata&>(*meta).value());
                    }
                    dict[py::str(it->first)] = obj;
                }
            }
            return dict.release();
        }
    };
}}

#endif // OPENVDB_PYTYPECASTERS_HAS_BEEN_INCLUDED
