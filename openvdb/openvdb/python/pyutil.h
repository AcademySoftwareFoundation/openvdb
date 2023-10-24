// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_PYUTIL_HAS_BEEN_INCLUDED
#define OPENVDB_PYUTIL_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include <sstream>
#include <string>
#include <utility> // for std::pair

namespace py = pybind11;

namespace pyutil {

template<class GridType>
struct GridTraitsBase
{
    /// @brief Return the name of the Python class that wraps this grid type
    /// (e.g., "FloatGrid" for openvdb::FloatGrid).
    ///
    /// @note This name is not the same as GridType::type().
    /// The latter returns a name like "Tree_float_5_4_3".
    static const char* name();

    /// Return the name of this grid type's value type ("bool", "float", "vec3s", etc.).
    static const char* valueTypeName()
    {
        return openvdb::typeNameAsString<typename GridType::ValueType>();
    }

    /// @brief Return a description of this grid type.
    ///
    /// @note This name is generated at runtime for each call to descr().
    static const std::string descr()
    {
        return std::string("OpenVDB grid with voxels of type ") + valueTypeName();
    }
}; // struct GridTraitsBase


template<class GridType>
struct GridTraits: public GridTraitsBase<GridType>
{
};

/// Map a grid type to a traits class that derives from GridTraitsBase
/// and that defines a name() method.
#define GRID_TRAITS(_typ, _name) \
    template<> struct GridTraits<_typ>: public GridTraitsBase<_typ> { \
        static const char* name() { return _name; } \
    }

GRID_TRAITS(openvdb::FloatGrid, "FloatGrid");
GRID_TRAITS(openvdb::Vec3SGrid, "Vec3SGrid");
GRID_TRAITS(openvdb::BoolGrid, "BoolGrid");
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
GRID_TRAITS(openvdb::DoubleGrid, "DoubleGrid");
GRID_TRAITS(openvdb::Int32Grid, "Int32Grid");
GRID_TRAITS(openvdb::Int64Grid, "Int64Grid");
GRID_TRAITS(openvdb::Vec3IGrid, "Vec3IGrid");
GRID_TRAITS(openvdb::Vec3DGrid, "Vec3DGrid");
GRID_TRAITS(openvdb::points::PointDataGrid, "PointDataGrid");
#endif

#undef GRID_TRAITS


////////////////////////////////////////


// Note that the elements are pointers to C strings (char**), because
// py::class_::def_readonly() requires a pointer to a static member.
typedef std::pair<const char* const*, const char* const*> CStringPair;


/// @brief Enum-like mapping from string keys to string values, with characteristics
/// of both (Python) classes and class instances (as well as NamedTuples)
/// @details
/// - (@e key, @e value) pairs can be accessed as class attributes (\"<tt>MyClass.MY_KEY</tt>\")
/// - (@e key, @e value) pairs can be accessed via dict lookup on instances
///   (\"<tt>MyClass()['MY_KEY']</tt>\")
/// - (@e key, @e value) pairs can't be modified or reassigned
/// - instances are iterable (\"<tt>for key in MyClass(): ...</tt>\")
///
/// A @c Descr class must implement the following interface:
/// @code
/// struct MyDescr
/// {
///     // Return the Python name for the enum class.
///     static const char* name();
///     // Return the docstring for the enum class.
///     static const char* doc();
///     // Return the ith (key, value) pair, in the form of
///     // a pair of *pointers* to C strings
///     static CStringPair item(int i);
/// };
/// @endcode
template<typename Descr>
struct StringEnum
{
    /// Return the (key, value) map as a Python dict.
    static py::dict items()
    {
        static std::mutex sMutex;
        static py::dict itemDict;
        if (!itemDict) {
            // The first time this function is called, populate
            // the static dict with (key, value) pairs.
            std::lock_guard<std::mutex> lock(sMutex);
            if (!itemDict) {
                for (int i = 0; ; ++i) {
                    const CStringPair item = Descr::item(i);
                    OPENVDB_START_THREADSAFE_STATIC_WRITE
                    if (item.first) {
                        itemDict[py::str(*item.first)] =
                            py::str(*item.second);
                    }
                    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
                    else break;
                }
            }
        }
        return itemDict;
    }

    /// Return the keys as a Python list of strings.
    static py::object keys() { return items().attr("keys")(); }
    /// Return the number of keys as a Python int.
    py::int_ numItems() const
    {
        return py::int_(py::len(items()));
    }
    /// Return the value (as a Python string) for the given key.
    py::object getItem(py::object keyObj) const { return items()[keyObj]; }
    /// Return a Python iterator over the keys.
    py::object iter() const { return items().attr("__iter__")(); }

    /// Register this enum.
    static void wrap(py::module_ m)
    {
        py::class_<StringEnum> cls(
            m,
            /*classname=*/Descr::name(),
            /*docstring=*/Descr::doc());
        cls.def_static("keys", &StringEnum::keys, "keys() -> list")
            .def("__len__", &StringEnum::numItems, "__len__() -> int")
            .def("__iter__", &StringEnum::iter, "__iter__() -> iterator")
            .def("__getitem__", &StringEnum::getItem, "__getitem__(str) -> str")
            /*end*/;
        // Add a read-only, class-level attribute for each (key, value) pair.
        for (int i = 0; ; ++i) {
            const CStringPair item = Descr::item(i);
            if (item.first) cls.def_readonly_static(*item.first, item.second);
            else break;
        }
    }
};


////////////////////////////////////////

/// Return the name of the given Python object's class.
inline std::string
className(py::handle h)
{
    return py::cast<std::string>(h.attr("__class__").attr("__name__"));
}

} // namespace pyutil

#endif // OPENVDB_PYUTIL_HAS_BEEN_INCLUDED
