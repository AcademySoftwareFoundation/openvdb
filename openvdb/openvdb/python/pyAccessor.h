// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED
#define OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <openvdb/openvdb.h>
#include "pyutil.h"

namespace pyAccessor {

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;


//@{
/// Type traits for grid accessors
template<typename _GridT>
struct AccessorTraits
{
    using GridT = _GridT;
    using NonConstGridT = GridT;
    using GridPtrT = typename NonConstGridT::Ptr;
    using AccessorT = typename NonConstGridT::Accessor;
    using ValueT = typename AccessorT::ValueType;

    static const bool IsConst = false;

    static const char* typeName() { return "Accessor"; }

    static void setActiveState(AccessorT& acc, const Coord& ijk, bool on) {
        acc.setActiveState(ijk, on);
    }
    static void setValueOnly(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOnly(ijk, val);
    }
    static void setValueOn(AccessorT& acc, const Coord& ijk) { acc.setValueOn(ijk); }
    static void setValueOn(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOn(ijk, val);
    }
    static void setValueOff(AccessorT& acc, const Coord& ijk) { acc.setValueOff(ijk); }
    static void setValueOff(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOff(ijk, val);
    }
};

// Partial specialization for const accessors
template<typename _GridT>
struct AccessorTraits<const _GridT>
{
    using GridT = const _GridT;
    using NonConstGridT = _GridT;
    using GridPtrT = typename NonConstGridT::ConstPtr;
    using AccessorT = typename NonConstGridT::ConstAccessor;
    using ValueT = typename AccessorT::ValueType;

    static const bool IsConst = true;

    static const char* typeName() { return "ConstAccessor"; }

    static void setActiveState(AccessorT&, const Coord&, bool) { notWritable(); }
    static void setValueOnly(AccessorT&, const Coord&, const ValueT&) { notWritable(); }
    static void setValueOn(AccessorT&, const Coord&) { notWritable(); }
    static void setValueOn(AccessorT&, const Coord&, const ValueT&) { notWritable(); }
    static void setValueOff(AccessorT&, const Coord&) { notWritable(); }
    static void setValueOff(AccessorT&, const Coord&, const ValueT&) { notWritable(); }

    static void notWritable()
    {
        throw nb::type_error("accessor is read-only");
    }
};
//@}


////////////////////////////////////////


/// @brief ValueAccessor wrapper class that also stores a grid pointer,
/// so that the grid doesn't get deleted as long as the accessor is live
///
/// @internal This class could have just been made to inherit from ValueAccessor,
/// but the method wrappers allow for more Pythonic error messages.  For example,
/// if we constructed the Python getValue() method directly from the corresponding
/// ValueAccessor method, as follows,
///
///    .def("getValue", &Accessor::getValue, ...)
///
/// then the conversion from a Python type to a Coord& would be done
/// automatically.  But if the Python method were called with an object of
/// a type that is not convertible to a Coord, then the TypeError message
/// would say something like "TypeError: No registered converter was able to
/// produce a C++ rvalue of type openvdb::math::Coord...".
/// Handling the type conversion manually is more work, but it allows us to
/// instead generate messages like "TypeError: expected tuple(int, int, int),
/// found str as argument to FloatGridAccessor.getValue()".
template<typename _GridType>
class AccessorWrap
{
public:
    using Traits = AccessorTraits<_GridType>;
    using Accessor = typename Traits::AccessorT;
    using ValueType = typename Traits::ValueT;
    using GridType = typename Traits::NonConstGridT;
    using GridPtrType = typename Traits::GridPtrT;

    AccessorWrap(GridPtrType grid): mGrid(grid), mAccessor(grid->getAccessor()) {}

    AccessorWrap copy() const { return *this; }

    void clear() { mAccessor.clear(); }

    GridPtrType parent() const { return mGrid; }

    ValueType getValue(const Coord& ijk)
    {
        return mAccessor.getValue(ijk);
    }

    int getValueDepth(const Coord& ijk)
    {
        return mAccessor.getValueDepth(ijk);
    }

    int isVoxel(const Coord& ijk)
    {
        return mAccessor.isVoxel(ijk);
    }

    std::tuple<ValueType, bool> probeValue(const Coord& ijk)
    {
        ValueType value;
        bool on = mAccessor.probeValue(ijk, value);
        return std::make_tuple(value, on);
    }

    bool isValueOn(const Coord& ijk)
    {
        return mAccessor.isValueOn(ijk);
    }

    void setActiveState(const Coord& ijk, bool on)
    {
        Traits::setActiveState(mAccessor, ijk, on);
    }

    void setValueOnly(const Coord& ijk, const ValueType& val)
    {
        Traits::setValueOnly(mAccessor, ijk, val);
    }

    void setValueOn(const Coord& ijk, const std::optional<ValueType>& val)
    {
        if (val)
            Traits::setValueOn(mAccessor, ijk, *val);
        else
            Traits::setValueOn(mAccessor, ijk);
    }

    void setValueOff(const Coord& ijk, const std::optional<ValueType>& val)
    {
        if (val)
            Traits::setValueOff(mAccessor, ijk, *val);
        else
            Traits::setValueOff(mAccessor, ijk);
    }

    int isCached(const Coord& ijk)
    {
        return mAccessor.isCached(ijk);
    }

    /// @brief Define a Python wrapper class for this C++ class.
    static void wrap(nb::module_ m)
    {
        const std::string
            pyGridTypeName = pyutil::GridTraits<GridType>::name(),
            pyAccessorTypeName = Traits::typeName();

        nb::class_<AccessorWrap>(m,
            (pyGridTypeName + pyAccessorTypeName).c_str(), //nanobind requires a unique class name for each template instantiation
            (std::string(Traits::IsConst ? "Read-only" : "Read/write")
                + " access by (i, j, k) index coordinates to the voxels\nof a "
                + pyGridTypeName).c_str())
            .def("copy", &AccessorWrap::copy,
                 "Return a copy of this accessor.")

            .def("clear", &AccessorWrap::clear,
                "Clear this accessor of all cached data.")

            .def_prop_ro("parent", &AccessorWrap::parent,
                ("this accessor's parent " + pyGridTypeName).c_str())

            //
            // Voxel access
            //
            .def("getValue", &AccessorWrap::getValue,
                nb::arg("ijk"),
                "Return the value of the voxel at coordinates (i, j, k).")

            .def("getValueDepth", &AccessorWrap::getValueDepth,
                nb::arg("ijk"),
                "Return the tree depth (0 = root) at which the value of voxel\n"
                "(i, j, k) resides.  If (i, j, k) isn't explicitly represented in\n"
                "the tree (i.e., it is implicitly a background voxel), return -1.")

            .def("isVoxel", &AccessorWrap::isVoxel,
                nb::arg("ijk"),
                "Return True if voxel (i, j, k) resides at the leaf level of the tree.")

            .def("probeValue", &AccessorWrap::probeValue,
                nb::arg("ijk"),
                "Return the value of the voxel at coordinates (i, j, k)\n"
                "together with the voxel's active state.")

            .def("isValueOn", &AccessorWrap::isValueOn,
                nb::arg("ijk"),
                "Return the active state of the voxel at coordinates (i, j, k).")
            .def("setActiveState", &AccessorWrap::setActiveState,
                nb::arg("ijk"), nb::arg("on"),
                "Mark voxel (i, j, k) as either active or inactive (True or False),\n"
                "but don't change its value.")

            .def("setValueOnly", &AccessorWrap::setValueOnly,
                nb::arg("ijk"), nb::arg("value"),
                "Set the value of voxel (i, j, k), but don't change its active state.")

            .def("setValueOn", &AccessorWrap::setValueOn,
                nb::arg("ijk"), nb::arg("value") = nb::none(),
                "Mark voxel (i, j, k) as active and set the voxel's value if specified.")

            .def("setValueOff", &AccessorWrap::setValueOff,
                nb::arg("ijk"), nb::arg("value") = nb::none(),
                "Mark voxel (i, j, k) as inactive and set the voxel's value if specified.")

            .def("isCached", &AccessorWrap::isCached,
                nb::arg("ijk"),
                "Return True if this accessor has cached the path to voxel (i, j, k).");
    }

private:
    const GridPtrType mGrid;
    Accessor mAccessor;
}; // class AccessorWrap

} // namespace pyAccessor

#endif // OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED
