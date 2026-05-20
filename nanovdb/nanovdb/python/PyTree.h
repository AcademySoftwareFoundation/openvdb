// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYTREE_HAS_BEEN_INCLUDED
#define NANOVDB_PYTREE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/NodeManager.h>
#include <nanovdb/HostBuffer.h>

#include <type_traits>  // std::is_arithmetic_v, std::enable_if

namespace nb = nanobind;

namespace pynanovdb {

// -------------------- NanoLeaf<BuildT> --------------------
//
// Binds the 8^3 leaf node. Methods that don't depend on whether the leaf
// stores a contiguous T[512] array are bound unconditionally; the
// zero-copy 512-element NumPy values() view is only bound when the leaf
// layout actually carries T mValues[512] (the BuildTraits<T>::is_special
// types use packed / index / mask layouts where a 512-element T view is
// either impossible or misleading).
template<typename BuildT> void defineNanoLeaf(nb::module_& m, const char* name)
{
    using LeafT = nanovdb::NanoLeaf<BuildT>;
    using ValueT = typename LeafT::ValueType;
    using CoordT = typename LeafT::CoordType;

    auto cls = nb::class_<LeafT>(m, name,
        "Leaf node — 8x8x8 voxels. Inherits stats and bbox from the same "
        "leaf-data block bound across BuildTs.");

    cls.def("origin",        &LeafT::origin)
       .def("bbox",          &LeafT::bbox)
       .def("hasBBox",       &LeafT::hasBBox)
       .def_static("dim",    &LeafT::dim)
       .def_static("voxelCount", &LeafT::voxelCount)
       .def("memUsage",      &LeafT::memUsage)
       .def("isActive",
            nb::overload_cast<const CoordT&>(&LeafT::isActive, nb::const_),
            nb::arg("ijk"))
       .def("isActive",
            [](const LeafT& leaf, uint32_t n) {
                // Underlying mValueMask.isOn(n) is unchecked; release builds
                // skip the C++ NANOVDB_ASSERT and would silently read OOB.
                if (n >= LeafT::voxelCount()) {
                    throw nb::index_error(
                        "Leaf.isActive(n): n out of range [0, voxelCount)");
                }
                return leaf.isActive(n);
            },
            nb::arg("n"))
       .def("getValue",
            [](const LeafT& leaf, uint32_t offset) {
                // mValues[offset] is unchecked in C++; guard the Python side.
                if (offset >= LeafT::voxelCount()) {
                    throw nb::index_error(
                        "Leaf.getValue(offset): offset out of range [0, voxelCount)");
                }
                return leaf.getValue(offset);
            },
            nb::arg("offset"))
       .def("getValue",
            nb::overload_cast<const CoordT&>(&LeafT::getValue, nb::const_),
            nb::arg("ijk"))
       .def("getFirstValue", &LeafT::getFirstValue)
       .def("getLastValue",  &LeafT::getLastValue)
       .def("minimum",       &LeafT::minimum)
       .def("maximum",       &LeafT::maximum)
       .def("average",       &LeafT::average)
       .def("stdDeviation",  &LeafT::stdDeviation)
       // NOTE: variance() omitted — NanoVDB.h line 4388 uses unqualified
       // Pow2() which fails ADL for non-float ValueTs (ValueIndex /
       // ValueMask / etc.). Users can compute it as stdDeviation() ** 2.
       .def("flags",         &LeafT::flags)
       .def("valueMask",     &LeafT::valueMask, nb::rv_policy::reference_internal)
       .def("probeValue",
            [](const LeafT& leaf, const CoordT& ijk) {
                ValueT v;
                bool   on = leaf.probeValue(ijk, v);
                return std::make_tuple(v, on);
            },
            nb::arg("ijk"));

    // Zero-copy 512-element NumPy view of mValues. Only enabled for
    // BuildTs whose ValueType is a primitive arithmetic type (float,
    // double, int*). For Fp* / Index / Mask / bool / Point the leaf uses
    // packed / mask / void layouts. For Vec3f / Vec3d / Vec4f / Vec4d /
    // Vec3u8 / Vec3u16 / Rgba8 the leaf carries a struct array that
    // nanobind's ndarray can't represent directly — those would need a
    // flattened (count, dim) component-typed view which a follow-up can
    // add. Users can still walk every leaf via the bound getValue().
    if constexpr (std::is_arithmetic_v<ValueT>
                  && !nanovdb::BuildTraits<BuildT>::is_special) {
        cls.def("values",
            [](nb::handle py_self) {
                auto& leaf = nb::cast<LeafT&>(py_self);
                size_t shape[1] = {LeafT::voxelCount()};
                return nb::cast(
                    nb::ndarray<nb::numpy, ValueT, nb::ndim<1>, nb::c_contig, nb::device::cpu>(
                        static_cast<void*>(leaf.data()->mValues),
                        size_t(1), shape, py_self),
                    nb::rv_policy::reference);
            },
            nb::keep_alive<0, 1>(),
            "Return a zero-copy NumPy view of the 512 leaf values. The view "
            "keeps the leaf (and transitively the GridHandle that owns the "
            "underlying buffer) alive.");
    }
}

// -------------------- NanoUpper / NanoLower<BuildT> --------------------
//
// Both internal node levels share the same C++ API surface (just different
// LOG2DIM). One helper templated on the concrete InternalNode type.
template<typename InternalT>
void defineInternalNodeBase(nb::class_<InternalT>& cls)
{
    using ValueT = typename InternalT::ValueType;
    using CoordT = typename InternalT::CoordType;
    cls.def("origin",       &InternalT::origin)
       .def("bbox",         &InternalT::bbox)
       .def_static("dim",   &InternalT::dim)
       .def_static("memUsage", []() { return InternalT::memUsage(); })
       .def("minimum",      &InternalT::minimum)
       .def("maximum",      &InternalT::maximum)
       .def("average",      &InternalT::average)
       .def("stdDeviation", &InternalT::stdDeviation)
       // variance() omitted for parity with the leaf binding; compute as
       // stdDeviation() ** 2 in Python.
       .def("valueMask",    &InternalT::valueMask, nb::rv_policy::reference_internal)
       .def("childMask",    &InternalT::childMask, nb::rv_policy::reference_internal)
       .def("getValue",
            nb::overload_cast<const CoordT&>(&InternalT::getValue, nb::const_),
            nb::arg("ijk"))
       .def("getFirstValue", &InternalT::getFirstValue)
       .def("getLastValue",  &InternalT::getLastValue)
       .def("isActive",
            nb::overload_cast<const CoordT&>(&InternalT::isActive, nb::const_),
            nb::arg("ijk"))
       .def("probeValue",
            [](const InternalT& node, const CoordT& ijk) {
                ValueT v;
                bool   on = node.probeValue(ijk, v);
                return std::make_tuple(v, on);
            },
            nb::arg("ijk"));
}

template<typename BuildT> void defineNanoUpper(nb::module_& m, const char* name)
{
    using UpperT = nanovdb::NanoUpper<BuildT>;
    nb::class_<UpperT> cls(m, name,
        "Upper internal node — 32x32x32 (covers a 4096^3 region in index space).");
    defineInternalNodeBase(cls);
}

template<typename BuildT> void defineNanoLower(nb::module_& m, const char* name)
{
    using LowerT = nanovdb::NanoLower<BuildT>;
    nb::class_<LowerT> cls(m, name,
        "Lower internal node — 16x16x16 (covers a 128^3 region in index space).");
    defineInternalNodeBase(cls);
}

// -------------------- NanoRoot<BuildT> --------------------
template<typename BuildT> void defineNanoRoot(nb::module_& m, const char* name)
{
    using RootT = nanovdb::NanoRoot<BuildT>;
    using ValueT = typename RootT::ValueType;
    using CoordT = typename RootT::CoordType;

    nb::class_<RootT>(m, name, "Root node — top of the tree, holds the tile table.")
        .def("background",    &RootT::background, nb::rv_policy::reference_internal)
        .def("tileCount",     &RootT::tileCount)
        .def("getTableSize",  &RootT::getTableSize)
        .def("isEmpty",       &RootT::isEmpty)
        .def("bbox",          &RootT::bbox, nb::rv_policy::reference_internal)
        .def("minimum",       &RootT::minimum, nb::rv_policy::reference_internal)
        .def("maximum",       &RootT::maximum, nb::rv_policy::reference_internal)
        .def("average",       &RootT::average, nb::rv_policy::reference_internal)
        .def("stdDeviation",  &RootT::stdDeviation, nb::rv_policy::reference_internal)
        .def("memUsage",
             nb::overload_cast<>(&RootT::memUsage, nb::const_))
        .def("getValue",
             nb::overload_cast<const CoordT&>(&RootT::getValue, nb::const_),
             nb::arg("ijk"))
        .def("isActive",
             nb::overload_cast<const CoordT&>(&RootT::isActive, nb::const_),
             nb::arg("ijk"))
        .def("probeValue",
             [](const RootT& root, const CoordT& ijk) {
                 ValueT v;
                 bool   on = root.probeValue(ijk, v);
                 return std::make_tuple(v, on);
             },
             nb::arg("ijk"));
}

// -------------------- NanoTree<BuildT> --------------------
template<typename BuildT> void defineNanoTree(nb::module_& m, const char* name)
{
    using TreeT = nanovdb::NanoTree<BuildT>;
    using ValueT = typename TreeT::ValueType;
    using CoordT = typename TreeT::CoordType;
    using RootT = typename TreeT::RootType;
    using UpperT = typename TreeT::UpperNodeType;
    using LowerT = typename TreeT::LowerNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    nb::class_<TreeT>(m, name,
        "Tree — owns the root and provides bulk metadata queries "
        "(node counts, active voxel count, extrema).")
        .def("root",
             nb::overload_cast<>(&TreeT::root, nb::const_),
             nb::rv_policy::reference_internal)
        .def("background", &TreeT::background, nb::rv_policy::reference_internal)
        .def("activeVoxelCount", &TreeT::activeVoxelCount)
        // activeTileCount(level): valid range is 1..3 (lower / upper / root
        // tile counts). C++ uses NANOVDB_ASSERT(level > 0 && level <= 3)
        // which is a no-op in release builds — so guard explicitly.
        .def("activeTileCount",
             [](const TreeT& tree, uint32_t level) -> uint32_t {
                 if (level < 1 || level > 3) {
                     throw nb::value_error(
                         "Tree.activeTileCount(level): level must be 1, 2, or 3");
                 }
                 return tree.activeTileCount(level);
             },
             nb::arg("level"))
        // nodeCount(level): valid range is 0..2 (leaf / lower / upper).
        // C++ uses NANOVDB_ASSERT(level < 3), again no-op in release.
        // The lambda's `int level` argument disambiguates the call against
        // Tree's templated nodeCount<NodeT>() overload at the C++ level,
        // so we don't need an overload_cast / static_cast wrapper here.
        .def("nodeCount",
             [](const TreeT& tree, int level) -> uint32_t {
                 if (level < 0 || level >= 3) {
                     throw nb::value_error(
                         "Tree.nodeCount(level): level must be 0, 1, or 2");
                 }
                 return tree.nodeCount(level);
             },
             nb::arg("level"))
        .def("totalNodeCount", &TreeT::totalNodeCount)
        .def_static("memUsage", &TreeT::memUsage)
        .def("getValue",
             nb::overload_cast<const CoordT&>(&TreeT::getValue, nb::const_),
             nb::arg("ijk"))
        .def("isActive", &TreeT::isActive, nb::arg("ijk"))
        .def("probeValue",
             [](const TreeT& tree, const CoordT& ijk) {
                 ValueT v;
                 bool   on = tree.probeValue(ijk, v);
                 return std::make_tuple(v, on);
             },
             nb::arg("ijk"))
        .def("extrema",
             [](const TreeT& tree) {
                 ValueT mn, mx;
                 tree.extrema(mn, mx);
                 return std::make_tuple(mn, mx);
             },
             "Return (min, max) of the active values over the whole tree.")
        .def("getFirstLeaf",
             nb::overload_cast<>(&TreeT::getFirstLeaf, nb::const_),
             nb::rv_policy::reference_internal,
             "First leaf node in breadth-first order, or None if the tree is empty.")
        .def("getFirstLower",
             nb::overload_cast<>(&TreeT::getFirstLower, nb::const_),
             nb::rv_policy::reference_internal)
        .def("getFirstUpper",
             nb::overload_cast<>(&TreeT::getFirstUpper, nb::const_),
             nb::rv_policy::reference_internal);
}

// -------------------- NodeManager<BuildT> --------------------
//
// NodeManager is heap-managed by a NodeManagerHandle (move-only, owns the
// underlying memory). We bind one NodeManager class per BuildT and one
// host-side NodeManagerHandle class. Users get a handle from
// nanovdb.createNodeManager(grid); they then call handle.mgr() to obtain a
// borrowed pointer to the typed NodeManager — its lifetime is anchored to
// the handle via reference_internal.
template<typename BuildT> void defineNodeManager(nb::module_& m, const char* name)
{
    using NMT = nanovdb::NodeManager<BuildT>;
    nb::class_<NMT>(m, name,
        "Sequential breadth-first accessor for the leaf / lower / upper "
        "internal nodes of a NanoGrid. Construct via "
        "nanovdb.createNodeManager(grid).")
        .def("isLinear",
             nb::overload_cast<>(&NMT::isLinear, nb::const_))
        .def("memUsage",
             nb::overload_cast<>(&NMT::memUsage, nb::const_))
        .def("nodeCount",
             [](const NMT& nm, int level) -> uint64_t {
                 // Mirror Tree.nodeCount bounds (NodeManager forwards to Tree).
                 if (level < 0 || level >= 3) {
                     throw nb::value_error(
                         "NodeManager.nodeCount(level): level must be 0, 1, or 2");
                 }
                 return nm.nodeCount(level);
             },
             nb::arg("level"))
        .def("leafCount",  &NMT::leafCount)
        .def("lowerCount", &NMT::lowerCount)
        .def("upperCount", &NMT::upperCount)
        // leaf / lower / upper: NANOVDB_ASSERT(i < nodeCount(LEVEL)) in C++ is
        // no-op in release, so guard explicitly to convert OOB access into a
        // Python IndexError instead of memory corruption.
        .def("leaf",
             [](const NMT& nm, uint32_t i) -> const nanovdb::NanoLeaf<BuildT>& {
                 if (i >= nm.leafCount()) {
                     throw nb::index_error(
                         "NodeManager.leaf(i): i out of range [0, leafCount)");
                 }
                 return nm.leaf(i);
             },
             nb::rv_policy::reference_internal, nb::arg("i"))
        .def("lower",
             [](const NMT& nm, uint32_t i) -> const nanovdb::NanoLower<BuildT>& {
                 if (i >= nm.lowerCount()) {
                     throw nb::index_error(
                         "NodeManager.lower(i): i out of range [0, lowerCount)");
                 }
                 return nm.lower(i);
             },
             nb::rv_policy::reference_internal, nb::arg("i"))
        .def("upper",
             [](const NMT& nm, uint32_t i) -> const nanovdb::NanoUpper<BuildT>& {
                 if (i >= nm.upperCount()) {
                     throw nb::index_error(
                         "NodeManager.upper(i): i out of range [0, upperCount)");
                 }
                 return nm.upper(i);
             },
             nb::rv_policy::reference_internal, nb::arg("i"));
}

void defineNodeManagerHandle(nb::module_& m);
void defineCreateNodeManager(nb::module_& m);

// -------------------- grid.leaf_values() bulk extractor --------------------
//
// For non-special BuildTs with breadth-first, fixed-size leaves, the leaf
// values can be reached as a contiguous (N_leaves, 512) array — every leaf
// occupies sizeof(NanoLeaf<T>) bytes and mValues starts at a known offset
// inside each leaf. We bind this on NanoGrid<T> as leaf_values() for
// efficient bulk analytics from Python.
template<typename BuildT, typename = void>
struct PyLeafValuesBinder
{
    template<typename ClsT> static void apply(ClsT&) {}
};

template<typename BuildT>
struct PyLeafValuesBinder<BuildT,
    typename std::enable_if<
        std::is_arithmetic_v<typename nanovdb::NanoLeaf<BuildT>::ValueType>
        && !nanovdb::BuildTraits<BuildT>::is_special>::type>
{
    template<typename ClsT>
    static void apply(ClsT& cls)
    {
        using GridT = nanovdb::NanoGrid<BuildT>;
        using LeafT = nanovdb::NanoLeaf<BuildT>;
        using ValueT = typename LeafT::ValueType;
        cls.def("leaf_values",
            [](nb::handle py_self) -> nb::object {
                auto& grid = nb::cast<GridT&>(py_self);
                const auto& tree = grid.tree();
                const uint32_t nLeaves = tree.template nodeCount<LeafT>();
                if (!grid.isBreadthFirst()) {
                    throw nb::value_error(
                        "leaf_values() requires a breadth-first grid "
                        "layout; rebuild via "
                        "nanovdb.tools.createNanoGrid(...).");
                }
                // For an empty grid (no leaves) we still return an ndarray
                // — shape (0, 512) — so callers can iterate / np.asarray()
                // / shape-test without branching on a None sentinel.
                LeafT* first = const_cast<LeafT*>(tree.getFirstLeaf());
                size_t  shape[2]   = {nLeaves, LeafT::voxelCount()};
                int64_t strides[2] = {
                    static_cast<int64_t>(sizeof(LeafT) / sizeof(ValueT)),
                    1
                };
                // first is non-null whenever nLeaves > 0; when nLeaves == 0
                // we pass a dummy non-null aligned pointer (the grid itself)
                // so nanobind has something to base the empty array on.
                // Nothing will be read since the leading shape is 0.
                void* data = (first != nullptr)
                    ? static_cast<void*>(first->data()->mValues)
                    : static_cast<void*>(&grid);
                return nb::cast(
                    nb::ndarray<nb::numpy, ValueT, nb::ndim<2>, nb::device::cpu>(
                        data, size_t(2), shape, py_self, strides),
                    nb::rv_policy::reference);
            },
            nb::keep_alive<0, 1>(),
            "Return a zero-copy (N_leaves, 512) NumPy view of every leaf's "
            "values, in breadth-first leaf order. Available only for "
            "BuildTs whose leaf layout carries T mValues[512] (i.e. not "
            "Fp*, Index, Mask, bool, or Point) and only on breadth-first "
            "grids. Returns an empty (0, 512) array for grids with no "
            "leaves. The view keeps the grid alive.");
    }
};

} // namespace pynanovdb

#endif
