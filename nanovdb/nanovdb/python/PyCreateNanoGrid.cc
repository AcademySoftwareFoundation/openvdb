// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyCreateNanoGrid.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#ifdef NANOVDB_USE_OPENVDB
#include <nanobind/stl/shared_ptr.h>
#endif

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

namespace {

template<typename BuildT>
GridHandle<HostBuffer> createNanoGridFromFunc(const BuildT&                                       background,
                                              const std::string&                                  name,
                                              GridClass                                           gridClass,
                                              const std::function<BuildT(const nanovdb::Coord&)>& func,
                                              const CoordBBox&                                    bbox)
{
    nanovdb::tools::build::Grid<BuildT> srcGrid(background, name, gridClass);
    srcGrid(func, bbox);
    return nanovdb::tools::createNanoGrid(srcGrid);
}

} // namespace

template<typename BuildT> void defineCreateNanoGrid(nb::module_& m, const char* name)
{
    m.def(name, &createNanoGridFromFunc<BuildT>, nb::call_guard<nb::gil_scoped_release>(), "background"_a, "name"_a, "gridClass"_a, "func"_a, "bbox"_a);
}

template<typename BufferT> void defineOpenToNanoVDB(nb::module_& m)
{
#ifdef NANOVDB_USE_OPENVDB
    m.def("openToNanoVDB", &tools::openToNanoVDB<BufferT>,
          "base"_a,
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "verbose"_a = 0);
#endif
}

// ============================================================================
// Phase 5b/5c conversion bindings: AbsDiff/RelDiff oracle classes, and the
// polymorphic createNanoGrid free functions for quantized + index destination
// BuildTs. Each accepts source = NanoGrid<SrcBuildT> OR build::Grid<SrcBuildT>.
// ============================================================================

namespace {

// ----- Quantized (Fp4/Fp8/Fp16) -----
//
// C++ signature: createNanoGrid<SrcGridT, DstBuildT, BufferT>(srcGrid,
// sMode, cMode, ditherOn, verbose, buffer).
//
// Try SrcBuildT against both NanoGrid<SrcBuildT> and build::Grid<SrcBuildT>
// and return an empty nb::object on no match so the caller can fall through
// to the next SrcBuildT.
//
// GIL is held for the isinstance / cast dispatch (which touches the Python
// object's type and reference graph) but released around the underlying
// tools::createNanoGrid traversal — the source data lives in stable C++
// storage whose lifetime is anchored by the Python wrapper passed in via
// py_src, so it's safe to read without holding the GIL.
template<typename SrcBuildT, typename DstBuildT>
nb::object tryQuantizeFpX(nb::handle       py_src,
                          tools::StatsMode sMode,
                          CheckMode        cMode,
                          bool             ditherOn,
                          int              verbose)
{
    using NanoSrcT  = NanoGrid<SrcBuildT>;
    using BuildSrcT = tools::build::Grid<SrcBuildT>;
    if (nb::isinstance<NanoSrcT>(py_src)) {
        const auto& src = nb::cast<const NanoSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<NanoSrcT, DstBuildT, HostBuffer>(
                src, sMode, cMode, ditherOn, verbose);
        }
        return nb::cast(std::move(handle));
    }
    if (nb::isinstance<BuildSrcT>(py_src)) {
        const auto& src = nb::cast<const BuildSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<BuildSrcT, DstBuildT, HostBuffer>(
                src, sMode, cMode, ditherOn, verbose);
        }
        return nb::cast(std::move(handle));
    }
    return nb::object();
}

template<typename DstBuildT>
nb::object createNanoGridFpX(nb::handle       py_src,
                             tools::StatsMode sMode,
                             CheckMode        cMode,
                             bool             ditherOn,
                             int              verbose,
                             const char*      pyFnName)
{
    // The C++ Fp{4,8,16,N} preProcess static_asserts SrcValueT == float;
    // double sources hit a compile-time error, so we accept float only.
    if (auto r = tryQuantizeFpX<float, DstBuildT>(
            py_src, sMode, cMode, ditherOn, verbose); r.is_valid()) return r;
    std::string msg(pyFnName);
    msg += ": source must be a FloatGrid or tools.build.FloatGrid "
           "(Fp4/Fp8/Fp16/FpN require a float source value type).";
    throw nb::type_error(msg.c_str());
}

// ----- FpN (variable bit-width) -----
//
// C++ signature: createNanoGrid<SrcGridT, FpN, OracleT, BufferT>(srcGrid,
// sMode, cMode, ditherOn, verbose, oracle, buffer). OracleT is AbsDiff or
// RelDiff; the binding exposes both as separate Python overloads.
template<typename SrcBuildT, typename OracleT>
nb::object tryQuantizeFpN(nb::handle       py_src,
                          tools::StatsMode sMode,
                          CheckMode        cMode,
                          bool             ditherOn,
                          int              verbose,
                          const OracleT&   oracle)
{
    using NanoSrcT  = NanoGrid<SrcBuildT>;
    using BuildSrcT = tools::build::Grid<SrcBuildT>;
    // Same GIL pattern as tryQuantizeFpX: hold the GIL through the
    // isinstance / cast dispatch, release it for the conversion.
    if (nb::isinstance<NanoSrcT>(py_src)) {
        const auto& src = nb::cast<const NanoSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<NanoSrcT, FpN, OracleT, HostBuffer>(
                src, sMode, cMode, ditherOn, verbose, oracle);
        }
        return nb::cast(std::move(handle));
    }
    if (nb::isinstance<BuildSrcT>(py_src)) {
        const auto& src = nb::cast<const BuildSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<BuildSrcT, FpN, OracleT, HostBuffer>(
                src, sMode, cMode, ditherOn, verbose, oracle);
        }
        return nb::cast(std::move(handle));
    }
    return nb::object();
}

template<typename OracleT>
nb::object createNanoGridFpNImpl(nb::handle       py_src,
                                 const OracleT&   oracle,
                                 tools::StatsMode sMode,
                                 CheckMode        cMode,
                                 bool             ditherOn,
                                 int              verbose)
{
    if (auto r = tryQuantizeFpN<float, OracleT>(
            py_src, sMode, cMode, ditherOn, verbose, oracle); r.is_valid()) return r;
    throw nb::type_error(
        "createNanoGridFpN: source must be a FloatGrid or "
        "tools.build.FloatGrid (FpN requires a float source value type).");
}

// ----- Index / OnIndex -----
//
// C++ signature: createNanoGrid<SrcGridT, DstBuildT, BufferT>(srcGrid,
// channels, includeStats, includeTiles, verbose, buffer). DstBuildT is
// ValueIndex or ValueOnIndex; the binding exposes both as separate
// named functions. Source set is wider than the quantized variants —
// any arithmetic or vector source can be re-cast as an index grid.
template<typename SrcBuildT, typename DstBuildT>
nb::object tryIndexify(nb::handle py_src,
                       uint32_t   channels,
                       bool       includeStats,
                       bool       includeTiles,
                       int        verbose)
{
    using NanoSrcT  = NanoGrid<SrcBuildT>;
    using BuildSrcT = tools::build::Grid<SrcBuildT>;
    // Same GIL pattern as tryQuantizeFpX.
    if (nb::isinstance<NanoSrcT>(py_src)) {
        const auto& src = nb::cast<const NanoSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<NanoSrcT, DstBuildT, HostBuffer>(
                src, channels, includeStats, includeTiles, verbose);
        }
        return nb::cast(std::move(handle));
    }
    if (nb::isinstance<BuildSrcT>(py_src)) {
        const auto& src = nb::cast<const BuildSrcT&>(py_src);
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            handle = tools::createNanoGrid<BuildSrcT, DstBuildT, HostBuffer>(
                src, channels, includeStats, includeTiles, verbose);
        }
        return nb::cast(std::move(handle));
    }
    return nb::object();
}

template<typename DstBuildT>
nb::object createIndexImpl(nb::handle  py_src,
                           uint32_t    channels,
                           bool        includeStats,
                           bool        includeTiles,
                           int         verbose,
                           const char* pyFnName)
{
    if (auto r = tryIndexify<float,    DstBuildT>(py_src, channels, includeStats, includeTiles, verbose); r.is_valid()) return r;
    if (auto r = tryIndexify<double,   DstBuildT>(py_src, channels, includeStats, includeTiles, verbose); r.is_valid()) return r;
    if (auto r = tryIndexify<int32_t,  DstBuildT>(py_src, channels, includeStats, includeTiles, verbose); r.is_valid()) return r;
    if (auto r = tryIndexify<Vec3f,    DstBuildT>(py_src, channels, includeStats, includeTiles, verbose); r.is_valid()) return r;
    std::string msg(pyFnName);
    msg += ": source must be a FloatGrid, DoubleGrid, Int32Grid, "
           "Vec3fGrid, or the matching tools.build.* mutable grid.";
    throw nb::type_error(msg.c_str());
}

} // namespace

void defineCreateNanoGridConversions(nb::module_& toolsModule)
{
    // ------ Oracle classes ------
    nb::class_<tools::AbsDiff>(toolsModule, "AbsDiff",
        "Compression oracle for FpN: accept the approximation when "
        "|exact - approx| <= tolerance. A tolerance of -1.0 (the "
        "default) means uninitialized; any non-negative value (including "
        "0.0) is treated as initialized by the operator bool() check, "
        "or the C++ create function can fill it in via init().")
        .def(nb::init<float>(), "tolerance"_a = -1.0f)
        .def("getTolerance", &tools::AbsDiff::getTolerance)
        .def("setTolerance", &tools::AbsDiff::setTolerance, "tolerance"_a)
        .def("__bool__",
            [](const tools::AbsDiff& self) { return bool(self); },
            "True iff the tolerance has been initialized (>= 0).");

    nb::class_<tools::RelDiff>(toolsModule, "RelDiff",
        "Compression oracle for FpN: accept the approximation when "
        "|exact - approx| / max(|exact|, |approx|) <= tolerance.")
        .def(nb::init<float>(), "tolerance"_a = -1.0f)
        .def("getTolerance", &tools::RelDiff::getTolerance)
        .def("setTolerance", &tools::RelDiff::setTolerance, "tolerance"_a)
        .def("__bool__",
            [](const tools::RelDiff& self) { return bool(self); },
            "True iff the tolerance has been initialized (>= 0).");

    // ------ Quantized fixed-width: Fp4 / Fp8 / Fp16 ------
    toolsModule.def("createNanoGridFp4",
        [](nb::handle src, tools::StatsMode sMode, CheckMode cMode,
           bool ditherOn, int verbose) {
            return createNanoGridFpX<Fp4>(src, sMode, cMode, ditherOn, verbose,
                                          "createNanoGridFp4");
        },
        "src"_a, "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "Quantize a NanoGrid<float> or tools.build.FloatGrid into a "
        "NanoGrid<Fp4> (4 bits per voxel). ditherOn adds sub-quantum "
        "noise to break up banding.");

    toolsModule.def("createNanoGridFp8",
        [](nb::handle src, tools::StatsMode sMode, CheckMode cMode,
           bool ditherOn, int verbose) {
            return createNanoGridFpX<Fp8>(src, sMode, cMode, ditherOn, verbose,
                                          "createNanoGridFp8");
        },
        "src"_a, "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "Quantize a NanoGrid<float> or tools.build.FloatGrid into a NanoGrid<Fp8> (8 bits "
        "per voxel). ditherOn adds sub-quantum noise.");

    toolsModule.def("createNanoGridFp16",
        [](nb::handle src, tools::StatsMode sMode, CheckMode cMode,
           bool ditherOn, int verbose) {
            return createNanoGridFpX<Fp16>(src, sMode, cMode, ditherOn, verbose,
                                           "createNanoGridFp16");
        },
        "src"_a, "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "Quantize a NanoGrid<float> or tools.build.FloatGrid into a NanoGrid<Fp16> (16 bits "
        "per voxel). ditherOn adds sub-quantum noise.");

    // ------ Variable bit-width: FpN with AbsDiff or RelDiff oracle ------
    //
    // Two overloads — one per oracle type. Python dispatch picks the
    // right one from the oracle argument's type. The createNanoGrid C++
    // template's parameter order is (src, sMode, cMode, ditherOn, verbose,
    // oracle, buffer); we reorder for Python so oracle comes second
    // (most callers want to specify it explicitly), then mode/dither
    // parameters as kwargs with defaults.
    toolsModule.def("createNanoGridFpN",
        [](nb::handle src, const tools::AbsDiff& oracle,
           tools::StatsMode sMode, CheckMode cMode, bool ditherOn, int verbose) {
            return createNanoGridFpNImpl(src, oracle, sMode, cMode, ditherOn, verbose);
        },
        "src"_a, "oracle"_a = tools::AbsDiff(),
        "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "Quantize a NanoGrid<float> or tools.build.FloatGrid into a NanoGrid<FpN> (variable "
        "bits per voxel; each leaf picks the smallest N that satisfies "
        "the oracle's tolerance). Pass an AbsDiff oracle for absolute "
        "error bound, or use the RelDiff overload for relative error.");

    toolsModule.def("createNanoGridFpN",
        [](nb::handle src, const tools::RelDiff& oracle,
           tools::StatsMode sMode, CheckMode cMode, bool ditherOn, int verbose) {
            return createNanoGridFpNImpl(src, oracle, sMode, cMode, ditherOn, verbose);
        },
        "src"_a, "oracle"_a,
        "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "FpN overload accepting a RelDiff oracle for relative error.");

    // ------ Index / OnIndex ------
    //
    // createOnIndexGrid (the test-scaffold factory from Phase 3 follow-up)
    // is now superseded by createNanoGridOnIndex. The legacy name keeps
    // working through PyVoxelBlockManager.cc; the official Phase 5 name
    // lives here.
    toolsModule.def("createNanoGridIndex",
        [](nb::handle src, uint32_t channels, bool includeStats,
           bool includeTiles, int verbose) {
            return createIndexImpl<ValueIndex>(
                src, channels, includeStats, includeTiles, verbose,
                "createNanoGridIndex");
        },
        "src"_a, "channels"_a = 0u, "includeStats"_a = true,
        "includeTiles"_a = true, "verbose"_a = 0,
        "Convert a source grid into a NanoGrid<ValueIndex>. Every voxel "
        "(active or inactive) gets a unique uint64 sequential index, with "
        "the original values stored as blind data when channels > 0.");

    toolsModule.def("createNanoGridOnIndex",
        [](nb::handle src, uint32_t channels, bool includeStats,
           bool includeTiles, int verbose) {
            return createIndexImpl<ValueOnIndex>(
                src, channels, includeStats, includeTiles, verbose,
                "createNanoGridOnIndex");
        },
        "src"_a, "channels"_a = 0u, "includeStats"_a = true,
        "includeTiles"_a = true, "verbose"_a = 0,
        "Convert a source grid into a NanoGrid<ValueOnIndex>. Only the "
        "active voxels get a sequential index — the canonical input to "
        "buildVoxelBlockManager.");
}

#define NANOVDB_PY_FOR_EACH_SAMPLEABLE_BUILDT(T, Suffix) \
    template void defineCreateNanoGrid<T>(nb::module_&, const char*);
#include "BuildTypes.def"

template void defineOpenToNanoVDB<HostBuffer>(nb::module_&);

} // namespace pynanovdb
