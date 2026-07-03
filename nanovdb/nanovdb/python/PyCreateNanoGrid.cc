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

#include <cstring>
#include <string>
#include <vector>

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
    m.def(name, &createNanoGridFromFunc<BuildT>, nb::call_guard<nb::gil_scoped_release>(), "background"_a, "name"_a, "gridClass"_a, "func"_a, "bbox"_a,
          "Construct a NanoGrid by evaluating func(Coord) over every voxel in bbox. "
          "Returns a GridHandle owning the freshly-built grid.");
}

template<typename BufferT> void defineOpenToNanoVDB(nb::module_& m)
{
#ifdef NANOVDB_USE_OPENVDB
    m.def("openToNanoVDB", &tools::openToNanoVDB<BufferT>,
          "base"_a,
          "sMode"_a = tools::StatsMode::Default,
          "cMode"_a = CheckMode::Default,
          "verbose"_a = 0,
          "Convert an OpenVDB base grid to a NanoVDB GridHandle.");
#endif
}

// ============================================================================
// Conversion bindings: AbsDiff/RelDiff oracle classes, and the polymorphic
// createNanoGrid free functions for quantized + index destination BuildTs.
// Each accepts source = NanoGrid<SrcBuildT> OR build::Grid<SrcBuildT>.
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
    msg += ": source must be a FloatGrid or nanovdb.tools.build.FloatGrid "
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
        "nanovdb.tools.build.FloatGrid (FpN requires a float source value type).");
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
           "Vec3fGrid, or the matching nanovdb.tools.build.* mutable grid.";
    throw nb::type_error(msg.c_str());
}

// ----- tools.CreateNanoGrid: converter class with blind-data authoring -----
//
// Mirrors nanovdb::tools::CreateNanoGrid<SrcGridT>. The C++ class is
// templated on the source grid type, so this binding stores the source
// Python object plus the recorded settings and addBlindData() calls, then
// dispatches over the supported SrcBuildTs when getHandle() is called —
// constructing the C++ converter, replaying the recorded state, and baking
// the handle. The destination BuildT is the source BuildT (the C++
// default); quantized and index destinations remain on the
// createNanoGridFp* / createNanoGridIndex free functions above. Authored
// channels come back zeroed — fill them through the writable NumPy view
// returned by grid.getBlindData(n).

// Byte size of one element of the given blind-data GridType, or 0 when the
// size cannot be derived (the caller must then pass it explicitly). Matches
// the per-type size table enforced by GridBlindMetaData::isValid().
uint32_t blindDataTypeSize(GridType dataType)
{
    switch (dataType) {
    case GridType::Float:   return 4u;
    case GridType::Double:  return 8u;
    case GridType::Int16:   return 2u;
    case GridType::Int32:   return 4u;
    case GridType::Int64:   return 8u;
    case GridType::UInt8:   return 1u;
    case GridType::UInt32:  return 4u;
    case GridType::Half:    return 2u;
    case GridType::RGBA8:   return 4u;
    case GridType::Fp8:     return 1u;
    case GridType::Fp16:    return 2u;
    case GridType::Vec3f:   return 12u;
    case GridType::Vec3d:   return 24u;
    case GridType::Vec4f:   return 16u;
    case GridType::Vec4d:   return 32u;
    case GridType::Vec3u8:  return 3u;
    case GridType::Vec3u16: return 6u;
    default: return 0u;
    }
}

class PyCreateNanoGrid
{
    struct BlindDataSpec
    {
        std::string           name;
        GridBlindDataSemantic semantic;
        GridBlindDataClass    dataClass;
        GridType              dataType;
        uint64_t              count;
        uint32_t              size;
    };

public:
    explicit PyCreateNanoGrid(nb::object src)
        : mSrc(std::move(src))
    {
        if (!(matches<float>() || matches<double>() ||
              matches<int32_t>() || matches<Vec3f>())) {
            throw nb::type_error(
                "CreateNanoGrid: source must be a FloatGrid, DoubleGrid, "
                "Int32Grid, Vec3fGrid, or the matching "
                "nanovdb.tools.build.* mutable grid.");
        }
    }

    // Validates eagerly (the C++ ctor only NANOVDB_ASSERTs, which release
    // builds skip) so mistakes surface here rather than as an invalid grid.
    uint64_t addBlindData(const std::string&    name,
                          uint64_t              count,
                          GridType              dataType,
                          GridBlindDataSemantic semantic,
                          GridBlindDataClass    dataClass,
                          uint32_t              size)
    {
        if (name.size() >= GridBlindMetaData::MaxNameSize) {
            throw nb::value_error(
                "addBlindData: name exceeds the 255 character limit.");
        }
        if (size == 0u) size = blindDataTypeSize(dataType);
        if (size == 0u) {
            throw nb::value_error(
                "addBlindData: the element size cannot be derived from this "
                "dataType — pass size explicitly.");
        }
        const GridBlindMetaData meta(0, count, size, semantic, dataClass, dataType);
        if (!meta.isValid()) {
            throw nb::value_error(
                "addBlindData: invalid combination of dataSemantic, "
                "dataClass, dataType, and size.");
        }
        mBlind.push_back(BlindDataSpec{name, semantic, dataClass, dataType, count, size});
        return static_cast<uint64_t>(mBlind.size() - 1);
    }

    void setStats(tools::StatsMode mode) { mStats = mode; }
    void setChecksum(CheckMode mode) { mChecksum = mode; }
    void setVerbose(int mode) { mVerbose = mode; }
    void enableDithering(bool on) { mDither = on; }

    nb::object getHandle() const
    {
        if (auto r = tryGetHandle<float>();   r.is_valid()) return r;
        if (auto r = tryGetHandle<double>();  r.is_valid()) return r;
        if (auto r = tryGetHandle<int32_t>(); r.is_valid()) return r;
        if (auto r = tryGetHandle<Vec3f>();   r.is_valid()) return r;
        throw nb::type_error("CreateNanoGrid: unsupported source grid type.");
    }

private:
    template<typename SrcBuildT> bool matches() const
    {
        return nb::isinstance<NanoGrid<SrcBuildT>>(mSrc) ||
               nb::isinstance<tools::build::Grid<SrcBuildT>>(mSrc);
    }

    template<typename SrcBuildT> nb::object tryGetHandle() const
    {
        using NanoSrcT  = NanoGrid<SrcBuildT>;
        using BuildSrcT = tools::build::Grid<SrcBuildT>;
        if (nb::isinstance<NanoSrcT>(mSrc)) return this->bake(nb::cast<const NanoSrcT&>(mSrc));
        if (nb::isinstance<BuildSrcT>(mSrc)) return this->bake(nb::cast<const BuildSrcT&>(mSrc));
        return nb::object();
    }

    // Same GIL pattern as tryQuantizeFpX: the dispatch above runs with the
    // GIL held, the traversal runs without it (the source's lifetime is
    // anchored by mSrc).
    template<typename SrcGridT> nb::object bake(const SrcGridT& src) const
    {
        GridHandle<HostBuffer> handle;
        {
            nb::gil_scoped_release release;
            tools::CreateNanoGrid<SrcGridT> converter(src);
            converter.setStats(mStats);
            converter.setChecksum(mChecksum);
            converter.setVerbose(mVerbose);
            converter.enableDithering(mDither);
            for (const auto& b : mBlind) {
                converter.addBlindData(b.name, b.semantic, b.dataClass, b.dataType,
                                       static_cast<size_t>(b.count),
                                       static_cast<size_t>(b.size));
            }
            handle = converter.getHandle();
            // The C++ converter allocates authored channels without clearing
            // them (C++ callers memcpy their payload in). Zero-fill here so
            // the NumPy view starts deterministic. The authored channels are
            // the first mBlind.size() blind-data entries — any converter-
            // added channel (e.g. a long grid name) is appended after them.
            if (!mBlind.empty()) {
                if (auto* dst = const_cast<GridData*>(handle.gridData())) {
                    for (size_t i = 0; i < mBlind.size(); ++i) {
                        const GridBlindMetaData* meta = dst->blindMetaData(uint32_t(i));
                        std::memset(const_cast<void*>(meta->blindData()), 0,
                                    meta->blindDataSize());
                    }
                }
            }
        }
        return nb::cast(std::move(handle));
    }

    nb::object                 mSrc;
    std::vector<BlindDataSpec> mBlind;
    tools::StatsMode           mStats    = tools::StatsMode::Default;
    CheckMode                  mChecksum = CheckMode::Default;
    int                        mVerbose  = 0;
    bool                       mDither   = false;
};

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
        .def(nb::init<float>(), "tolerance"_a = -1.0f,
             "Construct an AbsDiff oracle with the given absolute tolerance.")
        .def("getTolerance", &tools::AbsDiff::getTolerance,
             "Return the current absolute tolerance.")
        .def("setTolerance", &tools::AbsDiff::setTolerance, "tolerance"_a,
             "Replace the current absolute tolerance.")
        .def("__bool__",
            [](const tools::AbsDiff& self) { return bool(self); },
            "True iff the tolerance has been initialized (>= 0).");

    nb::class_<tools::RelDiff>(toolsModule, "RelDiff",
        "Compression oracle for FpN: accept the approximation when "
        "|exact - approx| / max(|exact|, |approx|) <= tolerance.")
        .def(nb::init<float>(), "tolerance"_a = -1.0f,
             "Construct a RelDiff oracle with the given relative tolerance.")
        .def("getTolerance", &tools::RelDiff::getTolerance,
             "Return the current relative tolerance.")
        .def("setTolerance", &tools::RelDiff::setTolerance, "tolerance"_a,
             "Replace the current relative tolerance.")
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
        "Quantize a NanoGrid<float> or nanovdb.tools.build.FloatGrid into a "
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
        "Quantize a NanoGrid<float> or nanovdb.tools.build.FloatGrid into a NanoGrid<Fp8> (8 bits "
        "per voxel). ditherOn adds sub-quantum noise.");

    toolsModule.def("createNanoGridFp16",
        [](nb::handle src, tools::StatsMode sMode, CheckMode cMode,
           bool ditherOn, int verbose) {
            return createNanoGridFpX<Fp16>(src, sMode, cMode, ditherOn, verbose,
                                           "createNanoGridFp16");
        },
        "src"_a, "sMode"_a = tools::StatsMode::Default,
        "cMode"_a = CheckMode::Default, "ditherOn"_a = false, "verbose"_a = 0,
        "Quantize a NanoGrid<float> or nanovdb.tools.build.FloatGrid into a NanoGrid<Fp16> (16 bits "
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
        "Quantize a NanoGrid<float> or nanovdb.tools.build.FloatGrid into a NanoGrid<FpN> (variable "
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
    // createNanoGridIndex / createNanoGridOnIndex are the canonical names
    // for the broad-source-coverage index conversion bindings. A narrower
    // createOnIndexGrid factory still lives in PyVoxelBlockManager.cc as
    // the test scaffolding entry point used by the VBM unit tests.
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

    // ------ CreateNanoGrid converter class (blind-data authoring) ------
    nb::class_<PyCreateNanoGrid>(toolsModule, "CreateNanoGrid",
        "Reusable converter mirroring nanovdb::tools::CreateNanoGrid. "
        "Construct from a NanoGrid or nanovdb.tools.build.* grid (float, "
        "double, int32, or Vec3f), optionally declare blind-data channels "
        "with addBlindData(), then bake a fresh NanoGrid of the same BuildT "
        "with getHandle(). Authored channels come back zero-filled — write "
        "their contents through the writable NumPy view returned by "
        "grid.getBlindData(n). For quantized (Fp*) or index destinations "
        "use the createNanoGridFp* / createNanoGridIndex functions instead.")
        .def(nb::init<nb::object>(), "srcGrid"_a,
             "Construct a converter reading from srcGrid (a NanoGrid or "
             "nanovdb.tools.build.* grid of BuildT float, double, int32, or "
             "Vec3f). The converter keeps srcGrid alive.")
        .def("addBlindData", &PyCreateNanoGrid::addBlindData,
             "name"_a, "count"_a, "dataType"_a = GridType::Float,
             "dataSemantic"_a = GridBlindDataSemantic::Unknown,
             "dataClass"_a = GridBlindDataClass::AttributeArray,
             "size"_a = 0u,
             "Declare a blind-data channel of count elements of dataType to "
             "be allocated in the destination grid, and return its channel "
             "index. size (bytes per element) is derived from dataType when "
             "omitted; pass it explicitly for dataTypes without a fixed "
             "element size. The C++ signature orders the parameters (name, "
             "dataSemantic, dataClass, dataType, count, size) — reordered "
             "here so the common case reads addBlindData(name, count).")
        .def("setStats", &PyCreateNanoGrid::setStats, "mode"_a,
             "Set the StatsMode used when baking the destination grid.")
        .def("setChecksum", &PyCreateNanoGrid::setChecksum, "mode"_a,
             "Set the CheckMode used when baking the destination grid.")
        .def("setVerbose", &PyCreateNanoGrid::setVerbose, "mode"_a = 1,
             "Set the verbosity level used when baking the destination grid.")
        .def("enableDithering", &PyCreateNanoGrid::enableDithering, "on"_a = true,
             "Toggle dithering of the destination grid (only meaningful for "
             "quantized BuildTs; kept for parity with the C++ class).")
        .def("getHandle", &PyCreateNanoGrid::getHandle,
             "Bake and return a GridHandle owning a NanoGrid of the source's "
             "BuildT, including any channels declared via addBlindData(). "
             "Each call bakes a fresh grid.");
}

#define NANOVDB_PY_FOR_EACH_SAMPLEABLE_BUILDT(T, Suffix) \
    template void defineCreateNanoGrid<T>(nb::module_&, const char*);
#include "BuildTypes.def"

template void defineOpenToNanoVDB<HostBuffer>(nb::module_&);

} // namespace pynanovdb
