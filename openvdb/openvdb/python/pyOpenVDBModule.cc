// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream> // must be included before python on macos
#include <cstring> // for strncmp(), strrchr(), etc.
#include <limits>
#include <string>
#include <utility> // for std::make_pair()
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <openvdb/openvdb.h>
#include "pyGrid.h"
#include "pyutil.h"
#include "pyTypeCasters.h"

#ifdef PY_OPENVDB_USE_AX
#include <openvdb_ax/ax.h>
#endif

namespace py = pybind11;

// Forward declarations
void exportTransform(py::module_ m);
void exportMetadata(py::module_ m);
void exportGridBase(py::module_ m);
void exportFloatGrid(py::module_ m);
void exportIntGrid(py::module_ m);
void exportVec3Grid(py::module_ m);
void exportPointGrid(py::module_ m);

namespace _openvdbmodule {

using namespace openvdb;

template<typename T> void translateException(const T&) {}

/// @brief Define a function that translates an OpenVDB exception into
/// the equivalent Python exception.
/// @details openvdb::Exception::what() typically returns a string of the form
/// "<exception>: <description>".  To avoid duplication of the exception name in Python
/// stack traces, the function strips off the "<exception>: " prefix.  To do that,
/// it needs the class name in the form of a string, hence the preprocessor macro.
#define PYOPENVDB_CATCH(_openvdbname, _pyname)                      \
    template<>                                                      \
    void translateException<_openvdbname>(const _openvdbname& e)    \
    {                                                               \
        const char* name = #_openvdbname;                           \
        if (const char* c = std::strrchr(name, ':')) name = c + 1;  \
        const int namelen = int(std::strlen(name));                 \
        const char* msg = e.what();                                 \
        if (0 == std::strncmp(msg, name, namelen)) msg += namelen;  \
        if (0 == std::strncmp(msg, ": ", 2)) msg += 2;              \
        PyErr_SetString(_pyname, msg);                              \
    }


/// Define an overloaded function that translate all OpenVDB exceptions into
/// their Python equivalents.
/// @todo LookupError is redundant and should someday be removed.
PYOPENVDB_CATCH(openvdb::ArithmeticError,       PyExc_ArithmeticError)
PYOPENVDB_CATCH(openvdb::IndexError,            PyExc_IndexError)
PYOPENVDB_CATCH(openvdb::IoError,               PyExc_IOError)
PYOPENVDB_CATCH(openvdb::KeyError,              PyExc_KeyError)
PYOPENVDB_CATCH(openvdb::LookupError,           PyExc_LookupError)
PYOPENVDB_CATCH(openvdb::NotImplementedError,   PyExc_NotImplementedError)
PYOPENVDB_CATCH(openvdb::ReferenceError,        PyExc_ReferenceError)
PYOPENVDB_CATCH(openvdb::RuntimeError,          PyExc_RuntimeError)
PYOPENVDB_CATCH(openvdb::TypeError,             PyExc_TypeError)
PYOPENVDB_CATCH(openvdb::ValueError,            PyExc_ValueError)

#undef PYOPENVDB_CATCH


////////////////////////////////////////

GridBase::Ptr
readFromFile(const std::string& filename, const std::string& gridName)
{
    io::File vdbFile(filename);
    vdbFile.open();

    if (!vdbFile.hasGrid(gridName)) {
        std::ostringstream os;
        os << "file " << filename << " has no grid named \"" << gridName << "\"";
        throw py::key_error(os.str());
    }

    GridBase::Ptr grid = vdbFile.readGrid(gridName);
    vdbFile.close();

    return grid;
}


std::tuple<GridPtrVec, MetaMap>
readAllFromFile(const std::string& filename)
{
    io::File vdbFile(filename);
    vdbFile.open();

    GridPtrVecPtr grids = vdbFile.getGrids();
    MetaMap::Ptr metadata = vdbFile.getMetadata();
    vdbFile.close();

    return std::make_tuple(*grids, *metadata);
}


MetaMap
readFileMetadata(const std::string& filename)
{
    io::File vdbFile(filename);
    vdbFile.open();

    MetaMap::Ptr metadata = vdbFile.getMetadata();
    vdbFile.close();

    return *metadata;
}


GridBase::Ptr
readGridMetadataFromFile(const std::string& filename, const std::string& gridName)
{
    io::File vdbFile(filename);
    vdbFile.open();

    if (!vdbFile.hasGrid(gridName)) {
        std::ostringstream os;
        os << "file " << filename << " has no grid named \"" << gridName <<"\"";
        throw py::key_error(os.str());
    }

    return vdbFile.readGridMetadata(gridName);
}


GridPtrVec
readAllGridMetadataFromFile(const std::string& filename)
{
    io::File vdbFile(filename);
    vdbFile.open();
    GridPtrVecPtr grids = vdbFile.readAllGridMetadata();
    vdbFile.close();

    return *grids;
}


void
writeToFile(const std::string& filename, GridBase::ConstPtr grid, MetaMap metadata)
{
    GridCPtrVec grids;
    grids.push_back(grid);

    io::File vdbFile(filename);
    if (metadata.metaCount()) {
        vdbFile.write(grids, metadata);
    } else {
        vdbFile.write(grids);
    }
    vdbFile.close();
}

void
writeToFile(const std::string& filename, const GridCPtrVec& grids, MetaMap metadata)
{
    io::File vdbFile(filename);
    if (metadata.metaCount()) {
        vdbFile.write(grids, metadata);
    } else {
        vdbFile.write(grids);
    }
    vdbFile.close();
}

#ifdef PY_OPENVDB_USE_AX
void axrun(const std::string& code, GridBase::Ptr grid)
{
    GridPtrVec grids;
    grids.push_back(grid);

    try {
        openvdb::ax::run(code.c_str(), grids);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void axrun(const std::string& code, GridPtrVec& grids)
{
    try {
        openvdb::ax::run(code.c_str(), grids);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
#endif

////////////////////////////////////////


std::string getLoggingLevel();
void setLoggingLevel(const std::string& loggingLevel);
void setProgramName(const std::string& name, bool color);


std::string
getLoggingLevel()
{
    switch (logging::getLevel()) {
        case logging::Level::Debug: return "debug";
        case logging::Level::Info:  return "info";
        case logging::Level::Warn:  return "warn";
        case logging::Level::Error: return "error";
        case logging::Level::Fatal: break;
    }
    return "fatal";
}


void
setLoggingLevel(const std::string& loggingLevel)
{
    std::string level = loggingLevel;
    std::transform(level.begin(), level.end(), level.begin(),
        [](unsigned char c) { return std::tolower(c); });
    auto start = level.begin();
    auto end = level.rbegin();
    while (std::isspace(*start)) ++start;
    while (std::isspace(*end)) ++end;
    std::string levelStr(start, end.base());

    if (levelStr == "debug") { logging::setLevel(logging::Level::Debug); return; }
    else if (levelStr == "info") { logging::setLevel(logging::Level::Info); return; }
    else if (levelStr == "warn") { logging::setLevel(logging::Level::Warn); return; }
    else if (levelStr == "error") { logging::setLevel(logging::Level::Error); return; }
    else if (levelStr == "fatal") { logging::setLevel(logging::Level::Fatal); return; }

    PyErr_Format(PyExc_ValueError,
        "expected logging level \"debug\", \"info\", \"warn\", \"error\", or \"fatal\","
        " got \"%s\"", levelStr.c_str());
    throw py::error_already_set();
}


void
setProgramName(const std::string& name, bool color)
{
    logging::setProgramName(name, color);
}


////////////////////////////////////////


// Descriptor for the openvdb::GridClass enum (for use with pyutil::StringEnum)
struct GridClassDescr
{
    static const char* name() { return "GridClass"; }
    static const char* doc()
    {
        return "Classes of volumetric data (level set, fog volume, etc.)";
    }
    static pyutil::CStringPair item(int i)
    {
        static const int sCount = 4;
        static const char* const sStrings[sCount][2] = {
            { "UNKNOWN",    strdup(GridBase::gridClassToString(GRID_UNKNOWN).c_str()) },
            { "LEVEL_SET",  strdup(GridBase::gridClassToString(GRID_LEVEL_SET).c_str()) },
            { "FOG_VOLUME", strdup(GridBase::gridClassToString(GRID_FOG_VOLUME).c_str()) },
            { "STAGGERED",  strdup(GridBase::gridClassToString(GRID_STAGGERED).c_str()) }
        };
        if (i >= 0 && i < sCount) return pyutil::CStringPair(&sStrings[i][0], &sStrings[i][1]);
        return pyutil::CStringPair(static_cast<char**>(nullptr), static_cast<char**>(nullptr));
    }
};


// Descriptor for the openvdb::VecType enum (for use with pyutil::StringEnum)
struct VecTypeDescr
{
    static const char* name() { return "VectorType"; }
    static const char* doc()
    {
        return
            "The type of a vector determines how transforms are applied to it.\n"
            "  - INVARIANT:\n"
            "      does not transform (e.g., tuple, uvw, color)\n"
            "  - COVARIANT:\n"
            "      apply inverse-transpose transformation with w = 0\n"
            "      and ignore translation (e.g., gradient/normal)\n"
            "  - COVARIANT_NORMALIZE:\n"
            "      apply inverse-transpose transformation with w = 0\n"
            "      and ignore translation, vectors are renormalized\n"
            "      (e.g., unit normal)\n"
            "  - CONTRAVARIANT_RELATIVE:\n"
            "      apply \"regular\" transformation with w = 0 and ignore\n"
            "      translation (e.g., displacement, velocity, acceleration)\n"
            "  - CONTRAVARIANT_ABSOLUTE:\n"
            "      apply \"regular\" transformation with w = 1 so that\n"
            "      vector translates (e.g., position)\n";
    }
    static pyutil::CStringPair item(int i)
    {
        static const int sCount = 5;
        static const char* const sStrings[sCount][2] = {
            { "INVARIANT", strdup(GridBase::vecTypeToString(openvdb::VEC_INVARIANT).c_str()) },
            { "COVARIANT", strdup(GridBase::vecTypeToString(openvdb::VEC_COVARIANT).c_str()) },
            { "COVARIANT_NORMALIZE",
                strdup(GridBase::vecTypeToString(openvdb::VEC_COVARIANT_NORMALIZE).c_str()) },
            { "CONTRAVARIANT_RELATIVE",
                strdup(GridBase::vecTypeToString(openvdb::VEC_CONTRAVARIANT_RELATIVE).c_str()) },
            { "CONTRAVARIANT_ABSOLUTE",
                strdup(GridBase::vecTypeToString(openvdb::VEC_CONTRAVARIANT_ABSOLUTE).c_str()) }
        };
        if (i >= 0 && i < sCount) return std::make_pair(&sStrings[i][0], &sStrings[i][1]);
        return pyutil::CStringPair(static_cast<char**>(nullptr), static_cast<char**>(nullptr));
    }
};

} // namespace _openvdbmodule


////////////////////////////////////////


#ifdef DWA_OPENVDB
#define PY_OPENVDB_MODULE_NAME  _openvdb
extern "C" { void init_openvdb(); }
#else
#define PY_OPENVDB_MODULE_NAME  pyopenvdb
extern "C" { void initpyopenvdb(); }
#endif

PYBIND11_MODULE(PY_OPENVDB_MODULE_NAME, m)
{
    // Don't auto-generate ugly, C++-style function signatures.
    py::options docOptions;
    docOptions.disable_function_signatures();
    docOptions.enable_user_defined_docstrings();

    using namespace openvdb::OPENVDB_VERSION_NAME;

    // Initialize OpenVDB.
    initialize();
#ifdef PY_OPENVDB_USE_AX
    openvdb::ax::initialize();
#endif

#define PYOPENVDB_TRANSLATE_EXCEPTION(_classname)                \
    py::register_exception_translator([](std::exception_ptr p) { \
        try {                                                    \
            if (p) std::rethrow_exception(p);                    \
        } catch (const _classname &e) {                          \
            _openvdbmodule::translateException<_classname>(e);   \
        } \
    });

    PYOPENVDB_TRANSLATE_EXCEPTION(ArithmeticError);
    PYOPENVDB_TRANSLATE_EXCEPTION(IndexError);
    PYOPENVDB_TRANSLATE_EXCEPTION(IoError);
    PYOPENVDB_TRANSLATE_EXCEPTION(KeyError);
    PYOPENVDB_TRANSLATE_EXCEPTION(LookupError);
    PYOPENVDB_TRANSLATE_EXCEPTION(NotImplementedError);
    PYOPENVDB_TRANSLATE_EXCEPTION(ReferenceError);
    PYOPENVDB_TRANSLATE_EXCEPTION(RuntimeError);
    PYOPENVDB_TRANSLATE_EXCEPTION(TypeError);
    PYOPENVDB_TRANSLATE_EXCEPTION(ValueError);

#undef PYOPENVDB_TRANSLATE_EXCEPTION

    // Export the python bindings.
    exportTransform(m);
    exportMetadata(m);
    exportGridBase(m);
    exportFloatGrid(m);
    exportIntGrid(m);
    exportVec3Grid(m);
    exportPointGrid(m);


    m.def("read",
        &_openvdbmodule::readFromFile,
        "read(filename, gridname) -> Grid\n\n"
        "Read a single grid from a .vdb file.",
        py::arg("filename"), py::arg("gridname"));

#ifdef PY_OPENVDB_USE_AX
    m.def("ax",
        py::overload_cast<const std::string&, GridBase::Ptr>(&_openvdbmodule::axrun),
        "ax(code, grids) -> Grid\n\n"
        "Run AX code on a VDB grid.",
        py::arg("code"), py::arg("grid"));

    m.def("ax",
        py::overload_cast<const std::string&, GridPtrVec&>(&_openvdbmodule::axrun),
        "ax(code, grids) -> Grid\n\n"
        "Run AX code on some VDB grids.",
        py::arg("code"), py::arg("grids"));
#endif

    m.def("readAll",
        &_openvdbmodule::readAllFromFile,
        "readAll(filename) -> list, dict\n\n"
        "Read a .vdb file and return a list of grids and\n"
        "a dict of file-level metadata.",
        py::arg("filename"));

    m.def("readMetadata",
        &_openvdbmodule::readFileMetadata,
        "readMetadata(filename) -> dict\n\n"
        "Read file-level metadata from a .vdb file.",
        py::arg("filename"));

    m.def("readGridMetadata",
        &_openvdbmodule::readGridMetadataFromFile,
        "readGridMetadata(filename, gridname) -> Grid\n\n"
        "Read a single grid's metadata and transform (but not its tree)\n"
        "from a .vdb file.",
        py::arg("filename"), py::arg("gridname"));

    m.def("readAllGridMetadata",
        &_openvdbmodule::readAllGridMetadataFromFile,
        "readAllGridMetadata(filename) -> list\n\n"
        "Read a .vdb file and return a list of grids populated with\n"
        "their metadata and transforms, but not their trees.",
        py::arg("filename"));

    m.def("write",
        py::overload_cast<const std::string&, GridBase::ConstPtr, MetaMap>(&_openvdbmodule::writeToFile),
        "write(filename, grids, metadata=None)\n\n"
        "Write a grid and (optionally) a dict\n"
        "of (name, value) metadata pairs to a .vdb file.",
        py::arg("filename"), py::arg("grid"), py::arg("metadata") = py::dict());

    m.def("write",
        py::overload_cast<const std::string&, const GridCPtrVec&, MetaMap>(&_openvdbmodule::writeToFile),
        "write(filename, grids, metadata=None)\n\n"
        "Write a sequence of grids and (optionally) a dict\n"
        "of (name, value) metadata pairs to a .vdb file.",
        py::arg("filename"), py::arg("grids"), py::arg("metadata") = py::dict());

    m.def("getLoggingLevel", &_openvdbmodule::getLoggingLevel,
        "getLoggingLevel() -> str\n\n"
        "Return the severity threshold (\"debug\", \"info\", \"warn\", \"error\",\n"
        "or \"fatal\") for error messages.");
    m.def("setLoggingLevel", &_openvdbmodule::setLoggingLevel,
        "setLoggingLevel(level)\n\n"
        "Specify the severity threshold (\"debug\", \"info\", \"warn\", \"error\",\n"
        "or \"fatal\") for error messages.  Messages of lower severity\n"
        "will be suppressed.",
        py::arg("level"));
    m.def("setProgramName", &_openvdbmodule::setProgramName,
        "setProgramName(name, color=True)\n\n"
        "Specify the program name to be displayed in error messages,\n"
        "and optionally specify whether to print error messages in color.",
        py::arg("name"), py::arg("color") = true);

    // Add some useful module-level constants.
    m.attr("LIBRARY_VERSION") = py::make_tuple(
        openvdb::OPENVDB_LIBRARY_MAJOR_VERSION,
        openvdb::OPENVDB_LIBRARY_MINOR_VERSION,
        openvdb::OPENVDB_LIBRARY_PATCH_VERSION);
    m.attr("FILE_FORMAT_VERSION") = openvdb::OPENVDB_FILE_VERSION;
    m.attr("COORD_MIN") = openvdb::Coord::min();
    m.attr("COORD_MAX") = openvdb::Coord::max();
    m.attr("LEVEL_SET_HALF_WIDTH") = openvdb::LEVEL_SET_HALF_WIDTH;

    pyutil::StringEnum<_openvdbmodule::GridClassDescr>::wrap(m);
    pyutil::StringEnum<_openvdbmodule::VecTypeDescr>::wrap(m);

} // PYBIND11_MODULE
