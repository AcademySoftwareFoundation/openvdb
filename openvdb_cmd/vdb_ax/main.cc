// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file cmd/openvdb_ax.cc
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  The command line vdb_ax binary which provides tools to
///   run and analyze AX code.
///

#include "cli.h"

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>
#include <openvdb_ax/ast/PrintTree.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/AttributeRegistry.h>
#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>
#include <openvdb_ax/compiler/Logger.h>

#include <openvdb/openvdb.h>
#include <openvdb/version.h>
#include <openvdb/io/File.h>
#include <openvdb/util/logging.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/Assert.h>
#include <openvdb/points/PointDelete.h>

#include <tbb/global_control.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

const char* gProgName = "";

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {

enum VDB_AX_MODE { Default, Execute, Analyze, Functions };
enum VDB_AX_COMPILATION { None, All, Points, Volumes };

namespace cli {

template <typename T> inline void ParamToStream(std::ostream& os, const T& v) { os << v; }
template <> inline void ParamToStream<bool>(std::ostream& os, const bool& v) { os << std::boolalpha << v; }

template <>
inline void ParamToStream<std::vector<std::string>>(
    std::ostream& os,
    const std::vector<std::string>& v)
{
    for (const auto& s : v) os << s << ", ";
}

template <>
inline void ParamToStream<std::pair<bool, std::string>>(
    std::ostream& os,
    const std::pair<bool, std::string>& v)
{
    os << v.first << ':' << v.second;
}

template <>
inline void ParamToStream<CompilerOptions::OptLevel>(
    std::ostream& os,
    const CompilerOptions::OptLevel& v)
{
    switch (v) {
        case  CompilerOptions::OptLevel::NONE  : { os << "NONE"; break; }
        case  CompilerOptions::OptLevel::O1    : { os << "O1"; break; }
        case  CompilerOptions::OptLevel::O2    : { os << "O2"; break; }
        case  CompilerOptions::OptLevel::Os    : { os << "Os"; break; }
        case  CompilerOptions::OptLevel::Oz    : { os << "Oz"; break; }
        case  CompilerOptions::OptLevel::O3    : { os << "O3"; break; }
        default : return;
    }
}

template <>
inline void ParamToStream<VDB_AX_MODE>(
    std::ostream& os,
    const VDB_AX_MODE& mode)
{
    switch (mode) {
        case VDB_AX_MODE::Execute   : { os << "execute"; break; }
        case VDB_AX_MODE::Analyze   : { os << "analyze"; break; }
        case VDB_AX_MODE::Functions : { os << "functions"; break; }
        default : os << "execute";
    }
}

template <>
inline void ParamToStream<VDB_AX_COMPILATION>(
    std::ostream& os,
    const VDB_AX_COMPILATION& mode)
{
    switch (mode) {
        case VDB_AX_COMPILATION::Points  : { os << "points"; break; }
        case VDB_AX_COMPILATION::Volumes : { os << "volumes"; break; }
        default :;
    }
}

} // namespace cli
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


void fatal [[noreturn]] (const char* msg = nullptr)
{
    if (msg) OPENVDB_LOG_FATAL(msg << ". See '" << gProgName << " --help'");
    std::exit(EXIT_FAILURE);
}

void fatal [[noreturn]] (const std::string msg) { fatal(msg.c_str()); }

void loadSnippetFile(const std::string& fileName, std::string& textString)
{
    std::ifstream in(fileName.c_str(), std::ios::in | std::ios::binary);

    if (!in) {
        OPENVDB_LOG_FATAL("File Load Error: " << fileName);
        fatal();
    }

    textString =
        std::string(std::istreambuf_iterator<char>(in),
                    std::istreambuf_iterator<char>());
}

// This is to handle the old style input paths
// @todo remove
std::string sOldInputFile;
bool sOldInputInit = false;

struct ProgOptions
{
    inline std::vector<openvdb::ax::cli::ParamBase*> positional()
    {
        std::vector<openvdb::ax::cli::ParamBase*> params {
            &this->mMode,
            &this->mOldInputFile,
            &this->mOldOutputFile
        };
        return params;
    }

    inline std::vector<openvdb::ax::cli::ParamBase*> optional()
    {
        std::vector<openvdb::ax::cli::ParamBase*> params {
            &this->mVerbose,
            &this->mHelp,
            // execute
            &this->mInputVDBFiles,
            &this->mInputCode,
            &this->mOutputVDBFile,
            &this->mOptLevel,
            &this->mThreads,
            &this->mWarningsAsErrors,
            &this->mMaxErrors,
            &this->mCopyFileMeta,
            // analyze
            &this->mPrintAST,
            &this->mReprint,
            &this->mAttribRegPrint,
            &this->mCompileFor,
            // function
            &this->mFunctionList,
            &this->mFunctionNamesOnly
        };
        return params;
    }

    // Global options

    openvdb::ax::cli::Param<openvdb::ax::VDB_AX_MODE> mMode =
        openvdb::ax::cli::ParamBuilder<openvdb::ax::VDB_AX_MODE>()
            .addOpt("execute")
            .addOpt("analyze")
            .addOpt("functions")
            .setCB([](openvdb::ax::VDB_AX_MODE& v, const char* arg) {
                const std::string s(arg);
                if (s == "analyze")        v = openvdb::ax::VDB_AX_MODE::Analyze;
                else if (s == "functions") v = openvdb::ax::VDB_AX_MODE::Functions;
                else if (s == "execute")   v = openvdb::ax::VDB_AX_MODE::Execute;
                else {
                    v = openvdb::ax::VDB_AX_MODE::Default;
                    sOldInputFile = arg;
                    sOldInputInit = true;
                }
            })
            .get();

    /// @deprecated
    openvdb::ax::cli::Param<std::string> mOldInputFile =
        openvdb::ax::cli::ParamBuilder<std::string>()
            .addOpt("input.vdb")
            .get();

    /// @deprecated
    openvdb::ax::cli::Param<std::string> mOldOutputFile =
        openvdb::ax::cli::ParamBuilder<std::string>()
            .addOpt("output.vdb")
            .get();

    openvdb::ax::cli::Param<bool> mVerbose =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("-v")
            .addOpt("--verbose")
            .setDoc("verbose (print timing and diagnostics).")
            .get();

    openvdb::ax::cli::Param<bool> mHelp =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("-h")
            .addOpt("--help")
            .addOpt("-help")
            .setDoc("print help and exit (use [command] --help for more information).")
            .get();

    // Execute options

    openvdb::ax::cli::Param<std::vector<std::string>> mInputVDBFiles =
        openvdb::ax::cli::ParamBuilder<std::vector<std::string>>()
            .addOpt("-i [file.vdb]")
            .setDoc("append an input vdb file to be read.")
            .setCB([](std::vector<std::string>& v, const char* arg) {
                v.emplace_back(arg);
            })
            .get();

    openvdb::ax::cli::Param<std::unique_ptr<std::string>> mInputCode =
        openvdb::ax::cli::ParamBuilder<std::unique_ptr<std::string>>()
            .addOpt("-s [code]")
            .addOpt("-f [file]")
            .setDoc("input code to execute as a string or from a file. Only the last file "
                "is used")
            .setCB([](std::unique_ptr<std::string>& v, const char* arg, const uint32_t idx) {
                if (v) OPENVDB_LOG_WARN("multiple code snippets provided, only using last input.");
                if (idx == 0) v.reset(new std::string(arg));
                else {
                    OPENVDB_ASSERT(idx == 1);
                    v.reset(new std::string());
                    loadSnippetFile(arg, *v);
                }
            })
            .get();

    openvdb::ax::cli::Param<std::string> mOutputVDBFile =
        openvdb::ax::cli::ParamBuilder<std::string>()
            .addOpt("-o [file.vdb]")
            .setDoc("write the result to a given vdb file. Note that providing the same "
                "file-path to both in/out arguments will overwrite the file. If no output "
                "is provided, the input will be processed but nothing will be written to "
                "disk (this is useful for testing the success status of code).")
            .get();

    // Compilation options

    openvdb::ax::cli::Param<openvdb::ax::CompilerOptions::OptLevel> mOptLevel =
        openvdb::ax::cli::ParamBuilder<openvdb::ax::CompilerOptions::OptLevel>()
            .addOpt("--opt [NONE|O0|O1|O2|Os|Oz|O3]")
            .setDoc("compilation optimization level (Default: 03). [03] ensures the most "
                "vigorus optimization passes are enabled. This should very rarely be changed "
                "but is useful for identifying issues with particular optimization passes.")
            .setCB([](openvdb::ax::CompilerOptions::OptLevel& v, const char* arg)
            {
                if (arg[0] == 'O') {
                    if      (arg[1] == '0')  v = openvdb::ax::CompilerOptions::OptLevel::O0;
                    else if (arg[1] == '1')  v = openvdb::ax::CompilerOptions::OptLevel::O1;
                    else if (arg[1] == '2')  v = openvdb::ax::CompilerOptions::OptLevel::O2;
                    else if (arg[1] == 's')  v = openvdb::ax::CompilerOptions::OptLevel::Os;
                    else if (arg[1] == 'z')  v = openvdb::ax::CompilerOptions::OptLevel::Oz;
                    else if (arg[1] == '3')  v = openvdb::ax::CompilerOptions::OptLevel::O3;
                }
                else if (std::string(arg) == "NONE") v = openvdb::ax::CompilerOptions::OptLevel::NONE;
                else {
                    fatal("invalid option given for --opt level");
                }
            })
            .setDefault(openvdb::ax::CompilerOptions::OptLevel::O3)
            .get();

    openvdb::ax::cli::Param<size_t> mThreads =
        openvdb::ax::cli::ParamBuilder<size_t>()
            .addOpt("--threads [n]")
            .setDoc("number of threads to use, 0 uses all available (Default: 0).")
            .setDefault(0)
            .get();

    openvdb::ax::cli::Param<bool> mWarningsAsErrors =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--werror")
            .setDoc("warnings as errors.")
            .get();

    openvdb::ax::cli::Param<size_t> mMaxErrors =
        openvdb::ax::cli::ParamBuilder<size_t>()
            .addOpt("--max-errors [n]")
            .setDoc("maximum error messages, 0 allows all error messages (Default: 0).")
            .setDefault(0)
            .get();

    openvdb::ax::cli::Param<bool> mCopyFileMeta =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--copy-file-metadata")
            .setDoc("copy the file level metadata of the first input to the output.")
            .get();

    // Analyze options

    openvdb::ax::cli::Param<bool> mPrintAST =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--ast-print")
            .setDoc("print the generated abstract syntax tree.")
            .get();

    openvdb::ax::cli::Param<bool> mReprint =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--re-print")
            .setDoc("re-interpret print of the code post ast traversal.")
            .get();

    openvdb::ax::cli::Param<bool> mAttribRegPrint =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--reg-print")
            .setDoc("print the attribute registry (name, types, access, dependencies).")
            .get();

    openvdb::ax::cli::Param<openvdb::ax::VDB_AX_COMPILATION> mCompileFor =
        openvdb::ax::cli::ParamBuilder<openvdb::ax::VDB_AX_COMPILATION>()
            .addOpt("--try-compile <points | volumes>")
            .setDoc("attempt compilation for points, volumes or both if no option is provided.")
            .setCB([](openvdb::ax::VDB_AX_COMPILATION& v) {
                v = openvdb::ax::VDB_AX_COMPILATION::All;
            })
            .setCB([](openvdb::ax::VDB_AX_COMPILATION& v, const char* arg) {
                const std::string s(arg);
                if (s == "points")        v = openvdb::ax::VDB_AX_COMPILATION::Points;
                else if (s == "volumes")  v = openvdb::ax::VDB_AX_COMPILATION::Volumes;
                else fatal("invalid option given for --try-compile level.");
            })
            .get();

    // Function Options

    openvdb::ax::cli::Param<std::pair<bool, std::string>> mFunctionList =
        openvdb::ax::cli::ParamBuilder<std::pair<bool, std::string>>()
            .addOpt("--list <filter-name>")
            .setDoc("list functions, their documentation and their signatures. "
                    "optionally only list those whose name includes a provided string.")
            .setCB([](std::pair<bool, std::string>& v) {
                v.first = true;
            })
            .setCB([](std::pair<bool, std::string>& v, const char* arg) {
                v.first = true;
                v.second = std::string(arg);
            })
            .get();

    openvdb::ax::cli::Param<bool> mFunctionNamesOnly =
        openvdb::ax::cli::ParamBuilder<bool>()
            .addOpt("--list-names")
            .setDoc("list all available functions names only.")
            .get();
};

template <typename Cb>
auto operator<<(std::ostream& os, const Cb& cb) -> decltype(cb(os)) { return cb(os); }

auto usage_execute(const ProgOptions& opts, const bool verbose)
{
    return [&, verbose](std::ostream& os) -> std::ostream& {
        os <<
        "[execute] read/process/write VDB file/streams (default command):\n";
        if (verbose) {
            os <<
            "\n" <<
            "    This command takes a list of positional arguments which represent VDB files\n" <<
            "    and runs AX code across their voxel or point values. Unique kernels are built\n" <<
            "    and run separately for volumes and point grids. All grids are written to the\n" <<
            "    same output file:\n" <<
            "\n" <<
            "         " << gProgName << " -i density.vdb -s \"@density += 1;\" -o out.vdb       // increment values by 1\n" <<
            "         " << gProgName << " -i a.vdb -i b.vdb -i c.vdb -s \"@c = @a + @b;\" -o out.vdb  // combine a,b into c\n" <<
            "         " << gProgName << " -i points.vdb -s \"@P += v@v * 2;\" -o out.vdb        // move points based on a vector attribute\n" <<
            "\n" <<
            "    For more examples and help with syntax, see the AX documentation:\n" <<
            "      https://www.openvdb.org/documentation/doxygen/openvdbax.html\n" <<
            "\n";
        }

        openvdb::ax::cli::usage(os, opts.mInputVDBFiles.opts(), opts.mInputVDBFiles.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mInputCode.opts(), opts.mInputCode.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mOutputVDBFile.opts(), opts.mOutputVDBFile.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mOptLevel.opts(), opts.mOptLevel.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mThreads.opts(), opts.mThreads.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mWarningsAsErrors.opts(), opts.mWarningsAsErrors.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mMaxErrors.opts(), opts.mMaxErrors.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mCopyFileMeta.opts(), opts.mCopyFileMeta.doc(), verbose);

        os << "  Volumes:\n";
        openvdb::ax::VolumeExecutable::CLI::usage(os, verbose);
        os << "  Points:\n";
        openvdb::ax::PointExecutable::CLI::usage(os, verbose);

        return os;
    };
}

auto usage_analyze(const ProgOptions& opts, const bool verbose)
{
    return [&, verbose](std::ostream& os) -> std::ostream& {
        os <<
        "[analyze] parse code and run analysis:\n";
        if (verbose) {
            os <<
            "\n" <<
            "    Examples:\n" <<
            "         " << gProgName << " analyze -s \"@density += 1;\" --try-compile points  // compile code for points\n" <<
            "\n";
        }

        openvdb::ax::cli::usage(os, opts.mPrintAST.opts(), opts.mPrintAST.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mReprint.opts(), opts.mReprint.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mAttribRegPrint.opts(), opts.mAttribRegPrint.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mCompileFor.opts(), opts.mCompileFor.doc(), verbose);

        return os;
    };
}

auto usage_functions(const ProgOptions& opts, const bool verbose)
{
    return [&, verbose](std::ostream& os) -> std::ostream& {
        os <<
        "[functions] query available function information:\n";
        if (verbose) {
            os <<
            "\n" <<
            "    Examples:\n" <<
            "         " << gProgName << " functions --list log  // print functions with 'log' in the name\n" <<
            "\n";
        }

        openvdb::ax::cli::usage(os, opts.mFunctionList.opts(), opts.mFunctionList.doc(), verbose);
        openvdb::ax::cli::usage(os, opts.mFunctionNamesOnly.opts(), opts.mFunctionNamesOnly.doc(), verbose);

        return os;
    };
}

void shortManPage [[noreturn]] (const ProgOptions& opts, int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
    "usage: " << gProgName << " [command] [--help|-h] [-v] [<args>]\n" <<
    '\n' <<
    "CLI utility for processing OpenVDB data using AX.\n" <<
    "Available [command] modes are: [execute|analyze|functions] (Default: execute).\n";
    openvdb::ax::cli::usage(std::cerr, opts.mHelp.opts(), opts.mHelp.doc(), false);
    openvdb::ax::cli::usage(std::cerr, opts.mVerbose.opts(), opts.mVerbose.doc(), false);
    std::cerr << '\n';
    std::cerr
    << usage_execute(opts, false) << '\n'
    << usage_analyze(opts, false) << '\n'
    << usage_functions(opts, false) << '\n'
    <<
    "Email bug reports, questions, discussions to <openvdb-dev@lists.aswf.io>\n" <<
    "and/or open issues at https://github.com/AcademySoftwareFoundation/openvdb.\n";
    std::exit(exitStatus);
}

void usage [[noreturn]] (const ProgOptions& opts, int exitStatus = EXIT_FAILURE)
{
    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Default)   shortManPage(opts, exitStatus);
    std::cerr << "usage: " << gProgName << " [";
    openvdb::ax::cli::ParamToStream(std::cerr, opts.mMode.get());
    std::cerr << "] [<args>]\n";
    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute)   std::cerr << usage_execute(opts, true) << std::endl;
    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Analyze)   std::cerr << usage_analyze(opts, true) << std::endl;
    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Functions) std::cerr << usage_functions(opts, true) << std::endl;
    std::exit(exitStatus);
}

/// @brief RAII obj for openvdb/ax initialization
struct ScopedInitialize
{
    ScopedInitialize(int argc, char *argv[]) {
        openvdb::logging::initialize(argc, argv);
        openvdb::initialize();
    }

    ~ScopedInitialize() {
        if (openvdb::ax::isInitialized()) {
            openvdb::ax::uninitialize();
        }
        openvdb::uninitialize();
    }

    inline void initializeCompiler() const { openvdb::ax::initialize(); }
    inline bool isInitialized() const { return openvdb::ax::isInitialized(); }
};

/// @brief  Print detailed information about OpenVDB AX functions by iterating
///   over them.
void printFunctions(const bool namesOnly,
                const std::string& search,
                std::ostream& os)
{
    static const size_t maxHelpTextWidth = 100;

    openvdb::ax::FunctionOptions opts;
    opts.mLazyFunctions = false;
    const openvdb::ax::codegen::FunctionRegistry::UniquePtr registry =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // convert to ordered map for alphabetical sorting
    // only include non internal functions and apply any search
    // criteria

    std::map<std::string, const openvdb::ax::codegen::FunctionGroup*> functionMap;
    for (const auto& iter : registry->map()) {
        if (iter.second.isInternal()) continue;
        if (!search.empty() && iter.first.find(search) == std::string::npos) {
            continue;
        }
        functionMap[iter.first] = iter.second.function();
    }

    if (functionMap.empty()) return;

    if (namesOnly) {

        const size_t size = functionMap.size();
        size_t pos = 0, count = 0;

        auto iter = functionMap.cbegin();
        for (; iter != functionMap.cend(); ++iter) {
            if (count == size - 1) break;
            const std::string& name = iter->first;
            if (count != 0) {
                if (pos > maxHelpTextWidth) {
                    os << '\n';
                    pos = 0;
                }
                else {
                    os << ' ';
                    ++pos;
                }
            }
            pos += name.size() + 1;
            os << name << ',';
            ++count;
        }

        os << iter->first << '\n';
    }
    else {

        llvm::LLVMContext C;

        for (const auto& iter : functionMap) {
            const openvdb::ax::codegen::FunctionGroup* const function = iter.second;
            const char* cdocs = function->doc();
            if (!cdocs || cdocs[0] == '\0') {
                cdocs = "<No documentation exists for this function>";
            }

            // do some basic formatting on the help text

            std::string docs(cdocs);
            size_t pos = maxHelpTextWidth;
            while (pos < docs.size()) {
                while (docs[pos] != ' ' && pos != 0) --pos;
                if (pos == 0) break;
                docs.insert(pos, "\n|  ");
                pos += maxHelpTextWidth;
            }

            os << iter.first << '\n' << '|' << '\n';
            os << "| - " << docs << '\n' << '|' << '\n';

            const auto& list = function->list();
            for (const openvdb::ax::codegen::Function::Ptr& decl : list) {
                os << "|  - ";
                decl->print(C, os);
                os << '\n';
            }
            os << '\n';
        }
    }
}

/// @brief  Parse cli parameters and return a tuple of program options and exe
///   cli settings. This is primarily required as cli::init throws CLIErrors
///   but the executable CLI interfaces only allow CLI options to be moved,
///   not copied. Once the CLI parameters become part of the public interface
///   (if ever) then this can be simplified.
inline auto parseCliComponents(int argc, char *argv[])
{
    ProgOptions opts;
    if (argc == 1) shortManPage(opts);

    // skip program argument
    argc-=1;
    ++argv;

    std::unique_ptr<bool[]> flags(new bool[argc]);
    std::fill(flags.get(), flags.get()+argc, false);

    int32_t optionalArgc = argc;
    const char** optionalArgv = const_cast<const char**>(argv);
    bool* optionalFlags = flags.get();
    // skip positional arguments (if any) for executables
    while (optionalArgc && optionalArgv[0][0] != '-') {
        optionalArgc-=1;
        ++optionalArgv;
        ++optionalFlags;
    }

    try {
        openvdb::ax::cli::init(argc, const_cast<const char**>(argv), opts.positional(), opts.optional(), flags.get());
        auto volumecli = openvdb::ax::VolumeExecutable::CLI::create(optionalArgc, optionalArgv, optionalFlags);
        auto pointcli = openvdb::ax::PointExecutable::CLI::create(optionalArgc, optionalArgv, optionalFlags);

        for (int i = 0; i < argc; ++i) {
            if (flags[i]) continue;
            OPENVDB_LOG_FATAL("\"" << argv[i] << "\" is not a valid option");
            shortManPage(opts);
        }

        return std::make_tuple(std::move(opts), std::move(volumecli), std::move(pointcli));
    }
    catch (const openvdb::CLIError& e) {
        fatal(e.what());
    }
}

int
main(int argc, char *argv[])
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE
    gProgName = argv[0];
    const char* ptr = ::strrchr(gProgName, '/');
    if (ptr != nullptr) gProgName = ptr + 1;
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE

    // create program options and exe ci settings
    auto cliComponents = parseCliComponents(argc, argv);
    ProgOptions opts = std::move(std::get<0>(cliComponents));

    openvdb::util::CpuTimer timer;
    auto getTime = [&timer]() -> std::string {
        const double msec = timer.milliseconds();
        std::ostringstream os;
        openvdb::util::printTime(os, msec, "", "", 1, 1, 0);
        return os.str();
    };

    auto& os = std::cerr;
#define axlog(message) { if (opts.mVerbose) os << message; }
#define axtimer() timer.restart()
#define axtime() getTime()

    // Handle program options
    if (opts.mHelp.get()) {
        usage(opts, EXIT_SUCCESS);
    }

    if (sOldInputInit || opts.mOldInputFile.isInit() || opts.mOldOutputFile.isInit()) {
        if (opts.mInputVDBFiles.isInit() || opts.mOutputVDBFile.isInit()) {
            fatal("position arguments [input.vdb <output.vdb>] are deprecated and "
                "cannot be used with -i and -o");
        }
        else {
            OPENVDB_LOG_WARN("position arguments [input.vdb <output.vdb>] are deprecated. use -i and -o");
            if (sOldInputInit)                opts.mInputVDBFiles.init(sOldInputFile.c_str());
            if (opts.mOldInputFile.isInit())  opts.mInputVDBFiles.init(opts.mOldInputFile.get().c_str());
            if (opts.mOldOutputFile.isInit()) opts.mOutputVDBFile.init(opts.mOutputVDBFile.get().c_str());
        }
    }

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Default) {
        opts.mMode.set(openvdb::ax::VDB_AX_MODE::Execute);
    }

    // figure out if we need to init ax/llvm. If we're printing function
    // signatures we need to init an llvm context
    bool initCompile = (opts.mFunctionList.get().first) ||
        (opts.mCompileFor.get() != openvdb::ax::VDB_AX_COMPILATION::None);

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute) {
        initCompile = true;
        if (opts.mInputVDBFiles.get().empty()) {
            fatal("no vdb files have been provided");
        }
    }
    else if (!opts.mInputVDBFiles.get().empty()) {
        std::ostringstream tmp;
        openvdb::ax::cli::ParamToStream(tmp, opts.mMode.get());
        fatal(tmp.str() + " does not take input vdb files");
    }

    if (opts.mMode.get() != openvdb::ax::VDB_AX_MODE::Functions) {
        if (!opts.mInputCode.get()) {
            fatal("expected at least one AX file or a code snippet");
        }
    }

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute) {
        if (opts.mOutputVDBFile.get().empty()) {
            OPENVDB_LOG_WARN("no output VDB File specified - nothing will be written to disk");
        }
    }

    if (opts.mVerbose.get()) {
        axlog("OpenVDB AX " << openvdb::getLibraryVersionString() << '\n');
        axlog("----------------\n");
        axlog("Inputs\n");

        std::ostringstream tmp;
        openvdb::ax::cli::ParamToStream(tmp, opts.mMode.get());
        axlog("  mode    : " << tmp.str());
        if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Analyze) {
            axlog(" (");
            if (opts.mPrintAST.get())       axlog("|ast out");
            if (opts.mReprint.get())        axlog("|reprint out");
            if (opts.mAttribRegPrint.get()) axlog("|registry out");
            if (initCompile)                axlog("|compilation");
            axlog("|)");
        }
        axlog('\n');

        if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute) {
            axlog("  vdb in  : \"");
            for (const auto& in : opts.mInputVDBFiles.get()) {
                const bool sep = (&in != &opts.mInputVDBFiles.get().back());
                axlog(in << (sep ? ", " : ""));
            }
            axlog("\"\n");
            axlog("  vdb out : \"" << opts.mOutputVDBFile.get() << "\"\n");
        }
        if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute ||
            opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Analyze) {
            axlog("  ax code : ");
            if (opts.mInputCode.get() && !opts.mInputCode.get()->empty()) {
                const bool containsnl =
                    opts.mInputCode.get()->find('\n') != std::string::npos;
                if (containsnl) axlog("\n    ");

                // indent output
                const char* c = opts.mInputCode.get()->c_str();
                while (*c != '\0') {
                    axlog(*c);
                    if (*c == '\n') axlog("    ");
                    ++c;
                }
            }
            else {
                axlog("\"\"");
            }
            axlog('\n');
            axlog('\n');
        }
        axlog(std::flush);
    }

    axtimer();
    axlog("[INFO] Initializing OpenVDB" << std::flush);
    ScopedInitialize initializer(argc, argv);
    axlog(": " << axtime() << '\n');

    std::unique_ptr<tbb::global_control> control;
    if (opts.mThreads.get() > 0) {
        axlog("[INFO] Initializing thread usage [" << opts.mThreads.get() << "]\n" << std::flush);
        control.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, opts.mThreads.get()));
    }

    openvdb::GridPtrVec grids;
    openvdb::MetaMap::Ptr meta;

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute) {
        // read vdb file data for
        axlog("[INFO] Reading VDB data"
            << (openvdb::io::Archive::isDelayedLoadingEnabled() ?
                " (delay-load)" : "") << '\n');
        for (const auto& filename : opts.mInputVDBFiles.get()) {
            openvdb::io::File file(filename);
            try {
                axlog("[INFO] | \"" << filename << "\"");
                axtimer();
                file.open();
                auto in = file.getGrids();
                grids.insert(grids.end(), in->begin(), in->end());
                // choose the first files metadata
                if (opts.mCopyFileMeta.get() && !meta) meta = file.getMetadata();
                file.close();
                axlog(": " << axtime() << '\n');
            } catch (openvdb::Exception& e) {
                axlog('\n');
                OPENVDB_LOG_ERROR(e.what() << " (" << filename << ")");
                return EXIT_FAILURE;
            }
        }
    }

    if (initCompile) {
        axtimer();
        axlog("[INFO] Initializing AX/LLVM" << std::flush);
        initializer.initializeCompiler();
        axlog(": " << axtime() << '\n');
    }

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Functions) {
        const bool printing =
            opts.mFunctionList.get().first ||
            opts.mFunctionNamesOnly.get();

        if (printing) {
            axlog("Querying available functions\n" << std::flush);
            OPENVDB_ASSERT(opts.mFunctionNamesOnly.get() || initializer.isInitialized());
            printFunctions(opts.mFunctionNamesOnly.get(),
                opts.mFunctionList.get().second,
                std::cout);
            return EXIT_SUCCESS;
        }
        else {
            fatal("vdb_ax functions requires a valid option");
        }
    }

    // set up logger

    openvdb::ax::Logger
        logs([](const std::string& msg) { std::cerr << msg << std::endl; },
             [](const std::string& msg) { std::cerr << msg << std::endl; });
    logs.setMaxErrors(opts.mMaxErrors.get());
    logs.setWarningsAsErrors(opts.mWarningsAsErrors.get());
    logs.setPrintLines(true);
    logs.setNumberedOutput(true);

    // parse

    axtimer();
    axlog("[INFO] Parsing input code" << std::flush);

    const openvdb::ax::ast::Tree::ConstPtr syntaxTree =
        openvdb::ax::ast::parse(opts.mInputCode.get()->c_str(), logs);
        axlog(": " << axtime() << '\n');
    if (!syntaxTree) {
        return EXIT_FAILURE;
    }

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Analyze) {
        axlog("[INFO] Running analysis options\n" << std::flush);
        if (opts.mPrintAST.get()) {
            axlog("[INFO] | Printing AST\n" << std::flush);
            openvdb::ax::ast::print(*syntaxTree, true, std::cout);
        }
        if (opts.mReprint.get()) {
            axlog("[INFO] | Reprinting code\n" << std::flush);
            openvdb::ax::ast::reprint(*syntaxTree, std::cout);
        }
        if (opts.mAttribRegPrint.get()) {
            axlog("[INFO] | Printing Attribute Registry\n" << std::flush);
            const openvdb::ax::AttributeRegistry::ConstPtr reg =
                openvdb::ax::AttributeRegistry::create(*syntaxTree);
            reg->print(std::cout);
            std::cout << std::flush;
        }

        if (!initCompile) {
            return EXIT_SUCCESS;
        }
    }

    OPENVDB_ASSERT(initCompile);

    std::ostringstream tmp;
    openvdb::ax::cli::ParamToStream(tmp, opts.mOptLevel.get());

    axtimer();
    axlog("[INFO] Creating Compiler\n");
    axlog("[INFO] | Optimization Level [" << tmp.str() << "]\n" << std::flush);

    openvdb::ax::CompilerOptions compOpts;
    compOpts.mOptLevel = opts.mOptLevel.get();

    openvdb::ax::Compiler::Ptr compiler =
        openvdb::ax::Compiler::create(compOpts);
    openvdb::ax::CustomData::Ptr customData =
        openvdb::ax::CustomData::create();
    axlog("[INFO] | " << axtime() << '\n' << std::flush);

    // Check what we need to compile for if performing execution

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Execute) {
        bool points = false;
        bool volumes = false;
        for (auto grid : grids) {
            points |= grid->isType<openvdb::points::PointDataGrid>();
            volumes |= !points;
            if (points && volumes) break;
        }
        if (points && volumes) opts.mCompileFor.set(openvdb::ax::VDB_AX_COMPILATION::All);
        else if (points)       opts.mCompileFor.set(openvdb::ax::VDB_AX_COMPILATION::Points);
        else if (volumes)      opts.mCompileFor.set(openvdb::ax::VDB_AX_COMPILATION::Volumes);
    }

    if (opts.mMode.get() == openvdb::ax::VDB_AX_MODE::Analyze) {

        bool psuccess = true;

        if (opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::All ||
            opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::Points) {
            axtimer();
            axlog("[INFO] Compiling for VDB Points\n" << std::flush);
            try {
                compiler->compile<openvdb::ax::PointExecutable>(*syntaxTree, logs, customData);
                if (logs.hasError()) {
                    axlog("[INFO] Compilation error(s)!\n");
                    psuccess = false;
                }
             }
            catch (std::exception& e) {
                psuccess = false;
                axlog("[INFO] Fatal error!\n");
                OPENVDB_LOG_ERROR(e.what());
            }
            const bool hasWarning = logs.hasWarning();
            if (psuccess) {
                axlog("[INFO] | Compilation successful");
                if (hasWarning) axlog(" with warning(s)");
                axlog('\n');
            }
            axlog("[INFO] | " << axtime() << '\n' << std::flush);
        }

        bool vsuccess = true;

        if (opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::All ||
            opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::Volumes) {
            axtimer();
            axlog("[INFO] Compiling for VDB Volumes\n" << std::flush);
            try {
                compiler->compile<openvdb::ax::VolumeExecutable>(*syntaxTree, logs, customData);
                if (logs.hasError()) {
                    axlog("[INFO] Compilation error(s)!\n");
                    vsuccess = false;
                }
            }
            catch (std::exception& e) {
                vsuccess = false;
                axlog("[INFO] Fatal error!\n");
                OPENVDB_LOG_ERROR(e.what());
            }
            const bool hasWarning = logs.hasWarning();
            if (vsuccess) {
                axlog("[INFO] | Compilation successful");
                if (hasWarning) axlog(" with warning(s)");
                axlog('\n');
            }
            axlog("[INFO] | " << axtime() << '\n' << std::flush);
        }

        return ((vsuccess && psuccess) ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    // Execute points

    if (opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::All ||
        opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::Points) {

        axlog("[INFO] VDB PointDataGrids Found\n" << std::flush);

        openvdb::ax::PointExecutable::Ptr pointExe;

        axtimer();
        try {
            axlog("[INFO] Compiling for VDB Points\n" << std::flush);
            pointExe = compiler->compile<openvdb::ax::PointExecutable>(*syntaxTree, logs, customData);
        } catch (std::exception& e) {
            OPENVDB_LOG_FATAL("Fatal error!\nErrors:\n" << e.what());
            return EXIT_FAILURE;
        }

        if (pointExe) {
            axlog("[INFO] | Compilation successful");
            if (logs.hasWarning()) axlog(" with warning(s)");
            axlog('\n');
        }
        else {
            if (logs.hasError()) {
                axlog("[INFO] Compilation error(s)!\n");
            }
            return EXIT_FAILURE;
        }
        axlog("[INFO] | " << axtime() << '\n' << std::flush);

        size_t total = 0, count = 1;
        if (opts.mVerbose.get()) {
            for (auto grid : grids) {
                if (!grid->isType<openvdb::points::PointDataGrid>()) continue;
                ++total;
            }
        }

        pointExe->setSettingsFromCLI(std::move(std::get<2>(cliComponents)));

        for (auto grid :grids) {
            if (!grid->isType<openvdb::points::PointDataGrid>()) continue;
            openvdb::points::PointDataGrid::Ptr points =
                openvdb::gridPtrCast<openvdb::points::PointDataGrid>(grid);
            axtimer();
            axlog("[INFO] Executing on \"" << points->getName() << "\" "
                  << count << " of " << total << '\n');

            axlog("[INFO] | Input attribute layout:\n")
            if (const auto leaf = points->tree().cbeginLeaf()) {
                const auto& desc = leaf->attributeSet().descriptor();
                for (const auto& entry : desc.map()) {
                    axlog("  " << entry.first << ":" << desc.valueType(entry.second) << '\n');
                }
                for (const auto& entry : desc.groupMap()) {
                    axlog("  " << entry.first << ":group\n");
                }
            }
            else {
                axlog("  <empty>");
            }
            axlog(std::flush);

            ++count;

            try {
                pointExe->execute(*points);
            }
            catch (std::exception& e) {
                OPENVDB_LOG_FATAL("Execution error!\nErrors:\n" << e.what());
                return EXIT_FAILURE;
            }

            axlog("[INFO] | Execution success.\n");
            axlog("[INFO] | " << axtime() << '\n' << std::flush);
        }
    }

    // Execute volumes

    if (opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::All ||
        opts.mCompileFor.get() == openvdb::ax::VDB_AX_COMPILATION::Volumes) {

        axlog("[INFO] VDB Volumes Found\n" << std::flush);

        openvdb::ax::VolumeExecutable::Ptr volumeExe;
        try {
            axlog("[INFO] Compiling for VDB Points\n" << std::flush);
            volumeExe = compiler->compile<openvdb::ax::VolumeExecutable>(*syntaxTree, logs, customData);
        } catch (std::exception& e) {
            OPENVDB_LOG_FATAL("Fatal error!\nErrors:\n" << e.what());
            return EXIT_FAILURE;
        }

        if (volumeExe) {
            axlog("[INFO] | Compilation successful");
            if (logs.hasWarning()) axlog(" with warning(s)");
            axlog('\n');
        }
        else {
            if (logs.hasError()) {
                axlog("[INFO] Compilation error(s)!\n");
            }
            return EXIT_FAILURE;
        }
        axlog("[INFO] | " << axtime() << '\n' << std::flush);

        if (opts.mVerbose.get()) {
            std::vector<const std::string*> names;
            axlog("[INFO] Executing using:\n");
            for (auto grid : grids) {
                if (grid->isType<openvdb::points::PointDataGrid>()) continue;
                axlog("  " << grid->getName() << '\n');
                axlog("    " << grid->valueType() << '\n');
                axlog("    " << grid->gridClassToString(grid->getGridClass()) << '\n');
            }
            axlog(std::flush);
        }

        volumeExe->setSettingsFromCLI(std::move(std::get<1>(cliComponents)));

        try { volumeExe->execute(grids); }
        catch (std::exception& e) {
            OPENVDB_LOG_FATAL("Execution error!\nErrors:\n" << e.what());
            return EXIT_FAILURE;
        }

        axlog("[INFO] | Execution success.\n");
        axlog("[INFO] | " << axtime() << '\n' << std::flush);
    }

    if (!opts.mOutputVDBFile.get().empty()) {
        axtimer();
        axlog("[INFO] Writing results" << std::flush);
        openvdb::io::File out(opts.mOutputVDBFile.get());
        try {
            if (meta) out.write(grids, *meta);
            else      out.write(grids);
        } catch (openvdb::Exception& e) {
            OPENVDB_LOG_ERROR(e.what() << " (" << out.filename() << ")");
            return EXIT_FAILURE;
        }
        axlog("[INFO] | " << axtime() << '\n' << std::flush);
    }

    return EXIT_SUCCESS;
}

