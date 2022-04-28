// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file cmd/openvdb_ax.cc
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  The command line vdb_ax binary which provides tools to
///   run and analyze AX code.
///

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
#include <openvdb/points/PointDelete.h>

// tbb/task_scheduler_init.h was removed in TBB 2021. The best construct to swap
// to is tbb/global_control (for executables). global_control was only officially
// added in TBB 2019U4 but exists in 2018 as a preview feature. To avoid more
// compile time branching (as we still support 2018), we use it in 2018 too by
// enabling the below define.
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

const char* gProgName = "";

void fatal [[noreturn]] (const char* msg = nullptr)
{
    if (msg) OPENVDB_LOG_FATAL(msg << ". See '" << gProgName << " --help'");
    std::exit(EXIT_FAILURE);
}

void fatal [[noreturn]] (const std::string msg) { fatal(msg.c_str()); }

struct ProgOptions
{
    enum Mode { Default, Execute, Analyze, Functions };
    enum Compilation { All, Points, Volumes };

    Mode mMode = Default;
    int32_t threads = 0;

    // Compilation options
    size_t mMaxErrors = 0;
    bool mWarningsAsErrors = false;

    // Execute options
    std::unique_ptr<std::string> mInputCode = nullptr;
    std::vector<std::string> mInputVDBFiles = {};
    std::string mOutputVDBFile = "";
    bool mCopyFileMeta = false;
    bool mVerbose = false;
    openvdb::ax::CompilerOptions::OptLevel mOptLevel =
        openvdb::ax::CompilerOptions::OptLevel::O3;

    // Analyze options
    bool mPrintAST = false;
    bool mReprint = false;
    bool mAttribRegPrint = false;
    bool mInitCompile = false;
    Compilation mCompileFor = All;

    // Function Options
    bool mFunctionList = false;
    bool mFunctionNamesOnly = false;
    std::string mFunctionSearch = "";
};

inline std::string modeString(const ProgOptions::Mode mode)
{
    switch (mode) {
        case ProgOptions::Execute   : return "execute";
        case ProgOptions::Analyze   : return "analyze";
        case ProgOptions::Functions : return "functions";
        default : return "";
    }
}

ProgOptions::Compilation
tryCompileStringToCompilation(const std::string& str)
{
    if (str == "points")   return ProgOptions::Points;
    if (str == "volumes")  return ProgOptions::Volumes;
    fatal("invalid option given for --try-compile level.");
}

openvdb::ax::CompilerOptions::OptLevel
optStringToLevel(const std::string& str)
{
    if (str == "NONE") return openvdb::ax::CompilerOptions::OptLevel::NONE;
    if (str == "O0")   return openvdb::ax::CompilerOptions::OptLevel::O0;
    if (str == "O1")   return openvdb::ax::CompilerOptions::OptLevel::O1;
    if (str == "O2")   return openvdb::ax::CompilerOptions::OptLevel::O2;
    if (str == "Os")   return openvdb::ax::CompilerOptions::OptLevel::Os;
    if (str == "Oz")   return openvdb::ax::CompilerOptions::OptLevel::Oz;
    if (str == "O3")   return openvdb::ax::CompilerOptions::OptLevel::O3;
    fatal("invalid option given for --opt level");
}

inline std::string
optLevelToString(const openvdb::ax::CompilerOptions::OptLevel level)
{
    switch (level) {
        case  openvdb::ax::CompilerOptions::OptLevel::NONE : return "NONE";
        case  openvdb::ax::CompilerOptions::OptLevel::O1 : return "O1";
        case  openvdb::ax::CompilerOptions::OptLevel::O2 : return "O2";
        case  openvdb::ax::CompilerOptions::OptLevel::Os : return "Os";
        case  openvdb::ax::CompilerOptions::OptLevel::Oz : return "Oz";
        case  openvdb::ax::CompilerOptions::OptLevel::O3 : return "O3";
        default : return "";
    }
}

template <typename Cb>
auto operator<<(std::ostream& os, const Cb& cb) -> decltype(cb(os)) { return cb(os); }

auto usage_execute(const bool verbose)
{
    return [=](std::ostream& os) -> std::ostream& {
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
            "         " << gProgName << " density.vdb -s \"@density += 1;\" -o out.vdb       // increment values by 1\n" <<
            "         " << gProgName << " a.vdb b.vdb c.vdb -s \"@c = @a + @b;\" -o out.vdb  // combine a,b into c\n" <<
            "         " << gProgName << " points.vdb -s \"@P += v@v * 2;\" -o out.vdb        // move points based on a vector attribute\n" <<
            "\n" <<
            "    For more examples and help with syntax, see the AX documentation:\n" <<
            "      https://academysoftwarefoundation.github.io/openvdb/openvdbax.html\n" <<
            "\n";
        }
        os <<
        "    -i [file.vdb]          append an input vdb file to be read\n"
        "    -s [code], -f [file]   input code to execute as a string or from a file.\n" <<
        "    -o [file.vdb]          write the result to a given vdb file\n" <<
        "    --opt [level]          optimization level [NONE, O0, O1, O2, Os, Oz, O3 (default)]\n" <<
        "    --werror               warnings as errors\n" <<
        "    --max-errors [n]       maximum error messages, 0 (default) allows all error messages\n" <<
        "    --threads [n]          number of threads to use, 0 (default) uses all available.\n" <<
        "    --copy-file-metadata   copy the file level metadata of the first input to the output.\n";
        if (verbose) {
            os << '\n' <<
            "Notes:\n" <<
            "    Providing the same file-path to both in/out arguments will overwrite the\n" <<
            "    file. If no output is provided, the input will be processed but will remain\n" <<
            "    unchanged on disk (this is useful for testing the success status of code).\n";
        }
        return os;
    };
}

auto usage_analyze(const bool verbose)
{
    return [=](std::ostream& os) -> std::ostream& {
        os <<
        "[analyze] parse the provided code and run analysis:\n";
        if (verbose) {
            os <<
            "\n" <<
            "    Examples:\n" <<
            "         " << gProgName << " analyze -s \"@density += 1;\" --try-compile points  // compile code for points\n" <<
            "\n";
        }
        return os <<
        "    -s [code], -f [file]  input code as a string or from a file.\n" <<
        "    --ast-print           print the generated abstract syntax tree\n" <<
        "    --re-print            re-interpret print of the code post ast traversal\n" <<
        "    --reg-print           print the attribute registry (name, types, access, dependencies)\n" <<
        "    --try-compile <points | volumes>\n" <<
        "                          attempt to compile code for points, volumes or both if no\n" <<
        "                          option is provided, reporting any failures or success.\n";
    };
}

auto usage_functions(const bool verbose)
{
    return [=](std::ostream& os) -> std::ostream& {
        os <<
        "[functions] query available function information:\n";
        if (verbose) {
            os <<
            "\n" <<
            "    Examples:\n" <<
            "         " << gProgName << " functions --list log  // print functions with 'log' in the name\n" <<
            "\n";
        }
        return os <<
        "    --list <name>  list all functions, their documentation and their signatures.\n" <<
        "                   optionally only list those whose name includes a provided string.\n" <<
        "    --list-names   list all available functions names only\n";
    };
}

void usage [[noreturn]] (int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
    "usage: " << gProgName << " [command] [--help|-h] [-v] [<args>]\n" <<
    '\n' <<
    "CLI utility for processing OpenVDB data using AX. Various commands are supported.\n" <<
    "    -h, --help  print help and exit. [command] -h prints extra information.\n" <<
    "    -v          verbose (print timing and diagnostics)\n" <<
    '\n'
    << usage_execute(false) <<
    '\n'
    << usage_analyze(false) <<
    '\n'
    << usage_functions(false) <<
    '\n' <<
    "Email bug reports, questions, discussions to <openvdb-dev@lists.aswf.io>\n" <<
    "and/or open issues at https://github.com/AcademySoftwareFoundation/openvdb.\n";

    std::exit(exitStatus);
}

void usage [[noreturn]] (const ProgOptions::Mode mode, int exitStatus = EXIT_FAILURE)
{
    if (mode == ProgOptions::Mode::Default)   usage(exitStatus);
    std::cerr << "usage: " << gProgName << " [" << modeString(mode) << "] [<args>]\n";
    if (mode == ProgOptions::Mode::Execute)   std::cerr << usage_execute(true) << std::endl;
    if (mode == ProgOptions::Mode::Analyze)   std::cerr << usage_analyze(true) << std::endl;
    if (mode == ProgOptions::Mode::Functions) std::cerr << usage_functions(true) << std::endl;
    std::exit(exitStatus);
}

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

struct OptParse
{
    int argc;
    char** argv;

    OptParse(int argc_, char* argv_[]): argc(argc_), argv(argv_) {}

    bool check(int idx, const std::string& name, int numArgs = 1) const
    {
        if (argv[idx] == name) {
            if (idx + numArgs >= argc) {
                OPENVDB_LOG_FATAL("option " << name << " requires "
                    << numArgs << " argument" << (numArgs == 1 ? "" : "s"));
                fatal();
            }
            return true;
        }
        return false;
    }
};

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

int
main(int argc, char *argv[])
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE
    gProgName = argv[0];
    const char* ptr = ::strrchr(gProgName, '/');
    if (ptr != nullptr) gProgName = ptr + 1;
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE

    if (argc == 1) usage();

    OptParse parser(argc, argv);
    ProgOptions opts;

    openvdb::util::CpuTimer timer;
    auto getTime = [&timer]() -> std::string {
        const double msec = timer.milliseconds();
        std::ostringstream os;
        openvdb::util::printTime(os, msec, "", "", 1, 1, 0);
        return os.str();
    };

    auto& os = std::cerr;
#define axlog(message) \
    { if (opts.mVerbose) os << message; }
#define axtimer() timer.restart()
#define axtime() getTime()

    bool multiSnippet = false;
    bool dashInputHasBeenUsed = false, positionalInputHasBeenUsed = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (parser.check(i, "-s")) {
                ++i;
                multiSnippet |= static_cast<bool>(opts.mInputCode);
                opts.mInputCode.reset(new std::string(argv[i]));
            } else if (parser.check(i, "-f")) {
                ++i;
                multiSnippet |= static_cast<bool>(opts.mInputCode);
                opts.mInputCode.reset(new std::string());
                loadSnippetFile(argv[i], *opts.mInputCode);
            } else if (parser.check(i, "-i")) {
                if (positionalInputHasBeenUsed) {
                    fatal("unrecognized positional argument: \"" + opts.mInputVDBFiles.back() + "\". use -i and -o for vdb files");
                }
                dashInputHasBeenUsed = true;
                opts.mInputVDBFiles.emplace_back(argv[++i]);
            } else if (parser.check(i, "-v", 0)) {
                opts.mVerbose = true;
            } else if (parser.check(i, "--threads")) {
                opts.threads = atoi(argv[++i]);
            } else if (parser.check(i, "--copy-file-metadata", 0)) {
                opts.mCopyFileMeta = true;
            } else if (parser.check(i, "-o")) {
                if (positionalInputHasBeenUsed) {
                    fatal("unrecognized positional argument: \"" + opts.mInputVDBFiles.back() + "\". use -i and -o for vdb files");
                }
                opts.mOutputVDBFile = argv[++i];
            } else if (parser.check(i, "--max-errors")) {
                opts.mMaxErrors = atoi(argv[++i]);
            } else if (parser.check(i, "--werror", 0)) {
                opts.mWarningsAsErrors = true;
            } else if (parser.check(i, "--list", 0)) {
                opts.mFunctionList = true;
                opts.mInitCompile = true; // need to intialize llvm
                opts.mFunctionNamesOnly = false;
                if (i + 1 >= argc) continue;
                if (argv[i+1][0] == '-') continue;
                ++i;
                opts.mFunctionSearch = std::string(argv[i]);
            } else if (parser.check(i, "--list-names", 0)) {
                opts.mFunctionList = true;
                opts.mFunctionNamesOnly = true;
            } else if (parser.check(i, "--ast-print", 0)) {
                opts.mPrintAST = true;
            } else if (parser.check(i, "--re-print", 0)) {
                opts.mReprint = true;
            } else if (parser.check(i, "--reg-print", 0)) {
                opts.mAttribRegPrint = true;
            } else if (parser.check(i, "--try-compile", 0)) {
                opts.mInitCompile = true;
                if (i + 1 >= argc) continue;
                if (argv[i+1][0] == '-') continue;
                ++i;
                opts.mCompileFor = tryCompileStringToCompilation(argv[i]);
            } else if (parser.check(i, "--opt")) {
                ++i;
                opts.mOptLevel = optStringToLevel(argv[i]);
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                usage(opts.mMode, EXIT_SUCCESS);
            } else {
                fatal("\"" + arg + "\" is not a valid option");
            }
        } else if (!arg.empty()) {
            if (dashInputHasBeenUsed) {
                fatal("unrecognized positional argument: \"" + arg + "\". use -i and -o for vdb files");
            }

            if (opts.mMode == ProgOptions::Mode::Default) {
                bool skip = true;
                if (arg == "analyze")        opts.mMode = ProgOptions::Analyze;
                else if (arg == "functions") opts.mMode = ProgOptions::Functions;
                else if (arg == "execute")   opts.mMode = ProgOptions::Execute;
                else {
                    skip = false;
                    opts.mMode = ProgOptions::Execute;
                }
                if (skip) continue;
            }

            // @todo remove positional vdb in/out file arg support
            if (opts.mInputVDBFiles.empty()) {
                positionalInputHasBeenUsed = true;
                OPENVDB_LOG_WARN("position arguments [input.vdb <output.vdb>] are deprecated. use -i and -o");
                opts.mInputVDBFiles.emplace_back(arg);
            }
            else if (opts.mOutputVDBFile.empty()) {
                opts.mOutputVDBFile = arg;
            }
            else {
                fatal("unrecognized positional argument: \"" + arg + "\"");
            }
        } else {
            usage();
        }
    }

    if (opts.mMode == ProgOptions::Mode::Default) {
        opts.mMode = ProgOptions::Mode::Execute;
    }

    if (opts.mMode == ProgOptions::Mode::Execute) {
        opts.mInitCompile = true;
        if (opts.mInputVDBFiles.empty()) {
            fatal("no vdb files have been provided");
        }
    }
    else if (!opts.mInputVDBFiles.empty()) {
        fatal(modeString(opts.mMode) + " does not take input vdb files");
    }

    if (opts.mVerbose) {
        axlog("OpenVDB AX " << openvdb::getLibraryVersionString() << '\n');
        axlog("----------------\n");
        axlog("Inputs\n");
        axlog("  mode    : " << modeString(opts.mMode));
        if (opts.mMode == ProgOptions::Analyze) {
            axlog(" (");
            if (opts.mPrintAST) axlog("|ast out");
            if (opts.mReprint)  axlog("|reprint out");
            if (opts.mAttribRegPrint) axlog("|registry out");
            if (opts.mInitCompile)  axlog("|compilation");
            axlog("|)");
        }
        axlog('\n');

        if (opts.mMode == ProgOptions::Execute) {
            axlog("  vdb in  : \"");
            for (const auto& in : opts.mInputVDBFiles) {
                const bool sep = (&in != &opts.mInputVDBFiles.back());
                axlog(in << (sep ? ", " : ""));
            }
            axlog("\"\n");
            axlog("  vdb out : \"" << opts.mOutputVDBFile << "\"\n");
        }
        if (opts.mMode == ProgOptions::Execute ||
            opts.mMode == ProgOptions::Analyze) {
            axlog("  ax code : ");
            if (opts.mInputCode && !opts.mInputCode->empty()) {
                const bool containsnl =
                    opts.mInputCode->find('\n') != std::string::npos;
                if (containsnl) axlog("\n    ");

                // indent output
                const char* c = opts.mInputCode->c_str();
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

    if (opts.mMode != ProgOptions::Functions) {
        if (!opts.mInputCode) {
            fatal("expected at least one AX file or a code snippet");
        }
        if (multiSnippet) {
            OPENVDB_LOG_WARN("multiple code snippets provided, only using last input.");
        }
    }

    if (opts.mMode == ProgOptions::Execute) {
        if (opts.mOutputVDBFile.empty()) {
            OPENVDB_LOG_WARN("no output VDB File specified - nothing will be written to disk");
        }
    }

    axtimer();
    axlog("[INFO] Initializing OpenVDB" << std::flush);
    ScopedInitialize initializer(argc, argv);
    axlog(": " << axtime() << '\n');

    std::unique_ptr<tbb::global_control> control;
    if (opts.threads > 0) {
        axlog("[INFO] Initializing thread usage [" << opts.threads << "]\n" << std::flush);
        control.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, opts.threads));
    }

    openvdb::GridPtrVec grids;
    openvdb::MetaMap::Ptr meta;

    if (opts.mMode == ProgOptions::Execute) {
        // read vdb file data for
        axlog("[INFO] Reading VDB data"
            << (openvdb::io::Archive::isDelayedLoadingEnabled() ?
                " (delay-load)" : "") << '\n');
        for (const auto& filename : opts.mInputVDBFiles) {
            openvdb::io::File file(filename);
            try {
                axlog("[INFO] | \"" << filename << "\"");
                axtimer();
                file.open();
                auto in = file.getGrids();
                grids.insert(grids.end(), in->begin(), in->end());
                // choose the first files metadata
                if (opts.mCopyFileMeta && !meta) meta = file.getMetadata();
                file.close();
                axlog(": " << axtime() << '\n');
            } catch (openvdb::Exception& e) {
                axlog('\n');
                OPENVDB_LOG_ERROR(e.what() << " (" << filename << ")");
                return EXIT_FAILURE;
            }
        }
    }

    if (opts.mInitCompile) {
        axtimer();
        axlog("[INFO] Initializing AX/LLVM" << std::flush);
        initializer.initializeCompiler();
        axlog(": " << axtime() << '\n');
    }

    if (opts.mMode == ProgOptions::Functions) {
        if (opts.mFunctionList) {
            axlog("Querying available functions\n" << std::flush);
            assert(opts.mFunctionNamesOnly || initializer.isInitialized());
            printFunctions(opts.mFunctionNamesOnly,
                opts.mFunctionSearch,
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
    logs.setMaxErrors(opts.mMaxErrors);
    logs.setWarningsAsErrors(opts.mWarningsAsErrors);
    logs.setPrintLines(true);
    logs.setNumberedOutput(true);

    // parse

    axtimer();
    axlog("[INFO] Parsing input code" << std::flush);

    const openvdb::ax::ast::Tree::ConstPtr syntaxTree =
        openvdb::ax::ast::parse(opts.mInputCode->c_str(), logs);
        axlog(": " << axtime() << '\n');
    if (!syntaxTree) {
        return EXIT_FAILURE;
    }

    if (opts.mMode == ProgOptions::Analyze) {
        axlog("[INFO] Running analysis options\n" << std::flush);
        if (opts.mPrintAST) {
            axlog("[INFO] | Printing AST\n" << std::flush);
            openvdb::ax::ast::print(*syntaxTree, true, std::cout);
        }
        if (opts.mReprint) {
            axlog("[INFO] | Reprinting code\n" << std::flush);
            openvdb::ax::ast::reprint(*syntaxTree, std::cout);
        }
        if (opts.mAttribRegPrint) {
            axlog("[INFO] | Printing Attribute Registry\n" << std::flush);
            const openvdb::ax::AttributeRegistry::ConstPtr reg =
                openvdb::ax::AttributeRegistry::create(*syntaxTree);
            reg->print(std::cout);
            std::cout << std::flush;
        }

        if (!opts.mInitCompile) {
            return EXIT_SUCCESS;
        }
    }

    assert(opts.mInitCompile);

    axtimer();
    axlog("[INFO] Creating Compiler\n");
    axlog("[INFO] | Optimization Level [" << optLevelToString(opts.mOptLevel) << "]\n" << std::flush);
    openvdb::ax::CompilerOptions compOpts;
    compOpts.mOptLevel = opts.mOptLevel;

    openvdb::ax::Compiler::Ptr compiler =
        openvdb::ax::Compiler::create(compOpts);
    openvdb::ax::CustomData::Ptr customData =
        openvdb::ax::CustomData::create();
    axlog("[INFO] | " << axtime() << '\n' << std::flush);

    // Check what we need to compile for if performing execution

    if (opts.mMode == ProgOptions::Execute) {
        bool points = false;
        bool volumes = false;
        for (auto grid : grids) {
            points |= grid->isType<openvdb::points::PointDataGrid>();
            volumes |= !points;
            if (points && volumes) break;
        }
        if (points && volumes) opts.mCompileFor = ProgOptions::Compilation::All;
        else if (points)       opts.mCompileFor = ProgOptions::Compilation::Points;
        else if (volumes)      opts.mCompileFor = ProgOptions::Compilation::Volumes;
    }

    if (opts.mMode == ProgOptions::Analyze) {

        bool psuccess = true;

        if (opts.mCompileFor == ProgOptions::Compilation::All ||
            opts.mCompileFor == ProgOptions::Compilation::Points) {
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

        if (opts.mCompileFor == ProgOptions::Compilation::All ||
            opts.mCompileFor == ProgOptions::Compilation::Volumes) {
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

    if (opts.mCompileFor == ProgOptions::Compilation::All ||
        opts.mCompileFor == ProgOptions::Compilation::Points) {

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
        if (opts.mVerbose) {
            for (auto grid : grids) {
                if (!grid->isType<openvdb::points::PointDataGrid>()) continue;
                ++total;
            }
        }

        for (auto grid :grids) {
            if (!grid->isType<openvdb::points::PointDataGrid>()) continue;
            openvdb::points::PointDataGrid::Ptr points =
                openvdb::gridPtrCast<openvdb::points::PointDataGrid>(grid);
            axtimer();
            axlog("[INFO] Executing on \"" << points->getName() << "\" "
                  << count << " of " << total << '\n' << std::flush);
            ++count;

            try {
                pointExe->execute(*points);
                if (openvdb::ax::ast::callsFunction(*syntaxTree, "deletepoint")) {
                    openvdb::points::deleteFromGroup(points->tree(), "dead", false, false);
                }
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

    if (opts.mCompileFor == ProgOptions::Compilation::All ||
        opts.mCompileFor == ProgOptions::Compilation::Volumes) {

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

        if (opts.mVerbose) {
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

        try { volumeExe->execute(grids); }
        catch (std::exception& e) {
            OPENVDB_LOG_FATAL("Execution error!\nErrors:\n" << e.what());
            return EXIT_FAILURE;
        }

        axlog("[INFO] | Execution success.\n");
        axlog("[INFO] | " << axtime() << '\n' << std::flush);
    }

    if (!opts.mOutputVDBFile.empty()) {
        axtimer();
        axlog("[INFO] Writing results" << std::flush);
        openvdb::io::File out(opts.mOutputVDBFile);
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

