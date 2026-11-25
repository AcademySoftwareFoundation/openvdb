// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file main.cc
///
/// @brief Benchmark of the core library tools that read neighbors, to compare
/// two builds of the library, for example with and without
/// tree::NeighborAccessor.
///
/// Every row goes through the public API only, so the same source builds
/// against the library before and after a change. Each row prints a hash of
/// its result, which must be the same in both builds. At the end of a run the
/// rows are summarized, and compared with the rows of an earlier run of
/// another build if there is a baseline file.

#include "Harness.h"
#include "Summary.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace {

void
usage()
{
    std::cerr <<
        "usage: vdb_bench <group> [options]\n"
        "\n"
        "groups: all levelset analysis operators filter poisson sampling meshing accessor\n"
        "\n"
        "options:\n"
        "  --threads N        threads to run on, default 8\n"
        "  --repeat N         timed runs of every row, the fastest is used, default 3\n"
        "  --passes N         how often every group runs, default 2 for all and 1 for a group\n"
        "  --voxel SIZE       voxel size of the level sets, default 1/1024\n"
        "  --only TEXT        only the rows with TEXT in \"tool/name\"\n"
        "  --dragon-vdb FILE  dragon.vdb, the rows that use it are skipped without it\n"
        "  --save FILE        write every row to a CSV file at the end\n"
        "  --baseline FILE    compare with the rows of a CSV file from another build\n"
        "  --summary FILE     write the summary to a file, default is standard output\n"
        "  --csv              print rows as CSV, for one group and one pass\n";
}

bench::Options
parseArguments(int argc, char* argv[])
{
    bench::Options options;
    if (argc < 2) throw std::runtime_error("missing group name");
    options.group = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[++i];
        };
        if (argument == "--threads") options.threads = std::stoi(next());
        else if (argument == "--repeat") options.repeat = std::stoi(next());
        else if (argument == "--passes") options.passes = std::stoi(next());
        else if (argument == "--voxel") options.voxelSize = std::stof(next());
        else if (argument == "--only") options.only = next();
        else if (argument == "--dragon-vdb") options.dragonPath = next();
        else if (argument == "--save") options.savePath = next();
        else if (argument == "--baseline") options.baselinePath = next();
        else if (argument == "--summary") options.summaryPath = next();
        else if (argument == "--csv") options.csv = true;
        else throw std::runtime_error("unknown option " + argument);
    }
    if (options.threads < 1 || options.repeat < 1) throw std::runtime_error("threads and repeat must be at least 1");
    if (options.passes < 0) throw std::runtime_error("passes cannot be negative");
    if (options.passes == 0) options.passes = (options.group == "all") ? 2 : 1;
    if (options.csv && (options.group == "all" || options.passes > 1)) {
        throw std::runtime_error("--csv is for one group and one pass, use --save instead");
    }
    return options;
}

void
runGroup(bench::Context& context, const std::string& group)
{
    if (group == "levelset") bench::runLevelSet(context);
    else if (group == "analysis") bench::runAnalysis(context);
    else if (group == "operators") bench::runOperators(context);
    else if (group == "filter") bench::runFilter(context);
    else if (group == "poisson") bench::runPoisson(context);
    else if (group == "sampling") bench::runSampling(context);
    else if (group == "meshing") bench::runMeshing(context);
    else if (group == "accessor") bench::runAccessor(context);
    else throw std::runtime_error("unknown group " + group);
}

} // anonymous namespace


int
main(int argc, char* argv[])
{
    try {
        const bench::Options options = parseArguments(argc, argv);
        openvdb::initialize();

        std::vector<std::string> groups;
        if (options.group == "all") groups.assign(std::begin(bench::kGroupNames), std::end(bench::kGroupNames));
        else groups.push_back(options.group);

        std::vector<bench::Row> rows;
        for (int pass = 0; pass < options.passes; ++pass) {
            for (const std::string& group : groups) {
                bench::Options groupOptions = options;
                groupOptions.group = group;
                bench::Context context(groupOptions, rows);
                runGroup(context, group);
            }
        }

        if (!options.savePath.empty()) bench::writeCsv(options.savePath, rows);
        if (options.csv) return EXIT_SUCCESS;

        std::vector<bench::Record> baseline;
        if (!options.baselinePath.empty()) baseline = bench::readCsv(options.baselinePath);
        std::ofstream summaryFile;
        if (!options.summaryPath.empty()) {
            summaryFile.open(options.summaryPath);
            if (!summaryFile) throw std::runtime_error("cannot write " + options.summaryPath);
        }
        std::ostream& out = options.summaryPath.empty() ? std::cout : summaryFile;
        const bool same = bench::writeSummary(out, options, bench::toRecords(rows), baseline);
        return same ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        usage();
        return EXIT_FAILURE;
    }
}
