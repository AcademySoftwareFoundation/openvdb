// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   nanovdb_validate.cc

    \author Ken Museth

    \date   October 13, 2020

    \brief  Command-line tool that validates Grids in nanovdb files
*/

#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/GridValidator.h>
#include <iomanip>
#include <sstream>

void usage [[noreturn]] (const std::string& progName, int exitStatus = EXIT_FAILURE)
{
    std::cerr << "\nUsage: " << progName << " [options] *.nvdb\n"
              << "Which: Validates grids in one or more NanoVDB files\n\n"
              << "Options:\n"
              << "-g,--grid name\tOnly validate grids matching the specified string name\n"
              << "-h,--help\tPrints this message\n"
              << "-v,--verbose\tPrint verbose information information useful for debugging\n"
              << "--version\tPrint version information to the terminal\n";
    exit(exitStatus);
}

void version [[noreturn]] (const char* progName, int exitStatus = EXIT_SUCCESS)
{
    printf("\n%s was build against NanoVDB version %s\n", progName, nanovdb::Version().c_str());
    exit(exitStatus);
}

int main(int argc, char* argv[])
{
    int exitStatus = EXIT_SUCCESS;

    bool                     verbose = false;
    bool                     detailed = true;
    std::string              gridName;
    std::vector<std::string> fileNames;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-h" || arg == "--help") {
                usage(argv[0], EXIT_SUCCESS);
            } else if (arg == "--version") {
                version(argv[0]);
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else if (arg == "-g" || arg == "--grid") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a grid name to follow the -g,--grid option\n";
                    usage(argv[0]);
                } else {
                    gridName.assign(argv[++i]);
                }
            } else {
                std::cerr << "\nIllegal option: \"" << arg << "\"\n";
                usage(argv[0]);
            }
        } else if (!arg.empty()) {
            fileNames.push_back(arg);
        }
    }
    if (fileNames.size() == 0) {
        std::cerr << "\nExpected at least one input NanoVDB file\n";
        usage(argv[0]);
    }

    const auto nameKey = nanovdb::io::stringHash(gridName);

    try {
        for (auto& file : fileNames) {
            auto list = nanovdb::io::readGridMetaData(file);
            if (!gridName.empty()) {
                std::vector<nanovdb::io::GridMetaData> tmp;
                for (auto& m : list) {
                    if (nameKey == m.nameKey && gridName == m.gridName)
                        tmp.emplace_back(m);
                }
                list = tmp;
            }
            if (list.size() == 0) {
                continue;
            }

            if (verbose) {
                std::cout << "\nThe file \"" << file << "\" contains the following matching " << list.size() << " grid(s):\n";
            }

            for (auto& m : list) {
                auto handle = nanovdb::io::readGrid(file, m.gridName);
                auto gridType = handle.gridType();
                bool test = false;
                if (gridType == nanovdb::GridType::End) {
                    std::cerr << "GridHandle was empty\n" << std::endl;
                    usage(argv[0]);
                } else if (auto* grid = handle.grid<float>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Vec3f>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<uint32_t>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<int32_t>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<int16_t>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<int64_t>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<int16_t>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<double>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Vec3d>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::ValueMask>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Rgba8>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Fp4>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Fp8>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::Fp16>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<nanovdb::FpN>()) {
                    test = isValid(*grid, detailed, verbose);
                } else if (auto* grid = handle.grid<bool>()) {
                    test = isValid(*grid, detailed, verbose);
                } else {
                    std::cerr << "Unsupported GridType: \"" << nanovdb::toStr(gridType) << "\"\n" << std::endl;
                    usage(argv[0]);
                }
                if (verbose) {
                    std::cout << "Grid named \"" << m.gridName << "\": " << (test ? "passed" : "failed") << std::endl;
                } else if (!test) {
                    std::cout << "Grid named \"" << m.gridName << "\": failed" << std::endl;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        exitStatus = EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Exception oof unexpected type caught" << std::endl;
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}
