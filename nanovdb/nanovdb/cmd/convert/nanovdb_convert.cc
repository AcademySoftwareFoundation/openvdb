// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   nanovdb_convert.cc

    \author Ken Museth

    \date   May 21, 2020

    \brief  Command-line tool that converts between openvdb and nanovdb files
*/

#include <string>
#include <algorithm>
#include <cctype>

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>

void usage [[noreturn]] (const std::string& progName, int exitStatus = EXIT_FAILURE)
{
    std::cerr << "\nUsage: " << progName << " [options] *.vdb output.nvdb\n"
              << "Which: converts one or more OpenVDB files to a single NanoVDB file\n\n"
              << "Usage: " << progName << " [options] *.nvdb output.vdb\n"
              << "Which: converts one or more NanoVDB files to a single OpenVDB file\n\n"
              << "Options:\n"
              << "-a,--abs-error float\t Absolute error tolerance used for variable bit depth quantization\n"
              << "-b,--blosc\tUse BLOSC compression on the output file\n"
              << "-c,--checksum mode\t where mode={none, partial, full}\n"
              << "-d,--dither\tApply dithering during blocked compression\n"
              << "-f,--force\tOverwrite output file if it already exists\n"
              << "--fp4\tQuantize float grids to 4 bits\n"
              << "--fp8\tQuantize float grids to 8 bits\n"
              << "--fp16\tQuantize float grids to 16 bits\n"
              << "--fpN\tQuantize float grids to variable bit depth (use -a or -r to specify a tolerance)\n"
              << "-g,--grid name\tConvert all grids matching the specified string name\n"
              << "-h,--help\tPrints this message\n"
              << "-r,--rel-error float\t Relative error tolerance used for variable bit depth quantization\n"
              << "-s,--stats mode\t where mode={none, bbox, extrema, all}\n"
              << "-v,--verbose\tPrint verbose information to the terminal\n"
              << "--version\tPrint version information to the terminal\n"
              << "-z,--zip\tUse ZIP compression on the output file\n";
    exit(exitStatus);
}

void version [[noreturn]] (const char* progName, int exitStatus = EXIT_SUCCESS)
{
    char str[8];
    nanovdb::toStr(str, nanovdb::Version());
    printf("\n%s was build against NanoVDB version %s\n", progName, str);
    exit(exitStatus);
}

int main(int argc, char* argv[])
{
    int exitStatus = EXIT_SUCCESS;

    nanovdb::io::Codec       codec = nanovdb::io::Codec::NONE;// compression codec for the file
    nanovdb::tools::StatsMode       sMode = nanovdb::tools::StatsMode::Default;
    nanovdb::CheckMode    cMode = nanovdb::CheckMode::Default;
    nanovdb::GridType        qMode = nanovdb::GridType::Unknown;//specify the quantization mode
    bool                     verbose = false, overwrite = false, dither = false, absolute = true;
    float                    tolerance = -1.0f;
    std::string              gridName;
    std::vector<std::string> fileNames;
    auto toLowerCase = [](std::string &str) {
        std::transform(str.begin(), str.end(), str.begin(),[](unsigned char c){return std::tolower(c);});
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else if (arg == "--version") {
                version(argv[0]);
            } else if (arg == "-f" || arg == "--force") {
                overwrite = true;
            } else if (arg == "--fp4") {
                qMode = nanovdb::GridType::Fp4;
            } else if (arg == "--fp8") {
                qMode = nanovdb::GridType::Fp8;
            } else if (arg == "--fp16") {
                qMode = nanovdb::GridType::Fp16;
            } else if (arg == "--fpN") {
                qMode = nanovdb::GridType::FpN;
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0], EXIT_SUCCESS);
            } else if (arg == "-b" || arg == "--blosc") {
                codec = nanovdb::io::Codec::BLOSC;
            } else if (arg == "-d" || arg == "--dither") {
                dither = true;
            } else if (arg == "-z" || arg == "--zip") {
                codec = nanovdb::io::Codec::ZIP;
            } else if (arg == "-c" || arg == "--checksum") {
                if (i + 1 == argc) {
                    std::cerr << "Expected a mode to follow the -c,--checksum option\n" << std::endl;
                    usage(argv[0]);
                } else {
                    std::string str(argv[++i]);
                    toLowerCase(str);
                    if (str == "none") {
                       cMode = nanovdb::CheckMode::Disable;
                    } else if (str == "partial") {
                       cMode = nanovdb::CheckMode::Partial;
                    } else if (str == "full") {
                       cMode = nanovdb::CheckMode::Full;
                    } else {
                      std::cerr << "Expected one of the following checksum modes: {none, partial, full}\n" << std::endl;
                      usage(argv[0]);
                    }
                }
            } else if (arg == "-s" || arg == "--stats") {
                if (i + 1 == argc) {
                    std::cerr << "Expected a mode to follow the -s,--stats option\n" << std::endl;
                    usage(argv[0]);
                } else {
                    std::string str(argv[++i]);
                    toLowerCase(str);
                    if (str == "none") {
                       sMode = nanovdb::tools::StatsMode::Disable;
                    } else if (str == "bbox") {
                       sMode = nanovdb::tools::StatsMode::BBox;
                    } else if (str == "extrema") {
                       sMode = nanovdb::tools::StatsMode::MinMax;
                    } else if (str == "all") {
                       sMode = nanovdb::tools::StatsMode::All;
                    } else {
                      std::cerr << "Expected one of the following stats modes: {none, bbox, extrema, all}\n" << std::endl;
                      usage(argv[0]);
                    }
                }
            } else if (arg == "-a" || arg == "--abs-error") {
                if (i + 1 == argc) {
                    std::cerr << "Expected a float to follow the -a,--abs-error option\n" << std::endl;
                    usage(argv[0]);
                } else {
                    qMode = nanovdb::GridType::FpN;
                    absolute = true;
                    tolerance = static_cast<float>(atof(argv[++i]));
                }
            } else if (arg == "-r" || arg == "--rel-error") {
                if (i + 1 == argc) {
                    std::cerr << "Expected a float to follow the -r,--rel-error option\n" << std::endl;
                    usage(argv[0]);
                } else {
                    qMode = nanovdb::GridType::FpN;
                    absolute = false;
                    tolerance = static_cast<float>(atof(argv[++i]));
                }
            } else if (arg == "-g" || arg == "--grid") {
                if (i + 1 == argc) {
                    std::cerr << "Expected a grid name to follow the -g,--grid option\n" << std::endl;
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

    if (fileNames.size() < 2) {
        std::cerr << "Expected at least one input file followed by exactly one output file\n" << std::endl;
        usage(argv[0]);
    }
    const std::string outputFile = fileNames.back();
    const std::string ext = outputFile.substr(outputFile.find_last_of(".") + 1);
    bool              toNanoVDB = false;
    if (ext == "nvdb") {
        toNanoVDB = true;
    } else if (ext != "vdb") {
        std::cerr << "Unrecognized file extension: \"" << ext << "\"\n" << std::endl;
        usage(argv[0]);
    }

    fileNames.pop_back();

    if (!overwrite) {
        std::ifstream is(outputFile, std::ios::in | std::ios::binary);
        if (is.peek() != std::ifstream::traits_type::eof()) {
            std::cout << "Overwrite the existing output file named \"" << outputFile << "\"? [Y]/N: ";
            std::string answer;
            getline(std::cin, answer);
            toLowerCase(answer);
            if (!answer.empty() && answer != "y" && answer != "yes") {
                std::cout << "Please specify a different output file" << std::endl;
                exit(EXIT_SUCCESS);
            }
        }
    }

    openvdb::initialize();

    // Note, unlike OpenVDB, NanoVDB allows for multiple write operations into the same output file stream.
    // Hence, NanoVDB grids can be read, converted and written to file one at a time whereas all
    // the OpenVDB grids has to be written to file in a single operation.

    auto openToNano = [&](const openvdb::GridBase::Ptr& base)
    {
        using SrcGridT = openvdb::FloatGrid;
        if (auto floatGrid = openvdb::GridBase::grid<SrcGridT>(base)) {
            nanovdb::tools::CreateNanoGrid<SrcGridT> s(*floatGrid);
            s.setStats(sMode);
            s.setChecksum(cMode);
            s.enableDithering(dither);
            s.setVerbose(verbose ? 1 : 0);
            switch (qMode) {
            case nanovdb::GridType::Fp4:
                return s.getHandle<nanovdb::Fp4>();
            case nanovdb::GridType::Fp8:
                return s.getHandle<nanovdb::Fp8>();
            case nanovdb::GridType::Fp16:
                return s.getHandle<nanovdb::Fp16>();
            case nanovdb::GridType::FpN:
                if (absolute) {
                    return s.getHandle<nanovdb::FpN>(nanovdb::tools::AbsDiff(tolerance));
                } else {
                    return s.getHandle<nanovdb::FpN>(nanovdb::tools::RelDiff(tolerance));
                }
            default:
                break;
            }// end of switch
        }
        return nanovdb::tools::openToNanoVDB(base, sMode, cMode, verbose ? 1 : 0);
    };
    try {
        if (toNanoVDB) { // OpenVDB -> NanoVDB
            std::ofstream os(outputFile, std::ios::out | std::ios::binary);
            for (auto& inputFile : fileNames) {
                if (inputFile.substr(inputFile.find_last_of(".") + 1) != "vdb") {
                    std::cerr << "Since the last file has extension .nvdb the remaining input files were expected to have extensions .vdb\n" << std::endl;
                    usage(argv[0]);
                }
                if (verbose)
                    std::cout << "Opening OpenVDB file named \"" << inputFile << "\"" << std::endl;
                openvdb::io::File file(inputFile);
                file.open(false); //disable delayed loading
                if (gridName.empty()) {// convert all grid in the file
                    auto grids = file.getGrids();
                    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer> > handles;
                    for (auto& grid : *grids) {
                        if (verbose) {
                            std::cout << "Converting OpenVDB grid named \"" << grid->getName() << "\" to NanoVDB" << std::endl;
                        }
                        handles.push_back(openToNano(grid));
                    } // loop over OpenVDB grids in file
                    auto handle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles);
                    nanovdb::io::writeGrid(os, handle, codec);
                } else {// convert only grid with matching name
                    auto grid = file.readGrid(gridName);
                    if (verbose) {
                        std::cout << "Converting OpenVDB grid named \"" << grid->getName() << "\" to NanoVDB" << std::endl;
                    }
                    auto handle = openToNano(grid);
                    nanovdb::io::writeGrid(os, handle, codec);
                }
            } // loop over input files
        } else { // NanoVDB -> OpenVDB
            openvdb::io::File      file(outputFile);
            openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec());
            for (auto& inputFile : fileNames) {
                if (inputFile.substr(inputFile.find_last_of(".") + 1) != "nvdb") {
                    std::cerr << "Since the last file has extension .vdb the remaining input files were expected to have extensions .nvdb\n" << std::endl;
                    usage(argv[0]);
                }
                if (verbose)
                    std::cout << "Opening NanoVDB file named \"" << inputFile << "\"" << std::endl;
                if (gridName.empty()) {
                    auto handles = nanovdb::io::readGrids(inputFile, verbose);
                    for (auto &h : handles) {
                        for (uint32_t i = 0; i < h.gridCount(); ++i) {
                            if (verbose)
                                std::cout << "Converting NanoVDB grid named \"" << h.gridMetaData(i)->shortGridName() << "\" to OpenVDB" << std::endl;
                            grids->push_back(nanovdb::tools::nanoToOpenVDB(h, 0, i));
                        }
                    }
                } else {
                    auto handle = nanovdb::io::readGrid(inputFile, gridName);
                    if (!handle) {
                        std::cerr << "File did not contain a NanoVDB grid named \"" << gridName << "\"\n" << std::endl;
                        usage(argv[0]);
                    }
                    if (verbose)
                        std::cout << "Converting NanoVDB grid named \"" << handle.gridMetaData()->shortGridName() << "\" to OpenVDB" << std::endl;
                    grids->push_back(nanovdb::tools::nanoToOpenVDB(handle));
                }
            } // loop over input files
            file.write(*grids);
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
