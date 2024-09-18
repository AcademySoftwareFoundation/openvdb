// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   nanovdb_print.cc

    \author Ken Museth

    \date   May 21, 2020

    \brief  Command-line tool that prints information about grids in a nanovdb file
*/

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <iomanip>
#include <sstream>

void usage [[noreturn]] (const std::string& progName, int exitStatus = EXIT_FAILURE)
{
    std::cerr << "\nUsage: " << progName << " [options] *.nvdb\n"
              << "Which: Prints grid information from one or more NanoVDB files\n\n"
              << "Options:\n"
              << "-g,--grid name\tPrint all grids matching the specified string name\n"
              << "-h,--help\tPrints this message\n"
              << "-l,--long\tPrints out extra grid information\n"
              << "-s,--short\tOnly prints out a minimum amount of grid information\n"
              << "-v,--verbose\tPrint information about the meaning of the labels\n"
              << "--version\tPrint version information to the terminal\n";
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

    enum Mode : int { Short = 0,
                      Default = 1,
                      Long = 2 } mode = Default;
    char str[32];
    bool verbose = false;
    std::string              gridName;
    std::vector<std::string> fileNames;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-h" || arg == "--help") {
                usage(argv[0], EXIT_SUCCESS);
            } else if (arg == "--version") {
                version(argv[0]);
            } else if (arg == "-s" || arg == "--short") {
                mode = Short;
            } else if (arg == "-l" || arg == "--long") {
                mode = Long;
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

    auto format = [](uint64_t b) {
        std::stringstream ss;
        ss << std::setprecision(4);
        if (b >> 40) {
            ss << double(b) / double(1ULL << 40) << " TB";
        } else if (b >> 30) {
            ss << double(b) / double(1ULL << 30) << " GB";
        } else if (b >> 20) {
            ss << double(b) / double(1ULL << 20) << " MB";
        } else if (b >> 10) {
            ss << double(b) / double(1ULL << 10) << " KB";
        } else {
            ss << b << " Bytes";
        }
        return ss.str();
    };

    const int padding = 2;

    auto width = [&](size_t& n, const std::string& str) {
        const auto size = str.length() + padding;
        if (size > n)
            n = size;
    };
    auto Vec3dToStr = [](const nanovdb::Vec3d& v) {
        std::stringstream ss;
        ss << std::setprecision(3);
        ss << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
        return ss.str();
    };
    auto wbboxToStr = [](const nanovdb::math::BBox<nanovdb::Vec3d>& bbox) {
        std::stringstream ss;
        if (bbox.empty()) {
            ss << "empty grid";
        } else {
            ss << std::setprecision(3);
            ss << "(" << bbox[0][0] << "," << bbox[0][1] << "," << bbox[0][2] << ")";
            ss << " -> ";
            ss << "(" << bbox[1][0] << "," << bbox[1][1] << "," << bbox[1][2] << ")";
        }
        return ss.str();
    };
    auto ibboxToStr = [](const nanovdb::CoordBBox& bbox) {
        std::stringstream ss;
        if (bbox.empty()) {
            ss << "empty grid";
        } else {
            ss << "(" << bbox[0][0] << "," << bbox[0][1] << "," << bbox[0][2] << ")";
            ss << " -> ";
            ss << "(" << bbox[1][0] << "," << bbox[1][1] << "," << bbox[1][2] << ")";
        }
        return ss.str();
    };
    auto resToStr = [](const nanovdb::CoordBBox& bbox) {
        std::stringstream ss;
        auto              dim = bbox.dim();
        ss << dim[0] << " x " << dim[1] << " x " << dim[2];
        return ss.str();
    };
    auto nodesToStr = [](const uint32_t* nodes) {
        std::stringstream ss;
        ss << nodes[2] << "->" << nodes[1] << "->" << nodes[0];
        return ss.str();
    };

    try {
        for (auto& file : fileNames) {
            auto list = nanovdb::io::readGridMetaData(file);
            if (!gridName.empty()) {
                std::vector<nanovdb::io::FileGridMetaData> tmp;
                for (auto& m : list) {
                    if (nameKey == m.nameKey && gridName == m.gridName)
                        tmp.emplace_back(m);
                }
                list = tmp;
            }
            if (list.size() == 0)
                continue;
            const auto numberWidth = std::to_string(list.size()).length() + padding;
            auto       nameWidth = std::string("Name").length() + padding;
            auto       typeWidth = std::string("Type").length() + padding;
            auto       classWidth = std::string("Class").length() + padding;
            auto       codecWidth = std::string("Codec").length() + padding;
            auto       ibboxWidth = std::string("Index Bounding Box").length() + padding;
            auto       wbboxWidth = std::string("World BBox").length() + padding;
            auto       sizeWidth = std::string("Size").length() + padding;
            auto       fileWidth = std::string("File").length() + padding;
            auto       voxelsWidth = std::string("# Voxels").length() + padding;
            auto       voxelSizeWidth = std::string("Scale").length() + padding;
            auto       versionWidth = std::string("Version").length() + padding;
            auto       configWidth = std::string("32^3->16^3->8^3").length() + padding;
            auto       tileWidth = std::string("# Active tiles").length() + padding;
            auto       resWidth = std::string("Resolution").length() + padding;
            for (auto& m : list) {
                width(nameWidth, m.gridName);
                width(typeWidth, nanovdb::toStr(str, m.gridType));
                width(classWidth, nanovdb::toStr(str, m.gridClass));
                width(codecWidth, nanovdb::io::toStr(str, m.codec));
                width(wbboxWidth, wbboxToStr(m.worldBBox));
                width(ibboxWidth, ibboxToStr(m.indexBBox));
                width(resWidth, resToStr(m.indexBBox));
                width(sizeWidth, format(m.gridSize));
                width(fileWidth, format(m.fileSize));
                width(versionWidth, nanovdb::toStr(str, m.version));
                width(configWidth, nodesToStr(m.nodeCount));
                width(tileWidth, nodesToStr(m.tileCount));
                width(voxelsWidth, std::to_string(m.voxelCount));
                width(voxelSizeWidth, Vec3dToStr(m.voxelSize));
            }
            std::cout << "\nThe file \"" << file << "\" contains the following ";
            if (list.size()>1) {
                std::cout << list.size() << " grids:\n";
            } else {
                std::cout << "grid:\n";
            }
            std::cout << std::left << std::setw(numberWidth) << "#"
                      << std::left << std::setw(nameWidth) << "Name"
                      << std::left << std::setw(typeWidth) << "Type";
            if (mode != Short) {
                std::cout << std::left << std::setw(classWidth) << "Class"
                          << std::left << std::setw(versionWidth) << "Version"
                          << std::left << std::setw(codecWidth) << "Codec"
                          << std::left << std::setw(sizeWidth) << "Size"
                          << std::left << std::setw(fileWidth) << "File"
                          << std::left << std::setw(voxelSizeWidth) << "Scale";
            }
            std::cout << std::left << std::setw(voxelsWidth) << "# Voxels"
                      << std::left << std::setw(resWidth) << "Resolution";
            if (mode == Long) {
                std::cout << std::left << std::setw(configWidth) << "32^3->16^3->8^3"
                          << std::left << std::setw(tileWidth)   << "# Active tiles"
                          << std::left << std::setw(ibboxWidth)  << "Index Bounding Box"
                          << std::left << std::setw(wbboxWidth)  << "World Bounding Box";
            }
            std::cout << std::endl;
            int n = 0;
            for (auto& m : list) {
                if (!gridName.empty() && (nameKey != m.nameKey || gridName != m.gridName))
                    continue;
                std::cout << std::left << std::setw(numberWidth) << ++n
                          << std::left << std::setw(nameWidth) << m.gridName
                          << std::left << std::setw(typeWidth) << nanovdb::toStr(str, m.gridType);
                if (mode != Short) {
                    std::cout << std::left << std::setw(classWidth) << nanovdb::toStr(str, m.gridClass)
                              << std::left << std::setw(versionWidth) << nanovdb::toStr(str+10, m.version)
                              << std::left << std::setw(codecWidth) << nanovdb::io::toStr(str + 20, m.codec)
                              << std::left << std::setw(sizeWidth) << format(m.gridSize)
                              << std::left << std::setw(fileWidth) << format(m.fileSize)
                              << std::left << std::setw(voxelSizeWidth) << Vec3dToStr(m.voxelSize);
                }
                std::cout << std::left << std::setw(voxelsWidth) << m.voxelCount
                          << std::left << std::setw(resWidth) << resToStr(m.indexBBox);
                if (mode == Long) {
                    std::cout << std::left << std::setw(configWidth) << nodesToStr(m.nodeCount)
                              << std::left << std::setw(tileWidth)   << nodesToStr(m.tileCount)
                              << std::left << std::setw(ibboxWidth)  << ibboxToStr(m.indexBBox)
                              << std::left << std::setw(wbboxWidth)  << wbboxToStr(m.worldBBox);
                }
                std::cout << std::endl;
            }
        }
        if (verbose) {
            size_t w = 0;
            switch (mode) {
            case Mode::Short:
                width(w, "\"Name\":");
                width(w, "\"Type\":");
                width(w, "\"# Voxels\":");
                width(w, "\"Resolution\":");
                std::cout << std::left << std::setw(w) << "\n\"Name\":"  << "name of a grid. Note that it is optional and hence might be empty."
                          << std::left << std::setw(w) << "\n\"Type\":"  << "static type of the values in a grid, e.g. float, vec3f etc."
                          << std::left << std::setw(w) << "\n\"# Voxels\":" << "total number of active values in a grid."
                          << std::left << std::setw(w) << "\n\"Resolution\":" << "Efficient resolution of all the active values in a grid!\n";
                break;
            case Mode::Default:
                width(w, "\"Name\":");
                width(w, "\"Type\":");
                width(w, "\"Class\":");
                width(w, "\"Version\":");
                width(w, "\"Codec\":");
                width(w, "\"Size\":");
                width(w, "\"File\":");
                width(w, "\"Scale\":");
                width(w, "\"# Voxels\":");
                width(w, "\"Resolution\":");
                std::cout << std::left << std::setw(w) << "\n\"Name\":"  << "name of a grid. Note that it is optional and hence might be empty."
                          << std::left << std::setw(w) << "\n\"Type\":"  << "static type of the values in a grid, e.g. float, vec3f etc."
                          << std::left << std::setw(w) << "\n\"Class\":"  << "class of the grid, e.g. FOG for Fog volume, LS for level set, etc."
                          << std::left << std::setw(w) << "\n\"Version\":"  << "major.minor.patch version numbers of the grid."
                          << std::left << std::setw(w) << "\n\"Codec\":"  << "codec of the optional compression applied to the out-of-core grid, i.e. on disk."
                          << std::left << std::setw(w) << "\n\"Size\":"  << "In-core memory footprint of the grid, i.e. in ram."
                          << std::left << std::setw(w) << "\n\"File\":"  << "Out-of-core memory footprint of the grid, i.e. on disk."
                          << std::left << std::setw(w) << "\n\"Scale\":"  << "Scale of the grid, i.e. the size of a voxel in world units."
                          << std::left << std::setw(w) << "\n\"# Voxels\":" << "total number of active values in a grid."
                          << std::left << std::setw(w) << "\n\"Resolution\":" << "Efficient resolution of all the active values in a grid!\n";
                break;
            case Mode::Long:
            width(w, "\"Name\":");
                width(w, "\"Type\":");
                width(w, "\"Class\":");
                width(w, "\"Version\":");
                width(w, "\"Codec\":");
                width(w, "\"Size\":");
                width(w, "\"File\":");
                width(w, "\"Scale\":");
                width(w, "\"# Voxels\":");
                width(w, "\"Resolution\":");
                width(w, "\"32^3->16^3->8^3\":");
                width(w, "\"# Active tiles\":");
                width(w, "\"Index Bounding Box\":");
                width(w, "\"World Bounding Box\":");
                std::cout << std::left << std::setw(w) << "\n\"Name\":"  << "name of a grid. Note that it is optional and hence might be empty."
                          << std::left << std::setw(w) << "\n\"Type\":"  << "static type of the values in a grid, e.g. float, vec3f etc."
                          << std::left << std::setw(w) << "\n\"Class\":"  << "class of the grid, e.g. FOG for Fog volume, LS for level set, etc."
                          << std::left << std::setw(w) << "\n\"Version\":" << "major.minor.patch version numbers of the grid."
                          << std::left << std::setw(w) << "\n\"Codec\":"  << "codec of the optional compression applied to the out-of-core grid, i.e. on disk."
                          << std::left << std::setw(w) << "\n\"Size\":"  << "In-core memory footprint of the grid, e.g. in RAM on the CPU."
                          << std::left << std::setw(w) << "\n\"File\":"  << "Out-of-core memory footprint of the grid, i.e. compressed on disk."
                          << std::left << std::setw(w) << "\n\"Scale\":"  << "Scale of the grid, i.e. the size of a voxel in world units."
                          << std::left << std::setw(w) << "\n\"# Voxels\":" << "total number of active values in a grid. Note this includes both active tiles and voxels."
                          << std::left << std::setw(w) << "\n\"Resolution\":" << "Efficient resolution of all the active values in a grid!"
                          << std::left << std::setw(w) << "\n\"32^3->16^3->8^3\":" << "Number of nodes at each level of the tree structure from the root to leaf level."
                          << std::left << std::setw(w) << "\n\"# Active tiles\":" << "Number of active tiles at each level of the tree structure from the root to leaf level."
                          << std::left << std::setw(w) << "\n\"Index Bounding Box\":" << "coordinate bounding box of all the active values in a grid. Note that both min and max coordinates are inclusive!"
                          << std::left << std::setw(w) << "\n\"World Bounding Box\":" << "world-space bounding box of all the active values in a grid. Note that min is inclusive and max is exclusive!\n";
                break;
            default:
                throw std::runtime_error("Internal error in switch!");
                break;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        exitStatus = EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Exception of unexpected type caught" << std::endl;
        exitStatus = EXIT_FAILURE;
    }

    return exitStatus;
}// main
