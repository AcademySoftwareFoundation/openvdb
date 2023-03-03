// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include "Viewer.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
#include <unistd.h>
#endif
#include <stdio.h>


inline void
usage [[noreturn]] (const char* progName, int status)
{
    (status == EXIT_SUCCESS ? std::cout : std::cerr) <<
"Usage: " << progName << " file.vdb [file.vdb ...] [options]\n" <<
"Which: displays OpenVDB grids\n" <<
"Options:\n" <<
"    -i                 print grid information\n" <<
"    -h, -help          print this usage message and exit\n" <<
"    -version           print version information\n" <<
"\n" <<
"Controls:\n" <<
"    Esc                exit\n" <<
"    -> (Right)         show next grid\n" <<
"    <- (Left)          show previous grid\n" <<
"    1                  toggle tree topology view on/off\n" <<
"    2                  toggle surface view on/off\n" <<
"    3                  toggle data view on/off\n" <<
"    G                  (\"geometry\") look at center of geometry\n" <<
"    H                  (\"home\") look at origin\n" <<
"    I                  toggle on-screen grid info on/off\n" <<
"    left mouse         tumble\n" <<
"    right mouse        pan\n" <<
"    mouse wheel        zoom\n" <<
"\n" <<
"    X + wheel          move right cut plane\n" <<
"    Shift + X + wheel  move left cut plane\n" <<
"    Y + wheel          move top cut plane\n" <<
"    Shift + Y + wheel  move bottom cut plane\n" <<
"    Z + wheel          move front cut plane\n" <<
"    Shift + Z + wheel  move back cut plane\n" <<
"    Ctrl + X + wheel   move both X cut planes\n" <<
"    Ctrl + Y + wheel   move both Y cut planes\n" <<
"    Ctrl + Z + wheel   move both Z cut planes\n";
    exit(status);
}


////////////////////////////////////////


int
main(int argc, char *argv[])
{
    const char* progName = argv[0];
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;

    int status = EXIT_SUCCESS;

    try {
        openvdb::initialize();
        openvdb::logging::initialize(argc, argv);

        bool printInfo = false, printGLInfo = false, printVersionInfo = false;

        // Parse the command line.
        std::vector<std::string> filenames;
        for (int n = 1; n < argc; ++n) {
            std::string str(argv[n]);
            if (str[0] != '-') {
                filenames.push_back(str);
            } else if (str == "-i") {
                printInfo = true;
            } else if (str == "-d") { // deprecated
                printGLInfo = true;
            } else if (str == "-h" || str == "-help" || str == "--help") {
                usage(progName, EXIT_SUCCESS);
            } else if (str == "-version" || str == "--version") {
                printVersionInfo = true;
                printGLInfo = true;
            } else {
                usage(progName, EXIT_FAILURE);
            }
        }
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        const bool hasInput = false;
#else
        const bool hasInput = !isatty(fileno(stdin));
#endif
        const size_t numFiles = filenames.size();

        if (printVersionInfo) {
            std::cout << "OpenVDB library version: "
                << openvdb::getLibraryAbiVersionString() << "\n";
            std::cout << "OpenVDB file format version: "
                << openvdb::OPENVDB_FILE_VERSION << std::endl;
            // If there are no files to view, don't print the OpenGL version,
            // since that would require opening a viewer window.
            if (!hasInput && numFiles == 0) return EXIT_SUCCESS;
        }
        if (!hasInput && numFiles == 0 && !printGLInfo) usage(progName, EXIT_FAILURE);

        openvdb_viewer::Viewer viewer = openvdb_viewer::init(progName, /*bg=*/false);

        if (printGLInfo) {
            // Now that the viewer window is open, we can get the OpenGL version, if requested.
            if (!printVersionInfo) {
                // Preserve the behavior of the deprecated -d option.
                std::cout << viewer.getVersionString() << std::endl;
            } else {
                // Print GLFW and OpenGL versions.
                std::cout << viewer.getGLFWVersionString() << std::endl;
                std::cout << viewer.getOpenGLVersionString() << std::endl;
            }
            if (!hasInput && numFiles == 0) return EXIT_SUCCESS;
        }

        std::string indent(numFiles == 1 ? "" : "    ");

        auto print_info = [&](openvdb::GridPtrVecPtr grids){
            for (size_t i = 0; i < grids->size(); ++i) {
                const std::string name = (*grids)[i]->getName();
                openvdb::Coord dim = (*grids)[i]->evalActiveVoxelDim();
                std::cout << indent << (name.empty() ? "<unnamed>" : name)
                          << " (" << dim[0] << " x " << dim[1] << " x " << dim[2]
                          << " voxels)" << std::endl;
            }
        };

        openvdb::GridCPtrVec allGrids;

        // Load VDB grids from stdin stream
        if (hasInput) {
            openvdb::io::Stream s(std::cin);
            openvdb::GridPtrVecPtr grids = s.getGrids();
            if (printInfo) print_info(grids);
            allGrids.insert(allGrids.end(), grids->begin(), grids->end());
        }

        // Load VDB grid from files.
        for (size_t n = 0; n < numFiles; ++n) {
            openvdb::io::File file(filenames[n]);
            file.open();

            openvdb::GridPtrVecPtr grids = file.getGrids();
            if (grids->empty()) {
                OPENVDB_LOG_WARN(filenames[n] << " is empty");
                continue;
            }
            allGrids.insert(allGrids.end(), grids->begin(), grids->end());

            if (printInfo) {
                if (numFiles > 1) std::cout << filenames[n] << ":\n";
                print_info(grids);
            }
        }

        viewer.open();
        viewer.view(allGrids);

        openvdb_viewer::exit();

    } catch (const char* s) {
        OPENVDB_LOG_FATAL(s);
        status = EXIT_FAILURE;
    } catch (std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        status = EXIT_FAILURE;
    } catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
        std::terminate();
    }

    return status;
}
