// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file main.cpp
///
/// @brief One-stop command-line tool for printing, converting, processing and
///        and rendering of VDB grids.
///
////////////////////////////////////////////////////////////////////////////////


#include "Tool.h"

#if defined(_WIN32)
#include <fcntl.h>// for _O_BINARY
#include <io.h>// for _setmode and _fileno
#endif

int main(int argc, char *argv[])
{
    int exitStatus = EXIT_SUCCESS;

    try {

#if defined(_WIN32)
        // Preserve binary stdin/stdout before any I/O; diagnostics use stderr.
        if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
            throw std::runtime_error("Failed to set stdin to binary mode");
        }
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
            throw std::runtime_error("Failed to set stdout to binary mode");
        }
#endif

        openvdb::vdb_tool::Tool tool(argc, argv);
        if (!tool.run()) exitStatus = EXIT_FAILURE;

    } catch (const std::exception& e) {

        std::cerr << "Fatal error in " << argv[0] << ":\n\t" << e.what() << std::endl;
        exitStatus = EXIT_FAILURE;

    } catch (...) {

        std::cerr << "Fatal error in " << argv[0] << ":\n\texception of unknown type caught" << std::endl;
        exitStatus = EXIT_FAILURE;
        std::terminate();

    }

    return exitStatus;
}
