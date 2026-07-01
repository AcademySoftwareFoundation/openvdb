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

int main(int argc, char *argv[])
{
    int exitStatus = EXIT_SUCCESS;

    try {

        openvdb::vdb_tool::Tool tool(argc, argv);
        tool.run();

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
