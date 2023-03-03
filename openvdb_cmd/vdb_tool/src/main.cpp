// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

        OPENVDB_LOG_FATAL(std::string("Tool: ")+e.what());
        exitStatus = EXIT_FAILURE;

    } catch (...) {

        OPENVDB_LOG_FATAL("Tool: exception of unknown type caught");
        exitStatus = EXIT_FAILURE;
        std::terminate();

    }

    return exitStatus;
}
