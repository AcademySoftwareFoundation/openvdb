// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/io/IO.h>
#include "VoxToNanoVDB.h"

/// @brief Convert an .vox file into a .nvdb file.
///
/// @note This example only depends on NanoVDB.
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.vox>"
                  << " (<out.nvdb>)" << std::endl;
        return 1;
    }

    std::string inFilename(argv[1]), gridName("Vox model");
    std::string outFilename("vox_to_nanovdb_output.nvdb");
    if (argc > 2)
        outFilename = std::string(argv[2]);

    try {
        auto handle = convertVoxToNanoVDB(inFilename, gridName);
        nanovdb::io::writeGrid<nanovdb::HostBuffer>(outFilename, handle, nanovdb::io::Codec::ZIP, 1); // Write the NanoVDB grid to file and throw if writing fails
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
