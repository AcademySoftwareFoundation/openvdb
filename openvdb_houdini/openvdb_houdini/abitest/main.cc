// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>

// include method declarations both inside and outside houdini namespace
#include "TestABI.h"
namespace houdini {
#include "TestABI.h"
} // namespace houdini

int test()
{
    {
        // verify the ABI matches
        const std::string abiTest = houdini::getABI();
        const std::string abiMain = getABI();
        if (abiTest != abiMain) {
            std::stringstream ss;
            ss << "Error: Mismatching ABIs for ABI Test - "
                << abiTest << " vs " << abiMain;
            throw std::runtime_error(ss.str());
        }

        // output a warning if the namespaces match
        const std::string namespaceTest = houdini::getNamespace();
        const std::string namespaceMain = getNamespace();
        if (namespaceTest == namespaceMain) {
            std::cerr << "Warning: Namespace names match, "
                << "so this test is not expected to fail." << std::endl;
        }
    }

    { // check ABI from Houdini to non-Houdini
        void* grid = houdini::createFloatGrid();
        validateFloatGrid(grid);
        houdini::cleanupFloatGrid(grid);

        grid = houdini::createPointsGrid();
        validatePointsGrid(grid);
        houdini::cleanupPointsGrid(grid);
    }

    { // check ABI from non-Houdini to Houdini
        void* grid = createFloatGrid();
        houdini::validateFloatGrid(grid);
        cleanupFloatGrid(grid);

        grid = createPointsGrid();
        houdini::validatePointsGrid(grid);
        cleanupPointsGrid(grid);
    }

    return 0;
}

int
main(int, char**)
{
    try {
        test();
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown Error " << std::endl;
        return 1;
    }

    return 0;
}
