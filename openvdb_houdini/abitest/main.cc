// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>

#define ADD_DECLARE_SUFFIX(name) name ## TEST
#define ADD_DEFINE_SUFFIX(name) name ## MAIN

#include "TestABI.h"

int test()
{
    {
        // verify the ABI matches
        const std::string abiTest = getABI_TEST();
        const std::string abiMain = getABI_MAIN();
        if (abiTest != abiMain) {
            std::stringstream ss;
            ss << "Error: Mismatching ABIs for ABI Test - "
                << abiTest << " vs " << abiMain;
            throw std::runtime_error(ss.str());
        }

        // output a warning if the namespaces match
        const std::string namespaceTest = getNamespace_TEST();
        const std::string namespaceMain = getNamespace_MAIN();
        if (namespaceTest == namespaceMain) {
            std::cerr << "Warning: Namespace names match, "
                << "so this test is not expected to fail." << std::endl;
        }
    }

    { // check ABI from TEST to MAIN
        void* grid = createGrid_TEST();
        validateGrid_MAIN(grid);
        cleanupGrid_TEST(grid);
    }

    { // check ABI from MAIN to TEST
        void* grid = createGrid_MAIN();
        validateGrid_TEST(grid);
        cleanupGrid_MAIN(grid);
    }

    return 0;
}

int
main(int, char**)
{
    openvdb::initialize();

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
