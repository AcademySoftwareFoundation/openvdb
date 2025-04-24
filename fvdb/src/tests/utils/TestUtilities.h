// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_TESTS_UTILS_TESTUTILITIES_H
#define FVDB_TESTS_UTILS_TESTUTILITIES_H

namespace fvdb::test {

// Prints empty green brackets followed by the specified name to match google test output
void
printSubtestPrefix(std::string const &name) {
    std::cout << "\033[32m[          ]\033[0m " << name << " ";
}

// Prints a green OK message to match google test output
void
printGreenOK() {
    std::cout << "\033[32mOK\033[0m" << std::endl;
}

} // namespace fvdb::test

#endif // FVDB_TESTS_UTILS_TESTUTILITIES_H
