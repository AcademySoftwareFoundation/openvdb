///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "util.h"

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class TestFunctionRegistry : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestFunctionRegistry);
    CPPUNIT_TEST(testCreateAllVerify);
    CPPUNIT_TEST_SUITE_END();

    void testCreateAllVerify();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFunctionRegistry);

void
TestFunctionRegistry::testCreateAllVerify()
{
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry();
    openvdb::ax::FunctionOptions opts;

    // check that no warnings are printed during registration
    // @todo  Replace this with a better logger once AX has one!

    std::streambuf* sbuf = std::cerr.rdbuf();

    try {
        // Redirect cerr
        std::stringstream buffer;
        std::cerr.rdbuf(buffer.rdbuf());
        reg->createAll(opts, true);
        const std::string& result = buffer.str();
        CPPUNIT_ASSERT(result.empty());
    }
    catch (...) {
        std::cerr.rdbuf(sbuf);
        throw;
    }

    std::cerr.rdbuf(sbuf);
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
