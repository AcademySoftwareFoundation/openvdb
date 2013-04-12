///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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

#ifdef DWA_OPENVDB

#include <pdevunit/pdevunit.h>
#include <logging/logging.h>

#else

#include <cstdlib> // for exit()
#include <cstring> // for strrchr()
#include <iostream>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#ifdef _WIN32
#include <openvdb/port/getopt.c>
#else
#include <unistd.h> // for getopt(), optarg
#endif


static void
dump(CppUnit::Test* test)
{
    if (test == NULL) {
        std::cerr << "Error: no tests found\n";
        return;
    }

    std::cout << test->getName() << std::endl;
    for (int i = 0; i < test->getChildTestCount(); i++) {
        dump(test->getChildTestAt(i));
    }
}


int
run(int argc, char* argv[])
{
    int verbose = 0;
    std::string tests;
    int c = -1;
    while ((c = getopt(argc, argv, "lt:v")) != -1) {
        switch (c) {
            case 'l':
            {
                dump(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
                return EXIT_SUCCESS;
            }
            case 'v': verbose = 1; break;
            case 't': if (optarg) tests = optarg; break;
            default:
            {
                const char* prog = argv[0];
                if (const char* ptr = ::strrchr(prog, '/')) prog = ptr + 1;
                std::cerr <<
"Usage: " << prog << " [options]\n" <<
"Which: runs OpenVDB library unit tests\n" <<
"Options:\n" <<
"    -l       list all available tests\n" <<
"    -t test  specific suite or test to run, e.g., \"-t TestGrid\"\n" <<
"             or \"-t TestGrid::testGetGrid\" (default: run all tests)\n" <<
"    -v       verbose output\n";
                return EXIT_FAILURE;
            }
        }
    }

    try {
        CppUnit::TestFactoryRegistry& registry =
            CppUnit::TestFactoryRegistry::getRegistry();

        CppUnit::TestRunner runner;
        runner.addTest(registry.makeTest());

        CppUnit::TestResult controller;

        CppUnit::TestResultCollector result;
        controller.addListener(&result);

        CppUnit::TextTestProgressListener progress;
        CppUnit::BriefTestProgressListener vProgress;
        if (verbose) {
            controller.addListener(&vProgress);
        } else {
            controller.addListener(&progress);
        }

        runner.run(controller, tests);

        CppUnit::CompilerOutputter outputter(&result, std::cerr);
        outputter.write();

        return result.wasSuccessful() ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
#endif


int
main(int argc, char *argv[])
{
#ifdef DWA_OPENVDB
    // Disable logging by default ("-quiet") unless overridden
    // with "-debug" or "-info".
    bool quiet = false;
    {
        std::vector<char*> args(argv, argv + argc);
        int numArgs = int(args.size());
        logging_base::Config config(numArgs, &args[0]);
        quiet = (!config.useInfo() && !config.useDebug());
    }
    char* quietArg = "-quiet";
    std::vector<char*> args(argv, argv + argc);
    if (quiet) args.insert(++args.begin(), quietArg);
    int numArgs = int(args.size());

    logging_base::Config config(numArgs, &args[0]);
    logging_base::configure(config);

    return pdevunit::run(numArgs, &args[0]);
#else
    return run(argc, argv);
#endif
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
