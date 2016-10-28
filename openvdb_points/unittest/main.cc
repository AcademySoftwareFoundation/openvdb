///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
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


#ifdef DWA_OPENVDB

#include <pdevunit/pdevunit.h>
#include <logging_base/logging.h>

#else

#include <cstdlib> // for EXIT_SUCCESS
#include <cstring> // for strrchr()
#include <iostream>
#include <string>
#include <vector>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#ifdef OPENVDB_USE_LOG4CPLUS
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#endif


void
usage(const char* progName)
{
    std::cerr <<
        "Usage: " << progName << " [options]\n" <<
        "Which: runs OpenVDB Points library unit tests\n" <<
        "Options:\n" <<
        "    -l       list all available tests\n" <<
        "    -t test  specific suite or test to run, e.g., \"-t TestGrid\"\n" <<
        "             or \"-t TestGrid::testGetGrid\" (default: run all tests)\n" <<
        "    -v       verbose output\n";
#ifdef OPENVDB_USE_LOG4CPLUS
    std::cerr << "\n" <<
        "    -error   log fatal and non-fatal errors (default: log only fatal errors)\n" <<
        "    -warn    log warnings and errors\n" <<
        "    -info    log info messages, warnings and errors\n" <<
        "    -debug   log debugging messages, info messages, warnings and errors\n";
#endif
}


static void
dump(CppUnit::Test* test)
{
    if (test == nullptr) {
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
    const char* progName = argv[0];
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;

    bool verbose = false;
    std::vector<std::string> tests;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-l") {
            dump(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
            return EXIT_SUCCESS;
        } else if (arg == "-v") {
            verbose = true;
        } else if (arg == "-t") {
            if (i + 1 < argc) {
                ++i;
                tests.push_back(argv[i]);
            } else {
                usage(progName);
                return EXIT_FAILURE;
            }
        } else if (arg == "-h" || arg == "-help" || arg == "--help") {
            usage(progName);
            return EXIT_SUCCESS;
        } else {
            std::cerr << progName << ": unrecognized option '" << arg << "'\n";
            usage(progName);
            return EXIT_FAILURE;
        }
    }
    if (tests.empty()) tests.push_back(""); // run all tests

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

        for (size_t i = 0; i < tests.size(); ++i) {
            runner.run(controller, tests[i]);
        }

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
    const std::string quietArg("-quiet");
    std::vector<const char*> args(argv, argv + argc);
    if (quiet) args.insert(++args.begin(), quietArg.c_str());
    int numArgs = int(args.size());

    logging_base::Config config(numArgs, &args[0]);
    logging_base::configure(config);

    return pdevunit::run(numArgs, const_cast<char**>(&args[0]));

#else // ifndef DWA_OPENVDB

#ifndef OPENVDB_USE_LOG4CPLUS
    return run(argc, argv);
#else
    log4cplus::BasicConfigurator::doConfigure();

    std::vector<char*> args{argv[0]};

    log4cplus::Logger log = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("main"));
    log.setLogLevel(log4cplus::FATAL_LOG_LEVEL);
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        if (std::string("-info") == arg)       log.setLogLevel(log4cplus::INFO_LOG_LEVEL);
        else if (std::string("-warn") == arg)  log.setLogLevel(log4cplus::WARN_LOG_LEVEL);
        else if (std::string("-error") == arg) log.setLogLevel(log4cplus::ERROR_LOG_LEVEL);
        else if (std::string("-debug") == arg) log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);
        else args.push_back(arg);
    }

    return run(int(args.size()), &args[0]);
#endif // OPENVDB_USE_LOG4CPLUS

#endif // DWA_OPENVDB
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
