// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file   nanovdb_viewer.cpp

	\author Wil Braithwaite

	\date   April 26, 2020

	\brief  nanovdb batch-renderer/viewer.
*/

#if defined(NANOVDB_USE_GLFW)
#include "Viewer.h"
#endif
#include "BatchRenderer.h"

#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host

void usage [[noreturn]] (const std::string& progName, int exitStatus = EXIT_FAILURE)
{
    std::cerr << "\n"
              << "Usage: " << progName << " [options] *.nvdb\n"
              << "Description: Render grids from one or more NanoVDB files\n"
              << "\n"
              << "Options:\n"
              << "-h,--help\tPrints this message\n"
              << "-g,--grid name\tView all grids matching the specified string name\n"
              << "-b,--batch\tUse headless batch render\n"
              << "-p,--render-platform\tThe rendering platform to use by name\n"
              << "-o,--output\tThe output filename prefix (format = ./<output>.frame.ext)\n"
              << "-l,--render-platform-list\tList the available rendering platforms\n"
              << "-n,--count\trender <count> frames\n"
              << "--turntable\tRender a 360 turntable within the frame count\n"
              << "--width\tThe render width\n"
              << "--height\tThe render height\n"
              << "--samples\tThe render sample count\n"
              << "\n"
              << "Examples:\n"
              << "* Render temperature grid using CUDA with 32 samples:\n"
              << "\t" << progName << " -p cuda --grid temperature --samples 32 explosion.0023.vdb\n"
              << "* Render density grid sequence:\n"
              << "\t" << progName << " --grid density explosion.%04d.vdb:0-100\n"
              << "\n";
    exit(exitStatus);
}

int main(int argc, char* argv[])
{
    int exitStatus = EXIT_SUCCESS;

    std::string                                      platformName;
    std::string                                      gridName;
    std::vector<std::pair<std::string, std::string>> fileNames;
    RendererParams                                   rendererParams;
    bool                                             batch = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-h" || arg == "--help") {
                usage(argv[0], EXIT_SUCCESS);
            } else if (arg == "-g" || arg == "--grid") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a grid name to follow the -g,--grid option\n";
                    usage(argv[0]);
                } else {
                    gridName.assign(argv[++i]);
                }
            } else if (arg == "-l" || arg == "--list") {
                RenderLauncher renderLauncher;
                auto           names = renderLauncher.getPlatformNames();
                for (const auto& it : names) {
                    std::cout << it << std::endl;
                }
                return 0;
            } else if (arg == "-b" || arg == "--batch") {
                batch = true;
            } else if (arg == "-o" || arg == "--output") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a filename to follow the -o,--output option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mOutputPrefix.assign(argv[++i]);
                }
            } else if (arg == "--gold") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a filename to follow the --gold option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mGoldPrefix.assign(argv[++i]);
                }
            } else if (arg == "--samples") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected an integer to follow the --samples option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mOptions.samplesPerPixel = atoi(argv[++i]);
                }
            } else if (arg == "--width") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected an integer to follow the --width option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mWidth = atoi(argv[++i]);
                }
            } else if (arg == "--height") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected an integer to follow the --height option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mHeight = atoi(argv[++i]);
                }
            } else if (arg == "-n" || arg == "--count") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected an integer to follow the -n,--count option\n";
                    usage(argv[0]);
                } else {
                    rendererParams.mFrameCount = atoi(argv[++i]);
                }
            } else if (arg == "--turntable") {
                rendererParams.mUseTurntable = true;
            } else if (arg == "-p" || arg == "--platform") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a string to follow the -p,--platform option\n";
                    usage(argv[0]);
                } else {
                    platformName.assign(argv[++i]);
                }
            } else {
                std::cerr << "\nUnrecognized option: \"" << arg << "\"\n";
                usage(argv[0]);
            }
        } else if (!arg.empty()) {
            // check for sequence...
            if (arg.find("%", 0) != std::string::npos) {
                auto pos = arg.find_last_of(':');
                auto range = arg.substr(pos + 1);
                auto filename = arg.substr(0, pos);

                int start = 0, end = rendererParams.mFrameCount;

                if ((pos = range.find('-', 0)) != std::string::npos) {
                    start = atoi(range.substr(0, pos).c_str());
                    end = atoi(range.substr(pos + 1).c_str());
                }

                if (end - start == 0) {
                    std::cerr << "Invalid filename range\n";
                    exit(1);
                }

                char fileNameBuf[FILENAME_MAX];
                for (int i = start; i < end; ++i) {
                    sprintf(fileNameBuf, filename.c_str(), i);
                    //std::cout << "filename: " << fileNameBuf << "\n";
                    fileNames.push_back(std::make_pair(arg, fileNameBuf));
                }
            } else {
                fileNames.push_back(std::make_pair(arg, arg));
            }
        }
    }

    if (fileNames.size() == 0) {
        fileNames.push_back(std::make_pair("__internal", "internal://points_sphere_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://points_box_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://points_torus_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://fog_sphere_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://fog_box_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://fog_torus_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://ls_sphere_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://ls_box_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://ls_bbox_100"));
        fileNames.push_back(std::make_pair("__internal", "internal://ls_torus_100"));
    }

#if defined(__EMSCRIPTEN__)
    // if we are using emscripten, then use embedded file.
    // and preset some parameters that we know will work.
    platformName = "host";
    rendererParams.mUseAccumulation = false;
    rendererParams.mWidth = 64;
    rendererParams.mHeight = 64;
#endif

    std::unique_ptr<RendererBase> renderer;

#if defined(NANOVDB_USE_GLFW)
    if (batch)
        renderer.reset(new BatchRenderer(rendererParams));
    else
        renderer.reset(new Viewer(rendererParams));
#else
    renderer.reset(new BatchRenderer(rendererParams));
#endif

    if (platformName.empty() == false) {
        if (renderer->setRenderPlatformByName(platformName) == false) {
            std::cerr << "Unrecognized platform: " << platformName << std::endl;
            return exitStatus;
        }
    }

    const auto nameKey = nanovdb::io::stringHash(gridName);

    try {
        for (auto& file : fileNames) {
            if (gridName.empty())
                renderer->addGrid(file.first, file.second);
            else
                renderer->addGrid(file.first, file.second, gridName);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unexpected type caught" << std::endl;
        return 1;
    }

    try {
        renderer->open();
        renderer->run();
        renderer->close();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    catch (...) {
        std::cerr << "Exception of unexpected type caught" << std::endl;
    }

    return exitStatus;
}
