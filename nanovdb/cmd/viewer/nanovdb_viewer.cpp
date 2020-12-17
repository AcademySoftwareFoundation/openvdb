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
#include "StringUtils.h"

#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <iomanip>
#include <numeric>

static std::string makeStringCsv(const char* const* strs, int n)
{
    std::string str;
    if (n > 0) {
        str = strs[0];
        for (int i = 1; i < n; ++i) {
            str += std::string(",") + strs[i];
        }
    }
    return str;
}


template<>
struct ToString<nanovdb::Vec3f>
{
    inline std::string operator()(const nanovdb::Vec3f& v) const
    {
        std::ostringstream ss;
        ss << '(' << v[0] << ',' << v[1] << ',' << v[1] << ')';
        return ss.str();
    }
};

template<>
struct FromString<nanovdb::Vec3f>
{
    inline nanovdb::Vec3f operator()(const std::string& s) const
    {
        std::istringstream ss(s);
        char skip;
        float x,y,z;
        ss >> skip >> x >> skip >> y >> skip >> z >> skip;
        return nanovdb::Vec3f(x,y,z);
    }
};

struct ParamInfo
{
    std::string type;
    std::string description;
};

static const std::map<std::string, ParamInfo> kRenderParamMap{
    {"width", {"integer", "framebuffer width"}},
    {"height", {"integer", "framebuffer height"}},
    {"start", {"integer", "start frame"}},
    {"end", {"integer", "end frame"}},
    {"background", {"boolean", "use background"}},
    {"lighting", {"boolean", "use lighting"}},
    {"sun-direction", {"vec3f", "sun direction"}},
    {"shadows", {"boolean", "use shadows"}},
    {"ground", {"boolean", "use ground-plane"}},
    {"ground-reflections", {"boolean", "use ground-plane reflections"}},
    {"tonemapping", {"boolean", "use tonemapping"}},
    {"tonemapping-whitepoint", {"scalar", "tonemapping whitepoint"}},
    {"camera-target", {"vec3f", "camera target position, e.g. \"(0,10,0)\""}},
    {"camera-rotation", {"vec3f", "camera rotation in degrees. e.g. \"(22.5,90,0)\""}},
    {"camera-distance", {"scalar", "distance from camera target position"}},
    {"camera-fov", {"scalar", "camera field-of-view in degrees"}},
    {"camera-samples", {"integer", "camera samples per ray"}},
    {"camera-turntable", {"boolean", "use camera turntable"}},
    {"camera-turntable-rate", {"scalar", "camera turntable revolutions per frame-sequence"}},
    {"material-override", {"string", "the render method override: {" + makeStringCsv(kMaterialClassTypeStrings, (int)MaterialClass::kNumTypes) + "}"}},
    {"camera-lens", {"string", "the camera lens type: {" + makeStringCsv(kCameraLensTypeStrings, (int)Camera::LensType::kNumTypes) + "}"}},
    {"iterations", {"integer", "number of proressive iterations per frame."}},
    {"gold", {"filename", "The input filename (e.g. \"./reference/gold.%04d.png\""}},
    {"output", {"filename", "The output filename (e.g. \"output.%04d.jpg\""}},
    {"output-format", {"string", "the output format override: {png, jpg, tga, bmp, hdr, pfm}"}},
    {"material-volume-density", {"scalar", "density scaling factor for volume materials"}},
    {"material-blackbody-temperature", {"scalar", "temperature scaling factor for blackbody materials"}},
};

void usage [[noreturn]] (const std::string& progName, int exitStatus = EXIT_FAILURE)
{
    std::cerr << "\n"
              << "Usage: " << progName << " [options] <url>...\n"
              << "\n"
              << "Where URL is:\n"
              << "(<nodename>=)<url>(#<gridname>)([<start>-<end>])\n"
              << "\n"
              << "Render grids from one or more NanoVDB files\n"
              << "\n"
              << "--- General Options ---\n"
              << "-h,--help\tPrints this message\n"
              << "-b,--batch\tUse headless batch render\n"
              << "-p,--platform\tThe rendering platform to use by name\n"
              << "-l,--platform-list\tList the available rendering platforms\n"
              << "\n"
              << "--- Render Options ---\n";

    for (auto& param : kRenderParamMap) {
        std::cerr << std::left << "--render-" << std::setw(16) << param.first << "\t" << std::setw(12) << param.second.type << "\t" << param.second.description << "\n";
    }

    std::cerr
        << "\n"
        << "--- Examples ---\n"
        << "* Render blackbody grid using CUDA with 32 samples:\n"
        << "\t"
        << progName
        << "-p cuda --render-camera-samples 32 explode=explosion.vdb#density explode=explosion.vdb#temperature --render-material-override BlackBodyVolumePathTracer --render-material-blackbody-temperature 5.0\n"
        << "* Render density grid sequence of frames 1-10:\n"
        << "\t"
        << progName
        << " explosion_anim.%04d.vdb#density[0-10]\n"
        << "* Render single grid sequence of frames 1-5:\n"
        << "\t"
        << progName
        << " explosion_anim.%04d.vdb#[0-5]\n"
        << "\n";
    exit(exitStatus);
}

void version [[noreturn]] (const char* progName, int exitStatus = EXIT_SUCCESS)
{
    printf("\n%s was build against NanoVDB version %s\n", progName, nanovdb::Version().c_str());
    exit(exitStatus);
}

void printPlatformList()
{
    RenderLauncher renderLauncher;
    auto           names = renderLauncher.getPlatformNames();
    for (const auto& it : names) {
        std::cout << it << std::endl;
    }
}

int main(int argc, char* argv[])
{
    std::string                                       platformName;
    std::vector<std::pair<std::string, GridAssetUrl>> urls;
    RendererParams                                    rendererParams;

    bool batch = false;

    // make an invalid range.
    rendererParams.mFrameStart = 0;
    rendererParams.mFrameEnd = -1;

    StringMap renderStringParams;

    // make an invalid range.
    rendererParams.mFrameStart = 0;
    rendererParams.mFrameEnd = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-h" || arg == "--help") {
                usage(argv[0], EXIT_SUCCESS);
            } else if (arg == "--version") {
                version(argv[0]);
            } else if (arg == "-l" || arg == "--list") {
                printPlatformList();
                return 0;
            } else if (arg == "-b" || arg == "--batch") {
                batch = true;
            } else if (arg == "-p" || arg == "--platform") {
                if (i + 1 == argc) {
                    std::cerr << "\nExpected a string to follow the -p,--platform option\n";
                    usage(argv[0]);
                } else {
                    platformName.assign(argv[++i]);
                }
            } else if (arg.substr(0, 9) == "--render-") {
                // collect render options...
                for (auto& param : kRenderParamMap) {
                    if (arg.substr(9) == param.first) {
                        if (i + 1 == argc) {
                            std::cerr << "\nExpected a " << param.second.type << " to follow --render-" << param.first << "\n";
                            usage(argv[0]);
                        } else {
                            renderStringParams.set(param.first, argv[++i]);
                        }
                    }
                }
            } else {
                std::cerr << "\nIllegal option: \"" << arg << "\"\n";
                usage(argv[0]);
            }
        } else if (!arg.empty()) {
            // <nodeName>=<GridAssetUrl>
            std::string urlStr = arg;
            std::string nodeName = "";
            auto        pos = arg.find('=');
            if (pos != std::string::npos) {
                urlStr = arg.substr(pos + 1);
                nodeName = arg.substr(0, pos);
            }

            GridAssetUrl url(urlStr);
            urls.push_back(std::make_pair(nodeName, url));

            // update frame range...
            if (url.isSequence()) {
                if (rendererParams.mFrameEnd < rendererParams.mFrameStart) {
                    rendererParams.mFrameStart = url.frameStart();
                    rendererParams.mFrameEnd = url.frameEnd();
                } else {
                    rendererParams.mFrameStart = nanovdb::Min(rendererParams.mFrameStart, url.frameStart());
                    rendererParams.mFrameEnd = nanovdb::Max(rendererParams.mFrameEnd, url.frameEnd());
                }
            }
        }
    }

    // collect the final parameters...
    rendererParams.mFrameStart = renderStringParams.get<int>("start", rendererParams.mFrameStart);
    rendererParams.mFrameEnd = renderStringParams.get<int>("end", rendererParams.mFrameEnd);
    rendererParams.mWidth = renderStringParams.get<int>("width", rendererParams.mWidth);
    rendererParams.mHeight = renderStringParams.get<int>("height", rendererParams.mHeight);
    rendererParams.mOutputFilePath = renderStringParams.get<std::string>("output", rendererParams.mOutputFilePath);
    rendererParams.mOutputExtension = renderStringParams.get<std::string>("output-format", rendererParams.mOutputExtension);
    rendererParams.mGoldPrefix = renderStringParams.get<std::string>("gold", rendererParams.mGoldPrefix);
    rendererParams.mMaterialOverride = renderStringParams.getEnum<MaterialClass>("material-override", kMaterialClassTypeStrings, (int)MaterialClass::kNumTypes, rendererParams.mMaterialOverride);
    rendererParams.mMaterialBlackbodyTemperature = renderStringParams.get<float>("material-blackbody-temperature", rendererParams.mMaterialBlackbodyTemperature);
    rendererParams.mMaterialVolumeDensity = renderStringParams.get<float>("material-volume-density", rendererParams.mMaterialVolumeDensity);

    rendererParams.mSceneParameters.sunDirection = renderStringParams.get<nanovdb::Vec3f>("sun-direction", rendererParams.mSceneParameters.sunDirection);
    rendererParams.mSceneParameters.samplesPerPixel = renderStringParams.get<int>("camera-samples", rendererParams.mSceneParameters.samplesPerPixel);
    rendererParams.mSceneParameters.useBackground = renderStringParams.get<bool>("background", rendererParams.mSceneParameters.useBackground);
    rendererParams.mSceneParameters.useLighting = renderStringParams.get<bool>("lighting", rendererParams.mSceneParameters.useLighting);
    rendererParams.mSceneParameters.useShadows = renderStringParams.get<bool>("shadows", rendererParams.mSceneParameters.useShadows);
    rendererParams.mSceneParameters.useGround = renderStringParams.get<bool>("ground", rendererParams.mSceneParameters.useGround);
    rendererParams.mSceneParameters.useGroundReflections = renderStringParams.get<bool>("ground-reflections", rendererParams.mSceneParameters.useGroundReflections);
    rendererParams.mSceneParameters.useTonemapping = renderStringParams.get<bool>("tonemapping", rendererParams.mSceneParameters.useTonemapping);
    rendererParams.mSceneParameters.tonemapWhitePoint = renderStringParams.get<float>("tonemapping-whitepoint", rendererParams.mSceneParameters.tonemapWhitePoint);
    rendererParams.mSceneParameters.camera.lensType() = renderStringParams.getEnum<Camera::LensType>("camera-lens", kCameraLensTypeStrings, (int)Camera::LensType::kNumTypes, rendererParams.mSceneParameters.camera.lensType());
    rendererParams.mUseTurntable = renderStringParams.get<bool>("camera-turntable", rendererParams.mUseTurntable);
    rendererParams.mTurntableRate = renderStringParams.get<float>("camera-turntable-rate", rendererParams.mTurntableRate);
    rendererParams.mCameraFov = renderStringParams.get<float>("camera-fov", rendererParams.mCameraFov);
    rendererParams.mCameraDistance = renderStringParams.get<float>("camera-distance", rendererParams.mCameraDistance);
    rendererParams.mCameraTarget = renderStringParams.get<nanovdb::Vec3f>("camera-target", rendererParams.mCameraTarget);
    rendererParams.mCameraRotation = renderStringParams.get<nanovdb::Vec3f>("camera-rotation", rendererParams.mCameraRotation);
    rendererParams.mMaxProgressiveSamples = renderStringParams.get<int>("iterations", rendererParams.mMaxProgressiveSamples);

    // if range still invalid, then make a default frame range...
    if (rendererParams.mFrameEnd < rendererParams.mFrameStart) {
        rendererParams.mFrameStart = 0;
        rendererParams.mFrameEnd = 100;
    }

#if defined(__EMSCRIPTEN__)
    // if we are using emscripten, then use embedded file.
    // and preset some parameters that we know will work.
    platformName = "host";
    rendererParams.mUseAccumulation = false;
    rendererParams.mWidth = 64;
    rendererParams.mHeight = 64;
#endif

    std::cout << R"foo(Starting NanoVDB Viewer...
-------------------------------------------------------------------------------
Please note that the first time CUDA is used for rendering, the application
may stall while CUDA is compiling code for your specific GPU architecture.
This is perfectly normal, and will only happen ONCE after source-code is built.
-------------------------------------------------------------------------------
)foo";

    try {
        std::unique_ptr<RendererBase> renderer;
        if (batch) {
            renderer.reset(new BatchRenderer(rendererParams));
        } else {
#if defined(NANOVDB_USE_GLFW)
            renderer.reset(new Viewer(rendererParams));
#else
            std::cerr << "Warning: GLFW was not enabled in your build configuration. Using batch mode.\n";
            renderer.reset(new BatchRenderer(rendererParams));
#endif
        }

        if (platformName.empty() == false) {
            if (renderer->setRenderPlatformByName(platformName) == false) {
                std::cerr << "Unrecognized platform: " << platformName << std::endl;
                return EXIT_FAILURE;
            }
        }

        if (urls.size() > 0) {
            // ensure only one node is made for each specified node name.
            std::map<std::string, std::vector<GridAssetUrl>> nodeGridMap;
            for (auto& nodeUrlPairs : urls) {
                nodeGridMap[nodeUrlPairs.first].push_back(nodeUrlPairs.second);
            }

            // attach the grids.
            for (auto& it : nodeGridMap) {
                if (it.first.length()) {
                    // attach grids to one scene node...
                    int attachmentIndex = 0;
                    auto nodeId = renderer->addSceneNode(it.first);
                    for (size_t i = 0; i < it.second.size(); ++i) {
                        auto& assetUrl = it.second[i];
                        if (assetUrl.scheme() == "file" && assetUrl.gridName().empty()) {
                            auto gridNames = renderer->getGridNamesFromFile(assetUrl);
                            for (auto& gridName : gridNames) {
                                assetUrl.gridName() = gridName;
                                renderer->addGridAsset(assetUrl);
                                renderer->setSceneNodeGridAttachment(nodeId, attachmentIndex++, assetUrl);
                            }
                        } else {
                            renderer->addGridAsset(assetUrl);
                            renderer->setSceneNodeGridAttachment(nodeId, attachmentIndex++, assetUrl);
                        }
                    }
                } else {
                    // create scene nodes for each grid...
                    int nodeIndex = renderer->addGridAssetsAndNodes("default", it.second);
                    if (nodeIndex == -1) {
                        throw std::runtime_error("Some assets have errors.");
                    }
                }
            }

            renderer->selectSceneNodeByIndex(0);
#if 1
            if (auto node = renderer->findNodeByIndex(0)) {
                // waiting for load will enable the frameing to work when we reset the camera!
                if (!renderer->updateNodeAttachmentRequests(node, true, true)) {
                    throw std::runtime_error("Some assets have errors. Unable to render scene node " + node->mName + "; bad asset");
                }
            }
#endif
            renderer->resetCamera();
        }

        renderer->open();
        renderer->run();
        renderer->close();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Exception of unexpected type caught" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
