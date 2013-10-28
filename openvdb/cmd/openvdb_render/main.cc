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
//
/// @file main.cc
///
/// @brief Simple ray tracer for OpenVDB volumes
///
/// @note This is intended mainly as an example of how to ray-trace OpenVDB volumes.
/// It is not a production-quality renderer, and it is currently limited to
/// rendering level set volumes.

#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/scoped_ptr.hpp>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>
#ifdef DWA_OPENVDB
#include <logging_base/logging.h>
#include <usagetrack.h>
#endif


namespace {

const char* gProgName = "";


struct RenderOpts
{
    std::string shader;
    std::string camera;
    float aperture, focal, frame, znear, zfar;
    openvdb::Vec3R rotate;
    openvdb::Vec3R translate;
    openvdb::Vec3R target;
    openvdb::Vec3R up;
    bool lookat;
    size_t samples;
    size_t width, height;
    std::string compression;
    int threads;
    bool verbose;

    RenderOpts():
        shader("diffuse"),
        camera("perspective"),
        aperture(41.2136),
        focal(50.0),
        frame(1.0),
        znear(1.0e-3),
        zfar(std::numeric_limits<double>::max()),
        rotate(0.0),
        translate(0.0),
        target(0.0),
        up(0.0, 1.0, 0.0),
        lookat(false),
        samples(1),
        width(1920),
        height(1080),
        compression("zip"),
        threads(0),
        verbose(false)
    {}

    std::string validate() const
    {
        if (shader != "diffuse" && shader != "matte" && shader != "normal") {
            return "expected diffuse, matte or normal shader, got \"" + shader + "\"";
        }
        if (!boost::starts_with(camera, "ortho") && !boost::starts_with(camera, "persp")) {
            return "expected perspective or orthographic camera, got \"" + camera + "\"";
        }
        if (compression != "none" && compression != "rle" && compression != "zip") {
            return "expected none, rle or zip compression, got \"" + compression + "\"";
        }
        if (width < 1 || height < 1) {
            std::ostringstream ostr;
            ostr << "expected width > 0 and height > 0, got " << width << "x" << height;
            return ostr.str();
        }
        return "";
    }

    std::ostream& put(std::ostream& os) const
    {
        os << "-aperture " << aperture
           << " -camera " << camera
           << " -compression " << compression
           << " -cpus " << threads
           << " -far " << zfar
           << " -focal " << focal
           << " -frame " << frame;
        if (lookat) os << " -lookat " << target[0] << "," << target[1] << "," << target[2];
        os << " -near " << znear
           << " -res " << width << "x" << height;
        if (!lookat) os << " -rotate " << rotate[0] << "," << rotate[1] << "," << rotate[2];
        os << " -shader " << shader
           << " -samples " << samples
           << " -translate " << translate[0] << "," << translate[1] << "," << translate[2];
        if (lookat) os << " -up " << up[0] << "," << up[1] << "," << up[2];
        if (verbose) os << " -v";
        return os;
    }
};

std::ostream& operator<<(std::ostream& os, const RenderOpts& opts) { return opts.put(os); }


void
usage(int exitStatus = EXIT_FAILURE)
{
    RenderOpts opts; // default options
    const float fov = openvdb::tools::PerspectiveCamera::focalLengthToFieldOfView(
        opts.focal, opts.aperture);

    std::ostringstream ostr;
    ostr << std::setprecision(3) <<
"Usage: " << gProgName << " in.vdb out.{exr,ppm} [options]\n" <<
"Which: ray-traces OpenVDB volumes\n" <<
"Options:\n" <<
"    -aperture F       perspective camera aperture in mm (default: " << opts.aperture << ")\n" <<
"    -camera S         camera type; either \"persp[ective]\" or \"ortho[graphic]\"\n" <<
"                      (default: " << opts.camera << ")\n" <<
"    -compression S    EXR compression scheme; either \"none\" (uncompressed),\n" <<
"                      \"rle\" or \"zip\" (default: " << opts.compression << ")\n" <<
"    -cpus N           number of rendering threads, or 1 to disable threading,\n" <<
"                      or 0 to use all available CPUs (default: " << opts.threads << ")\n" <<
"    -far F            camera far plane depth (default: " << opts.zfar << ")\n" <<
"    -focal F          perspective camera focal length in mm (default: " << opts.focal << ")\n" <<
"    -fov F            perspective camera field of view in degrees\n" <<
"                      (default: " << fov << ")\n" <<
"    -frame F          ortho camera frame width in world units (default: "
    << opts.frame << ")\n" <<
"    -lookat X,Y,Z     rotate the camera to point to (X, Y, Z)\n" <<
"    -name S           name of the grid to be rendered (default: render\n" <<
"                      the first floating-point grid found in in.vdb)\n" <<
"    -near F           camera near plane depth (default: " << opts.znear << ")\n" <<
"    -res WxH          image dimensions in pixels (default: "
    << opts.width << "x" << opts.height << ")\n" <<
"    -r X,Y,Z                                    \n" <<
"    -rotate X,Y,Z     camera rotation in degrees\n" <<
"                      (default: look at the center of the grid)\n" <<
"    -shader S         shader name; either \"diffuse\", \"matte\" or \"normal\"\n" <<
"                      (default: " << opts.shader << ")\n" <<
"    -samples N        number of samples (rays) per pixel (default: " << opts.samples << ")\n" <<
"    -t X,Y,Z                            \n" <<
"    -translate X,Y,Z  camera translation\n" <<
"    -up X,Y,Z         vector that should point up after rotation with -lookat\n" <<
"                      (default: " << opts.up << ")\n" <<
"\n" <<
"    -v                verbose (print timing and diagnostics)\n" <<
"    -h, -help         print this usage message and exit\n" <<
"\n" <<
"Example:\n" <<
"    " << gProgName << " crawler.vdb crawler.exr -shader diffuse -res 1920x1080 \\\n" <<
"        -focal 35 -samples 4 -translate 0,210.5,400 -compression rle -v\n" <<
"\n" <<
"This is not (and is not intended to be) a production-quality renderer,\n" <<
"and it is currently limited to rendering level set volumes.\n";

    std::cerr << ostr.str();
    exit(exitStatus);
}


void
saveEXR(const std::string& fname, const openvdb::tools::Film& film, const RenderOpts& opts)
{
    typedef openvdb::tools::Film::RGBA RGBA;

    std::string filename = fname;
    if (!boost::iends_with(filename, ".exr")) filename += ".exr";

    if (opts.verbose) {
        std::cout << gProgName << ": writing " << filename << "..." << std::endl;
    }

    const tbb::tick_count start = tbb::tick_count::now();

    int threads = (opts.threads == 0 ? 8 : opts.threads);
    Imf::setGlobalThreadCount(threads);

    Imf::Header header(film.width(), film.height());
    if (opts.compression == "none") {
        header.compression() = Imf::NO_COMPRESSION;
    } else if (opts.compression == "rle") {
        header.compression() = Imf::RLE_COMPRESSION;
    } else if (opts.compression == "zip") {
        header.compression() = Imf::ZIP_COMPRESSION;
    } else {
        OPENVDB_THROW(openvdb::ValueError,
            "expected none, rle or zip compression, got \"" << opts.compression << "\"");
    }
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
    header.channels().insert("A", Imf::Channel(Imf::FLOAT));

    const size_t pixelBytes = sizeof(RGBA), rowBytes = pixelBytes * film.width();
    RGBA& pixel0 = const_cast<RGBA*>(film.pixels())[0];
    Imf::FrameBuffer framebuffer;
    framebuffer.insert("R",
        Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.r), pixelBytes, rowBytes));
    framebuffer.insert("G",
        Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.g), pixelBytes, rowBytes));
    framebuffer.insert("B",
        Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.b), pixelBytes, rowBytes));
    framebuffer.insert("A",
        Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&pixel0.a), pixelBytes, rowBytes));

    Imf::OutputFile imgFile(filename.c_str(), header);
    imgFile.setFrameBuffer(framebuffer);
    imgFile.writePixels(film.height());

    if (opts.verbose) {
        std::ostringstream ostr;
        ostr << gProgName << ": ...completed in " << std::setprecision(3)
            << (tbb::tick_count::now() - start).seconds() << " sec";
        std::cout << ostr.str() << std::endl;
    }
}


template<typename GridType>
void
render(const GridType& grid, const std::string& imgFilename, const RenderOpts& opts)
{
    using namespace openvdb;

    const GridClass cls = grid.getGridClass();
    if (cls != GRID_LEVEL_SET) {
        OPENVDB_THROW(ValueError, "expected level set, got " << GridBase::gridClassToString(cls));
    }

    tools::Film film(opts.width, opts.height);

    boost::scoped_ptr<tools::BaseCamera> camera;
    if (boost::starts_with(opts.camera, "persp")) {
        camera.reset(new tools::PerspectiveCamera(film, opts.rotate, opts.translate,
            opts.focal, opts.aperture, opts.znear, opts.zfar));
    } else if (boost::starts_with(opts.camera, "ortho")) {
        camera.reset(new tools::OrthographicCamera(film, opts.rotate, opts.translate,
            opts.frame, opts.znear, opts.zfar));
    } else {
        OPENVDB_THROW(ValueError,
            "expected perspective or orthographic camera, got \"" << opts.camera << "\"");
    }
    if (opts.lookat) camera->lookAt(opts.target, opts.up);

    boost::scoped_ptr<tools::BaseShader> shader;
    if (opts.shader == "diffuse") {
        shader.reset(new tools::DiffuseShader);
    } else if (opts.shader == "matte") {
        shader.reset(new tools::MatteShader);
    } else if (opts.shader == "normal") {
        shader.reset(new tools::NormalShader);
    } else {
        OPENVDB_THROW(ValueError,
            "expected diffuse, matte or normal shader, got \"" << opts.shader << "\"");
    }

    if (opts.verbose) {
        std::cout << gProgName << ": ray-tracing..." << std::endl;
    }
    const tbb::tick_count start = tbb::tick_count::now();

    tools::rayTrace(grid, *shader, *camera, opts.samples, 0, (opts.threads != 1));

    if (opts.verbose) {
        std::ostringstream ostr;
        ostr << gProgName << ": ...completed in " << std::setprecision(3)
            << (tbb::tick_count::now() - start).seconds() << " sec";
        std::cout << ostr.str() << std::endl;
    }

    if (boost::iends_with(imgFilename, ".ppm")) {
        // Save as PPM (fast, but large file size).
        std::string filename = imgFilename;
        filename.erase(filename.size() - 4); // strip .ppm extension
        film.savePPM(filename);
    } else if (boost::iends_with(imgFilename, ".exr")) {
        // Save as EXR (slow, but small file size).
        saveEXR(imgFilename, film, opts);
    } else {
        OPENVDB_THROW(ValueError, "unsupported image file format (" + imgFilename + ")");
    }
}


void
strToSize(const std::string& s, size_t& x, size_t& y)
{
    std::vector<std::string> elems;
    boost::split(elems, s, boost::algorithm::is_any_of(",x"));
    const size_t numElems = elems.size();
    if (numElems > 0) x = size_t(std::max(0, atoi(elems[0].c_str())));
    if (numElems > 1) y = size_t(std::max(0, atoi(elems[1].c_str())));
}


openvdb::Vec3R
strToVec3R(const std::string& s)
{
    openvdb::Vec3R result;
    std::vector<std::string> elems;
    boost::split(elems, s, boost::algorithm::is_any_of(","));
    for (size_t i = 0, N = elems.size(); i < N; ++i) {
        result[i] = atof(elems[i].c_str());
    }
    return result;
}


struct OptParse
{
    int argc;
    char** argv;

    OptParse(int argc_, char* argv_[]): argc(argc_), argv(argv_) {}

    bool check(int idx, const std::string& name, int numArgs = 1) const
    {
        if (argv[idx] == name) {
            if (idx + numArgs >= argc) {
                std::cerr << gProgName << ": option " << name << " requires "
                    << numArgs << " argument" << (numArgs == 1 ? "" : "s") << "\n";
                usage();
            }
            return true;
        }
        return false;
    }
};

} // unnamed namespace


int
main(int argc, char *argv[])
{
#ifdef DWA_OPENVDB
    USAGETRACK_report_basic_tool_usage(argc, argv, /*duration=*/0);
    logging_base::configure(argc, argv);
#endif

    OPENVDB_START_THREADSAFE_STATIC_WRITE
    gProgName = argv[0];
    if (const char* ptr = ::strrchr(gProgName, '/')) gProgName = ptr + 1;
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE

    int retcode = EXIT_SUCCESS;

    if (argc == 1) usage();

    std::string vdbFilename, imgFilename, gridName;
    RenderOpts opts;

    bool hasFocal = false, hasFov = false, hasRotate = false, hasLookAt = false;
    float fov = 0.0;

    OptParse parser(argc, argv);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (parser.check(i, "-aperture")) {
                ++i;
                opts.aperture = atof(argv[i]);
            } else if (parser.check(i, "-camera")) {
                ++i;
                opts.camera = argv[i];
            } else if (parser.check(i, "-compression")) {
                ++i;
                opts.compression = argv[i];
            } else if (parser.check(i, "-cpus")) {
                ++i;
                opts.threads = std::max(0, atoi(argv[i]));
            } else if (parser.check(i, "-far")) {
                ++i;
                opts.zfar = atof(argv[i]);
            } else if (parser.check(i, "-focal")) {
                ++i;
                opts.focal = atof(argv[i]);
                hasFocal = true;
            } else if (parser.check(i, "-fov")) {
                ++i;
                fov = atof(argv[i]);
                hasFov = true;
            } else if (parser.check(i, "-frame")) {
                ++i;
                opts.frame = atof(argv[i]);
            } else if (parser.check(i, "-lookat")) {
                ++i;
                opts.lookat = true;
                opts.target = strToVec3R(argv[i]);
                hasLookAt = true;
            } else if (parser.check(i, "-name")) {
                ++i;
                gridName = argv[i];
            } else if (parser.check(i, "-near")) {
                ++i;
                opts.znear = atof(argv[i]);
            } else if (parser.check(i, "-r") || parser.check(i, "-rotate")) {
                ++i;
                opts.rotate = strToVec3R(argv[i]);
                hasRotate = true;
            } else if (parser.check(i, "-res")) {
                ++i;
                strToSize(argv[i], opts.width, opts.height);
            } else if (parser.check(i, "-shader")) {
                ++i;
                opts.shader = argv[i];
            } else if (parser.check(i, "-samples")) {
                ++i;
                opts.samples = size_t(std::max(0, atoi(argv[i])));
            } else if (parser.check(i, "-t") || parser.check(i, "-translate")) {
                ++i;
                opts.translate = strToVec3R(argv[i]);
            } else if (parser.check(i, "-up")) {
                ++i;
                opts.up = strToVec3R(argv[i]);
            } else if (arg == "-v") {
                opts.verbose = true;
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                usage(EXIT_SUCCESS);
            } else {
                std::cerr << gProgName << ": \"" << arg << "\" is not a valid option\n";
                usage();
            }
        } else if (vdbFilename.empty()) {
            vdbFilename = arg;
        } else if (imgFilename.empty()) {
            imgFilename = arg;
        } else {
            usage();
        }
    }
    if (vdbFilename.empty() || imgFilename.empty()) {
        usage();
    }
    if (hasFov) {
        if (hasFocal) {
            OPENVDB_LOG_FATAL("specify -focal or -fov, but not both");
            usage();
        }
        opts.focal =
            openvdb::tools::PerspectiveCamera::fieldOfViewToFocalLength(fov, opts.aperture);
    }
    if (hasLookAt && hasRotate) {
        OPENVDB_LOG_FATAL("specify -lookat or -r[otate], but not both");
        usage();
    }
    {
        const std::string err = opts.validate();
        if (!err.empty()) {
            OPENVDB_LOG_FATAL(err);
            usage();
        }
    }

    try {
        tbb::task_scheduler_init schedulerInit(
            (opts.threads == 0) ? tbb::task_scheduler_init::automatic : opts.threads);

        openvdb::initialize();

        const tbb::tick_count start = tbb::tick_count::now();
        if (opts.verbose) {
            std::cout << gProgName << ": reading ";
            if (!gridName.empty()) std::cout << gridName << " from ";
            std::cout << vdbFilename << "..." << std::endl;
        }

        openvdb::FloatGrid::Ptr grid;
        {
            openvdb::io::File file(vdbFilename);

            file.open();
            if (!gridName.empty()) {
                grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
                if (!grid) {
                    OPENVDB_THROW(openvdb::ValueError,
                        gridName + " is not a scalar, floating-point grid");
                }
            } else {
                // If no grid was specified by name, retrieve the first float grid from the file.
                openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
                for (size_t i = 0; i < grids->size(); ++i) {
                    grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grids->at(i));
                    if (grid) {
                        gridName = grid->getName();
                        file.close();
                        file.open();
                        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
                        break;
                    }
                }
                if (!grid) {
                    OPENVDB_THROW(openvdb::ValueError,
                        "no scalar, floating-point grids in file " + vdbFilename);
                }
            }
        }

        if (opts.verbose) {
            std::ostringstream ostr;
            ostr << gProgName << ": ...completed in " << std::setprecision(3)
                << (tbb::tick_count::now() - start).seconds() << " sec";
            std::cout << ostr.str() << std::endl;
        }

        if (grid) {
            if (!hasLookAt && !hasRotate) {
                // If the user specified neither the camera rotation nor a target
                // to look at, orient the camera to point to the center of the grid.
                opts.target = grid->evalActiveVoxelBoundingBox().getCenter();
                opts.target = grid->constTransform().indexToWorld(opts.target);
                opts.lookat = true;
            }

            if (opts.verbose) std::cout << opts << std::endl;

            render<openvdb::FloatGrid>(*grid, imgFilename, opts);
        }
    } catch (std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        retcode = EXIT_FAILURE;
    } catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
        std::unexpected();
    }

    return retcode;
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
