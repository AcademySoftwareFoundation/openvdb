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
#include <tbb/tick_count.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>
#ifdef DWA_OPENVDB
#include <logging_base/logging.h>
#endif


namespace {

const char* gProgName = "";


struct RenderOpts
{
    std::string shader;
    std::string camera;
    float aperture, focal, frame, znear, zfar;
    openvdb::Vec3R rotation;
    openvdb::Vec3R translation;
    size_t samples;
    size_t width, height;
    std::string compression;
    bool threaded;
    bool verbose;

    RenderOpts():
        shader("diffuse"),
        camera("perspective"),
        aperture(41.2136),
        focal(50.0),
        frame(1.0),
        znear(1.0e-3),
        zfar(std::numeric_limits<double>::max()),
        rotation(0.0),
        translation(0.0),
        samples(1),
        width(2048),
        height(1024),
        compression("zip"),
        threaded(true),
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
            << " -cpus " << (threaded ? 0 : 1)
            << " -far " << zfar
            << " -focal " << focal
            << " -frame " << frame
            << " -near " << znear
            << " -rotate " << rotation[0] << "," << rotation[1] << "," << rotation[2]
            << " -res " << width << "x" << height
            << " -shader " << shader
            << " -samples " << samples
            << " -translate " << translation[0] << "," << translation[1] << "," << translation[2];
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

    std::cerr <<
"Usage: " << gProgName << " in.vdb out.{exr,ppm} [options]\n" <<
"Which: ray-traces OpenVDB volumes\n" <<
"Options:\n" <<
"    -aperture F       camera aperture (default: " << opts.aperture << ")\n" <<
"    -camera S         camera type; either \"persp[ective]\" or \"ortho[graphic]\"\n" <<
"                      (default: " << opts.camera << ")\n" <<
"    -compression S    EXR compression scheme; either \"none\" (uncompressed),\n" <<
"                      \"rle\" or \"zip\" (default: " << opts.compression << ")\n" <<
"    -cpus N           specify N = 1 to disable threading, otherwise (for now)\n" <<
"                      all available CPUs are used (default: N = 0)\n" <<
"    -far F            camera far plane depth (default: " << opts.zfar << ")\n" <<
"    -focal F          camera focal length (default: " << opts.focal << ")\n" <<
"    -fov F            camera field of view (default: " << fov << ")\n" <<
"    -frame F          camera world-space frame width (default: " << opts.frame << ")\n" <<
"    -near F           camera near plane depth (default: " << opts.znear << ")\n" <<
"    -res WxH          image width and height (default: "
    << opts.width << "x" << opts.height << ")\n" <<
"    -r X,Y,Z                                    \n" <<
"    -rotate X,Y,Z     camera rotation in degrees\n" <<
"    -shader S         shader name; either \"diffuse\", \"matte\" or \"normal\"\n" <<
"                      (default: " << opts.shader << ")\n" <<
"    -samples N        number of samples (rays) per pixel\n" <<
"                      (default: " << opts.samples << ")\n" <<
"    -t X,Y,Z                            \n" <<
"    -translate X,Y,Z  camera translation\n" <<
"\n" <<
"    -v                verbose (print diagnostics)\n" <<
"    -h, -help         print this usage message and exit\n" <<
"\n" <<
"Example:\n" <<
"    " << gProgName << " bunny.vdb bunny.exr -shader normal -res 1920x1080\n" <<
"        -focal 40 -rotate 0,45,0 -translate 50,12.5,50 -compression rle\n" <<
"\n" <<
"This is not (and is not intended to be) a production-quality renderer,\n" <<
"and it is currently limited to rendering level set volumes.\n";

    exit(exitStatus);
}


void
saveEXR(const std::string& fname, const openvdb::tools::Film& film, const RenderOpts& opts)
{
    typedef openvdb::tools::Film::RGBA RGBA;

    std::string filename = fname;
    if (!boost::iends_with(filename, ".exr")) filename += ".exr";

    if (opts.verbose) {
        std::cout << gProgName << ": writing " << filename << "..." << std::flush;
    }

    tbb::tick_count start = tbb::tick_count::now();

    if (opts.threaded) Imf::setGlobalThreadCount(8);

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
    Imf::FrameBuffer framebuffer;
    framebuffer.insert("R", Imf::Slice(Imf::FLOAT,
        (char*)(&(film.pixels()[0].r)), pixelBytes, rowBytes));
    framebuffer.insert("G", Imf::Slice(Imf::FLOAT,
        (char*)(&(film.pixels()[0].g)), pixelBytes, rowBytes));
    framebuffer.insert("B", Imf::Slice(Imf::FLOAT,
        (char*)(&(film.pixels()[0].b)), pixelBytes, rowBytes));
    framebuffer.insert("A", Imf::Slice(Imf::FLOAT,
        (char*)(&(film.pixels()[0].a)), pixelBytes, rowBytes));

    Imf::OutputFile imgFile(filename.c_str(), header);
    imgFile.setFrameBuffer(framebuffer);
    imgFile.writePixels(film.height());

    if (opts.verbose) {
        std::cout << (tbb::tick_count::now() - start).seconds() << " sec" << std::endl;
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
        camera.reset(new tools::PerspectiveCamera(film, opts.rotation, opts.translation,
            opts.focal, opts.aperture, opts.znear, opts.zfar));
    } else if (boost::starts_with(opts.camera, "ortho")) {
        camera.reset(new tools::OrthographicCamera(film, opts.rotation, opts.translation,
            opts.frame, opts.znear, opts.zfar));
    } else {
        OPENVDB_THROW(ValueError,
            "expected perspective or orthographic camera, got \"" << opts.camera << "\"");
    }

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

    //tools::LevelSetRayIntersector<GridType> lsri(grid);
    //tools::LevelSetRayTracer<FloatGrid> tracer(lsri, *shader, *camera, opts.samples);
    tools::LevelSetRayTracer<FloatGrid> tracer(grid, *shader, *camera, opts.samples);
    tracer.trace(opts.threaded);

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


struct ArgCounter
{
    int argc;
    ArgCounter(int argc_): argc(argc_) {}
    bool last(int i) const
    {
        if (i + 1 >= argc) usage();
        return false;
    }
};

} // unnamed namespace


int
main(int argc, char *argv[])
{
    int retcode = EXIT_SUCCESS;

    gProgName = argv[0];
    if (const char* ptr = ::strrchr(gProgName, '/')) gProgName = ptr + 1;

#ifdef DWA_OPENVDB
    logging_base::configure(argc, argv);
#endif

    if (argc == 1) usage();

    std::string vdbFilename, imgFilename;
    RenderOpts opts;

    bool hasFocal = false, hasFov = false;
    float fov = 0.0;

    ArgCounter counter(argc);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-aperture" && !counter.last(i)) {
                ++i;
                opts.aperture = atof(argv[i]);
            } else if (arg == "-camera" && !counter.last(i)) {
                ++i;
                opts.camera = argv[i];
            } else if (arg == "-compression" && !counter.last(i)) {
                ++i;
                opts.compression = argv[i];
            } else if (arg == "-cpus" && !counter.last(i)) {
                ++i;
                opts.threaded = (atoi(argv[i]) != 1);
            } else if (arg == "-far" && !counter.last(i)) {
                ++i;
                opts.zfar = atof(argv[i]);
            } else if (arg == "-focal" && !counter.last(i)) {
                ++i;
                opts.focal = atof(argv[i]);
                hasFocal = true;
            } else if (arg == "-fov" && !counter.last(i)) {
                ++i;
                fov = atof(argv[i]);
                hasFov = true;
            } else if (arg == "-frame" && !counter.last(i)) {
                ++i;
                opts.frame = atof(argv[i]);
            } else if (arg == "-near" && !counter.last(i)) {
                ++i;
                opts.znear = atof(argv[i]);
            } else if ((arg == "-r" || arg == "-rotate") && !counter.last(i)) {
                ++i;
                opts.rotation = strToVec3R(argv[i]);
            } else if (arg == "-res" && !counter.last(i)) {
                ++i;
                strToSize(argv[i], opts.width, opts.height);
            } else if (arg == "-shader" && !counter.last(i)) {
                ++i;
                opts.shader = argv[i];
            } else if (arg == "-samples" && !counter.last(i)) {
                ++i;
                opts.samples = size_t(std::max(0, atoi(argv[i])));
            } else if ((arg == "-t" || arg == "-translate") && !counter.last(i)) {
                ++i;
                opts.translation = strToVec3R(argv[i]);
            } else if (arg == "-v") {
                opts.verbose = true;
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                usage(EXIT_SUCCESS);
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
    {
        const std::string err = opts.validate();
        if (!err.empty()) {
            OPENVDB_LOG_FATAL(err);
            usage();
        }
    }

    if (opts.verbose) std::cout << opts << std::endl;

    try {
        openvdb::initialize();

        openvdb::io::File file(vdbFilename);
        if (opts.verbose) std::cout << "reading " << vdbFilename << "...\n";
        file.open();
        if (openvdb::FloatGrid::Ptr grid =
            openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0)))
        {
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
