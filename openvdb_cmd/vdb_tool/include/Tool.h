// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Tool.h
///
/// @brief Defines the Tool class, which chains together any sequence of high-level
///        OpenVDB operations exposed by the vdb_tool command-line utility.
///
/// @details Tool ties Parser (command-line action registry) and Geometry (polygon
///          mesh and point storage) together with the internal stacks of VDB grids
///          and Geometry instances. For example, it can convert a sequence of polygon
///          meshes and particles to level sets, perform a large number of operations
///          on these level set surfaces, generate adaptive polygon meshes from level
///          sets, render images, and write particles, meshes, or VDBs to disk.
///
/// @warning All prints are directed to cerr since cout is used for piping!
///
/// @todo expose LevelSetMeasure
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_HAS_BEEN_INCLUDED
#define VDB_TOOL_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/Formats.h>
#include <openvdb/util/Assert.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Count.h>// for tools::minMax (used by -print level=2)
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetDilatedMesh.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetMorph.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/PolySoupToLevelSet.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/PointsToMask.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/Mask.h> // for tools::interiorMask()
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

#ifdef VDB_TOOL_USE_NANO
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#endif

#ifdef VDB_TOOL_USE_EXR
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>
#endif

#ifdef VDB_TOOL_USE_PNG
#include <png.h>
#endif

#ifdef VDB_TOOL_USE_PDAL
#include <pdal/pdal.hpp>
#endif

#ifdef VDB_TOOL_USE_JPG
#include <jpeglib.h>
#endif

#include <tbb/blocked_range2d.h>
#include <tbb/enumerable_thread_specific.h>

#include "Calculator.h"
#include "Parser.h"
#include "Geometry.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef VDB_TOOL_USE_MPEG
#include <cstdlib>// for std::system
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

/// @brief Top-level command-line tool that chains OpenVDB high-level operations.
/// @details Owns the action parser, the stacks of in-flight Geometry and VDB grids,
///          and the log-redirection state. A Tool instance is constructed from argv,
///          which registers and parses all command-line actions; run() then executes
///          them in order. The class is non-copyable and non-movable.
class Tool
{
public:

    /// @brief Construct a Tool from command-line arguments.
    /// @param argc Argument count, as received by main().
    /// @param argv Argument vector, as received by main(). argv[0] is taken as the command name.
    /// @throw std::invalid_argument if parsing fails (unknown action, malformed option, etc.).
    Tool(int argc, char *argv[]);

    /// @brief Destructor; restores std::clog if a log file was opened.
    ~Tool() {this->endLog();}

    Tool(const Tool&) = delete;            ///< Copy construction is disabled.
    Tool(Tool&&) = delete;                 ///< Move construction is disabled.
    Tool& operator=(const Tool&) = delete; ///< Copy assignment is disabled.
    Tool& operator=(Tool&&) = delete;      ///< Move assignment is disabled.

    /// @brief Execute every action that was registered during construction, in order.
    /// @note  On a fatal exception inside an action this method writes the message to
    ///        std::cerr and calls std::exit(EXIT_FAILURE) rather than propagating.
    void run();

    /// @brief Redirect std::clog/std::cerr/std::cout to a log file for the remainder of this Tool's lifetime.
    /// @param logFile Path of the log file. If empty, a timestamped name is generated.
    /// @param append  If true, append to the existing file; otherwise truncate it (default).
    /// @param tee     If true (default), output is also written to the original terminal stream so the
    ///                user keeps interactive feedback while the file accumulates the same data. If false,
    ///                output is routed exclusively to the log file (the pre-#6 behaviour).
    /// @throw std::invalid_argument if the file cannot be opened or the standard stream buffers cannot be captured.
    /// @note Subsequent calls are no-ops while a redirection is already active.
    void startLog(std::string logFile, bool append = false, bool tee = true);

    /// @brief Restore std::clog/std::cerr/std::cout to their original buffers and close the log file (if any).
    void endLog() {
      if (mOldClogBuffer) std::clog.rdbuf(mOldClogBuffer);
      if (mOldCerrBuffer) std::cerr.rdbuf(mOldCerrBuffer);
      if (mOldCoutBuffer) std::cout.rdbuf(mOldCoutBuffer);
      if (mLogFile.is_open()) mLogFile.close();
      mOldClogBuffer = nullptr;
      mOldCerrBuffer = nullptr;
      mOldCoutBuffer = nullptr;
      mClogTee.reset();
      mCerrTee.reset();
      mCoutTee.reset();
    }

    /// @brief Print a summary of the current VDB-grid and Geometry stacks to @a os.
    void print(std::ostream& os = std::clog) const;

    /// @brief Print the canonical "-action option=value ..." form of every queued action to @a os.
    void print_args(std::ostream& os = std::clog) const;

    /// @brief Return the current version of this tool as "major.minor.patch".
    static std::string version() {return std::to_string(sMajor)+"."+std::to_string(sMinor)+"."+std::to_string(sPatch);}
    /// @brief Return the major version (incremented on incompatible option/file changes).
    static int major() {return sMajor;}
    /// @brief Return the minor version (incremented on backwards-compatible new features).
    static int minor() {return sMinor;}
    /// @brief Return the patch version (incremented on backwards-compatible bug fixes).
    static int patch() {return sPatch;}

private:

    static const int sMajor = 10; ///< Major version: incremented on incompatible option/file changes.
    static const int sMinor =  8; ///< Minor version: incremented on backwards-compatible new features.
    static const int sPatch =  0; ///< Patch version: incremented on backwards-compatible bug fixes.

    using GridT   = FloatGrid;                                       ///< Scalar grid type used by most level-set operations.
    using FilterT = std::unique_ptr<tools::LevelSetFilter<GridT>>;   ///< Owned pointer to a LevelSetFilter over GridT.
    struct Points; ///< Forward declaration of the helper points wrapper used by particlesToSdf.
    struct Header; ///< Forward declaration of the config-file header record.

    mutable util::CpuTimer   mTimer;          ///< Reusable timer for verbose timing reports.
    std::string              mCmdName;        ///< Base name of this command-line tool (argv[0]).
    std::string              mRawCmdLine;     ///< Verbatim argv joined by spaces — used in the log header.
    std::list<Geometry::Ptr> mGeom;           ///< Stack of Geometry instances owned by this tool (back = top).
    std::list<GridBase::Ptr> mGrid;           ///< Stack of VDB grids owned by this tool (back = top).
    Parser                   mParser;         ///< Command-line action parser and processor.
    bool                     mErrorOnWarning; ///< If true, warning() escalates to a fatal error.
    std::ofstream            mLogFile;        ///< Backing file used by startLog/endLog when active.
    std::streambuf          *mOldClogBuffer;  ///< Cached std::clog buffer for restoring after logging.
    std::streambuf          *mOldCerrBuffer;  ///< Cached std::cerr buffer for restoring after logging.
    std::streambuf          *mOldCoutBuffer;  ///< Cached std::cout buffer for restoring after logging.
    std::unique_ptr<TeeBuf>  mClogTee;        ///< Tee streambuf for std::clog (terminal + log file) when -log tee=true.
    std::unique_ptr<TeeBuf>  mCerrTee;        ///< Tee streambuf for std::cerr.
    std::unique_ptr<TeeBuf>  mCoutTee;        ///< Tee streambuf for std::cout.

    /// @brief Delete all queued Geometry, VDB grids, and local variables.
    void clear();

    /// @brief Clip an input VDB grid against another grid, a bbox, or a frustum.
    /// @tparam GridType Type of the input grid being clipped.
    /// @param v     Numeric parameters defining the clipping region (interpretation depends on mode).
    /// @param age   Stack age of the secondary clipping grid, or sentinel meaning "use bbox/frustum".
    /// @param input The grid being clipped (left unchanged).
    /// @return Shared pointer to the clipped grid.
    template <typename GridType>
    GridBase::Ptr clip(const VecF &v, int age, const GridType &input);
    /// @brief Action callback for "-clip"; dispatches to the templated clip() above.
    void clip();

    /// @brief Composite two grids using a binary op (min, max, or sum).
    void composite();

    /// @brief Generate a derived grid (e.g. gradient, curl, divergence) from another grid.
    void compute();

    /// @brief Import and process one or more configuration files.
    void config();

    /// @brief Perform CSG operations (union/intersection/difference) between two level-set surfaces.
    void csg();

    /// @brief Run the Enright advection benchmark on a level set.
    void enright();

    /// @brief Expand the narrow band of a level set.
    void expandLevelSet();

    /// @brief Perform filtering (convolution) of a level-set surface.
    void filterLevelSet();

    /// @brief Signed flood-fill of a level-set VDB.
    void floodLevelSet();

    /// @brief Print documentation for one, multiple, or all available actions.
    void help();

    /// @brief Convert an iso-surface of a scalar field into a level set (i.e. SDF).
    void isoToLevelSet();

    /// @brief Convert a volume into an adaptive polygon mesh.
    void volumeToMesh();

    /// @brief Create a level-set sphere, i.e. a narrow-band signed distance to a sphere.
    void levelSetSphere();

    /// @brief  Convert signed distance field into a unsigned distance field
    void sdf2udf();

    /// @brief Apply a simple function to each voxel in a grid
    void forValues();

    /// @brief Create a level-set platonic solid with the specified number of polygon faces.
    void levelSetPlatonic();

    /// @brief Convert a level-set VDB into a fog volume (normalized density).
    void levelSetToFog();

    /// @brief Convert a polygon mesh into a symmetric or asymmetric narrow-band level set.
    void meshToLevelSet();

    /// @brief Convert a polygon mesh into a symmetric narrow-band unsigned distance field.
    void meshToUnsignedDistanceField();

#ifdef VDB_TOOL_USE_SHRINKWRAP
    /// @brief Convert an arbitrary (possibly non-watertight) polygon soup into a narrow-band level set.
    /// @note  Gated behind VDB_TOOL_USE_SHRINKWRAP (temporarily disabled; not exposed via CMake).
    ///        Enable a local build with: cmake -DCMAKE_CXX_FLAGS="-DVDB_TOOL_USE_SHRINKWRAP" ..
    void soupToLevelSet();
#endif

    /// @brief Generate a dx-offset surface from a polygon soup.
    void soupToOffset();

    /// @brief Convert every quad in the current mesh into two triangles.
    void quadsToTriangles();

    /// @brief Convert a sequence of image files into a single MPEG movie file.
    void movie();

    /// @brief Construct a level-of-detail sequence of VDB trees with powers-of-two refinements.
    void multires();

    /// @brief Perform morphological dilation/erosion on a level-set surface.
    void offsetLevelSet();

    /// @brief Convert geometry points into a narrow-band level set.
    void particlesToLevelSet();

    /// @brief Encode geometry points into a VDB PointDataGrid.
    void pointsToVdb();

    /// @brief Prune away inactive values in a VDB grid.
    void pruneLevelSet();

    /// @brief Read one or more geometry or VDB files from disk or STDIN.
    void read();
    /// @brief Read a geometry file (mesh or point cloud) and push it onto the geometry stack.
    void readGeo(  const std::string &fileName);
    /// @brief Read an OpenVDB file and push every selected grid onto the grid stack.
    void readVDB(  const std::string &fileName);
    /// @brief Read a NanoVDB file (.nvdb) and push every selected grid onto the grid stack.
    void readNVDB( const std::string &fileName);

    /// @brief Ray-trace level-set surfaces or volume-render fog volumes.
    void render();

    /// @brief Resample one VDB grid into another VDB grid or onto a transformed copy of itself.
    void resample();

    /// @brief Segment an input VDB into a list of topologically disconnected VDB grids.
    void segment();

    /// @brief Scatter points into the active values of an input VDB grid.
    void scatter();

    /// @brief Generate images of axis-aligned volume slices.
    void slice();

    /// @brief Apply affine transformations (uniform scale -> rotation -> translation) to VDB grids and geometry.
    void transform();

    /// @brief Extract points encoded in a VDB into geometry-format point lists.
    void vdbToPoints();

    /// @brief Write the list of geometries, VDB grids, or config files to disk or STDOUT.
    void write();
    /// @brief Write a single geometry to disk in the format implied by the file extension.
    void writeGeo( const std::string &fileName);
    /// @brief Write a single VDB grid to disk.
    void writeVDB( const std::string &fileName);
    /// @brief Write a single VDB grid as a NanoVDB (.nvdb) file.
    void writeNVDB(const std::string &fileName);
    /// @brief Write the currently parsed action list as a config file.
    void writeConf(const std::string &fileName);

    /// @brief Estimate the voxel size of a level set from a desired grid dimension.
    /// @param maxDimension Maximum voxel resolution along the longest axis of the bbox.
    /// @param exWidth      Exterior half-width of the narrow band, in voxel units.
    /// @param inWidth      Interior half-width of the narrow band, in voxel units.
    /// @param geo_age      Stack age of the geometry whose bbox drives the estimate.
    /// @return Voxel size in world units.
    float estimateVoxelSize(int maxDimension, float exWidth, float inWidth, int geo_age);
    /// @brief Convenience overload: symmetric narrow band (exWidth == inWidth == halfWidth).
    float estimateVoxelSize(int maxDim,  float halfWidth, int geo_age) {return this->estimateVoxelSize(maxDim, halfWidth, halfWidth, geo_age);}

    /// @brief Build a LevelSetFilter configured with the given spatial and temporal schemes.
    /// @param grid  Grid the filter will operate on.
    /// @param space Spatial discretization order.
    /// @param time  Temporal discretization order.
    FilterT createFilter(GridT &grid,  int space, int time);

    /// @brief Return a formatted string of usage examples (for the -examples action).
    std::string examples() const;

    /// @brief Emit a banner-framed warning to @a os. Escalates to error if mErrorOnWarning is true.
    void warning(const std::string &msg, std::ostream& os = std::clog) const;

    /// @brief Register every available action with the parser. Called from the constructor.
    void init();

    /// @brief Return an iterator to the VDB grid at stack age @a age (0 = most recent).
    /// @throw std::invalid_argument if @a age exceeds the current stack depth.
    inline auto getGrid(size_t age) const;
    /// @brief Return an iterator to the Geometry at stack age @a age (0 = most recent).
    /// @throw std::invalid_argument if @a age exceeds the current stack depth.
    inline auto getGeom(size_t age) const;

    /// @brief Convert the output of a VolumeToMesh pass into a Geometry instance.
    Geometry::Ptr mesherToGeometry(tools::VolumeToMesh&) const;
    /// @brief Adaptively mesh a scalar grid at @a isoValue and return the result as a Geometry.
    /// @param grid       Input scalar grid.
    /// @param isoValue   Iso-value at which to extract the surface (default 0 for SDFs).
    /// @param adaptivity Adaptivity parameter passed to VolumeToMesh (0 = uniform quads).
    Geometry::Ptr volumeToGeometry(const GridT &grid, float isoValue=0.0f, float adaptivity=0.0f) const;

};// Tool class

// ==============================================================================================================

Tool::Tool(int argc, char *argv[])
    : mTimer(std::clog)
    , mCmdName(getBase(argv[0]))// name of executable
    , mRawCmdLine([&]{
        std::string s;
        for (int i = 0; i < argc; ++i) {
            if (i > 0) s += ' ';
            s += argv[i];
        }
        return s;
      }())
    , mParser({{"dim", "256", "256", "default grid resolution along the longest axis"},
               {"voxel", "0.0", "0.01", "default voxel size in world units. A value of zero indicates that dim is used to derive the voxel size."},
               {"width", "3.0", "3.0", "default narrow-band width of level sets in voxel units"},
               {"time", "1", "1|2|3", "default temporal discretization order"},
               {"space", "5", "1|2|3|5", "default spatial discretization order"},
               {"keep", "false", "1|0|true|false", "by default delete the input"}})
    , mErrorOnWarning(false)
    , mLogFile()
    , mOldClogBuffer(nullptr)
    , mOldCerrBuffer(nullptr)
    , mOldCoutBuffer(nullptr)
{
    openvdb::initialize();
    this->init();// fast: less than 1 ms
    try {
        mParser.finalize();
        mParser.parse(argc, argv);// extremely fast, but might throw
    } catch (const std::exception& e) {
        this->endLog();
        throw std::invalid_argument(e.what());
    }
}// Tool::Tool

// ==============================================================================================================

void Tool::startLog(std::string logFile, bool append, bool tee)
{
    if (mOldClogBuffer != nullptr) return;// handles repeated calls
    if (logFile.empty()) logFile = "vdb_tool_" + dateStamp() + ".log";
    const auto mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
    mLogFile.open(logFile, mode);
    if (!mLogFile.is_open()) {
        throw std::invalid_argument("startLog: failed to open log file \"" + logFile + "\"");
    }
    // Unit-buffered output so `watch -n 0.5 vdb_tool.log` (and tail -f) see
    // each line as soon as it's written, instead of waiting for the 4 KB
    // block buffer to flush. The help text recommends this workflow.
    mLogFile.setf(std::ios::unitbuf);

    // Redirect all three text streams so warnings/errors (cerr) and any
    // stdout-bound output also land in the log — not just clog.
    mOldClogBuffer = std::clog.rdbuf();
    mOldCerrBuffer = std::cerr.rdbuf();
    mOldCoutBuffer = std::cout.rdbuf();
    if (mOldClogBuffer == nullptr || mOldCerrBuffer == nullptr || mOldCoutBuffer == nullptr) {
        throw std::invalid_argument("startLog: failed to cache standard stream buffers");
    }
    if (tee) {
        // Each TeeBuf fans output to (original terminal stream, log file) so
        // the user keeps live console feedback while the log accumulates.
        mClogTee = std::make_unique<TeeBuf>(mOldClogBuffer, mLogFile.rdbuf());
        mCerrTee = std::make_unique<TeeBuf>(mOldCerrBuffer, mLogFile.rdbuf());
        mCoutTee = std::make_unique<TeeBuf>(mOldCoutBuffer, mLogFile.rdbuf());
        std::clog.rdbuf(mClogTee.get());
        std::cerr.rdbuf(mCerrTee.get());
        std::cout.rdbuf(mCoutTee.get());
    } else {
        // Exclusive log mode (tee=false): nothing goes to the terminal.
        std::clog.rdbuf(mLogFile.rdbuf());
        std::cerr.rdbuf(mLogFile.rdbuf());
        std::cout.rdbuf(mLogFile.rdbuf());
    }

    // Self-describing log header — timestamp, vdb_tool version, and the full
    // command line. Makes a stored log readable days later without needing
    // to remember what was invoked. In append mode a blank line separates
    // this run's header from the previous run's output.
    if (append) mLogFile << "\n";
    mLogFile << "# vdb_tool log\n"
             << "# date     : " << dateStamp() << "\n"
             << "# version  : " << Tool::version() << "\n"
             << "# command  : " << mRawCmdLine << "\n"
             << std::flush;
}

// ==============================================================================================================

auto Tool::getGrid(size_t age) const
{
    if (age>=mGrid.size()) {
      throw std::invalid_argument("-"+mParser.getAction().names[0]+" called getGrid("+std::to_string(age)+"), but grid count = "+std::to_string(mGrid.size()));
    }
    auto it = mGrid.crbegin();
    std::advance(it, age);
    return it;
}// Tool::getGrid

// ==============================================================================================================

auto Tool::getGeom(size_t age) const
{
    if (age>=mGeom.size()) {
      throw std::invalid_argument("-"+mParser.getAction().names[0]+" called getGeom("+std::to_string(age)+"), but geometry count = "+std::to_string(mGeom.size()));
    }
    auto it = mGeom.crbegin();
    std::advance(it, age);
    return it;
}// Tool::getGeom

// ==============================================================================================================

void Tool::run()
{
    if (mParser.verbose>1) this->print_args();
    try {
        mParser.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error in Tool::run: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}// Tool::run

// ==============================================================================================================

void Tool::warning(const std::string &msg, std::ostream& os) const
{
    if (mParser.verbose) {
        os << "\n" << std::setw(static_cast<int>(msg.size())) << std::setfill('*') << "\n" << msg
           << "\n" << std::setw(static_cast<int>(msg.size())) << std::setfill('*') << "\n";
    }
}// Tool::warning

// ==============================================================================================================

/// @brief Header record prepended to every vdb_tool config (.txt) file.
/// @details Identifies a config file with the magic string "vdb_tool" followed by
///          the major.minor.patch version that produced it. Used to gate loading
///          configs from incompatible tool versions.
struct Tool::Header {
    /// @brief Construct a header for the current tool version.
    Header() : mMagic("vdb_tool"), mMajor(sMajor), mMinor(sMinor), mPatch(sPatch) {}
    /// @brief Parse a header from the first line of a config file.
    /// @param line First line of the config file, e.g. "vdb_tool 10.8.0".
    /// @throw std::invalid_argument if @a line does not match "vdb_tool MAJOR.MINOR.PATCH".
    Header(const std::string &line) : mMagic("vdb_tool") {
      const VecS header = tokenize(line, " .");
      if (header.size()!=4 || header[0]!=mMagic ||
         !isInt(header[1], mMajor) ||
         !isInt(header[2], mMinor) ||
         !isInt(header[3], mPatch)) throw std::invalid_argument("Header: incompatible: \""+line+"\"");
    }
    /// @brief Format the header as the string written at the top of a config file.
    std::string str() const {
      return mMagic+" "+std::to_string(mMajor)+"."+std::to_string(mMinor)+"."+std::to_string(mPatch);
    }
    /// @brief Returns true if this header's major version matches the running tool.
    bool isCompatible() const {return mMajor == sMajor;}

    std::string mMagic; ///< Magic identifier; always "vdb_tool" for a valid header.
    int mMajor;         ///< Major version recorded in (or expected by) the config file.
    int mMinor;         ///< Minor version recorded in (or expected by) the config file.
    int mPatch;         ///< Patch version recorded in (or expected by) the config file.
};// Header struct

// ==============================================================================================================

/// @brief Lightweight adapter exposing a std::vector<Vec3s> as the point-source interface
///        expected by tools::ParticlesToLevelSet.
/// @details ParticlesToLevelSet requires the source to define a PosType alias and to provide
///          size() and getPos() member functions. This wrapper supplies them over a borrowed
///          vector of vertices without copying the data.
struct Tool::Points {
    using PosType = Vec3R; ///< Position type required by ParticlesToLevelSet (double precision).

    /// @brief Construct a Points adapter over an existing vertex array (stored by reference).
    Points(const std::vector<Vec3s> &vtx) : mPoints(vtx) {}
    /// @brief Number of points exposed by this adapter.
    size_t size() const { return mPoints.size(); }
    /// @brief Write the n'th point into @a p, converting from Vec3s to PosType (Vec3R).
    void getPos(size_t n, PosType &p) const { p = mPoints[n]; }

    const std::vector<Vec3s> &mPoints; ///< Borrowed reference to the underlying vertex array.
};// Points struct

// ==============================================================================================================

void Tool::init()
{
  // note, the following actions were added when mParser was constructed: -quiet,-verbose,-debug,-default,-for,-each,-end

  //  mParser.addAction({"name",.. "alias"}, "documentation of action",
  //                    {{"option name", "default value", "expected values", "documentation of option"}},
  //                     {more options}...});
  mParser.addAction(
     {"config", "c"}, "Import and process one or more configuration files",
    {{"files", "",  "config1.txt,config2.txt...", "list of configuration files to load and execute"},
     {"execute", "true", "1|0|true|false", "toggle wether to execute the actions in the config file"},
     {"update", "false", "1|0|true|false", "toggle wether to update the version number of the config file"}},
     [&](){this->config();}, [](){}, 0); // anonymous options are appended to "files"

  mParser.addAction(
     {"help", "h"}, "Print documentation for one, multiple or all available actions",
    {{"actions", "", "read,write,...", "list of actions to document. If the list is empty documentation is printed for all available actions and if other actions proceed this action, documentation is printed for those actions only"},
     {"exit", "true", "1|0|true|false", "toggle wether to terminate after this action or not"},
     {"brief", "false", "1|0|true|false", "toggle brief or detailed documentation"},
     {"format", "text", "text|md", "output format. 'text' (default) is the usual human-readable help; 'md' emits a single Markdown table of action names + descriptions, used to regenerate the action list in README.md so it can't drift from the registered actions."}},
     [](){}, [&](){this->help();}, 0); // anonymous options are appended to "actions"

  mParser.addAction(
     {"read", "import", "load", "i"}, "Read one or more geometry or VDB files from disk or STDIN.",
    {{"files", "", "{file|stdin}.{obj|ply|abc|stl|off|pts|xyz|e57|vdb|nvdb|gltf|glb|geo|usd|usda|usdc|usdz}", "list of files or the input stream, e.g. file.vdb,stdin.vdb. Note that \"files=\" is optional since any argument without \"=\" is intrepreted as a file and appended to \"files\""},
     {"grids", "*", "*|grid_name,...", "list of VDB grids name to be imported (defaults to \"*\", i.e. import all available grids)"},
     {"delayed", "true", "1|0|true|false", "toggle delayed loading of VDB grids (enabled by default). This option is ignored by other file types"}},
     [](){}, [&](){this->read();}, 0);//  anonymous options are treated as to the first option,i.e. "files"

  mParser.addAction(
     {"write", "export", "save", "o"}, "Write list of geometry, VDB or config files to disk or STDOUT",
    {{"files", "", "{file|stdout}.{obj|ply|stl|off|geo|abc|vdb|nvdb|txt}", "list of files or the output stream, e.g. file.vdb or stdin.vdb. Note that \"files=\" is optional since any argument without the \"=\" character is intrepreted as a file and appended to \"files\"."},
     {"geo", "0", "0|1...", "geometry to write (defaults to \"0\" which is the latest)."},
     {"vdb", "*", "0,1,...", "list of VDB grids to write (defaults to \"*\", i.e. all available grids)."},
     {"keep", "", "1|0|true|false", "toggle wether to preserved or deleted geometry and grids after they have been written."},
     {"codec", "", "none|zip|blosc|active", "compression codec for the file or stream"},
     {"bits", "32", "32|16|8|4|N", "bit-width of floating point numbers during quantization of VDB and NanoVDB grids, i.e. 32 is full, 16, is half (defaults to 32). NanoVDB also supports 8, 4 and N which is adaptive bit-width"},// VDB: 32, 16 + for NVDB 8, 4 or N
     {"dither", "false", "1|0|true|false", "toggle dithering of quantized NanoVDB grids (disabled by default)"},
     {"absolute", "true", "1|0|true|false", "toggle absolute or relative error tolerance during quantization of NanoVDBs. Only used if bits=N. Defaults to absolute"},// absolute or relative error for N bits in NVDB
     {"tolerance", "-1", "1.0", "absolute or relative error tolerance used during quantization of NanoVDBs. Only used if bits=N."},// error tolerance for N bits in NVDB
     {"stats", "", "none|bbox|extrema|all", "specify the statistics to compute for NanoVDBs."},
     {"ascii", "false", "1|0|true|false", "for ascii vs binary output format when available (e.g. for ply files). Defaults to false, i.e. binary is preferred over ascii when available"},
     {"checksum", "", "none|partial|full", "specify the type of checksum to compute for NanoVDBs"}},
     [&](){mParser.setDefaults();}, [&](){this->write();}, 0);// anonymous options are treated as to the first option,i.e. "files"

  mParser.addAction(
     {"clear"}, "Deletes geometry, VDB grids and local variables",
    {{"geo", "*", "*|0,1,...", "list of geometries to delete (defaults to all)"},
     {"vdb", "*", "*|0,1,...", "list of VDB grids to delete (defaults to all)"},
     {"variables", "0", "1|0|true|false", "clear all the local variables (defaults to off)"}},
     [](){}, [&](){this->clear();});

  mParser.addAction(
     {"sphere"}, "Create a level set sphere, i.e. a narrow-band signed distance to a sphere",
    {{"dim", "", "256", "largest dimension in voxel units of the sphere (defaults to 256). If \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.0", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\". Defaults to 0.0, i.e. this option is disabled"},
     {"radius", "1.0", "1.0", "radius of sphere in world units"},
     {"center", "(0,0,0)", "(0.0,0.0,0.0)", "center of sphere in world units"},
     {"signed", "true", "1|0|true|false", "toggle wether the output volume should be a signed vs unsigned distance field"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"name", "sphere", "sphere", "name assigned to the level set sphere"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetSphere();});

  mParser.addAction(
     {"forAllValues"}, "Applied a simple computational kernel to ALL values in a grid.",
    {{"keep", "", "1|0|true|false", "toggle wether the input volume is preserved or deleted after the conversion"},
     {"vdb", "0", "0|0,1", "age(s) (i.e. stack index) of grid(s) to be processed. Defaults to 0, i.e. most recently inserted VDB. Accepts a comma-separated list to use multiple grids in the kernel: the FIRST grid is written (iterated), the rest are read-only inputs. Length must match use=."},
     {"kernel", "", "sin(v)+2*v*v", "user-defined math expression to apply to each value. The \"kernel=\" prefix is OPTIONAL; the kernel may also be supplied as a bare positional argument, e.g. \"-forOnValues 'sin(v)+1'\" or \"-forOnValues 'sin(v)+1' keep=true\" — other named options of the same action still parse normally. Supports infix (e.g. \"sin(v)+2*v*v\"), RPN (e.g. \"$v:sin:$v:pow2:2:*:+\"), and infix multi-statement programs with assignment (e.g. \"t = v*v; t + sin(t)\"). The variable that holds the current voxel value is configurable via the \"use\" option (defaults to \"v\"). Stencil kernels: write \"v(dx,dy,dz)\" with integer-literal offsets to read a relative neighbor voxel through a per-thread ConstAccessor (e.g. \"v(1,0,0)-v(-1,0,0)\" computes a finite-difference x-derivative). The grid is internally deep-copied so reads come from a stable snapshot. Any other identifier in the kernel is looked up once in the Processor's string memory (the same namespace used by -eval / -calc) and used as a per-voxel constant — kernels like \"a*v + b\" therefore require -eval / -calc to have set \"a\" and \"b\" beforehand, else an error is thrown. An empty kernel is a no-op."},
     {"use", "v", "v|x|x,y", "name(s) of the kernel variable(s) bound to the voxel value(s) of the input grid(s). Defaults to \"v\". Accepts a comma-separated list matching vdb=: use=x,y vdb=0,1 makes \"x\" the output grid and \"y\" a read-only input. Each name is excluded from the Processor-memory lookup and may be called as a function (e.g. \"x(1,0,0)\") to read a relative neighbor through a per-thread ConstAccessor."},
     {"class", "", "ls", "class label of the output volume."},
     {"background", "", "1.5,2.0", "background value(s) of the output volume. If two values are provided they are assumed to be outside, inside"},
     {"name", "", "foo-bar", "name assigned to the output volume"}},
     [&](){mParser.setDefaults();}, [&](){this->forValues();},
     /*anonymous=*/2, /*greedy=*/true);// kernel (index 2) may itself contain '='; accept bare "x+1" or "t=x*x; t+1" alongside kernel='...'

  mParser.addAction(
     {"forOnValues"}, "Applied a simple computational kernel to ON values in a grid.",
    {{"keep", "", "1|0|true|false", "toggle wether the input volume is preserved or deleted after the conversion"},
     {"vdb", "0", "0|0,1", "age(s) (i.e. stack index) of grid(s) to be processed. Defaults to 0, i.e. most recently inserted VDB. Accepts a comma-separated list to use multiple grids in the kernel: the FIRST grid is written (iterated), the rest are read-only inputs. Length must match use=."},
     {"kernel", "", "sin(v)+2*v*v", "user-defined math expression to apply to each value. The \"kernel=\" prefix is OPTIONAL; the kernel may also be supplied as a bare positional argument, e.g. \"-forOnValues 'sin(v)+1'\" or \"-forOnValues 'sin(v)+1' keep=true\" — other named options of the same action still parse normally. Supports infix (e.g. \"sin(v)+2*v*v\"), RPN (e.g. \"$v:sin:$v:pow2:2:*:+\"), and infix multi-statement programs with assignment (e.g. \"t = v*v; t + sin(t)\"). The variable that holds the current voxel value is configurable via the \"use\" option (defaults to \"v\"). Stencil kernels: write \"v(dx,dy,dz)\" with integer-literal offsets to read a relative neighbor voxel through a per-thread ConstAccessor (e.g. \"v(1,0,0)-v(-1,0,0)\" computes a finite-difference x-derivative). The grid is internally deep-copied so reads come from a stable snapshot. Any other identifier in the kernel is looked up once in the Processor's string memory (the same namespace used by -eval / -calc) and used as a per-voxel constant — kernels like \"a*v + b\" therefore require -eval / -calc to have set \"a\" and \"b\" beforehand, else an error is thrown. An empty kernel is a no-op."},
     {"use", "v", "v|x|x,y", "name(s) of the kernel variable(s) bound to the voxel value(s) of the input grid(s). Defaults to \"v\". Accepts a comma-separated list matching vdb=: use=x,y vdb=0,1 makes \"x\" the output grid and \"y\" a read-only input. Each name is excluded from the Processor-memory lookup and may be called as a function (e.g. \"x(1,0,0)\") to read a relative neighbor through a per-thread ConstAccessor."},
     {"class", "", "ls", "class label of the output volume."},
     {"background", "", "1.5,2.0", "background value(s) of the output volume. If two values are provided they are assumed to be outside, inside"},
     {"name", "", "foo-bar", "name assigned to the output volume"}},
     [&](){mParser.setDefaults();}, [&](){this->forValues();},
     /*anonymous=*/2, /*greedy=*/true);// kernel (index 2) may itself contain '='; accept bare "x+1" or "t=x*x; t+1" alongside kernel='...'

  mParser.addAction(
     {"forOffValues"}, "Applied a simple computational kernel to OFF values in a grid.",
    {{"keep", "", "1|0|true|false", "toggle wether the input volume is preserved or deleted after the conversion"},
     {"vdb", "0", "0|0,1", "age(s) (i.e. stack index) of grid(s) to be processed. Defaults to 0, i.e. most recently inserted VDB. Accepts a comma-separated list to use multiple grids in the kernel: the FIRST grid is written (iterated), the rest are read-only inputs. Length must match use=."},
     {"kernel", "", "sin(v)+2*v*v", "user-defined math expression to apply to each value. The \"kernel=\" prefix is OPTIONAL; the kernel may also be supplied as a bare positional argument, e.g. \"-forOnValues 'sin(v)+1'\" or \"-forOnValues 'sin(v)+1' keep=true\" — other named options of the same action still parse normally. Supports infix (e.g. \"sin(v)+2*v*v\"), RPN (e.g. \"$v:sin:$v:pow2:2:*:+\"), and infix multi-statement programs with assignment (e.g. \"t = v*v; t + sin(t)\"). The variable that holds the current voxel value is configurable via the \"use\" option (defaults to \"v\"). Stencil kernels: write \"v(dx,dy,dz)\" with integer-literal offsets to read a relative neighbor voxel through a per-thread ConstAccessor (e.g. \"v(1,0,0)-v(-1,0,0)\" computes a finite-difference x-derivative). The grid is internally deep-copied so reads come from a stable snapshot. Any other identifier in the kernel is looked up once in the Processor's string memory (the same namespace used by -eval / -calc) and used as a per-voxel constant — kernels like \"a*v + b\" therefore require -eval / -calc to have set \"a\" and \"b\" beforehand, else an error is thrown. An empty kernel is a no-op."},
     {"use", "v", "v|x|x,y", "name(s) of the kernel variable(s) bound to the voxel value(s) of the input grid(s). Defaults to \"v\". Accepts a comma-separated list matching vdb=: use=x,y vdb=0,1 makes \"x\" the output grid and \"y\" a read-only input. Each name is excluded from the Processor-memory lookup and may be called as a function (e.g. \"x(1,0,0)\") to read a relative neighbor through a per-thread ConstAccessor."},
     {"class", "", "ls", "class label of the output volume."},
     {"background", "", "1.5,2.0", "background value(s) of the output volume. If two values are provided they are assumed to be outside, inside"},
     {"name", "", "foo-bar", "name assigned to the output volume"}},
     [&](){mParser.setDefaults();}, [&](){this->forValues();},
     /*anonymous=*/2, /*greedy=*/true);// kernel (index 2) may itself contain '='; accept bare "x+1" or "t=x*x; t+1" alongside kernel='...'

  mParser.addAction(
     {"sdf2udf"}, "Converts a signed distance field into an unsigned distance field, i.e. performs the Abs of all values and changes GridClass to UNKNOWN.",
    {{"keep", "", "1|0|true|false", "toggle wether the input volume is preserved or deleted after the conversion"},
     {"vdb", "0", "0|0,1", "age(s) (i.e. stack index) of grid(s) to be processed. Defaults to 0, i.e. most recently inserted VDB. Accepts a comma-separated list to use multiple grids in the kernel: the FIRST grid is written (iterated), the rest are read-only inputs. Length must match use=."},
     {"name", "sphere", "sphere", "name assigned to the output volume"}},
     [&](){mParser.setDefaults();}, [&](){this->sdf2udf();});

  mParser.addAction(
     {"quad2tri", "q2t"}, "Convert all quads in mesh to triangles, assuming they are both planar and convex",
    {{"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"}},
     [&](){mParser.setDefaults();}, [&](){this->quadsToTriangles();});

  mParser.addAction(
     {"movie", "img2mpeg", "mov2mpeg", "mov2gif", "img2gif"}, "Convert image and movie files to mpeg or animated gif files",
    {{"fps", "24", "24", "desired frame rate of mpeg movie"},
     {"input", "slice_*.ppm", "slice_*.ppm|input.avi", "input image files or movie file to get converted"},
     {"output", "slices.mp4", "output.mp4|output.gif", "name of output mpeg or gif file"},
     {"scale", "", "1280x720|640", "scale of the output movie or gif."},
//     {"keep", "true", "1|0|true|false", "toggle wether the input images are preserved or deleted after the conversion"},
     {"flip", "", "vertical|horizontal|180", "flip output video vertical or horizontal or rotate it by 180"}},
     [&](){mParser.setDefaults();}, [&](){this->movie();});

  mParser.addAction(
     {"mesh2ls", "mesh2sdf"}, "Convert a watertight polygon surface into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh",
    {{"dim", "", "256", "largest dimension in voxel units of the mesh bbox (defaults to 256). If \"vdb\" or \"voxel\" is defined then \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"exWidth", "0.0", "3.0", "half-width in voxel units of the output narrow-band level set (disabled by default)"},
     {"inWidth", "0.0", "3.0", "half-width in voxel units of the input narrow-band level set (disabled by default)"},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"vdb", "-1", "0", "age (i.e. stack index) of reference grid used to define the transform. Defaults to -1, i.e. disabled. If specified this option takes precedence over \"dim\" and \"voxel\"!"},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"},
     {"name", "", "mesh2ls_input", "specify the name of the resulting vdb (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->meshToLevelSet();});

  mParser.addAction(
     {"soup2udf", "mesh2udf"}, "Convert a polygon soup into a to a unsigned distance field with an symmetrical narrow band",
    {{"dim", "", "256", "largest dimension in voxel units of the mesh bbox (defaults to 256). If \"vdb\" or \"voxel\" is defined then \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"vdb", "-1", "0", "age (i.e. stack index) of reference grid used to define the transform. Defaults to -1, i.e. disabled. If specified this option takes precedence over \"dim\" and \"voxel\"!"},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"},
     {"name", "", "mesh2udf_input", "specify the name of the resulting vdb (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->meshToUnsignedDistanceField();});

#ifdef VDB_TOOL_USE_SHRINKWRAP
  // Temporarily gated out of the PR. Not exposed via CMake by design; enable a
  // local build with: cmake -DCMAKE_CXX_FLAGS="-DVDB_TOOL_USE_SHRINKWRAP" ..
  mParser.addAction(
     {"soup2ls", "soup2sdf", "shrinkwrap"}, "Convert a polygon soup into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh",
    {{"dim", "", "256", "largest dimension in voxel units of the mesh bbox (defaults to 256). If \"vdb\" or \"voxel\" is defined then \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"mode", "0", "0", "mode of offset operator: 0) old method (using mesh -> UDF -> mesh -> SDF), 1) Mihai's signed-flood-fill and 2) Greg's createLevelSetDilatedMesh. Defaults to 0, i.e. paper."},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"erode", "8", "2", "number of iterations of constrained erosion. Defaults to 8."},
     {"thres", "0", "0.01", "closing (or engineering) threshold. Defaults to 0, i.e. it\'s diabled."},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"},
     {"name", "", "soup2ls_input", "specify the name of the resulting vdb (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->soupToLevelSet();});
#endif

  mParser.addAction(
     {"soup2offset"}, "Convert a polygon soup into an offset narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh",
    {{"dim", "", "256", "largest dimension in voxel units of the mesh bbox (defaults to 256). If \"vdb\" or \"voxel\" is defined then \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     //{"offset", "1.0", "1.0", "Offset in voxel units. Defaults to one, i.e. offset surface corresponds to one voxel dilation from mesh."},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"mode", "0", "0", "mode of offset operator: 0) old method (using mesh -> UDF -> mesh -> SDF), 1) Mihai's signed-flood-fill and 2) Greg's createLevelSetDilatedMesh. Defaults to 0, i.e. paper."},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"},
     {"name", "", "soup2ls_input", "specify the name of the resulting vdb (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->soupToOffset();});

  mParser.addAction(
     {"vol2mesh", "vdb2mesh"}, "Convert a scalar volume to an adaptive polygon mesh",
    {{"adapt", "0.0", "0.005", "normalized metric for the adaptive meshing. 0 is uniform and 1 is extreme adaptivity. Defaults to 0."},
     {"iso", "0.0", "0.1", "iso-value used to define the implicit surface. Defaults to zero."},
     {"vdb", "0", "0", "age (i.e. stack index) of the level set VDB grid to be meshed. Defaults to 0, i.e. most recently inserted VDB."},
     {"mask","-1", "1", "age (i.e. stack index) of the level set VDB grid used as a surface mask during meshing. Defaults to -1, i.e. it's disabled."},
     {"invert", "false", "1|0|true|false", "boolean toggle to mesh the complement of the mask. Defaults to false and ignored if no mask is specified."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing. The mask is never removed!"},
     {"name", "", "vol2mesh_input", "specify the name of the resulting vdb (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->volumeToMesh();});

  mParser.addAction(
     {"ls2mesh", "sdf2mesh"}, "Convert a level set to an adaptive polygon mesh",
    {{"adapt", "0.0", "0.005", "normalized metric for the adaptive meshing. 0 is uniform and 1 is extreme adaptivity. Defaults to 0."},
     {"iso", "0.0", "0.1", "iso-value used to define the implicit surface. Defaults to zero."},
     {"vdb", "0", "0", "age (i.e. stack index) of the level set VDB grid to be meshed. Defaults to 0, i.e. most recently inserted VDB."},
     {"mask","-1", "1", "age (i.e. stack index) of the level set VDB grid used as a surface mask during meshing. Defaults to -1, i.e. it's disabled."},
     {"invert", "false", "1|0|true|false", "boolean toggle to mesh the complement of the mask. Defaults to false and ignored if no mask is specified."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing. The mask is never removed!"},
     {"name", "", "ls2mesh_input", "specify the name of the resulting vdb (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->volumeToMesh();});

  mParser.addAction(
     {"fog2mesh"}, "Convert a fog volume to an adaptive polygon mesh",
    {{"adapt", "0.0", "0.005", "normalized metric for the adaptive meshing. 0 is uniform and 1 is extreme adaptivity. Defaults to 0."},
     {"iso", "0.5", "0.5", "iso-value used to define the implicit surface. Defaults to zero."},
     {"vdb", "0", "0", "age (i.e. stack index) of the level set VDB grid to be meshed. Defaults to 0, i.e. most recently inserted VDB."},
     {"mask","-1", "1", "age (i.e. stack index) of the level set VDB grid used as a surface mask during meshing. Defaults to -1, i.e. it's disabled."},
     {"invert", "false", "1|0|true|false", "boolean toggle to mesh the complement of the mask. Defaults to false and ignored if no mask is specified."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing. The mask is never removed!"},
     {"name", "", "fog2mesh_input", "specify the name of the resulting vdb (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->volumeToMesh();});

  mParser.addAction(
     {"ls2fog", "l2f", "sdf2fog"}, "Convert a level set VDB into a VDB with a fog volume, i.e. normalized density.",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"cutoff", "0.0", "3.0", "cut-off in voxel units so fog = sdf >=0 ? 0 : -sdf/|cutoff|*dx (defaults to 0, i.e. cutoff = max for smoothest ramp"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "ls2fog_input", "specify the name of the resulting VDB (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetToFog();});

  mParser.addAction(
     {"points2ls", "points2sdf", "p2l", "pts2sdf"}, "Convert geometry points into a narrow-band level set",
    {{"dim", "", "256", "largest dimension in voxel units of the bbox of all the points (defaults to 256). If \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"radius", "2.0", "2.0", "radius in voxel units of the input points"},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input points are preserved or deleted after the processing"},
     {"name", "", "points2ls_input", "specify the name of the resulting VDB (by default it's derived from the input points)"}},
     [&](){mParser.setDefaults();}, [&](){this->particlesToLevelSet();});

  mParser.addAction(
     {"iso2ls", "lsRebuild", "i2l"}, "Convert an iso-surface of a scalar field into a level set (i.e. SDF)",
    {{"vdb", "0", "0,1", "age (i.e. stack index) of the VDB grid to be processed and an optional reference grid. Defaults to 0, i.e. most recently inserted VDB."},
     {"iso", "0.0", "0.0", "value of the iso-surface from which to compute the level set"},
     {"voxel", "", "0.0", "voxel size in world units (defaults to zero, i.e the transform out the output matches the input)"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "iso2ls_input", "specify the name of the resulting VDB (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->isoToLevelSet();});

  mParser.addAction(
     {"points2vdb", "p2v"}, "Encode geometry points into a VDB grid",
    {{"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input points are preserved or deleted after the processing"},
     {"ppv", "8", "8", "the number of points per voxel in the output VDB grid (defaults to 8)"},
     {"bits", "16", "16|8|32", "the number of bits used to represent a single point in the VDB grid (defaults to 16, i.e. half precision)"},
     {"name", "", "points_2vdb_input", "specify the name of the resulting VDB (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->pointsToVdb();});

  mParser.addAction(
     {"vdb2points", "v2p"}, "Extract points encoded in a VDB to points in a geometry format",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "vdb2points_input", "specify the name of the resulting points (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->vdbToPoints();});

  mParser.addAction(
     {"scatter"}, "Scatter point into the active values of an input VDB grid",
    {{"count", "0", "0", "fixed number of points to randomly scatter (disabled by default)"},
     {"density", "0.0", "0.0", "uniform density of points per active voxel (disabled by default)"},
     {"ppv", "8", "8", "number of points per active voxel (defaults to 8)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be scatter points into. Defaults to 0, i.e. most recently inserted VDB"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "scatter_input", "specify the name of the resulting points (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->scatter();});

  mParser.addAction(
     {"platonic"}, "Create a level set shape with the specified number of polygon faces",
    {{"dim", "", "256", "largest dimension in voxel units of the bbox of all the shape (defaults to 256). In \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"faces", "4", "{4|6|8|12|20}", "number of polygon faces of the shape to generate the level set VDB from"},
     {"scale", "1.0", "1.0", "scale of the shape in world units. E.g. if faces=6 and scale=1.0 the result is a unit cube"},
     {"center", "(0,0,0)", "(0.0,0.0,0.0)", "center of the shape in world units. defaults to the origin"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"name", "", "Tetrahedron", "specify the name of the resulting VDB (by default it's derived from face count)"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetPlatonic();});

  mParser.addAction(
     {"enright"}, "Performs Enright advection benchmark test on a level set",
    {{"translate", "(0,0,0)", "(0.0,0.0,0.0)", "defines the origin of the Enright velocity field"},
     {"scale", "1.0", "1.0", "defined the scale of the Enright velocity field"},
     {"dt", "0.05", "0.05", "time-step the input level set is advected"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->enright();});

  mParser.addAction(
     {"dilate", "dilateLS"}, "dilate level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is dilated"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
     {"erode", "erodeLS"}, "erode level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is eroded"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
     {"open", "openLS"}, "morphological opening, i.e. erosion followed by dilation, of a level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is opened"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
     {"close", "closeLS"}, "morphological closing, i.e. dilation followed by erosion, of level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is closed"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
     {"gauss", "gaussLS"}, "gaussian convolution of a level set surface",
    {{"iter",  "1", "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
     {"mean", "meanLS"}, "mean value filtering of a level set surface",
    {{"iter",  "1",  "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
     {"median", "medianLS"}, "median value filtering of a level set surface",
    {{"iter",  "1",  "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
     {"cpt"}, "generate a vector grid with the closest-point-transform to a level set surface",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"div"}, "generate a scalar grid with the divergence of a vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"curl"}, "generate a vector grid with the curl of another vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"grad"}, "generate a vector grid with the gradient of a scalar grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"curvature"}, "generate scalar grid with the mean curvature of a level set surface",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"length"}, "generate a scalar grid with the magnitude of a vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
     {"union"}, "CSG union of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to union. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true", "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
     {"intersection"}, "CSG intersection of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to intersect. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true", "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
     {"difference"}, "CSG difference of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to difference. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true",  "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
     {"min"}, "Given grids A and B, compute min(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
     {"max"}, "Given grids A and B, compute max(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
     {"sum"}, "Given grids A and B, compute sum(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
     {"multires"}, "construct a LoD sequences of VDB trees with powers of two refinements",
    {{"levels", "2", "2", "number of multi-resolution grids in the output LoD sequence"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->multires();});

  mParser.addAction(
     {"resample"}, "resample one VDB grid into another VDB grid or a transformation of the input grid",
    {{"vdb", "0,1", "0,1", "pair of input and optional output grids (i.e. stack index) to be processed. Defaults to 0,1, i.e. most recent VDB is resampled to match the transform of the second to most recent VDB."},
     {"scale", "0", "0", "scale use to transform the input grid (ignored if two grids are specified with vdb)"},
     {"translate", "(0,0,0)", "(0,0,0)", "translation use to transform the input grid (ignored if two grids are specified with vdb)"},
     {"order", "1", "1", "order of the polynomial interpolation kernel used during resampling"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->resample();});

  mParser.addAction(
     {"clip"}, "Clip a VDB grid against another grid, a bbox or frustum",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"bbox", "", "(0,0,0),(1,1,1)", "min and max of the world-space bounding-box used for clipping. Defaults to empty, i.e. disabled"},
     {"taper", "-1", "1", "taper of the frustum (requires bbox and depth to be specified). Defaults to -1, i.e. disabled"},
     {"depth", "-1", "1", "depth in world units of the frustum (requires bbox and taper to be specified). Defaults to -1, i.e. disabled"},
     {"mask", "-1", "1", "age (i.e. stack index) of a mask VDB used for clipping. Defaults to -1, i.e. disabled"}},
     [&](){mParser.setDefaults();}, [&](){this->clip();});

  mParser.addAction(
     {"slice"}, "Generate images of slices of a VDB grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "true", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"file", "slice", "slice", "name of ppm file(s) of slices"},
     {"force", "0", "1|0|true|false", "force computations of min/max, else use expected values for LS and FOG volumes (default)"},
     {"scale", "512", "1920x1080", "pixel size of image (aspect ratio is derived from the vdb unless both dimensions are given)"},
     {"X", "0.5", "1", "One or more X-slices in range 0 -> 1. Defaults to 0.5, i.e. mid-point"},
     {"Y", "", "1", "One or more Y-slices in range 0 -> 1. Defaults to 0.5, i.e. mid-point"},
     {"Z", "", "1", "One or more Z-slices in range 0 -> 1. Defaults to 0.5, i.e. mid-point"}},
     [&](){mParser.setDefaults();}, [&](){this->slice();});

  mParser.addAction(
    {"prune"}, "prune away inactive values in a VDB grid",
   {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB"}},
     [](){},[&](){this->pruneLevelSet();});

  mParser.addAction(
     {"flood"}, "signed-flood filling of a level set VDB",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB"}},
     [](){},[&](){this->floodLevelSet();});

  mParser.addAction(
     {"expand"}, "expand narrow band of level set",
    {{"dilate", "1", "1", "number of integer voxels that the narrow band of the input SDF will be dilated"},
     {"iter", "1", "1", "number of iterations of the fast sweeping algorithm (each using 8 sweeps)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->expandLevelSet();});

  mParser.addAction(
     {"segment"}, "segment an input VDB into a list if topologically disconnected VDB grids",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->segment();});

  mParser.addAction(
     {"transform"}, "apply affine transformations (uniform scale -> rotation -> translation) to a VDB grids and geometry",
    {{"rotate", "(0.0,0.0,0.0)", "(0.0,0.0,0.0)", "rotation in radians around x,y,z axis"},
     {"translate", "(0.0,0.0,0.0)", "(0.0,0.0,0.0)", "translation in world units along x,y,z axis"},
     {"scale", "1.0", "1.0", "uniform scaling in world units"},
     {"vdb", "", "0,2,..", "age (i.e. stack index) of the VDB grid to be processed. Defaults to empty."},
     {"geo", "", "0,2,..", "age (i.e. stack index) of the Geometry to be processed. Defaults to empty."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or overwritten"}},
     [&](){mParser.setDefaults();}, [&](){this->transform();});

  mParser.addAction(
     {"render"}, "ray-tracing of level set surfaces and volume rendering of fog volumes",
    {{"files", "", "output.{jpg|png|ppm|exr}", "file used to save the rendered image to disk"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the rendering"},
     {"camera", "perspective", "persp|ortho", "perspective or orthographic camera"},
     {"aperture", "41.2136", "41.2136", "width in mm of the frame of a perspective camera, i.e., the visible field (defaults to 41.2136mm)"},
     {"focal", "50", "50", "focal length of a perspective camera in mm (defaults to 50mm)"},
     {"isovalue", "0.0", "0.0", "iso-value use during ray-intersection of level set surfaces"},
     {"samples", "1", "1", "number of samples (rays) per pixel"},
     {"image", "1920x1080", "1920x1080", "image size defined in terms of pixel resolution"},
     {"translate", "(0,0,0)", "(0,0,0)", "translation of the camera in world-space units, applied after rotation"},
     {"rotate", "(0,0,0)", "(0,0,0)", "rotation in degrees of the camera in world space (applied in x, y, z order)"},
     {"target", "(0,0,0)", "", "target point in world pace that the camera will point at (if undefined target is set to the center of the bbox of the grid)"},
     {"up", "(0,1,0)", "(0,1,0)", "vector that should point up after rotation with lookat"},
     {"lookat", "true", "true", "rotate the camera so it looks at the center of the shape uses up as the horizontal direction"},
     {"near", "0.001", "0.001", "depth of the near clipping plane in world-space units"},
     {"far",  "3.4e+38", "3.4e+38", "depth of the far clipping plane in world-space units"},
     {"shader", "diffuse", "diffuse|normal|position|matte", "shader type; either \"diffuse\", \"matte\", \"normal\" or \"position\""},
     {"light", "(0.3,0.3,0.0),(0.7,0.7,0.7)", "(0.3,0.3,0.0),(0.7,0.7,0.7)", "light source direction and optional color"},
     {"frame", "1.0", "1.0", "orthographic camera frame width in world units"},
     {"cutoff", "0.005", "0.005", "density and transmittance cutoff value (ignored for level sets)"},
     {"gain", "0.2", "0.2", "amount of scatter along the shadow ray (ignored for level sets)"},
     {"absorb", "(0.1,0.1,0.1)", "(0.1,0.1,0.1)", "absorption coefficients for RGB (ignored for level sets)"},
     {"scatter", "(1.5,1.5,1.5)", "(1.5,1.5,1.5)", "scattering coefficients for RGB (ignored for level sets)"},
     {"step", "1.0,3.0", "1.0,3.0", "step size in voxels for integration along the primary ray (ignored for level sets)"},
     {"colorgrid", "-1", "1", "age of a vec3s VDB grid to be used to set material colors. Defaults to -1, i.e. disabled"}},
     [&](){mParser.setDefaults();}, [&](){this->render();}, 0);

  mParser.addAction(
       {"print", "p"}, "prints information to the terminal about the current stack of VDB grids and Geometry",
      {{"vdb", "*", "*", "print information about VDB grids"},
       {"geo", "*", "*", "print information about geometries"},
       {"mem", "0", "0|1|false|true", "print a list of all stored variables"},
       {"level", "0", "0|1|2", "detail level: 0=base table, 1=+bbox column, 2=+value range column"}},
      [](){}, [&](){this->print();});

  mParser.addAction(
      {"version"}, "write timing information to the terminal", {},
      [&](){std::clog << mCmdName << ": version " << Tool::version() << std::endl;std::exit(EXIT_SUCCESS);}, [](){});

  mParser.addAction(
      {"examples"}, "print examples to the terminal and terminate", {},
      [&](){std::clog << this->examples() << std::endl; std::exit(EXIT_SUCCESS);}, [](){});

  mParser.addAction(
      {"errorOnWarning", "stopOnWarning"}, "stop on warnings, i.e. treat warnings as errors", {},
      [&](){mErrorOnWarning = true;}, [](){});

  mParser.addAction(
      {"log"}, "enable logging to file",
      {{"file",   "",      "vdb_tool.log",   "file used for logging. Use \"watch -n 0.5 vdb_tool.log\" to see updates in real-time."},
       {"append", "false", "0|1|false|true", "if true, append to the existing log file instead of truncating it"},
       {"tee",    "true",  "0|1|false|true", "if true (default), also write to the original terminal stream so interactive feedback is preserved"}},
      [&](){this->startLog(mParser.get<std::string>("file"),
                            mParser.get<bool>("append"),
                            mParser.get<bool>("tee"));}, [](){}, 0);

  Processor &proc = mParser.processor;

  // operations related to VDB grids
  proc.add("voxelSize", "voxel size of specified vdb grid, e.g. {0:voxelSize} -> {0.01}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->voxelSize()[0]);});

  proc.add("voxelCount", "number of active voxels of specified VDB grid, e.g. {0:voxelCount} -> {3269821}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->activeVoxelCount());});

  proc.add("gridCount", "push the number of loaded VDB grids onto the stack, e.g. {gridCount} -> {1}",
      [&](){proc.push(mGrid.size());});

  proc.add("gridName", "name of a specified VDB grid, e.g. {0:gridName} -> {sphere}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->getName());});

  proc.add("isGridEmpty", "test if a specified VDB grid is empty or not, e.g. {0:isGridEmpty} -> {0}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->empty());});

  proc.add("gridType", "value type of a specified VDB grid, e.g. {0:gridType} -> {float}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->valueType());});

  proc.add("gridClass", "class of a specified VDB grid, e.g. {0:gridClass} -> {ls}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            switch ((*it)->getGridClass()) {
                case GRID_LEVEL_SET: proc.set("ls"); break;
                case GRID_FOG_VOLUME: proc.set("fog"); break;
                default: proc.set("unknown");}});

  proc.add("isLS", "test if a specified VDB grid is a level set or not, e.g. {0:isLS} -> {1}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->getGridClass()==GRID_LEVEL_SET);});

  proc.add("isFOG", "test if a specified VDB grid is a fog volume or not, e.g. {0:isFOG} -> {0}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            proc.set((*it)->getGridClass()==GRID_FOG_VOLUME);});

  proc.add("gridDim", "voxel dimension of specified VDB grid, e.g. {0:gridDim} -> {[255,255,255]}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            const CoordBBox bbox = (*it)->evalActiveVoxelBoundingBox();
            std::stringstream ss;
            ss << bbox.dim();
            proc.set(ss.str());});

  proc.add("gridBBox", "world space bounding box of specified VDB grid, e.g. {0:gridBBox} -> {[-1.016,-1.016,-1.016] [1.016,1.016,1.016]}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            const CoordBBox bbox = (*it)->evalActiveVoxelBoundingBox();
            const math::BBox<Vec3d> bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
            const math::BBox<Vec3R> bboxWorld = bboxIndex.applyMap(*((*it)->transform().baseMap()));
            const auto &min = bboxWorld.min(), &max = bboxWorld.max();
            std::stringstream ss;
            ss << "["<<min[0]<<","<<min[1]<<","<<min[2]<<"] "
               << "["<<max[0]<<","<<max[1]<<","<<max[2]<<"]";
            proc.set(ss.str());});

  proc.add("gridCenter", "world space center of bounding box of specified VDB grid, e.g. {0:gridCenter} -> {[0.0,0.0,0.0]}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            const CoordBBox bbox = (*it)->evalActiveVoxelBoundingBox();
            const math::BBox<Vec3d> bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
            const math::BBox<Vec3R> bboxWorld = bboxIndex.applyMap(*((*it)->transform().baseMap()));
            const auto center = 0.5*(bboxWorld.max() + bboxWorld.min());
            std::stringstream ss;
            ss << "["<<center[0]<<","<<center[1]<<","<<center[2]<<"]";
            proc.set(ss.str());});

  proc.add("gridRadius", "world space radius of bounding box of specified VDB grid, e.g. {0:gridRadius} -> {1.73}",
      [&](){auto it = this->getGrid(strToInt(proc.get()));
            const CoordBBox bbox = (*it)->evalActiveVoxelBoundingBox();
            const math::BBox<Vec3d> bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
            const math::BBox<Vec3R> bboxWorld = bboxIndex.applyMap(*((*it)->transform().baseMap()));
            proc.set(0.5*(bboxWorld.max() - bboxWorld.min()).length());});

  // operations related to geometry
  proc.add("vtxCount", "number of voxels of a specified geometry, e.g. {0:vtxCount} -> {2461023}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            proc.set((*it)->vtxCount());});

  proc.add("polyCount", "number of polygons of a specified geometry, e.g. {0:polyCount} -> {23560}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            proc.set((*it)->polyCount());});

  proc.add("geomCount", "push the number of loaded geometries onto the stack, e.g. {geomCount} -> {1}",
      [&](){proc.push(mGrid.size());});

  proc.add("geomName", "name of a specified geometry, e.g. {0:geomName} -> {bunny}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            proc.set((*it)->getName());});

  proc.add("isGeomEmpty", "test if a specified VDB grid is empty or not, e.g. {0:isGridEmpty} -> {0}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            proc.set((*it)->isEmpty());});

  proc.add("geomClass", "class of a specified geometry, e.g. {0:geomClass} -> {points}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            if ((*it)->isPoints()) {
                proc.set("points");
            } else if ((*it)->isMesh()) {
                proc.set("mesh");
            } else {
                proc.set("unknown");}});

  proc.add("geomBBox", "world space bounding box of specified geometry, e.g. {0:geomBBox} -> {[-1.016,-1.016,-1.016] [1.016,1.016,1.016]}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            const auto &min = (*it)->bbox().min(), &max = (*it)->bbox().max();
            std::stringstream ss;
            ss << "["<<min[0]<<","<<min[1]<<","<<min[2]<<"] "
               << "["<<max[0]<<","<<max[1]<<","<<max[2]<<"]";
            proc.set(ss.str());});

  proc.add("geomCenter", "world space center of bounding box of specified geometry, e.g. {0:geomCenter} -> {[0.0,0.0,0.0]}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            const auto center = 0.5*((*it)->bbox().max() + (*it)->bbox().min());
            std::stringstream ss;
            ss << "["<<center[0]<<","<<center[1]<<","<<center[2]<<"]";
            proc.set(ss.str());});

  proc.add("geomRadius", "world space radius of bounding box of specified geometry, e.g. {0:geomRadius} -> {1.73}",
      [&](){auto it = this->getGeom(strToInt(proc.get()));
            proc.set(0.5*((*it)->bbox().max() - (*it)->bbox().min()).length());});

}// Tool::init()

// ==============================================================================================================

void Tool::help()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "help");
  try {
    mParser.printAction();
    const VecS actions = mParser.getVec<std::string>("actions");
    const bool stop = mParser.get<bool>("exit");
    const bool brief = mParser.get<bool>("brief");
    const std::string format = mParser.get<std::string>("format");

    if (format == "md") {
      // Emit a single Markdown table of registered actions for the README's
      // big "action list" section. Sorted alphabetically by primary name
      // (matches the order Parser::finalize() establishes). Generated output
      // should be diffed against README.md so it can't silently drift.
      std::clog << "| Action | Description |\n";
      std::clog << "|---|---|\n";
      for (const auto &a : mParser.available) {
        std::string desc = a.documentation;
        // Markdown table cells can't contain raw newlines or unescaped '|'.
        for (char &c : desc) if (c == '\n' || c == '\r') c = ' ';
        size_t bar = 0;
        while ((bar = desc.find('|', bar)) != std::string::npos) {
          desc.replace(bar, 1, "\\|");
          bar += 2;
        }
        std::clog << "| **" << a.names[0] << "** | " << desc << " |\n";
      }
      if (stop) std::exit(EXIT_SUCCESS);
      return;
    }

    if (actions.empty()) {
      if (mParser.actions.size()==1) {// ./vdb_tool -help
        if (!brief) {
          std::clog << "\nThis command-line tool can perform a use-defined, and possibly\n"
                    << "non-linear, sequence of high-level tasks available in openvdb.\n"
                    << "For instance, it can convert polygon meshes and particles to level\n"
                    << "sets, and subsequently perform a large number of operations on these\n"
                    << "level set surfaces. It can also generate adaptive polygon meshes from\n"
                    << "level sets, write them to disk and even render them to image files.\n\n"
                    << "Version: " + Tool::version() + "\n" + this->examples() + "\n";
        }
        mParser.usage_all(brief);
        if (!brief) {
          std::clog << "\nNote that actions always start with one or more \"-\", and (except for file names)\n"
                    << "its options always contain a \"=\" and an optional number of characters\n"
                    << "used for identification, e.g. \"-erode r=2\" is identical to \"-erode radius=2.0\"\n"
                    << "but \"-erode rr=2\" will produce an error since \"rr\" does not match\n"
                    << "the first two character of \"radius\". Also note that this tool maintains two\n"
                    << "lists of primitives, namely geometry (i.e. points and meshes) and level sets.\n"
                    << "They can be referenced with \"geo=n\" and \"vdb=n\" where the integer \"n\" refers\n"
                    << "to age (i.e stack index) of the primitive with \"n=0\" meaning most recent. E.g.\n"
                    << "-mesh2ls g=1\" means convert the second to last geometry (here polygon mesh) to a\n"
                    << "level set. Likewise \"-gauss v=0\" means perform a gaussian filter on the most\n"
                    << "recent level set (default).\n";
        }
      } else {// e.g. ./vdb_tool -sphere -dilate -help
        mParser.usage(brief);
      }
    } else {// ./vdb_tool -help sphere dilate
      mParser.usage(actions, brief);
    }

    if (stop) std::exit(EXIT_SUCCESS);
  } catch (const std::exception& e) {
    throw std::invalid_argument(action_name+": "+e.what());
  }
}// Tool::help()

// ==============================================================================================================

std::string Tool::examples() const
{
    const int w = 16;
    std::stringstream ss;
    ss << std::left << std::setw(w) << "Surface points:" << mCmdName << " -read points.[obj/ply/stl/off/pts] -points2ls d=256 r=2.0 w=3 -dilate r=2 -gauss i=1 w=1 -erode r=2 -ls2m a=0.25 -write output.[ply/obj/stl]\n";
    ss << std::left << std::setw(w) << "Convert mesh:  " << mCmdName << " -read mesh.[ply/obj/off] -mesh2ls d=256 -write output.vdb config.txt\n";
    ss << std::left << std::setw(w) << "Config example:" << mCmdName << " -config config.txt\n";
    return ss.str();
}

// ==============================================================================================================

void Tool::clear()
{
  OPENVDB_ASSERT(mParser.getAction().names[0] == "clear");
  if (mParser.get<std::string>("geo") == "*") {
    mGeom.clear();
  } else {
    for (int a : mParser.getVec<int>("geo")) {
      auto it = this->getGeom(a);
      mGeom.erase(std::next(it).base());
    }
  }
  if (mParser.get<std::string>("vdb")  == "*") {
    mGrid.clear();
  } else {
    for (int a : mParser.getVec<int>("vdb")) {
      auto it = this->getGrid(a);
      mGrid.erase(std::next(it).base());
    }
  }
  if (mParser.get<bool>("variables")) {
    mParser.processor.memory().clear();
  }
}// Tool::clear

// ==============================================================================================================

void Tool::read()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "read");
  try {
    for (auto &fileName : mParser.getVec<std::string>("files")) {
      switch (findFileExt(fileName, {"geo,obj,ply,abc,pts,off,stl,xyz,usd,usda,usdc,usdz,gltf,glb", "vdb", "nvdb"})) {
      case 1:
        this->readGeo(fileName);
        break;
      case 2:
        this->readVDB(fileName);
        break;
      case 3:
        this->readNVDB(fileName);
        break;
      default:
#if VDB_TOOL_USE_PDAL
        pdal::StageFactory factory;
        if (factory.inferReaderDriver(fileName) != "") {
          this->readGeo(fileName);
          break;
        }
#endif
        throw std::invalid_argument("File \""+fileName+"\" has an invalid extension");
      }// end switch
    }// end for loop over files
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping do to " << e.what() << std::endl;
    }
  }
}// Tool::read

// ==============================================================================================================

void Tool::readGeo(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().names[0] == "read");
  if (mParser.verbose>1) std::clog << "Reading geometry from \"" << fileName << "\"\n";
  if (mParser.verbose) mTimer.start("Read geometry file \"" + fileName + "\"");
  Geometry::Ptr geom(new Geometry());
  geom->read(fileName, mParser.verbose);
  if (geom->vtxCount()) {
    geom->setName(getBase(fileName));
    mGeom.push_back(geom);
  }
  if (mParser.verbose) {
    mTimer.stop();
    if (mParser.verbose>1) geom->print();
  }
}// Tool::readGeo

// ==============================================================================================================

void Tool::readVDB(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().names[0] == "read");
  const VecS gridNames = mParser.getVec<std::string>("grids");
  if (gridNames.empty()) throw std::invalid_argument("readVDB: no grids names specified");
  GridPtrVecPtr grids;
  if (fileName=="stdin.vdb") {
    if (isatty(fileno(stdin))) throw std::invalid_argument("readVDB: stdin is not connected to the terminal!");
    if (mParser.verbose) mTimer.start("Reading VDB grid(s) from input stream");
    io::Stream s(std::cin);
    grids = s.getGrids();
  } else {
    if (mParser.verbose) mTimer.start("Reading VDB grid(s) from file named \""+fileName+"\"");
    io::File file(fileName);
    file.open(mParser.get<bool>("delayed"));
    grids = file.getGrids();
  }
  const size_t count = mGrid.size();
  if (grids) {
    for (GridBase::Ptr grid : *grids) {
      if (gridNames[0]=="*" || findMatch(grid->getName(), gridNames)) mGrid.push_back(grid);
    }
  } else if (mParser.verbose) {
    std::clog << "readVDB: no vdb grids in \"" << fileName << "\"";
  }
  if (mParser.verbose) {
    mTimer.stop();
    if (mGrid.size() == count) std::clog << "readVDB: no vdb grids were loaded\n";
    if (mParser.verbose>1) for (GridBase::Ptr grid : *grids) grid->print();
  }
}// Tool::readVDB

// ==============================================================================================================

#ifdef VDB_TOOL_USE_NANO
void Tool::readNVDB(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().names[0] == "read");
  const VecS gridNames = mParser.getVec<std::string>("grids");
  if (gridNames.empty()) throw std::invalid_argument("readNVDB: no grids names specified");
  std::vector<nanovdb::GridHandle<>> grids;
  if (fileName=="stdin.nvdb") {
    if (isatty(fileno(stdin))) throw std::invalid_argument("readNVDB: stdin is not connected to the terminal!");
    if (mParser.verbose) mTimer.start("Reading NanoVDB grid(s) from input stream");
    grids = nanovdb::io::readGrids(std::cin);
    throw std::runtime_error("Not implemented");
  } else {
    if (mParser.verbose) mTimer.start("Reading NanoVDB grid(s) from file named \""+fileName+"\"");
    grids = nanovdb::io::readGrids(fileName);
  }
  const size_t count = mGrid.size();
  if (grids.size()) {
    for (const auto& gridHandle : grids) {
      if (gridNames[0]=="*" || findMatch(gridHandle.gridMetaData()->shortGridName(), gridNames)) mGrid.push_back(nanovdb::tools::nanoToOpenVDB(gridHandle));
    }
  } else if (mParser.verbose>0) {
    std::clog << "readVDB: no vdb grids in \"" << fileName << "\"";
  }
  if (mParser.verbose) {
    mTimer.stop();
    if (mGrid.size() == count) std::clog << "readNVDB: no NanoVDB grids were loaded\n";
    if (mParser.verbose>1) for (auto it = std::next(mGrid.cbegin(), count); it != mGrid.cend(); ++it) (*it)->print();
  }
}// Tool::readNVDB
#else
void Tool::readNVDB(const std::string&)
{
    throw std::runtime_error("NanoVDB support was disabled during compilation!");
}// Tool::readNVDB
#endif

// ==============================================================================================================

void Tool::config()
{
    OPENVDB_ASSERT(mParser.getAction().names[0] == "config");
    const bool update  = mParser.get<bool>("update");
    const bool execute = mParser.get<bool>("execute");
    std::string line;
    for (auto &fileName : mParser.getVec<std::string>("files")) {
        if (update) {
            std::fstream file(fileName, std::fstream::in | std::fstream::out);
            if (!file.is_open() || !getline (file, line)) throw std::invalid_argument("updateConf: failed to open file \""+fileName+"\"");
            const Header old_header(line), new_header;
            if (!old_header.isCompatible()) {
                std::stringstream ss;
                ss << new_header.str() << std::endl;
                ss << file.rdbuf();// load the rest of the config file
                file.clear();
                file.seekg(0);// rewind to start
                file << ss.rdbuf();// write back to the config file
            }
            file.close();
        }
        if (execute) {
            std::ifstream file(fileName);
            if (!file.is_open()) throw std::invalid_argument("readConf: unable to open \""+fileName+"\"");
            if (mParser.verbose>1) std::clog << "Reading configuration from \"" << fileName << "\"\n";
            if (mParser.verbose) mTimer.start("Read config file \"" + fileName + "\"");
            if (!getline (file,line)) throw std::invalid_argument("readConf: empty file \""+fileName+"\"");
            Header header(line);
            if (!header.isCompatible()) throw std::invalid_argument("readConf: incompatible version \""+line+"\"");
            std::vector<char*> args({&header.mMagic[0]});//parser is expecting first argument to the name of the executable
            while (getline(file, line)) {
                const size_t start = line.find_first_not_of(" \t"), stop = line.find_first_of("#%");
                if (start >= stop) continue;// line is empty or starts with a comment
                line = line.substr(start, stop - start);// remove leading whitespaces and tailing comments
                line = line.substr(0, line.find_last_not_of(" \t") + 1);// remove tailing whitespaces
                VecS tmp = vdb_tool::tokenize(line, " ");
                tmp[0].insert (0, 1, '-');// first token is an action
                std::transform(tmp.begin(), tmp.end(), std::back_inserter(args), [](const std::string &s){
                    char *c = new char[s.size()+1];
                    std::strcpy(c, s.c_str());
                    return c;
                });
            }
            file.close();
            mParser.parse(static_cast<int>(args.size()), args.data());
            if (mParser.verbose) mTimer.stop();
        }
    }
}// Tool::config

// ==============================================================================================================

void Tool::write()
{
  const std::string &action_name = mParser.getAction().names[0]; 
  OPENVDB_ASSERT(action_name == "write");
  try {
    for (std::string &fileName : mParser.getVec<std::string>("files")) {
      switch (findFileExt(fileName, {"geo,obj,ply,stl,off,abc", "vdb", "nvdb", "txt"})) {
      case 1:
        this->writeGeo(fileName);
        break;
      case 2:
        this->writeVDB(fileName);
        break;
      case 3:
        this->writeNVDB(fileName);
        break;
      case 4:
        this->writeConf(fileName);
        break;
      default:
        throw std::invalid_argument("File \""+fileName+"\" has an invalid extension");
        break;
      }
    }
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::write

// ==============================================================================================================

void Tool::writeVDB(const std::string &fileName)
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "write");
  try {
    mParser.printAction();
    const std::string age = mParser.get<std::string>("vdb");
    const bool keep = mParser.get<bool>("keep");
    const std::string codec = toLowerCase(mParser.get<std::string>("codec"));
    bool half;
    switch (mParser.get<int>("bits")) {
      case 16:
        half = true; break;
      case 32:
        half = false; break;
      default:
        throw std::invalid_argument("writeVDB: bits should either be 32 or 16, not "+mParser.get<std::string>("bits"));
    }
    GridPtrVec grids;// vector of grids to be written and possibly removed from mGrid
    if (age == "*") {
      for (auto it = mGrid.crbegin(); it != mGrid.crend(); ++it) grids.push_back(*it);
      if (!keep) mGrid.clear();
    } else {
      for (int a : vectorize<int>(age, ",")) grids.push_back(*this->getGrid(a));
      if (!keep) for (auto &g : grids) mGrid.remove(g);
    }

    if (grids.empty()) throw std::invalid_argument("no vdb grids to write");

    auto setCodec = [&](io::Archive &a) {
      if (codec=="zip") {
        a.setCompression(io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
      } else if (codec=="blosc") {
        a.setCompression(io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);
      } else if (codec=="active") {
        a.setCompression(io::COMPRESS_ACTIVE_MASK);
      } else if (codec=="none") {
        a.setCompression(io::COMPRESS_NONE);
      } else if (!codec.empty()) {
        throw std::invalid_argument("writeVDB: unsupported codec \""+codec+"\"");
      }
    };
    for (size_t i=0; half && i<grids.size(); ++i) grids[i]->setSaveFloatAsHalf(true);
    if (fileName=="stdout.vdb") {
      if (isatty(fileno(stdout)))  throw std::invalid_argument("writeVDB: stdout is not connected to the terminal");
      if (mParser.verbose) mTimer.start("Streaming VDB grid(s) to output stream");
      io::Stream stream(std::cout);
      setCodec(stream);
      stream.write(grids);
    } else {
      if (mParser.verbose) mTimer.start("Writing VDB grid(s) to file named \""+fileName+"\"");
      io::File file(fileName);
      setCodec(file);
      file.write(grids);
      file.close();
    }
    for (size_t i=0; half && i<grids.size(); ++i) grids[i]->setSaveFloatAsHalf(false);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("writeVDB: ") + e.what());// catch in Tool::write
  }
}// Tool::writeVDB

// ==============================================================================================================

#ifdef VDB_TOOL_USE_NANO
void Tool::writeNVDB(const std::string &fileName)
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "write");
  try {
    mParser.printAction();
    const std::string age = mParser.get<std::string>("vdb");
    const bool keep = mParser.get<bool>("keep");
    const std::string codec_str = toLowerCase(mParser.get<std::string>("codec"));
    const std::string bits = mParser.get<std::string>("bits");
    const bool dither = mParser.get<bool>("dither");
    const bool absolute = mParser.get<bool>("absolute");
    const float tolerance = mParser.get<float>("tolerance");// negative values means derive it from the grid class (eg ls or fog)
    const std::string stats = mParser.get<std::string>("stats");
    const std::string checksum = mParser.get<std::string>("checksum");
    const int verbose = mParser.verbose ? 1 : 0;

    nanovdb::io::Codec codec = nanovdb::io::Codec::NONE;// compression codec for the file
    if (codec_str == "zip") {
      codec = nanovdb::io::Codec::ZIP;
    } else if (codec_str == "blosc") {
      codec = nanovdb::io::Codec::BLOSC;
    } else if (!codec_str.empty() && codec_str != "none") {
      throw std::invalid_argument("writeNVDB: unsupported codec \""+codec_str+"\"");
    }

    nanovdb::GridType qMode = nanovdb::GridType::Unknown;// output grid type defaults to input grid type
    if (bits == "4") {
      qMode = nanovdb::GridType::Fp4;
    } else if (bits == "8") {
      qMode = nanovdb::GridType::Fp8;
    } else if (bits == "16") {
      qMode = nanovdb::GridType::Fp16;
    } else if (bits == "N") {
      qMode = nanovdb::GridType::FpN;
    } else if (bits != "" && bits != "32") {
      throw std::invalid_argument("writeNVDB: unsupported bits \""+bits+"\"");
    }

    nanovdb::tools::StatsMode sMode = nanovdb::tools::StatsMode::Default;
    if (stats == "none") {
      sMode = nanovdb::tools::StatsMode::Disable;
    } else if (stats == "bbox") {
      sMode = nanovdb::tools::StatsMode::BBox;
    } else if (stats == "extrema") {
      sMode = nanovdb::tools::StatsMode::MinMax;
    } else if (stats == "all") {
      sMode = nanovdb::tools::StatsMode::All;
    } else if (stats != "") {
      throw std::invalid_argument("writeNVDB: unsupported stats \""+stats+"\"");
    }

    nanovdb::CheckMode cMode = nanovdb::CheckMode::Default;
    if (checksum == "none") {
      cMode = nanovdb::CheckMode::Disable;
    } else if (checksum == "partial") {
      cMode = nanovdb::CheckMode::Partial;
    } else if (checksum == "full") {
      cMode = nanovdb::CheckMode::Full;
    } else if (checksum != "") {
      throw std::invalid_argument("writeNVDB: unsupported checksum \""+checksum+"\"");
    }

    GridPtrVec grids;// vector of grids to be written and possibly removed from mGrid
    if (age == "*") {
      for (auto it = mGrid.crbegin(); it != mGrid.crend(); ++it) grids.push_back(*it);
      if (!keep) mGrid.clear();
    } else {
      for (int a : vectorize<int>(age, ",")) grids.push_back(*this->getGrid(a));
      if (!keep) for (auto &g : grids) mGrid.remove(g);
    }

    if (grids.empty()) throw std::invalid_argument(action_name+": no vdb grids to write");

    auto openToNano = [&](const GridBase::Ptr& base) {
      if (auto floatGrid = GridBase::grid<FloatGrid>(base)) {
        using SrcGridT = openvdb::FloatGrid;
        switch (qMode){
        case nanovdb::GridType::Fp4:
          return nanovdb::tools::createNanoGrid<SrcGridT, nanovdb::Fp4>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::Fp8:
          return nanovdb::tools::createNanoGrid<SrcGridT, nanovdb::Fp8>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::Fp16:
          return nanovdb::tools::createNanoGrid<SrcGridT, nanovdb::Fp16>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::FpN:
          if (absolute) {
            return nanovdb::tools::createNanoGrid<SrcGridT, nanovdb::FpN>(*floatGrid, sMode, cMode, dither, verbose, nanovdb::tools::AbsDiff(tolerance));
          } else {
            return nanovdb::tools::createNanoGrid<SrcGridT, nanovdb::FpN>(*floatGrid, sMode, cMode, dither, verbose, nanovdb::tools::RelDiff(tolerance));
          }
        default: break;// 32 bit float grids are handled below
        }// end of switch
      }
      return nanovdb::tools::openToNanoVDB(base, sMode, cMode, verbose);// float and other grids
    };// openToNano

    if (fileName=="stdout.nvdb") {
      if (isatty(fileno(stdout)))  throw std::invalid_argument("writeNVDB: stdout is not connected to the terminal");
      if (mParser.verbose) mTimer.start("Streaming NanoVDB to stdout");
      for (auto grid: grids) {
        auto handle = openToNano(grid);
        nanovdb::io::writeGrid(std::cout, handle, codec);
      }
    } else {
      if (mParser.verbose) mTimer.start("Writing NanoVDB to file");
      std::ofstream os(fileName, std::ios::out | std::ios::binary);
      for (auto grid: grids) {
        auto handle = openToNano(grid);
        nanovdb::io::writeGrid(os, handle, codec);
      }
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(action_name+": "+e.what());
  }
}// Tool::writeNVDB
#else
void Tool::writeNVDB(const std::string&)
{
    throw std::runtime_error("NanoVDB support was disabled during compilation!");
}// Tool::writeNVDB
#endif

// ==============================================================================================================

void Tool::writeGeo(const std::string &fileName)
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "write");
  const int age = mParser.get<int>("geo");
  const bool keep = mParser.get<bool>("keep");
  const bool ascii = mParser.get<bool>("ascii");
  if (mParser.verbose>1) std::clog << "Writing geometry to \"" << fileName << "\"\n";
  auto it = this->getGeom(age);
  if (mParser.verbose) mTimer.start("Write geometry file \"" + fileName + "\"");
  (*it)->write(fileName, ascii);
  if (!keep) mGeom.erase(std::next(it).base());
  if (mParser.verbose) mTimer.stop();
}// Tool::writeGeo

// ==============================================================================================================

void Tool::writeConf(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().names[0] == "write");
  if (mParser.verbose>1) std::clog << "Writing configuration to \"" << fileName << "\"\n";
  std::ofstream file(fileName);
  if (!file.is_open()) throw std::invalid_argument("writeConf: unable to open \""+fileName+"\"");
  if (mParser.verbose) mTimer.start("Write config file \"" + fileName + "\"");
  const Header header;
  file << header.str() << std::endl;
  for (auto &a : mParser.actions) if (a.names[0] != "config") a.print(file);// exclude "-config" to avoid infinite loop
  file.close();
  if (mParser.verbose) mTimer.stop();
}// Tool::writeConf

// ==============================================================================================================

void Tool::vdbToPoints()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "vdb2points");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age);
    auto grid = gridPtrCast<points::PointDataGrid>(*it);
    if (!grid || grid->getGridClass() != GRID_UNKNOWN) {
      throw std::invalid_argument("no PointDataGrid with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("VDB to points");
    const size_t count = points::pointCount(grid->tree());
    if (count==0) throw std::invalid_argument("empty PointDataGrid with age "+std::to_string(age));
    Geometry::Ptr geom(new Geometry());
    geom->vtx().resize(count);
    Vec3s *points = geom->vtx().data();
    for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
      const points::AttributeArray& array = leafIter->constAttributeArray("P");
      points::AttributeHandle<Vec3f> positionHandle(array);
      for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
        Vec3f voxelPosition = positionHandle.get(*indexIter);
        const Vec3d xyz = indexIter.getCoord().asVec3d();
        Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);
        *points++ = worldPosition;
      }// loop over points in leaf node
    }// loop over leaf nodes
    if (!keep) mGrid.erase(std::next(it).base());
    if (geom->isPoints()) {
      if (grid_name.empty()) grid_name = "vdb2points_"+grid->getName();
      geom->setName(grid_name);
      mGeom.push_back(geom);
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(action_name+": "+e.what());
  }
}// Tool::vdbToPoints

// ==============================================================================================================

void Tool::pointsToVdb()
{
  const std::string &name = mParser.getAction().names[0];
  OPENVDB_ASSERT(name == "points2vdb");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    const int pointsPerVoxel = mParser.get<int>("ppv");
    const int bits = mParser.get<int>("bits");
    std::string grid_name = mParser.get<std::string>("name");
    using GridT = points::PointDataGrid;
    using IdGridT = tools::PointIndexGrid;
    if (mParser.verbose) mTimer.start("Points to VDB");
    auto it = this->getGeom(age);
    Points points((*it)->vtx());
    const float voxelSize = points::computeVoxelSize(points, pointsPerVoxel);
    auto xform = math::Transform::createLinearTransform(voxelSize);

    GridT::Ptr grid;
    IdGridT::Ptr indexGrid;

    points::PointAttributeVector<openvdb::Vec3s> positionsWrapper((*it)->vtx());
    openvdb::NamePair rgbAttribute ;
    switch (bits) {
    case 8:
      indexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(positionsWrapper, *xform);
      grid = points::createPointDataGrid<points::FixedPointCodec</*1-byte=*/true>, GridT>(*indexGrid, positionsWrapper, *xform);
      openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::registerType();
      rgbAttribute =
        openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::attributeType();
      openvdb::points::appendAttribute(grid->tree(), "Cd", rgbAttribute);
      break;
    case 16:
      indexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(positionsWrapper, *xform);
      grid = points::createPointDataGrid<points::FixedPointCodec</*1-byte=*/false>, GridT>(*indexGrid, positionsWrapper, *xform);
      openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::registerType();
      rgbAttribute =
        openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::attributeType();
      openvdb::points::appendAttribute(grid->tree(), "Cd", rgbAttribute);
      break;
    case 32:
      indexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(positionsWrapper, *xform);
      grid = points::createPointDataGrid<points::NullCodec, GridT>(*indexGrid, positionsWrapper, *xform);

      openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::registerType();
      rgbAttribute =
        openvdb::points::TypedAttributeArray<Vec3s, points::NullCodec>::attributeType();
      openvdb::points::appendAttribute(grid->tree(), "Cd", rgbAttribute);
      break;
    default:
      throw std::invalid_argument("pointsToVdb: unsupported bit-width: "+std::to_string(bits));
    }

    if ((*it)->rgb().size() == (*it)->vtx().size()) {

      points::PointAttributeVector<Vec3s> rgbWrapper((*it)->rgb());
      points::populateAttribute<openvdb::points::PointDataTree,
        openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<Vec3s>>(
            grid->tree(), indexGrid->tree(), "Cd", rgbWrapper);

    }
    if (grid_name.empty()) grid_name = "points2vdb_"+(*it)->getName();
    grid->setName(grid_name);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::pointsToVdb

// ==============================================================================================================

void Tool::transform()
{
  const std::string &name = mParser.getAction().names[0];
  OPENVDB_ASSERT(name == "transform");
  try {
    mParser.printAction();
    const auto vdb_age = mParser.getVec<int>("vdb");
    const auto geo_age = mParser.getVec<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    const Vec3d trans = mParser.getVec3<double>("translate");
    const Vec3d rot = mParser.getVec3<double>("rotate");
    const double scale = mParser.get<double>("scale");
    if (scale<=0.0) throw std::invalid_argument(name+": invalid scale: "+std::to_string(scale));

    for (int age : vdb_age) {
      auto it = this->getGrid(age);
      GridBase::Ptr grid(nullptr);
      if (keep) {
        grid = (*it)->copyGrid();// transform and tree are shared
        if (!grid->getName().empty()) grid->setName("xform_"+grid->getName());
        grid->setTransform((*it)->transform().copy());// new transform
        mGrid.push_back(grid);
      } else {
        grid = *it;
      }
      // Order of translations: scale -> rotate -> translate
      if (scale!=1.0)  grid->transform().postScale(scale);
      if (rot[0]!=0.0) grid->transform().postRotate(rot[0], math::X_AXIS);
      if (rot[1]!=0.0) grid->transform().postRotate(rot[1], math::Y_AXIS);
      if (rot[2]!=0.0) grid->transform().postRotate(rot[2], math::Z_AXIS);
      if (trans.length()>0.0) grid->transform().postTranslate(trans);
    }
    if (geo_age.empty()) return;

    // Order of translations: scale -> rotate -> translate
    math::Transform::Ptr xform = math::Transform::createLinearTransform(scale);
    if (rot[0]!=0.0) xform->postRotate(rot[0], math::X_AXIS);
    if (rot[1]!=0.0) xform->postRotate(rot[1], math::Y_AXIS);
    if (rot[2]!=0.0) xform->postRotate(rot[2], math::Z_AXIS);
    if (trans.length()>0.0) xform->postTranslate(trans);
    for (int age : geo_age) {
      auto it = this->getGeom(age);
      Geometry::Ptr geom(nullptr);
      if (keep) {
        geom = (*it)->deepCopy();
        if (!geom->getName().empty()) geom->setName("xform_"+geom->getName());
        mGeom.push_back(geom);
      } else {
        geom = *it;
      }
      geom->transform(*xform);
    }
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::transform

// ==============================================================================================================

void Tool::levelSetToFog()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "ls2fog");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    const float cutoff = mParser.get<float>("cutoff");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age);
    auto sdf = gridPtrCast<FloatGrid>(*it);
    if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument("no Level Set with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("SDF to FOG");
    FloatGrid::Ptr fog = keep ? sdf->deepCopy() : sdf;
    const float cutoffDistance = cutoff <= 0.0f ? sdf->background() : cutoff * sdf->voxelSize()[0];
    tools::sdfToFogVolume(*fog, cutoffDistance);// fog <- sdf > 0 ? 0 : -sdf / |cutoffDistance|
    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = "ls2fog_"+sdf->getName();
    fog->setName(grid_name);
    mGrid.push_back(fog);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::levelSetToFog

// ==============================================================================================================

void Tool::isoToLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "iso2ls");
  try {
    mParser.printAction();
    const VecI age = mParser.getVec<int>("vdb");
    if (age.size()!=1 && age.size()!=2) throw std::invalid_argument(action_name+": expected one or two vdb grids, not "+std::to_string(age.size()));
    const float isoValue = mParser.get<float>("iso");
    const float voxel = mParser.get<float>("voxel");
    const float width = mParser.get<float>("width");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age[0]);
    auto grid = gridPtrCast<FloatGrid>(*it);
    if (!grid) throw std::invalid_argument("no VDB with age " + std::to_string(age[0]));
    if (mParser.verbose) mTimer.start("Iso to SDF");
    math::Transform::Ptr xform(nullptr);
    if (age.size()==2) {
      auto it = this->getGrid(age[1]);
      xform = (*it)->transform().copy();
    } else if (voxel>0.0f) {
      xform = math::Transform::createLinearTransform(voxel);
    }
    auto sdf = tools::levelSetRebuild(*grid, isoValue, width, xform.get());
    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = "iso2ls_"+grid->getName();
    sdf->setName(grid_name);
    mGrid.push_back(sdf);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::isoToLevelSet

// ==============================================================================================================

float Tool::estimateVoxelSize(int maxDim,  float exWidth, float inWidth, int geo_age)
{
  auto it = this->getGeom(geo_age);
  const auto bbox = (*it)->bbox();
  if (!bbox) {
    throw std::invalid_argument("estimateVoxelSize: invalid bbox");
  } else if (maxDim <= 0) {
    throw std::invalid_argument("estimateVoxelSize: invalid maxDim");
  }
  const auto d = bbox.extents()[bbox.maxExtent()];// longest extent of bbox along any coordinate axis
  return static_cast<float>(static_cast<double>(d)/static_cast<double>(maxDim - static_cast<int>(exWidth + inWidth)));
}// Tool::estimateVoxelSize

// ==============================================================================================================

void Tool::quadsToTriangles()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "quad2tri");
  try {
    mParser.printAction();
    const int geo_age = mParser.get<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    Geometry::Ptr mesh = *this->getGeom(geo_age);
    if (mesh->isPoints()) throw std::invalid_argument("called on points, i.e. no quads!");
    if (keep) {
      Geometry::Ptr meshCopy = mesh->deepCopy();
      mGeom.push_back(meshCopy);
      mesh = meshCopy;
    }
    if (mParser.verbose) mTimer.start("Quads -> Triangles");
    mesh->triangulateQuads();
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::quadsToTriangles

// ==============================================================================================================

void Tool::meshToLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "mesh2ls");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const float width = mParser.get<float>("width");
    const float exWidth = mParser.get<float>("exWidth");
    const float inWidth = mParser.get<float>("inWidth");
    const int geo_age = mParser.get<int>("geo");
    const int vdb_age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    math::Transform::Ptr xform(nullptr);
    if (vdb_age>=0) {// use xform from reference VDB
      auto it = this->getGrid(vdb_age);
      xform = (*it)->transform().copy();
    } else if (exWidth <= 0.0 || inWidth <= 0.0) {
      if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, width, geo_age);
      xform = math::Transform::createLinearTransform(voxel);
    } else {
      if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, exWidth, inWidth, geo_age);
      xform = math::Transform::createLinearTransform(voxel);
    }
    auto it = this->getGeom(geo_age);
    const Geometry &mesh = **it;
    if (mesh.isPoints()) throw std::invalid_argument("Warning: -mesh2ls/mesh2sdf was called on points, not a mesh! Hint: use -points2ls instead!");
    if (exWidth <= 0.0 || inWidth <= 0.0) {// symmetric narrow-band
        if (mParser.verbose) mTimer.start("Mesh -> LS");
        auto grid  = tools::meshToLevelSet<GridT>(*xform, mesh.vtx(), mesh.tri(), mesh.quad(), width);
        if (grid_name.empty()) grid_name = "mesh2ls_" + mesh.getName();
        grid->setName(grid_name);
        mGrid.push_back(grid);
    } else {// asymmetric narrow-band
        if (mParser.verbose) mTimer.start("Mesh -> SDF");
        auto grid  = tools::meshToSignedDistanceField<GridT>(*xform, mesh.vtx(), mesh.tri(), mesh.quad(), exWidth, inWidth);
        if (grid_name.empty()) grid_name = "mesh2sdf_" + mesh.getName();
        grid->setName(grid_name);
        mGrid.push_back(grid);
    }
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::meshToLevelSet

// ==============================================================================================================

void Tool::meshToUnsignedDistanceField()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "mesh2udf");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const float width = mParser.get<float>("width");
    const int geo_age = mParser.get<int>("geo");
    const int vdb_age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    math::Transform::Ptr xform(nullptr);
    if (vdb_age>=0) {// use xform from reference VDB
      auto it = this->getGrid(vdb_age);
      xform = (*it)->transform().copy();
    } else {
      if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, width, geo_age);
      xform = math::Transform::createLinearTransform(voxel);
    }
    auto it = this->getGeom(geo_age);
    const Geometry &mesh = **it;
    if (mesh.isPoints()) throw std::invalid_argument("only points, expected mesh! Hint: use -points2ls instead!");
    if (mParser.verbose) mTimer.start("Mesh -> UDF");
    auto grid = tools::meshToUnsignedDistanceField<GridT>(*xform, mesh.vtx(), mesh.tri(), mesh.quad(), width);
    if (grid_name.empty()) grid_name = action_name + "_" + mesh.getName();
    grid->setName(grid_name);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::meshToUnsignedDistanceField

// ==============================================================================================================

#ifdef VDB_TOOL_USE_SHRINKWRAP
void Tool::soupToLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "soup2ls");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");// final dimension
    float voxel = mParser.get<float>("voxel");// final voxel size
    const float width = mParser.get<float>("width");
    const int offset_mode = mParser.get<int>("mode");
    const int geo_age = mParser.get<int>("geo");
    const int nErode = mParser.get<int>("erode");
    const float thres = mParser.get<float>("thres");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    auto it = this->getGeom(geo_age);
    Geometry::Ptr mesh = *it;
    if (mesh->isPoints()) {
      if (!keep) mGeom.erase(std::next(it).base());
      throw std::invalid_argument("got points, expected mesh! Hint: use -points2ls instead!");
    }
    if (keep) mesh = mesh->deepCopy();// deep copy since mesh will be modified below
    if (mParser.verbose) mTimer.start("Soup -> SDF");

    Spinner spin, *progress = mParser.verbose ? &spin : nullptr;
    const tools::ShrinkWrapLimit D(nErode, thres);
    tools::PolySoup poly{std::move(mesh->vtx()), std::move(mesh->tri()), std::move(mesh->quad()), mesh->bbox()};
    auto grid = tools::polySoupToLevelSet<GridT>(std::move(poly), dim, voxel, D, width, progress, offset_mode);

    if (mParser.verbose) mTimer.stop();

    if (grid_name.empty()) grid_name = "soup2ls_" + mesh->getName();
    grid->setName(grid_name);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());

  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::soupToLevelSet
#endif// VDB_TOOL_USE_SHRINKWRAP

// ==============================================================================================================

void Tool::soupToOffset()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "soup2offset");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");// final dimension
    float voxel = mParser.get<float>("voxel");// final voxel size
    const float width = mParser.get<float>("width");
    //const float offset = mParser.get<float>("offset");
    const int offset_mode = mParser.get<int>("mode");
    const int geo_age = mParser.get<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    auto it = this->getGeom(geo_age);
    if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, width, geo_age);
    Geometry::Ptr mesh = *it;
    if (mesh->isPoints()) {
      if (!keep) mGeom.erase(std::next(it).base());
      throw std::invalid_argument("got points, expected mesh! Hint: use -points2ls instead!");
    }
    if (keep) mesh = mesh->deepCopy();// deep copy since mesh will be modified below
    if (mParser.verbose) mTimer.start("Soup -> Offset");

    tools::PolySoup poly{std::move(mesh->vtx()), std::move(mesh->tri()), std::move(mesh->quad()), mesh->bbox()};
    tools::PolySoupToLevelSet<GridT> tmp(std::move(poly), voxel, width);
    auto grid = tmp.offset(voxel, offset_mode);

    if (mParser.verbose) mTimer.stop();

    if (grid_name.empty()) grid_name = "soup2offset_" + mesh->getName();
    grid->setName(grid_name);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());

  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::soupToOffset

// ==============================================================================================================

void Tool::particlesToLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "points2ls");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const float width = mParser.get<float>("width");
    const float radius = mParser.get<float>("radius");
    const int age = mParser.get<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, width, age);
    auto it = this->getGeom(age);
    const Geometry &points = **it;
    if (points.isMesh()) throw std::invalid_argument("got mesh, expected points! Hint: use -mesh2ls instead!");
    if (mParser.verbose) mTimer.start("Points->SDF");
    GridT::Ptr grid = createLevelSet<GridT>(voxel, width);
    if (grid_name.empty()) grid_name = action_name + "_"+points.getName();
    grid->setName(grid_name);
    tools::particlesToSdf(Points(points.vtx()), *grid, voxel*radius);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::particlesToLevelSet

// ==============================================================================================================

typename Tool::FilterT Tool::createFilter(GridT &grid, int space, int time)
{
  auto filter = std::make_unique<tools::LevelSetFilter<GridT>>(grid);

  switch (space) {
  case 1:
    filter->setSpatialScheme(math::FIRST_BIAS);
    break;
  case 2:
    filter->setSpatialScheme(math::SECOND_BIAS);
    break;
  case 3:
    filter->setSpatialScheme(math::THIRD_BIAS);
    break;
  case 5:
#if 0
    filter->setSpatialScheme(math::WENO5_BIAS);
#else
    filter->setSpatialScheme(math::HJWENO5_BIAS);
#endif
    break;
  default:
    throw std::invalid_argument("createFilter: invalid space discretization scheme \""+std::to_string(space)+"\"");
  }

  switch (time) {
  case 1:
    filter->setTemporalScheme(math::TVD_RK1);
    break;
  case 2:
    filter->setTemporalScheme(math::TVD_RK2);
    break;
  case 3:
    filter->setTemporalScheme(math::TVD_RK3);
    break;
  default:
    throw std::invalid_argument("createFilter: invalid time discretization scheme \""+std::to_string(time)+"\"");
  }
  return filter;
}// Tool::createFilter

// ==============================================================================================================

void Tool::offsetLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(findMatch(action_name, {"dilate", "erode", "open", "close"}));
  try {
    mParser.printAction();
    float radius = mParser.get<float>("radius");
    const int space = mParser.get<int>("space");
    const int time = mParser.get<int>("time");
    const int age = mParser.get<int>("vdb");
    if (radius<0) throw std::invalid_argument("offsetLevelSet: invalid radius");
    if (radius==0) return;
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument("no level set with age " + std::to_string(age));
    }
    auto filter = this->createFilter(*grid, space, time);
    radius *= static_cast<float>((*it)->voxelSize()[0]);// voxel to world units
    if (action_name == "dilate") {
      if (mParser.verbose) mTimer.start("Dilate  SDF");
      filter->offset(-radius);
    } else if (action_name == "erode") {
      if (mParser.verbose) mTimer.start("Erode   SDF");
      filter->offset( radius);
    } else if (action_name == "open") {
      if (mParser.verbose) mTimer.start("Open   SDF");
      filter->offset( radius);
      filter->offset(-radius);
    } else if (action_name == "close") {
      if (mParser.verbose) mTimer.start("Close   SDF");
      filter->offset(-radius);
      filter->offset( radius);
    } else {
      throw std::invalid_argument("offsetLevelSet: invalid operation type");
    }
    grid->setName(action_name + "_" + grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::offsetLevelSet

// ==============================================================================================================

void Tool::filterLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(findMatch(action_name, {"gauss", "mean", "median"}));
  try {
    mParser.printAction();
    const int nIter = mParser.get<int>("iter");
    const int space = mParser.get<int>("space");
    const int time = mParser.get<int>("time");
    const int age = mParser.get<int>("vdb");
    const int size = mParser.get<int>("size");
    if (size<0) throw std::invalid_argument("filterLevelSet: invalid filter size");
    if (size==0) return;
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
    }
    auto filter = this->createFilter(*grid, space, time);

    if (action_name == "gauss") {
      if (mParser.verbose) mTimer.start("Gauss   SDF");
      for (int i=0; i<nIter; ++i) filter->gaussian(size);
    } else if (action_name == "mean") {
      if (mParser.verbose) mTimer.start("Mean SDF ");
      for (int i=0; i<nIter; ++i) filter->mean(size);
    } else if (action_name == "median") {
      if (mParser.verbose) mTimer.start("Median SDF");
      for (int i=0; i<nIter; ++i) filter->median(size);
    } else {
      throw std::invalid_argument("filterLevelSet: invalid filter type");
    }
    grid->setName(action_name + "_" + grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::filterLevelSet

// ==============================================================================================================

void Tool::pruneLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "prune");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("Prune   SDF");
    tools::pruneLevelSet(grid->tree());
    grid->setName("prune_"+grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::pruneLevelSet

// ==============================================================================================================

void Tool::floodLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "flood");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("Flood   SDF");
    tools::signedFloodFill(grid->tree());
    grid->setName("flood_"+grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::floodLevelSet

// ==============================================================================================================

void Tool::compute()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(findMatch(action_name, {"cpt","div","curl","length","grad","curvature"}));
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    if (action_name == "cpt") {
      if (mParser.verbose) mTimer.start("CPT of SDF");
      auto sdf = gridPtrCast<FloatGrid>(*it);
      if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("cpt: no level set with age "+std::to_string(age));
      auto grid = tools::cpt(*sdf);
      grid->setName("cpt_"+sdf->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (action_name == "div") {
      if (mParser.verbose) mTimer.start("Divergence");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("div: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::divergence(*vec);
      grid->setName("div_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (action_name == "curl") {
      if (mParser.verbose) mTimer.start("Curl of Vec3");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("curl: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::curl(*vec);
      grid->setName("curl_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (action_name == "length") {
      if (mParser.verbose) mTimer.start("Length of Vec3");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("length: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::magnitude(*vec);
      grid->setName("length_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (action_name == "grad") {
      if (mParser.verbose) mTimer.start("Gradient");
      auto scalar = gridPtrCast<FloatGrid>(*it);
      if (!scalar) throw std::invalid_argument("grad: no float grid with age "+std::to_string(age));
      auto grid = tools::gradient(*scalar);
      grid->setName("grad_"+scalar->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (action_name == "curvature") {
      if (mParser.verbose) mTimer.start("Curvature");
      auto scalar = gridPtrCast<FloatGrid>(*it);
      if (!scalar) throw std::invalid_argument("curv: no float grid with age "+std::to_string(age));
      auto grid = tools::meanCurvature(*scalar);
      grid->setName("curv_"+scalar->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else {
      throw std::invalid_argument("csg: invalid type");
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping due to: " << e.what() << std::endl;
    }
  }
}// Tool::compute

// ==============================================================================================================

void Tool::composite()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(findMatch(action_name, {"min","max","sum"}));
  try {
    mParser.printAction();
    const VecI ij = mParser.getVec<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    if (ij.size()!=2) throw std::invalid_argument(action_name+": expected two vdb ages, but got "+std::to_string(ij.size()));
    if (ij[0] == ij[1]) throw std::invalid_argument(action_name+": identical inputs: volume1=volume2="+std::to_string(ij[0]));
    auto itA = this->getGrid(ij[0]), itB = this->getGrid(ij[1]);
    GridT::Ptr gridA = gridPtrCast<GridT>(*itA);
    if (!gridA) throw std::invalid_argument(action_name + ": no float grid with age " + std::to_string(ij[0]));
    GridT::Ptr gridB = gridPtrCast<GridT>(*itB);
    if (!gridB) throw std::invalid_argument(action_name + ": no float grid with age " + std::to_string(ij[1]));
    if (gridA->transform() != gridB->transform()) throw std::invalid_argument(action_name+": grids have different transforms");
    GridT::Ptr tmpA, tmpB;
    if (keep) {
      tmpA = gridA->deepCopy();
      tmpB = gridB->deepCopy();
      mGrid.push_back(tmpA);
    } else {
      tmpA = gridA;
      tmpB = gridB;
      mGrid.erase(std::next(itB).base());// remove B from mGrids since it will be destroyed
    }
    tmpA->setName(action_name+"_"+tmpA->getName());
    if (mParser.verbose) mTimer.start(action_name);
    if (action_name == "min") {
      tools::compMin(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else if (action_name == "max") {
      tools::compMax(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else if (action_name == "sum") {
      tools::compSum(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else {
      throw std::invalid_argument(action_name+": invalid operation");
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name + ": " + e.what());
    } else {
      std::clog << action_name << ": skipping do to " << e.what() << std::endl;
    }
  }
}// Tool::composite

// ==============================================================================================================

void Tool::csg()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(findMatch(action_name, {"union", "intersection", "difference"}));
  try {
    mParser.printAction();
    const VecI ij = mParser.getVec<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    const bool prune = mParser.get<bool>("prune");
    const bool rebuild = mParser.get<bool>("rebuild");
    if (ij.size()!=2) throw std::invalid_argument("csg: expected two vdb ages, but got "+std::to_string(ij.size()));
    if (ij[0] == ij[1]) throw std::invalid_argument("csg: identical inputs: volume1=volume2="+std::to_string(ij[0]));
    auto itA = this->getGrid(ij[0]), itB = this->getGrid(ij[1]);
    GridT::Ptr gridA = gridPtrCast<GridT>(*itA);
    if (!gridA || gridA->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(ij[0]));
    }
    GridT::Ptr gridB = gridPtrCast<GridT>(*itB);
    if (!gridB || gridB->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(ij[1]));
    }
    if (gridA->transform() != gridB->transform()) {
      if (gridA->voxelSize()[0]<gridB->voxelSize()[0]) {// use the smallest voxel size
        const float halfWidth = static_cast<float>(gridA->background()/gridA->voxelSize()[0]);
        if (mParser.verbose) mTimer.start("Rebuilding "+std::to_string(ij[1]));
        gridB = tools::levelSetRebuild(*gridB, 0.0f, halfWidth, &(gridA->transform()));
      } else {
        const float halfWidth = static_cast<float>(gridB->background()/gridB->voxelSize()[0]);
        if (mParser.verbose) mTimer.start("Rebuilding "+std::to_string(ij[0]));
        gridA = tools::levelSetRebuild(*gridA, 0.0f, halfWidth, &(gridB->transform()));
      }
      if (mParser.verbose) mTimer.stop();
    }
    if (action_name == "union") {
      if (mParser.verbose) mTimer.start("Union");
      if (keep) {
        GridT::Ptr grid = tools::csgUnionCopy(*gridA, *gridB);
        if (rebuild) grid = tools::sdfToSdf(*grid);
        grid->setName("union_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgUnion(*gridA, *gridB, prune);// overwrites A and cannibalizes B
        if (rebuild) gridA = tools::sdfToSdf(*gridA);
        gridA->setName("union_"+gridA->getName());
      }
    } else if (action_name == "intersection") {
      if (mParser.verbose) mTimer.start("Intersection");
      if (keep) {
        GridT::Ptr grid = tools::csgIntersectionCopy(*gridA, *gridB);
        if (rebuild) grid = tools::sdfToSdf(*grid);
        grid->setName("intersection_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgIntersection(*gridA, *gridB, prune);// overwrites A and cannibalizes B
        if (rebuild) gridA = tools::sdfToSdf(*gridA);
        gridA->setName("intersection_"+gridA->getName());
      }
    } else if (action_name == "difference") {
      if (mParser.verbose) mTimer.start("Difference");
      if (keep) {
        GridT::Ptr grid = tools::csgDifferenceCopy(*gridA, *gridB);
        if (rebuild) grid = tools::sdfToSdf(*grid);
        grid->setName("difference_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgDifference(*gridA, *gridB, prune);// overwrites A and deletes B
        if (rebuild) gridA = tools::sdfToSdf(*gridA);
        gridA->setName("difference_"+gridA->getName());
      }
    } else {
      throw std::invalid_argument("csg: invalid type");
    }
    if (!keep) mGrid.erase(std::next(itB).base());// remove B since it was corrupted
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::csg

// ==============================================================================================================

void Tool::volumeToMesh()
{
  const std::string &action_name = mParser.getAction().names[0];
  const int mode = findMatch(action_name, {"ls2mesh", "fog2mesh", "vol2mesh"});// 1-based index
  OPENVDB_ASSERT(mode);// mode = 0 for no match
  try {
    mParser.printAction();
    const double adaptivity = mParser.get<float>("adapt");
    const double iso = mParser.get<float>("iso");
    const int age = mParser.get<int>("vdb");
    const int mask = mParser.get<int>("mask");
    const bool invert = mParser.get<bool>("invert");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    auto it = this->getGrid(age);// will throw if grid doesn't exist
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument("no FloatGrid with age " + std::to_string(age));
    if (mode==1 && grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument("no level set with age "+std::to_string(age));
    } else if (mode==2 && grid->getGridClass() != GRID_FOG_VOLUME) {
      throw std::invalid_argument("no fog volume with age "+std::to_string(age));
    }
    
    if (mParser.verbose) mTimer.start(action_name);

    tools::VolumeToMesh mesher(iso, adaptivity, /*relaxDisorientedTriangles*/true);
    if (mask >= 0) {
      auto base = *this->getGrid(mask);// might throw
      if (base->isType<BoolGrid>()) {
        mesher.setSurfaceMask(base, invert);
      } else if (base->isType<FloatGrid>()) {
        mesher.setSurfaceMask(tools::interiorMask(*gridPtrCast<FloatGrid>(base), 0.0), invert);
      } else if (base->isType<Vec3fGrid>()) {
        mesher.setSurfaceMask(tools::interiorMask(*gridPtrCast<Vec3fGrid>(base)), invert);
      } else {
        throw std::invalid_argument("unsupported mask type with age "+std::to_string(mask));
      }
    }
    mesher(*grid);
    Geometry::Ptr geom = this->mesherToGeometry(mesher);

    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = action_name + "_" + grid->getName();
    geom->setName(grid_name);
    mGeom.push_back(geom);

    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::volumeToMesh

// ==============================================================================================================

void Tool::forValues()
{
  const std::string &action_name = mParser.getAction().names[0];
  const int mode = findMatch(action_name, {"forAllValues", "forOnValues", "forOffValues"});// 1-based index
  OPENVDB_ASSERT(mode);// mode = 0 for no match
  try {
    // Multi-grid: vdb= and use= accept comma-separated lists. The FIRST
    // grid in vdb= is the OUTPUT (iterated and written); the rest are
    // read-only inputs accessible through their respective use= names.
    std::vector<int> ages = mParser.getVec<int>("vdb");
    std::vector<std::string> voxel_vars = mParser.getVec<std::string>("use");
    const bool keep = mParser.get<bool>("keep");
    const std::string kernel = mParser.get<std::string>("kernel");
    const std::string cls = mParser.get<std::string>("class");
    std::vector<float> back = mParser.getVec<float>("background");
    std::string grid_name = mParser.get<std::string>("name");
    if (ages.empty()) ages.push_back(0);
    if (voxel_vars.empty()) voxel_vars.push_back("v");
    if (ages.size() != voxel_vars.size()) {
        throw std::invalid_argument(action_name + ": vdb= and use= must have the "
            "same number of entries (got vdb=" + std::to_string(ages.size()) +
            " entries and use=" + std::to_string(voxel_vars.size()) + " entries)");
    }
    for (const std::string &v : voxel_vars) {
        if (v.empty()) {
            throw std::invalid_argument(action_name+": each use= entry must be a non-empty identifier");
        }
    }
    // The first (output) grid; secondary grids gathered below into `inputs`.
    auto it0 = this->getGrid(ages[0]);
    GridT::Ptr grid = gridPtrCast<GridT>(*it0);
    if (!grid) throw std::invalid_argument("no FloatGrid with age " + std::to_string(ages[0]));
    if (keep) {
      GridT::Ptr tmp = grid->deepCopy();
      mGrid.push_back(tmp);
      grid = tmp;
    }
    if (grid_name.empty()) grid_name = action_name + "_" + grid->getName();
    grid->setName(grid_name);
    const std::string &voxel_var = voxel_vars[0];// the OUTPUT grid's kernel name

    if (mParser.verbose) mTimer.start(action_name);
    if (!kernel.empty()) {
      // Gather every grid referenced via use=/vdb= as a "read source".
      // grids[0] is the OUTPUT (being iterated and written); grids[1..]
      // are read-only inputs.
      struct GridRef { std::string name; int age; GridT::Ptr grid; };
      std::vector<GridRef> grids;
      grids.push_back({voxel_vars[0], ages[0], grid});
      for (size_t k = 1; k < ages.size(); ++k) {
          auto itK = this->getGrid(ages[k]);
          GridT::Ptr gK = gridPtrCast<GridT>(*itK);
          if (!gK) {
              throw std::invalid_argument(action_name +
                  ": vdb=" + std::to_string(ages[k]) + " is not a FloatGrid");
          }
          // Disallow accidentally listing the same name twice (the second
          // would shadow the first and create confusing kernels).
          for (size_t j = 0; j < k; ++j) {
              if (voxel_vars[j] == voxel_vars[k]) {
                  throw std::invalid_argument(action_name + ": duplicate use= "
                      "name \"" + voxel_vars[k] + "\" in use=...");
              }
          }
          grids.push_back({voxel_vars[k], ages[k], gK});
      }

      Calculator calc;
      // Configure the neighbor-function rewriter BEFORE compile so calls
      // like "x(1,0,0)" or "y(0,1,0)" are recognized for ANY grid name in
      // voxel_vars and synthesized into "<name>(dx,dy,dz)" variables.
      calc.setNeighborFunctions(voxel_vars);
      calc.compile(kernel);// throws on syntax error / unknown op

      const auto &mem = mParser.processor.memory();

      // Parse a synthesized neighbor name "<prefix>(dx,dy,dz)" into the
      // grid index (via prefix lookup) and integer offsets. Returns -1 if
      // the name doesn't match any registered grid prefix.
      auto parseNeighbor = [&grids](const std::string &name,
                                     int &gridIdx, int &dx, int &dy, int &dz) -> bool {
          for (size_t k = 0; k < grids.size(); ++k) {
              const std::string prefix = grids[k].name + "(";
              if (name.size() <= prefix.size() + 1) continue;
              if (name.compare(0, prefix.size(), prefix) != 0) continue;
              if (name.back() != ')') continue;
              const std::string inner = name.substr(
                  prefix.size(), name.size() - prefix.size() - 1);
              int vals[3] = {0, 0, 0};
              int idx = 0;
              size_t start = 0;
              bool ok = true;
              for (size_t i = 0; i <= inner.size(); ++i) {
                  if (i == inner.size() || inner[i] == ',') {
                      if (idx >= 3) { ok = false; break; }
                      const std::string num = inner.substr(start, i - start);
                      try { vals[idx++] = std::stoi(num); }
                      catch (...) { ok = false; break; }
                      start = i + 1;
                  }
              }
              if (!ok || idx != 3) continue;
              gridIdx = static_cast<int>(k);
              dx = vals[0]; dy = vals[1]; dz = vals[2];
              return true;
          }
          return false;
      };

      // Classify each Calculator input variable into one of three buckets:
      //   - center binding: bare name matches one of the use= names; bound
      //     per-voxel to grid[k]'s value at the current coord (or *it if k==0).
      //   - neighbor binding: synthesized "<name>(dx,dy,dz)"; bound via
      //     thread-local ConstAccessor to grid[k] at (i+dx, j+dy, k+dz).
      //   - constant binding: looked up once in Processor memory.
      struct CenterBinding   { int idx; int gridIdx; };
      struct NeighborBinding { int idx; int gridIdx; int dx, dy, dz; };
      std::vector<CenterBinding>   centers;
      std::vector<NeighborBinding> neighbors;
      std::vector<float> base(calc.variables().size());
      for (size_t i = 0; i < calc.variables().size(); ++i) {
          const std::string &name = calc.variables()[i];
          // Center reference: bare name == one of the registered grid names.
          int matchedGrid = -1;
          for (size_t k = 0; k < grids.size(); ++k) {
              if (grids[k].name == name) { matchedGrid = static_cast<int>(k); break; }
          }
          if (matchedGrid >= 0) {
              centers.push_back({static_cast<int>(i), matchedGrid});
              continue;
          }
          // Neighbor reference: "<name>(dx,dy,dz)" for a known grid.
          int gIdx, dx, dy, dz;
          if (parseNeighbor(name, gIdx, dx, dy, dz)) {
              neighbors.push_back({static_cast<int>(i), gIdx, dx, dy, dz});
              continue;
          }
          // Otherwise: looked up once in Processor memory.
          if (!mem.isSet(name)) {
              std::string nameList;
              for (size_t k = 0; k < grids.size(); ++k) {
                  if (k) nameList += ", ";
                  nameList += "\"" + grids[k].name + "\"";
              }
              throw std::invalid_argument(
                  action_name+": kernel references undefined variable \""+name+
                  "\" (set it first with -eval / -calc, or use one of the grid "
                  "names [" + nameList + "] for the current voxel value, or "
                  "<name>(dx,dy,dz) for a relative neighbor)");
          }
          base[i] = strTo<float>(mem.get(name));
      }

      // Snapshot the OUTPUT grid only if the kernel reads non-zero offsets
      // from it: otherwise parallel writes to the iterator's grid would
      // race with neighbor reads from the same grid. Input grids are
      // read-only so they don't need a snapshot.
      const bool needSnapshot = std::any_of(neighbors.begin(), neighbors.end(),
          [](const NeighborBinding &n) {
              return n.gridIdx == 0 && (n.dx || n.dy || n.dz);
          });
      GridT::Ptr out_snap = needSnapshot ? grid->deepCopy() : nullptr;

      // One thread-local ConstAccessor per grid. accessors[0] reads the
      // output's snapshot (or live grid if no snapshot is needed);
      // accessors[k>0] read each input grid directly.
      using ConstAcc = GridT::ConstAccessor;
      using AccTLS  = tbb::enumerable_thread_specific<ConstAcc>;
      std::vector<std::unique_ptr<AccTLS>> accessors;
      accessors.reserve(grids.size());
      for (size_t k = 0; k < grids.size(); ++k) {
          GridT::Ptr src = (k == 0 && out_snap) ? out_snap : grids[k].grid;
          accessors.push_back(std::make_unique<AccTLS>(
              [src]{ return src->getConstAccessor(); }));
      }

      // Per-voxel lambda. Each TBB worker reuses its thread-local cached
      // accessors via accessors[k]->local() for fast sequential reads.
      auto kernel_fn = [&calc, &base, &centers, &neighbors, &accessors](auto &it) {
          constexpr size_t kMaxVars = 64;
          OPENVDB_ASSERT(base.size() <= kMaxVars);
          float values[kMaxVars];
          for (size_t i = 0; i < base.size(); ++i) values[i] = base[i];
          const openvdb::Coord c = it.getCoord();
          const float center0 = static_cast<float>(*it);
          // Bind bare-name center references: gridIdx==0 uses *it (no
          // accessor cost), others go through their accessor at the
          // current coord.
          for (const CenterBinding &cb : centers) {
              if (cb.gridIdx == 0) {
                  values[cb.idx] = center0;
              } else {
                  values[cb.idx] = static_cast<float>(
                      accessors[cb.gridIdx]->local().getValue(c));
              }
          }
          // Bind neighbor references via the corresponding accessor.
          for (const NeighborBinding &nb : neighbors) {
              if (nb.gridIdx == 0 && nb.dx == 0 && nb.dy == 0 && nb.dz == 0) {
                  values[nb.idx] = center0;
              } else {
                  values[nb.idx] = static_cast<float>(
                      accessors[nb.gridIdx]->local().getValue(
                          c.offsetBy(nb.dx, nb.dy, nb.dz)));
              }
          }
          it.setValue(calc.eval(values));
      };
      switch (mode) {
      case 1: tools::foreach(grid->beginValueAll(), kernel_fn); break;
      case 2: tools::foreach(grid->beginValueOn(),  kernel_fn); break;
      case 3: tools::foreach(grid->beginValueOff(), kernel_fn); break;
      default:
        throw std::invalid_argument("forEachKenel: invalid mode = " + std::to_string(mode));
        break;
      }
    }
    if (int n = findMatch(cls, {"ls", "fog", "unknown"})) {
      auto class_tag = n==1 ? GRID_LEVEL_SET : n==2 ? GRID_FOG_VOLUME : GRID_UNKNOWN;
      grid->setGridClass(class_tag);
    }
    if (back.size()==1) {
      tools::changeBackground(grid->tree(), back[0]);// +/- outside (sign is preserved)
    } else if (back.size()==2) {
      tools::changeAsymmetricLevelSetBackground(grid->tree(), back[0], back[1]);// outside, inside
    }
    if (mParser.verbose) mTimer.stop();

  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::forValues

// ==============================================================================================================

void Tool::sdf2udf()
{
  const std::string &action_name = mParser.getAction().names[0];
  const int mode = findMatch(action_name, {"sdf2udf"});// 1-based index
  OPENVDB_ASSERT(mode);// mode = 0 for no match
  try {
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    /// get the relevant grid to be processed
    auto it = this->getGrid(age);// will throw if grid doesn't exist
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument("no FloatGrid with age " + std::to_string(age));
    if (keep) {
      GridT::Ptr tmp = grid->deepCopy();
      mGrid.push_back(tmp);
      grid = tmp;
    }
    if (grid_name.empty()) grid_name = action_name + "_" + grid->getName();
    grid->setName(grid_name);
    if (grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("no level set with age "+std::to_string(age));
    grid->setGridClass(GRID_UNKNOWN);// GRID_LEVEL_SET -> GRID_UNKNOW

    if (mParser.verbose) mTimer.start(action_name);
    tools::foreach(grid->beginValueAll(), [](auto &it){it.setValue(math::Abs(*it));});
    if (mParser.verbose) mTimer.stop();

  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::sdf2udf

// ==============================================================================================================

void Tool::levelSetSphere()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "sphere");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const float radius = mParser.get<float>("radius");
    const Vec3f center = mParser.getVec3<float>("center");
    const float width = mParser.get<float>("width");
    const std::string grid_name = mParser.get<std::string>("name");
    if (voxel == 0.0f) voxel = 2.0f*radius/(static_cast<float>(dim) - 2.0f*width);
    if (mParser.verbose) mTimer.start(action_name);
    GridT::Ptr grid = tools::createLevelSetSphere<GridT>(radius, center, voxel, width);
    if (mParser.verbose) mTimer.stop();
    grid->setName(grid_name);
    mGrid.push_back(grid);
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::levelSetSphere

// ==============================================================================================================

void Tool::levelSetPlatonic()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "platonic");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const int faces = mParser.get<int>("faces");
    const float scale = mParser.get<float>("scale");
    const Vec3f center = mParser.getVec3<float>("center");
    const float width = mParser.get<float>("width");
    const std::string grid_name = mParser.get<std::string>("name");
    if (voxel == 0.0f) voxel = 2.0f*scale/(static_cast<float>(dim) - 2*width);
    std::string shape;
    switch (faces) {// TETRAHEDRON=4, CUBE=6, OCTAHEDRON=8, DODECAHEDRON=12, ICOSAHEDRON=20
      case  4: shape = "Tetrahedron"; break;
      case  6: shape = "Cube"; break;
      case  8: shape = "Octahedron"; break;
      case 12: shape = "Dodecahedron"; break;
      case 20: shape = "Icosahedron"; break;
      default: throw std::invalid_argument("levelSetPlatonic: invalid face count: "+std::to_string(faces));
    }
    if (mParser.verbose) mTimer.start("Create "+shape);
    GridT::Ptr grid = tools::createLevelSetPlatonic<GridT>(faces, scale, center, voxel, width);
    if (mParser.verbose) mTimer.stop();
    if (grid_name.empty()) {
      grid->setName(shape);
    } else {
      grid->setName(grid_name);
    }
    mGrid.push_back(grid);
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::levelSetPlatonic

// ==============================================================================================================

void Tool::multires()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "multires");
  try {
    mParser.printAction();
    const int levels = mParser.get<int>("levels");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument(action_name + ": no VDB with age " + std::to_string(age));
    if (mParser.verbose) mTimer.start("MultiResGrid");
    if (keep) {
      tools::MultiResGrid<GridT::TreeType> mrg(levels+1, *grid);
      for (size_t level=1; level<mrg.numLevels(); ++level) mGrid.push_back(mrg.grid(level));
    } else {
      tools::MultiResGrid<GridT::TreeType> mrg(levels+1, grid);
      mGrid.erase(std::next(it).base());
      for (size_t level=1; level<mrg.numLevels(); ++level) mGrid.push_back(mrg.grid(level));
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::multires

// ==============================================================================================================

void Tool::expandLevelSet()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "expand");
  try {
    mParser.printAction();
    const int dilate = mParser.get<int>("dilate");
    const int iter = mParser.get<int>("iter");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    GridT::Ptr sdf = gridPtrCast<GridT>(*it);
    if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("Expand SDF");
    auto grid = tools::dilateSdf(*sdf, dilate, tools::NN_FACE, iter);
    if (!keep) mGrid.erase(std::next(it).base());
    grid->setName("expand_"+grid->getName());
    mGrid.push_back(grid);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::expandLevelSet

// ==============================================================================================================

void Tool::segment()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "segment");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    if (mParser.verbose) mTimer.start("Segmenting VDB");
    std::vector<GridBase::Ptr> grids;
    if (auto grid = gridPtrCast<GridT>(*it)) {
      std::vector<GridT::Ptr> segments;
      if (grid->getGridClass() == GRID_LEVEL_SET) {
        tools::segmentSDF(*grid, segments);
      } else {
        tools::segmentActiveVoxels(*grid, segments);
      }
      for (auto g : segments) grids.push_back(g);
    } else {
      throw std::invalid_argument(action_name + ": no VDB with age " + std::to_string(age));
    }
    if (!keep) mGrid.erase(std::next(it).base());
    for (auto g : grids) mGrid.push_back(g);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::segment

// ==============================================================================================================

// for simplicity we are restricting this resampler to only work on float grids!
void Tool::resample()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "resample");
  try {
    mParser.printAction();
    const VecI age = mParser.getVec<int>("vdb");
    const float scale = mParser.get<float>("scale");
    const Vec3d translate = mParser.getVec3<double>("translate");
    const int order = mParser.get<int>("order");
    const bool keep = mParser.get<bool>("keep");

    if (age.size()!=1 && age.size()!=2) throw std::invalid_argument("resample: expected one or two arguments to \"vdb\"");
    auto itIn = this->getGrid(age[0]);
    FloatGrid::Ptr inGrid = gridPtrCast<FloatGrid>(*itIn), outGrid;
    if (age.size()==2) {
      auto itOut = this->getGrid(age[1]);
      outGrid = gridPtrCast<FloatGrid>(*itOut);
      if (!outGrid) throw std::invalid_argument(action_name+": no reference grid of type float with age "+std::to_string(age[1]));
    } else {
      if (scale<=0.0f) throw std::invalid_argument("resample: invalid scale: "+std::to_string(scale));
      auto map = math::MapBase::Ptr(new math::UniformScaleTranslateMap(scale, translate));
      auto xform = math::Transform::Ptr(new math::Transform(map));
      outGrid = FloatGrid::create();
      outGrid->setTransform(xform);
    }

    if (!inGrid) throw std::invalid_argument(action_name+": no grid of type float with age "+std::to_string(age[0]));

    if (mParser.verbose) mTimer.start("Resampling VDB");
    switch (order) {
    case 0:
      tools::resampleToMatch<tools::PointSampler>(*inGrid, *outGrid);
      break;
    case 1:
      tools::resampleToMatch<tools::BoxSampler>(*inGrid, *outGrid);
      break;
    case 2:
      tools::resampleToMatch<tools::QuadraticSampler>(*inGrid, *outGrid);
      break;
    default:
      throw std::invalid_argument("resample: invalid interpolation order: "+std::to_string(order));
    }
    if (!keep) mGrid.erase(std::next(itIn).base());
    if (age.size()==1) mGrid.push_back(outGrid);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::resample

// ==============================================================================================================

void Tool::scatter()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "scatter");
  try {
    mParser.printAction();
    const Index64 count = mParser.get<int>("count");
    const float density = mParser.get<float>("density");
    const int pointsPerVoxel = mParser.get<int>("ppv");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument(action_name + ": no VDB with age " + std::to_string(age));
    if (mParser.verbose) mTimer.start("SDF -> mesh");
    Geometry::Ptr geom(new Geometry());
    struct PointWrapper {
      std::vector<Vec3f> &xyz;
      PointWrapper(std::vector<Vec3f> &_xyz) : xyz(_xyz) {}
      Index64 size() const { return Index64(xyz.size()); }
      void add(const Vec3d &p) { xyz.emplace_back(float(p[0]), float(p[1]), float(p[2])); }
    } points(geom->vtx());
    using RandGenT = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
          0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>; // mt11213b
    RandGenT mtRand;

    if (count>0) {// fixed point count scattering
      tools::UniformPointScatter<PointWrapper, RandGenT> tmp(points, count, mtRand);
      tmp(*grid);
    } else if (density>0.0f) {// uniform density scattering
      tools::UniformPointScatter<PointWrapper, RandGenT> tmp(points, density, mtRand);
      tmp(*grid);
    }   else if (pointsPerVoxel>0) {// dense uniform scattering
      tools::DenseUniformPointScatter<PointWrapper, RandGenT> tmp(points, static_cast<float>(pointsPerVoxel), mtRand);
      tmp(*grid);
    } else {
      throw std::invalid_argument("scatter: internal error");
    }
    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = "scatter_"+grid->getName();
    geom->setName(grid_name);
    mGeom.push_back(geom);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::scatter

// ==============================================================================================================

void Tool::slice()
{
  using RangeT = tbb::blocked_range2d<int>;
  struct Axis {
    const std::string label;// string name of this axis
    const VecF        slices;// fractional slices along the current axis
    const Vec3I       abc;// indics of the three axis
    Axis(const Parser &p, char c, int i, int j, int k) : label(1,c), slices(p.getVec<float>(label)), abc(i,j,k) {}
  };

  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "slice");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    const bool force = mParser.get<bool>("force");
    const std::string file = mParser.get<std::string>("file");
    const VecI scale = mParser.getVec<int>("scale", "x");
    const std::vector<Axis> axes = {{mParser, 'X', 0, 1, 2}, {mParser, 'Y', 1, 0, 2}, {mParser, 'Z', 2, 0, 1}};

    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument(action_name + ": no VDB with age " + std::to_string(age));
    const auto &tree = grid->tree();
    if (mParser.verbose) mTimer.start(action_name);

    // color LUT from https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a (Apache-2.0)
    const unsigned char LUT[256][3] = {{48,18,59},{50,21,67},{51,24,74},{52,27,81},{53,30,88},{54,33,95},{55,36,102},{56,39,109},{57,42,115},{58,45,121},{59,47,128},{60,50,134},{61,53,139},{62,56,145},{63,59,151},{63,62,156},{64,64,162},{65,67,167},{65,70,172},{66,73,177},{66,75,181},{67,78,186},{68,81,191},{68,84,195},{68,86,199},{69,89,203},{69,92,207},{69,94,211},{70,97,214},{70,100,218},{70,102,221},{70,105,224},{70,107,227},{71,110,230},{71,113,233},{71,115,235},{71,118,238},{71,120,240},{71,123,242},{70,125,244},{70,128,246},{70,130,248},{70,133,250},{70,135,251},{69,138,252},{69,140,253},{68,143,254},{67,145,254},{66,148,255},{65,150,255},{64,153,255},{62,155,254},{61,158,254},{59,160,253},{58,163,252},{56,165,251},{55,168,250},{53,171,248},{51,173,247},{49,175,245},{47,178,244},{46,180,242},{44,183,240},{42,185,238},{40,188,235},{39,190,233},{37,192,231},{35,195,228},{34,197,226},{32,199,223},{31,201,221},{30,203,218},{28,205,216},{27,208,213},{26,210,210},{26,212,208},{25,213,205},{24,215,202},{24,217,200},{24,219,197},{24,221,194},{24,222,192},{24,224,189},{25,226,187},{25,227,185},{26,228,182},{28,230,180},{29,231,178},{31,233,175},{32,234,172},{34,235,170},{37,236,167},{39,238,164},{42,239,161},{44,240,158},{47,241,155},{50,242,152},{53,243,148},{56,244,145},{60,245,142},{63,246,138},{67,247,135},{70,248,132},{74,248,128},{78,249,125},{82,250,122},{85,250,118},{89,251,115},{93,252,111},{97,252,108},{101,253,105},{105,253,102},{109,254,98},{113,254,95},{117,254,92},{121,254,89},{125,255,86},{128,255,83},{132,255,81},{136,255,78},{139,255,75},{143,255,73},{146,255,71},{150,254,68},{153,254,66},{156,254,64},{159,253,63},{161,253,61},{164,252,60},{167,252,58},{169,251,57},{172,251,56},{175,250,55},{177,249,54},{180,248,54},{183,247,53},{185,246,53},{188,245,52},{190,244,52},{193,243,52},{195,241,52},{198,240,52},{200,239,52},{203,237,52},{205,236,52},{208,234,52},{210,233,53},{212,231,53},{215,229,53},{217,228,54},{219,226,54},{221,224,55},{223,223,55},{225,221,55},{227,219,56},{229,217,56},{231,215,57},{233,213,57},{235,211,57},{236,209,58},{238,207,58},{239,205,58},{241,203,58},{242,201,58},{244,199,58},{245,197,58},{246,195,58},{247,193,58},{248,190,57},{249,188,57},{250,186,57},{251,184,56},{251,182,55},{252,179,54},{252,177,54},{253,174,53},{253,172,52},{254,169,51},{254,167,50},{254,164,49},{254,161,48},{254,158,47},{254,155,45},{254,153,44},{254,150,43},{254,147,42},{254,144,41},{253,141,39},{253,138,38},{252,135,37},{252,132,35},{251,129,34},{251,126,33},{250,123,31},{249,120,30},{249,117,29},{248,114,28},{247,111,26},{246,108,25},{245,105,24},{244,102,23},{243,99,21},{242,96,20},{241,93,19},{240,91,18},{239,88,17},{237,85,16},{236,83,15},{235,80,14},{234,78,13},{232,75,12},{231,73,12},{229,71,11},{228,69,10},{226,67,10},{225,65,9},{223,63,8},{221,61,8},{220,59,7},{218,57,7},{216,55,6},{214,53,6},{212,51,5},{210,49,5},{208,47,5},{206,45,4},{204,43,4},{202,42,4},{200,40,3},{197,38,3},{195,37,3},{193,35,2},{190,33,2},{188,32,2},{185,30,2},{183,29,2},{180,27,1},{178,26,1},{175,24,1},{172,23,1},{169,22,1},{167,20,1},{164,19,1},{161,18,1},{158,16,1},{155,15,1},{152,14,1},{149,13,1},{146,11,1},{142,10,1},{139,9,2},{136,8,2},{133,7,2},{129,6,2},{126,5,2},{122,4,3}};

    const CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    const Coord dim = bbox.dim();
    math::Extrema ex;
    if (!force && grid->getGridClass() == GRID_LEVEL_SET) {
      ex.add( tree.background());
      ex.add(-tree.background());
    } else if (!force && grid->getGridClass() == GRID_FOG_VOLUME) {
      ex.add(0.0f);
      ex.add(1.0f);
    } else {
      ex = tools::extrema(grid->cbeginValueOn());
    }

    for (const Axis &axis : axes) {
      tools::Film image(scale[0], scale.size()==2 ? scale[1] : scale[0]*dim[axis.abc[2]]/dim[axis.abc[1]]);
      for (const float slice : axis.slices) {
        tbb::parallel_for(RangeT(0, image.width(), 0, image.height()), [&](const RangeT &range){
          const int a = axis.abc[0], b = axis.abc[1], c = axis.abc[2];
          Vec3R xyz;
          xyz[a] = slice * (dim[a]+1) + bbox.min()[a];
          auto acc = grid->getAccessor();// thread local copy
          for (auto row=range.rows().begin(); row!=range.rows().end(); ++row) {
            xyz[b] = row/float(image.width())*(dim[b]+1) + bbox.min()[b];
            for (int col=range.cols().begin(); col<range.cols().end(); ++col) {
              xyz[c] = col/float(image.height())*(dim[c]+1) + bbox.min()[c];
              const float v = tools::BoxSampler::sample(acc, xyz);
              // Clamp before the integer cast: float-to-uint8_t conversion is
              // UB when the value falls outside [0,255], and clang on ARM64
              // emits an unmasked fcvtzu — values outside ex's range (e.g. the
              // background of a level set whose active values have been
              // rewritten by forOnValues) yield huge indices and an OOB read.
              const float t = (v - ex.min()) / (ex.max() - ex.min());
              const int   k = int(255.0f * math::Clamp(t, 0.0f, 1.0f));
              const unsigned char *p = LUT[k];
              image.pixel(row,col) = tools::Film::RGBA(p[0]/255.0f, p[1]/255.0f, p[2]/255.0f);
            }// loop over colums in image
          }// loop over rows in image
        });// end parallel_for
        image.savePPM(file + "_" + axis.label + "_" + std::to_string(slice)+ ".ppm");
      }// loop over slices within an axis (singular)
    }// loop over axes (plural)

    if (!keep) mGrid.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::slice

// ==============================================================================================================

/// @brief Convert multiple image files to a mpeg movie file
// vdb_tool -sphere -for x=0,1,0.01 -slice X='{$x}' -end -img2mpeg input="slice_*.ppm" output=slices.mp4
// vdb_tool -sphere -for x=0,1,0.01 -slice X='{$x}' -end -img2mpeg && open slices.mp4
void Tool::movie()
{
#ifdef VDB_TOOL_USE_MPEG
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "movie");
  try {
    const int fps = mParser.get<int>("fps");
    const std::string input = mParser.get<std::string>("input");
    const std::string output = mParser.get<std::string>("output");
    const VecI scale = mParser.getVec<int>("scale", "x");
    //const bool keep = mParser.get<bool>("keep");
    const std::string flip = mParser.get<std::string>("flip");
    if (mParser.verbose) mTimer.start(action_name);

    std::string cmd("ffmpeg"), log("log.txt");
    cmd += " -loglevel error";// only log error messages
    if (contains(input, '*')) cmd += " -pattern_type glob";// support expanding shell-like wildcard patterns (globbing)
    cmd += " -i \'" + input + "\'";// specify multiple input files as "input_*.ppm"
    cmd += " -vf \"fps=" + std::to_string(fps);
    if (findMatch(flip,{"vertical",  "180"})) cmd += ",vflip";// flip vertical (up/down)
    if (findMatch(flip,{"horizontal","180"})) cmd += ",hflip";// flip horizontal (left/right)
    if (flip!="" && !contains(cmd,"flip")) throw std::invalid_argument("Tool::movie: invalid argument flip=\""+flip+"\", expected \"vertical\", \"horizontal\" or \"180\"");
    if (scale.size()==2) {
      cmd += ",scale=" + std::to_string(scale[0]) + ":" + std::to_string(scale[1]) + ":flags=lanczos";
    } else if (scale.size()==1) {
      cmd += ",scale=" + std::to_string(scale[0]) + ":-1:flags=lanczos";
    }
    cmd += "\"";// end "-vf \"fps=..."
    if (getExt(output) == "gif") cmd += " -c:v gif";// create animated gif
    cmd += " -y " + output;// overwrite output files without asking
    cmd += " > " + log + " 2>&1";// redirect stdout and stderr to log file
    if (mParser.verbose>1) std::clog << cmd << std::endl;
    auto mySystem = [&](const std::string &cmd){
      if (int code = std::system(cmd.c_str())) {
        std::stringstream ss;
        ss << code << "\n\"" << cmd << "\"\n" << std::ifstream(log).rdbuf();
        throw std::runtime_error(ss.str());
      }
    };
    mySystem(cmd);
    mySystem("rm " + log);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
#else
  throw std::runtime_error("MPEG support was disabled during compilation!");
#endif
}// Tool::movie

// ==============================================================================================================
// LeVeque, R., High-Resolution Conservative Algorithms For Advection In Incompressible Flow, SIAM J. Numer. Anal. 33, 627–665 (1996)
// https://faculty.washington.edu/rjl/pubs/hiresadv/0733033.pdf
void Tool::enright()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "enright");
  try {
    mParser.printAction();
    const Vec3d translate = mParser.getVec3<double>("translate");
    const float scale = mParser.get<float>("scale");
    const float dt = mParser.get<float>("dt");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    math::ScaleTranslateMap::Ptr map(new math::ScaleTranslateMap(Vec3d(scale) ,translate));
    const math::Transform xform(map);
    struct LeVequeField {
      using VectorType = Vec3f;
      const math::Transform xform;
      LeVequeField(const math::Transform &_xform) : xform(_xform) {}
      const math::Transform& transform() const { return xform; }
      Vec3f operator() (const Vec3d& xyz, float time) const {
        static const float pi = math::pi<float>(), phase = pi / 3.0f;
        const Vec3d p = xform.worldToIndex(xyz);
        const float Px =  pi * float(p[0]), Py = pi * float(p[1]), Pz = pi * float(p[2]);
        const float tr =  math::Cos(time * phase);
        const float a  =  math::Sin(2.0f*Py);
        const float b  = -math::Sin(2.0f*Px);
        const float c  =  math::Sin(2.0f*Pz);
        return tr * Vec3f(2.0f * math::Pow2(math::Sin(Px)) * a * c,
                                      b * math::Pow2(math::Sin(Py)) * c,
                                  b * a * math::Pow2(math::Sin(Pz)) );
      }
      Vec3f operator() (const Coord& ijk, float time) const {return (*this)(ijk.asVec3d(), time);}
    } field(xform);
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (keep) {
      auto tmp = grid->deepCopy();
      mGrid.push_back(tmp);
      grid = tmp;
    }
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
      throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
    }
    if (mParser.verbose) mTimer.start("Enright SDF");
    tools::LevelSetAdvection<GridT, LeVequeField> advect(*grid, field);
    advect.setSpatialScheme(math::HJWENO5_BIAS);
    advect.setTemporalScheme(math::TVD_RK2);
    advect.setTrackerSpatialScheme(math::HJWENO5_BIAS);
    advect.setTrackerTemporalScheme(math::TVD_RK1);
    advect.advect(0.0f, dt);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}// Tool::enright

// ==============================================================================================================

template <typename GridType>
GridBase::Ptr Tool::clip(const VecF &v, int age, const GridType &input)
{
  using Vec3T = Vec3d;
  GridBase::Ptr output;
  switch (v.size()) {
  case 0: {// clip against a mask
    auto it = this->getGrid(age);
    if (auto mask = gridPtrCast<FloatGrid>(*it)) {
      output = tools::clip(input, *mask);
    } else if (auto mask = gridPtrCast<Vec3fGrid>(*it)) {
      output = tools::clip(input, *mask);
    } else if (auto tmp = gridPtrCast<points::PointDataGrid>(*it)) {
      output = tools::clip(input, *mask);
    } else {
      throw std::invalid_argument("clip: unsupported mask type with "+std::to_string(age));
    }
    break;
  } case 6: {// clip against a bbox
    if (age>=0) throw std::invalid_argument("clip: both mask and bbox were specified");
    BBoxd bbox(Vec3T(v[0],v[1],v[2]), Vec3T(v[3],v[4],v[5]));
    output = tools::clip(input, bbox);
    break;
  } case 8: {// clip against a frustrum
  if (age>=0) throw std::invalid_argument("clip: both mask and frustrum were specified");
    BBoxd bbox(Vec3T(v[0],v[1],v[2]), Vec3T(v[3],v[4],v[5]));
    math::NonlinearFrustumMap frustum(bbox,v[6],v[7]);
    output = tools::clip(input, frustum);
    break;
  } default:
    throw std::invalid_argument("clip: expected either a mask, bbox or frustum");
  }
  return output;
}// Tool::clip

// ==============================================================================================================

void Tool::clip()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "clip");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    VecF vec = mParser.getVec<float>("bbox");
    float tmp;
    if ((tmp = mParser.get<float>("taper")) > 0.0f) vec.push_back(tmp);
    if ((tmp = mParser.get<float>("depth")) > 0.0f) vec.push_back(tmp);
    const int mask = mParser.get<int>("mask");
    auto it = this->getGrid(age);
    GridBase::Ptr grid;
    if (mParser.verbose) mTimer.start("Clip VDB grid");
    if (auto floatGrid = gridPtrCast<FloatGrid>(*it)) {
      grid =this->clip(vec, mask, *floatGrid);
    } else if (auto vec3Grid = gridPtrCast<Vec3fGrid>(*it)) {
      grid = this->clip(vec, mask, *vec3Grid);
    } else {
      throw std::invalid_argument(action_name + ": unsupported grid type with " + std::to_string(age));
    }
    if (!(*it)->getName().empty()) grid->setName("clip_"+(*it)->getName());
    if (!keep) mGrid.erase(std::next(it).base());
    mGrid.push_back(grid);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    if (mErrorOnWarning) {
      throw std::invalid_argument(action_name+": "+e.what());
    } else {
      std::clog << action_name << ": skipping due to " << e.what() << std::endl;
    }
  }
}

// ==============================================================================================================

#ifdef VDB_TOOL_USE_PNG
void savePNG(const std::string& fname, const tools::Film& film)
{
  png_structp png = png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) OPENVDB_THROW(RuntimeError, "png_create_write_struct failed");
  png_infop info = info = png_create_info_struct(png);
  if (!info) OPENVDB_THROW(RuntimeError, "png_create_info_struct failed.")

  FILE* fp = std::fopen(fname.c_str(), "wb");
  if (!fp) {
    OPENVDB_THROW(IoError,"Unable to open '" + fname + "' for writing");
  }

  if (setjmp(png_jmpbuf(png))) {
    OPENVDB_THROW(IoError, "Error during initialization of PNG I/O.");
  }
  png_init_io(png, fp);

  if (setjmp(png_jmpbuf(png))) {
    OPENVDB_THROW(IoError, "Error writing PNG file header.");
  }
  // Output is 8bit depth, RGB format.
  png_set_IHDR(png, info,
            int(film.width()), int(film.height()),
            8,
            PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);

  const size_t channels = 3; // 3 = RGB, 4 = RGBA
  auto buffer = film.convertToBitBuffer<uint8_t>(/*alpha=*/false);
  uint8_t* tmp = buffer.get();

  std::unique_ptr<png_bytep[]> row_pointers(new png_bytep[film.height()]);

  for (size_t row = 0; row < film.height(); row++) {
    row_pointers[row] = tmp + (row * film.width() * channels);
  }

  if (setjmp(png_jmpbuf(png))) {
    OPENVDB_THROW(IoError, "Error writing PNG data buffers.");
  }
  /* write out the entire image data in one call */
  png_write_image(png, row_pointers.get());
  png_write_end(png, nullptr);

  std::fclose(fp);
  png_destroy_write_struct(&png, &info);
}// savePNG
#else
void savePNG(const std::string&, const tools::Film&)
{
  OPENVDB_THROW(RuntimeError, "vdb_tool has not been compiled with .png support.");
}// savePNG
#endif

// ==============================================================================================================

#ifdef VDB_TOOL_USE_JPG
void saveJPG(const std::string& fname, const tools::Film& film)
{
  jpeg_error_mgr jerr;
  jpeg_compress_struct cinfo;
  jpeg_create_compress(&cinfo);
  cinfo.err = jpeg_std_error(&jerr);
  FILE* fp = std::fopen(fname.c_str(), "wb");
  if (!fp) OPENVDB_THROW(IoError,"Unable to open '" + fname + "' for writing");
  jpeg_stdio_dest(&cinfo, fp);
  cinfo.image_width      = film.width();
  cinfo.image_height     = film.height();
  cinfo.input_components = 3;
  cinfo.in_color_space   = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);
  auto buf = film.convertToBitBuffer<uint8_t>(/*alpha=*/false);
  uint8_t *row = buf.get();
  const uint32_t stride = film.width() * 3;
  for (int y = 0; y < film.height(); ++y) {
      jpeg_write_scanlines(&cinfo, &row, 1);
      row += stride;
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  std::fclose(fp);
}// saveJPG
#else
void saveJPG(const std::string&, const tools::Film&)
{
  OPENVDB_THROW(RuntimeError, "vdb_tool has not been compiled with .jpg support.");
}// saveJPG
#endif

// ==============================================================================================================

#ifdef VDB_TOOL_USE_EXR
void saveEXR(const std::string& filename, const tools::Film& film, const std::string &compression = "zip")
{
    Imf::setGlobalThreadCount(8);
    Imf::Header header(int(film.width()), int(film.height()));
    if (compression == "none") {
        header.compression() = Imf::NO_COMPRESSION;
    } else if (compression == "rle") {
        header.compression() = Imf::RLE_COMPRESSION;
    } else if (compression == "zip") {
        header.compression() = Imf::ZIP_COMPRESSION;
    } else {
        OPENVDB_THROW(ValueError,
            "expected none, rle or zip compression, got \"" << compression << "\"");
    }
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
    header.channels().insert("A", Imf::Channel(Imf::FLOAT));
    using RGBA = tools::Film::RGBA;
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
    imgFile.writePixels(int(film.height()));
}// saveEXR
#else
void saveEXR(const std::string&, const tools::Film&, const std::string& = "zip")
{
    OPENVDB_THROW(RuntimeError, "vdb_tool has not been compiled with .exr support.");
}// saveEXR
#endif

// ==============================================================================================================

void Tool::render()
{
  const std::string &action_name = mParser.getAction().names[0];
  OPENVDB_ASSERT(action_name == "render");
  const VecS fileNames = mParser.getVec<std::string>("files");
  const int age = mParser.get<int>("vdb");
  const bool keep = mParser.get<bool>("keep");
  const std::string camType = mParser.get<std::string>("camera");
  const float aperture = mParser.get<float>("aperture");
  const float focal = mParser.get<float>("focal");
  const float isovalue = mParser.get<float>("isovalue");
  const int samples = mParser.get<int>("samples");
  const VecI image = mParser.getVec<int>("image", "x");
  Vec3d translate = mParser.getVec3<double>("translate");
  const Vec3d rotate = mParser.getVec3<double>("rotate");
  Vec3d target = mParser.getVec3<double>("target");
  const Vec3d up = mParser.getVec3<double>("up");
  const bool lookat = mParser.get<bool>("lookat");
  const float znear = mParser.get<float>("near");
  const float zfar = mParser.get<float>("far");
  const std::string shader = mParser.get<std::string>("shader");
  VecF light = mParser.getVec<float>("light");
  const float frame = mParser.get<float>("frame");
  const  float cutoff = mParser.get<float>("cutoff");
  const float gain = mParser.get<float>("gain");
  const Vec3f absorb = mParser.getVec3<float>("absorb");
  const Vec3f scatter = mParser.getVec3<float>("scatter");
  const VecF step = mParser.getVec<float>("step");
  const int colorgrid = mParser.get<int>("colorgrid");

  if (light.size()==3) {
    for (size_t i=0; i<3; ++i) light.push_back(0.7f);
  } else if (light.size()!=6) {
    throw std::invalid_argument("render: \"light\" option expected 3 or 6 values, got "+std::to_string(light.size()));
  }
  if (image.size()!=2) throw std::invalid_argument("render: expected width and height,  e.g. image=1920x1080");
  auto it = this->getGrid(age);
  GridT::Ptr grid = gridPtrCast<GridT>(*it);
  if (!grid || grid->getGridClass() != GRID_LEVEL_SET) {
    throw std::invalid_argument(action_name + ": no level set with age " + std::to_string(age));
  }
  if (step.size()!=2) throw std::invalid_argument("render: \"step\" option expected 2 values, but got "+std::to_string(step.size()));

  tools::Film film(image[0], image[1]);
  const CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
  const math::BBox<Vec3d> bboxIndex(bbox.min().asVec3d(), bbox.max().asVec3d());
  const math::BBox<Vec3R> bboxWorld = bboxIndex.applyMap(*(grid->transform().baseMap()));

  if (lookat && target == Vec3d(0.0) && translate == Vec3d(0.0)) {
    target = bboxWorld.getCenter();
    translate = 3.0*bboxWorld.max();
  }
  Vec3SGrid::Ptr colorgridPtr;
  if (colorgrid>=0) {
    auto it2 = this->getGrid(colorgrid);
    colorgridPtr = gridPtrCast<Vec3SGrid>(*it2);
    if (!colorgridPtr) throw std::invalid_argument("render: no colorgrid of type Vec3f with age "+std::to_string(colorgrid));
  }
  std::unique_ptr<tools::BaseCamera> camera;
  if (startsWith(camType, "persp")) {
    camera.reset(new tools::PerspectiveCamera(film, rotate, translate, focal, aperture, znear, zfar));
  } else if (startsWith(camType, "ortho")) {
    camera.reset(new tools::OrthographicCamera(film, rotate, translate, frame, znear, zfar));
  } else {
    throw std::invalid_argument("render: expected perspective or orthographic camera, got \""+camType+"\"");
  }
  if (lookat) camera->lookAt(target, up);

  // Define the shader for level set rendering.  The default shader is a diffuse shader.
  std::unique_ptr<tools::BaseShader> shaderPtr;
  if (shader == "matte") {
    if (colorgridPtr) {
      shaderPtr.reset(new tools::MatteShader<Vec3SGrid>(*colorgridPtr));
    } else {
      shaderPtr.reset(new tools::MatteShader<>());
    }
  } else if (shader == "normal") {
    if (colorgridPtr) {
      shaderPtr.reset(new tools::NormalShader<Vec3SGrid>(*colorgridPtr));
    } else {
      shaderPtr.reset(new tools::NormalShader<>());
    }
  } else if (shader == "position") {
    if (colorgridPtr) {
      shaderPtr.reset(new tools::PositionShader<Vec3SGrid>(bboxWorld, *colorgridPtr));
    } else {
      shaderPtr.reset(new tools::PositionShader<>(bboxWorld));
    }
  } else if (shader == "diffuse") { // default
    if (colorgridPtr) {
      shaderPtr.reset(new tools::DiffuseShader<Vec3SGrid>(*colorgridPtr));
    } else {
      shaderPtr.reset(new tools::DiffuseShader<>());
    }
  } else {
    throw std::invalid_argument("render: unsupported value of shader=\""+shader+"\"");
  }

  if (grid->getGridClass() == GRID_LEVEL_SET) {
    if (mParser.verbose) mTimer.start("ray-tracing");
    tools::LevelSetRayIntersector<GridT> intersector(*grid, static_cast<GridT::ValueType>(isovalue));
    tools::rayTrace(*grid, intersector, *shaderPtr, *camera, samples, 0, true);
  } else {// volume rendering
    if (mParser.verbose) mTimer.start("volumerendering");
    using IntersectorT = tools::VolumeRayIntersector<GridT>;
    IntersectorT intersector(*grid);
    tools::VolumeRender<IntersectorT> renderer(intersector, *camera);
    renderer.setLightDir(  light[0], light[1], light[2]);
    renderer.setLightColor(light[3], light[4], light[5]);
    renderer.setPrimaryStep(step[0]);
    renderer.setShadowStep(step[1]);
    renderer.setScattering(scatter[0], scatter[1], scatter[2]);
    renderer.setAbsorption(absorb[0], absorb[1], absorb[2]);
    renderer.setLightGain(gain);
    renderer.setCutOff(cutoff);
    renderer.render(true);
  }

#ifdef VDB_TOOL_USE_JPG
  std::string fileName("test.jpg");
#elif VDB_TOOL_USE_PNG
  std::string fileName("test.png");
#else
  std::string fileName("test.ppm");
#endif

  if (fileNames.empty()) {
    if (!grid->getName().empty()) fileName = grid->getName() + "." + getExt(fileName);
  } else {
    fileName = fileNames[0];
  }

  switch (findFileExt(fileName, {"ppm", "png", "jpg", "exr"})) {
  case 1:
    film.savePPM(fileName);
    break;
  case 2:
    savePNG(fileName, film);
    break;
  case 3:
    saveJPG(fileName, film);
    break;
  case 4:
    saveEXR(fileName, film);
    break;
  default:
    throw std::invalid_argument("Image file \""+fileName+"\" has an unrecognized extension");
    break;
  }

  if (!keep) mGrid.erase(std::next(it).base());
  if (mParser.verbose) mTimer.stop();
}

// ==============================================================================================================

void Tool::print_args(std::ostream& os) const
{
  os << "\n" << std::setw(40) << std::setfill('=') << "> Actions <" << std::setw(40) << "\n";
  mParser.print(os);
  os << std::setw(80) << std::setfill('=') << "\n" << std::endl;
}// Tool::print_args

// ==============================================================================================================

// ----- Helpers for the pretty-printed -print output ------------------------
namespace print_detail {

enum Align { LEFT, RIGHT };

// Print a centered title between U+2550 (BOX DRAWINGS DOUBLE HORIZONTAL) chars
// to a fixed visible width (default 80 columns).
inline void printBanner(std::ostream& os, const std::string& title, int width = 80)
{
    static const std::string kBar = "═";// U+2550, 3 bytes in UTF-8
    const int titleLen = static_cast<int>(title.size());
    const int left  = std::max(0, (width - titleLen) / 2);
    const int right = std::max(0, width - titleLen - left);
    for (int i = 0; i < left;  ++i) os << kBar;
    os << title;
    for (int i = 0; i < right; ++i) os << kBar;
    os << "\n";
}

// 1234567 -> "1,234,567" — thousands separators for big counts.
inline std::string formatCommas(uint64_t n)
{
    std::string s = std::to_string(n);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0) { s.insert(pos, ","); pos -= 3; }
    return s;
}

// Wrap util::printBytes to produce a trim string ("15.326 MB").
inline std::string formatBytes(uint64_t bytes)
{
    std::stringstream ss;
    util::printBytes(ss, bytes, /*head=*/"", /*tail=*/"", /*exact=*/false,
                     /*width=*/0, /*precision=*/3);
    return ss.str();
}

// Compact "[a,b,c]->[d,e,f]" form for the bbox column. ASCII arrow so column
// widths computed from byte sizes line up correctly.
inline std::string formatBBox(const openvdb::CoordBBox& bbox)
{
    std::stringstream ss;
    const auto& mn = bbox.min();
    const auto& mx = bbox.max();
    ss << "[" << mn.x() << "," << mn.y() << "," << mn.z() << "]->["
       << mx.x() << "," << mx.y() << "," << mx.z() << "]";
    return ss.str();
}

// Float-valued bbox (world-space, used by Geometry). Trimmed to 3 decimals so
// the column doesn't bloat to seven-significant-digit floats.
template <typename BBoxT>
inline std::string formatBBoxd(const BBoxT& bbox)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    const auto& mn = bbox.min();
    const auto& mx = bbox.max();
    ss << "[" << mn[0] << "," << mn[1] << "," << mn[2] << "]->["
       << mx[0] << "," << mx[1] << "," << mx[2] << "]";
    return ss.str();
}

template <typename GridT>
inline std::string gridRangeStr(const GridT& grid)
{
    if (grid.activeVoxelCount() == 0) return "(empty)";
    const auto mm = tools::minMax(grid.tree());
    std::stringstream ss;
    ss << "[" << mm.min() << ", " << mm.max() << "]";
    return ss.str();
}

// Dispatch min/max stringification across the scalar grid types we know how
// to compare. Vec3-typed grids etc. get "(n/a)" because ordering isn't
// well-defined for them.
inline std::string formatRange(const openvdb::GridBase& grid)
{
    if (auto p = dynamic_cast<const openvdb::FloatGrid*>(&grid))  return gridRangeStr(*p);
    if (auto p = dynamic_cast<const openvdb::DoubleGrid*>(&grid)) return gridRangeStr(*p);
    if (auto p = dynamic_cast<const openvdb::Int32Grid*>(&grid))  return gridRangeStr(*p);
    if (auto p = dynamic_cast<const openvdb::Int64Grid*>(&grid))  return gridRangeStr(*p);
    if (auto p = dynamic_cast<const openvdb::BoolGrid*>(&grid))   return gridRangeStr(*p);
    return "(n/a)";
}

// Background value, dispatched by grid type. Falls back to the typed grid's
// own background() accessor, which prints scalars as numbers and Vec3s as
// "[x, y, z]".
template <typename GridT>
inline std::string gridBgStr(const GridT& grid)
{
    std::stringstream ss;
    ss << grid.background();
    return ss.str();
}
inline std::string formatBackground(const openvdb::GridBase& grid)
{
    if (auto p = dynamic_cast<const openvdb::FloatGrid*>(&grid))  return gridBgStr(*p);
    if (auto p = dynamic_cast<const openvdb::DoubleGrid*>(&grid)) return gridBgStr(*p);
    if (auto p = dynamic_cast<const openvdb::Int32Grid*>(&grid))  return gridBgStr(*p);
    if (auto p = dynamic_cast<const openvdb::Int64Grid*>(&grid))  return gridBgStr(*p);
    if (auto p = dynamic_cast<const openvdb::BoolGrid*>(&grid))   return gridBgStr(*p);
    if (auto p = dynamic_cast<const openvdb::Vec3SGrid*>(&grid))  return gridBgStr(*p);
    return "(n/a)";
}

// Print a table with column-width auto-sizing. Cells must be pure ASCII
// (column widths use byte size = visible width). The header separator uses
// U+2500 BOX DRAWINGS LIGHT HORIZONTAL and is emitted directly so the
// multibyte glyphs don't interact with std::setw byte-counting.
inline void printTable(std::ostream& os,
                       const std::vector<std::string>& headers,
                       const std::vector<std::vector<std::string>>& rows,
                       const std::vector<Align>& aligns,
                       const std::string& indent = "  ")
{
    const std::size_t nc = headers.size();
    std::vector<std::size_t> w(nc, 0);
    for (std::size_t c = 0; c < nc; ++c) w[c] = headers[c].size();
    for (const auto& r : rows) {
        for (std::size_t c = 0; c < nc && c < r.size(); ++c) {
            w[c] = std::max(w[c], r[c].size());
        }
    }
    auto emit = [&](const std::vector<std::string>& r) {
        os << indent;
        for (std::size_t c = 0; c < nc; ++c) {
            const std::string& cell = c < r.size() ? r[c] : std::string();
            os << (aligns[c] == RIGHT ? std::right : std::left)
               << std::setw(static_cast<int>(w[c])) << cell;
            if (c + 1 < nc) os << "  ";
        }
        os << "\n";
    };
    emit(headers);
    // Separator row — repeat U+2500 to match each column's width.
    os << indent;
    for (std::size_t c = 0; c < nc; ++c) {
        for (std::size_t i = 0; i < w[c]; ++i) os << "─";
        if (c + 1 < nc) os << "  ";
    }
    os << "\n";
    for (const auto& r : rows) emit(r);
}

}// namespace print_detail
// ----- end helpers ---------------------------------------------------------

void Tool::print(std::ostream& os) const
{
  using namespace print_detail;
  OPENVDB_ASSERT(mParser.getAction().names[0] == "print");
  const int level = mParser.get<int>("level");

  if (mParser.verbose>1) {
    os << "\n";
    printBanner(os, " Actions ");
    mParser.print(os);
    printBanner(os, "");
    os << "\n";
    printBanner(os, " Variables ");
    mParser.processor.memory().print(os);
    printBanner(os, "");
    os << std::endl;
  }

  if (mParser.verbose>0) {
    os << "\n";
    printBanner(os, " Primitives ");

    // Geometry — same column-aligned table style as the VDB grid table below.
    std::vector<std::string> geomHeaders = {"age", "name", "vtx", "tri", "quad", "size"};
    std::vector<Align>       geomAligns  = {RIGHT, LEFT,   RIGHT, RIGHT, RIGHT,  RIGHT};
    if (level >= 1) { geomHeaders.push_back("bbox"); geomAligns.push_back(LEFT); }
    if (level >= 2) { geomHeaders.push_back("rgb");  geomAligns.push_back(RIGHT); }

    auto buildGeomRow = [&](int age, const Geometry& geom) {
        const std::uint64_t mem = sizeof(geom)
            + geom.vtx().size()  * sizeof(openvdb::Vec3s)
            + geom.tri().size()  * sizeof(openvdb::Vec3I)
            + geom.quad().size() * sizeof(openvdb::Vec4I)
            + geom.rgb().size()  * sizeof(openvdb::Vec3s);
        std::vector<std::string> row = {
            std::to_string(age),
            geom.getName(),
            formatCommas(geom.vtx().size()),
            formatCommas(geom.tri().size()),
            formatCommas(geom.quad().size()),
            formatBytes(mem),
        };
        if (level >= 1) row.push_back(formatBBoxd(geom.bbox()));
        if (level >= 2) row.push_back(formatCommas(geom.rgb().size()));
        return row;
    };

    std::vector<std::vector<std::string>> geomRows;
    if (mParser.getStr("geo")=="*") {
      for (auto begin = mGeom.crbegin(), it = begin, end = mGeom.crend(); it != end; ++it) {
        geomRows.push_back(buildGeomRow(static_cast<int>(std::distance(begin, it)), **it));
      }
    } else {
      for (int age : mParser.getVec<int>("geo")) {
        geomRows.push_back(buildGeomRow(age, **this->getGeom(age)));
      }
    }
    if (geomRows.empty()) {
      os << "Geometry: none\n";
    } else {
      os << "Geometry:\n";
      printTable(os, geomHeaders, geomRows, geomAligns);
    }

    // VDB grids — column-aligned table.
    std::vector<std::string> headers = {"age", "name", "type", "class", "dim", "voxels", "dx", "size"};
    std::vector<Align>       aligns  = {RIGHT, LEFT,   LEFT,   LEFT,    LEFT,  RIGHT,    RIGHT, RIGHT};
    if (level >= 1) { headers.push_back("bbox");       aligns.push_back(LEFT); }
    if (level >= 2) { headers.push_back("background"); aligns.push_back(LEFT); }
    if (level >= 2) { headers.push_back("range");      aligns.push_back(LEFT); }

    auto buildRow = [&](int age, const GridBase& grid) {
        const auto bbox = grid.evalActiveVoxelBoundingBox();
        const auto dim  = bbox.dim();
        std::stringstream dimSS;
        dimSS << "[" << dim.x() << ", " << dim.y() << ", " << dim.z() << "]";
        std::stringstream dxSS;
        dxSS << grid.voxelSize()[0];
        std::vector<std::string> row = {
            std::to_string(age),
            grid.getName(),
            grid.valueType(),
            GridBase::gridClassToString(grid.getGridClass()),
            dimSS.str(),
            formatCommas(grid.activeVoxelCount()),
            dxSS.str(),
            formatBytes(grid.memUsage()),
        };
        if (level >= 1) row.push_back(formatBBox(bbox));
        if (level >= 2) row.push_back(formatBackground(grid));
        if (level >= 2) row.push_back(formatRange(grid));
        return row;
    };

    std::vector<std::vector<std::string>> rows;
    if (mParser.getStr("vdb")=="*") {
      for (auto begin = mGrid.crbegin(), it = begin, end = mGrid.crend(); it != end; ++it) {
        rows.push_back(buildRow(static_cast<int>(std::distance(begin, it)), **it));
      }
    } else {
      for (int age : mParser.getVec<int>("vdb")) {
        rows.push_back(buildRow(age, **this->getGrid(age)));
      }
    }
    if (rows.empty()) {
      os << "VDB grids: none\n";
    } else {
      os << "VDB grids:\n";
      printTable(os, headers, rows, aligns);
    }

    if (mParser.get<bool>("mem")) {
      os << "\n";
      printBanner(os, " Variables ");
      mParser.processor.memory().print(os);
    }

    printBanner(os, "");
    os << std::endl;
  }
}// Tool::print

// ==============================================================================================================

Geometry::Ptr Tool::mesherToGeometry(tools::VolumeToMesh &mesher) const
{
  Geometry::Ptr geom(new Geometry());

  {// allocate and copy vertices
    auto &vtx = geom->vtx();
    vtx.resize(mesher.pointListSize());
    tools::volume_to_mesh_internal::PointListCopy ptnCpy(mesher.pointList(), vtx);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vtx.size()), ptnCpy);
    mesher.pointList().reset(nullptr);
  }

  {// allocate and copy polygons
    auto& polygonPoolList = mesher.polygonPoolList();
    size_t numQuad = 0, numTri = 0;
    for (size_t i = 0, N = mesher.polygonPoolListSize(); i < N; ++i) {
      auto &polygons = polygonPoolList[i];
      numTri  += polygons.numTriangles();
      numQuad += polygons.numQuads();
    }
    auto &tri  = geom->tri();
    auto &quad = geom->quad();
    tri.resize(numTri);
    quad.resize(numQuad);
    size_t qIdx = 0, tIdx = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
      auto &poly = polygonPoolList[n];
      for (size_t i = 0, I = poly.numQuads(); i < I; ++i) quad[qIdx++] = poly.quad(i);
      for (size_t i = 0, I = poly.numTriangles(); i < I; ++i) tri[tIdx++] = poly.triangle(i);
    }
  }
  return geom;
}// Tool::mesherToGeometry

// ==============================================================================================================

Geometry::Ptr Tool::volumeToGeometry(const GridT &grid, float isoValue, float adaptivity) const
{
  tools::VolumeToMesh mesher(isoValue, adaptivity, /*relaxDisorientedTriangles*/true);
  mesher(grid);
  return this->mesherToGeometry(mesher);
}

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_TOOL_HAS_BEEN_INCLUDED
