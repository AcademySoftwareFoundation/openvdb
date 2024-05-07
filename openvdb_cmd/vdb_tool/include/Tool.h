// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Tool.h
///
/// @brief This tool can combine any sequence of the of high-level tools available in openvdb.
///        For instance, it can convert a sequence of polygon meshes and particles to level sets,
///        and perform a large number of operations on these level set surfaces. It can also
///        generate adaptive polygon meshes from level sets, render images and write particles,
///        meshes or VDBs them to disk.
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
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
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
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

#ifdef VDB_TOOL_USE_NANO
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/NanoToOpenVDB.h>
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

#ifdef VDB_TOOL_USE_JPG
#include <jpeglib.h>
#endif

#include "Parser.h"
#include "Geometry.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

class Tool
{
public:

    /// @brief Constructor from command-line arguments
    Tool(int argc, char *argv[]);

    Tool(const Tool&) = delete;// disallow copy construction
    Tool(Tool&&) = delete;// disallow move construction
    Tool& operator=(const Tool&) = delete;// disallow assignment
    Tool& operator=(Tool&&) = delete;// disallow move assignment

    /// @brief Executes all the actions defined during construction
    void run();

    /// @brief prints information to the terminal about the current stack of VDB grids and Geometry
    void print(std::ostream& os = std::cerr) const;
    void print_args(std::ostream& os = std::cerr) const;

    /// @brief return a string with the current version number of this tool
    static std::string version() {return std::to_string(sMajor)+"."+std::to_string(sMinor)+"."+std::to_string(sPatch);}
    static int major() {return sMajor;}
    static int minor() {return sMinor;}
    static int patch() {return sPatch;}

private:

    static const int sMajor =10;// incremented for incompatible changes options or file.
    static const int sMinor = 6;// incremented for new functionality that is backwards-compatible.
    static const int sPatch = 1;// incremented for backwards-compatible bug fixes.

    using GridT = FloatGrid;
    using FilterT = std::unique_ptr<tools::LevelSetFilter<GridT>>;
    struct Points;// defined below
    struct Header;// defined below
 
    mutable util::CpuTimer   mTimer;
    std::string              mCmdName;// name of this command-line tool
    std::list<Geometry::Ptr> mGeom;// list of geometries owned by this tool
    std::list<GridBase::Ptr> mGrid;// list of based grids owned by this tool
    Parser                   mParser;

    /// @brief Deletes geometry, VDB grids and local variables
    void clear();

    /// @brief Clip a VDB grid against another grid, a bbox or frustum
    template <typename GridType>
    GridBase::Ptr clip(const VecF &v, int age, const GridType &input);
    void clip();

    /// @brief composit two grids, e.g. min, max, sum
    void composite();

    /// @brief generate a grid of selected properties from another grid
    void compute();

    /// @brief Import and process one or more configuration files
    void config();

    /// @brief "perform CSG operations between of two level sets surfaces
    void csg();

    /// @brief Performs Enright advection benchmark test on a level set
    void enright();

    /// @brief expand narrow band of level set
    void expandLevelSet();

    /// @brief perform filtering (convolution) of a level set surface
    void filterLevelSet();

    /// @brief signed-flood filling of a level set VDB
    void floodLevelSet();

    /// @brief Print documentation for one, multiple or all available actions
    void help();

    /// @brief Converts an iso-surface of a scalar field into a level set (i.e. SDF)
    void isoToLevelSet();

    /// @brief Convert a level set to an adaptive polygon mesh
    void levelSetToMesh();

    /// @brief Create a level set sphere, i.e. a narrow-band signed distance to a sphere
    void levelSetSphere();

    /// @brief Create a level set shape with the specified number of polygon faces
    void levelSetPlatonic();

    /// @brief Converts a level set VDB into a VDB with a fog volume, i.e. normalized density
    void levelSetToFog();

    /// @brief Converts a polygon mesh into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh
    void meshToLevelSet();

    /// @brief construct a LoD sequences of VDB trees with powers of two refinements
    void multires();

    /// @brief perform morphology operations on a level set surface
    void offsetLevelSet();

    /// @brief Converts geometry points into a narrow-band level set
    void particlesToLevelSet();

    /// @brief Encode geometry points into a VDB grid
    void pointsToVdb();

    /// @brief prune away inactive values in a VDB grid
    void pruneLevelSet();

    /// @brief Read one or more geometry or VDB files from disk or STDIN
    void read();
    void readGeo(  const std::string &fileName);
    void readVDB(  const std::string &fileName);
    void readNVDB( const std::string &fileName);

    /// @brief ray-tracing of level set surfaces and volume rendering of fog volumes
    void render();

    /// @brief resample one VDB grid into another VDB grid or a transformation of the input grid
    void resample();

    /// @brief "segment an input VDB into a list if topologically disconnected VDB grids
    void segment();

    /// @brief Scatters point into the active values of an input VDB grid
    void scatter();

    /// @brief apply affine transformations (uniform scale -> rotation -> translation) to a VDB grids and geometry
    void transform();

    /// @brief Extracts points encoded in a VDB to points in a geometry format
    void vdbToPoints();

    /// @brief Write list of geometry, VDB or config files to disk or STDOUT
    void write();
    void writeGeo( const std::string &fileName);
    void writeVDB( const std::string &fileName);
    void writeNVDB(const std::string &fileName);
    void writeConf(const std::string &fileName);

    /// @brief return the voxel-size of  a LS estimated from a desired grid dimension of a specific geometry
    float estimateVoxelSize(int maxDimension,  float halfWidth, int geo_age);

    FilterT createFilter(GridT &grid,  int space, int time);

    /// @brief print examples to the terminal and terminate
    std::string examples() const;

    void warning(const std::string &msg, std::ostream& os = std::cerr) const;

    /// @brief Initialize this parser, i.e. register available actions
    void init();

    inline auto getGrid(size_t age) const;
    inline auto getGeom(size_t age) const;

};// Tool class

// ==============================================================================================================

Tool::Tool(int argc, char *argv[])
    : mTimer(std::cerr)
    , mCmdName(getBase(argv[0]))// name of executable
    , mParser({{"dim", "256", "256", "default grid resolution along the longest axis"},
               {"voxel", "0.0", "0.01", "default voxel size in world units. A value of zero indicates that dim is used to derive the voxel size."},
               {"width", "3.0", "3.0", "default narrow-band width of level sets in voxel units"},
               {"time", "1", "1|2|3", "default temporal discretization order"},
               {"space", "5", "1|2|3|5", "default spatial discretization order"},
               {"keep", "false", "1|0|true|false", "by default delete the input"}})
{
    openvdb::initialize();
    this->init();// fast: less than 1 ms
    mParser.finalize();
    mParser.parse(argc, argv);// extremely fast
}// Tool::Tool

// ==============================================================================================================

auto Tool::getGrid(size_t age) const
{
    if (age>=mGrid.size()) throw std::invalid_argument("-"+mParser.getAction().name+" called getGrid("+std::to_string(age)+"), but grid count = "+std::to_string(mGrid.size()));
    auto it = mGrid.crbegin();
    std::advance(it, age);
    return it;
}// Tool::getGrid

// ==============================================================================================================

auto Tool::getGeom(size_t age) const
{
    if (age>=mGeom.size()) throw std::invalid_argument("-"+mParser.getAction().name+" called getGeom("+std::to_string(age)+"), but geometry count = "+std::to_string(mGeom.size()));
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
    if (mParser.verbose>1) this->print();
}// Tool::run

// ==============================================================================================================

void Tool::warning(const std::string &msg, std::ostream& os) const
{
    if (mParser.verbose>0) {
        os << "\n" << std::setw(static_cast<int>(msg.size())) << std::setfill('*') << "\n" << msg
           << "\n" << std::setw(static_cast<int>(msg.size())) << std::setfill('*') << "\n";
    }
}// Tool::warning

// ==============================================================================================================

/// @brief Private struct for the header of config files
struct Tool::Header {
    Header() : mMagic("vdb_tool"), mMajor(sMajor), mMinor(sMinor), mPatch(sPatch) {}
    Header(const std::string &line) : mMagic("vdb_tool") {
      const VecS header = tokenize(line, " .");
      if (header.size()!=4 || header[0]!=mMagic ||
         !isInt(header[1], mMajor) ||
         !isInt(header[2], mMinor) ||
         !isInt(header[3], mPatch)) throw std::invalid_argument("Header: incompatible: \""+line+"\"");
    }
    std::string str() const {
      return mMagic+" "+std::to_string(mMajor)+"."+std::to_string(mMinor)+"."+std::to_string(mPatch);
    }
    bool isCompatible() const {return mMajor == sMajor;}

    std::string mMagic;
    int mMajor, mMinor, mPatch;
};// Header struct

// ==============================================================================================================

/// @brief Private wrapper struct for points required by particlesToSdf
struct Tool::Points {
    using PosType = Vec3R;

    Points(const std::vector<Vec3s> &vtx) : mPoints(vtx) {}
    size_t size() const { return mPoints.size(); }
    void getPos(size_t n, PosType &p) const { p = mPoints[n]; }

    const std::vector<Vec3s> &mPoints;
};// Points struct

// ==============================================================================================================

void Tool::init()
{
  // note, the following actions were added when mParser was constructed: -quiet,-verbose,-debug,-default,-for,-each,-end

  mParser.addAction(
      "config", "c", "Import and process one or more configuration files",
    {{"files", "",  "config1.txt,config2.txt...", "list of configuration files to load and execute"},
     {"execute", "true", "1|0|true|false", "toggle wether to execute the actions in the config file"},
     {"update", "false", "1|0|true|false", "toggle wether to update the version number of the config file"}},
     [&](){this->config();}, [](){}, 0); // anonymous options are appended to "files"

  mParser.addAction(
      "help", "h", "Print documentation for one, multiple or all available actions",
    {{"actions", "", "read,write,...", "list of actions to document. If the list is empty documentation is printed for all available actions and if other actions proceed this action, documentation is printed for those actions only"},
     {"exit", "true", "1|0|true|false", "toggle wether to terminate after this action or not"},
     {"brief", "false", "1|0|true|false", "toggle brief or detailed documentation"}},
     [](){}, [&](){this->help();}, 0); // anonymous options are appended to "actions"

  mParser.addAction(
      "read", "i", "Read one or more geometry or VDB files from disk or STDIN.",
    {{"files", "", "{file|stdin}.{abc|obj|ply|stl|vdb}", "list of files or the input stream, e.g. file.vdb,stdin.vdb. Note that \"files=\" is optional since any argument without \"=\" is intrepreted as a file and appended to \"files\""},
     {"grids", "*", "*|grid_name,...", "list of VDB grids name to be imported (defaults to \"*\", i.e. import all available grids)"},
     {"delayed", "true", "1|0|true|false", "toggle delayed loading of VDB grids (enabled by default). This option is ignored by other file types"}},
     [](){}, [&](){this->read();}, 0);//  anonymous options are treated as to the first option,i.e. "files"

  mParser.addAction(
      "write", "o", "Write list of geometry, VDB or config files to disk or STDOUT",
    {{"files", "", "{file|stdout}.{obj|ply|stl|vdb|nvdb}", "list of files or the output stream, e.g. file.vdb or stdin.vdb. Note that \"files=\" is optional since any argument without the \"=\" character is intrepreted as a file and appended to \"files\"."},
     {"geo", "0", "0|1...", "geometry to write (defaults to \"0\" which is the latest)."},
     {"vdb", "*", "0,1,...", "list of VDB grids to write (defaults to \"*\", i.e. all available grids)."},
     {"keep", "", "1|0|true|false", "toggle wether to preserved or deleted geometry and grids after they have been written."},
     {"codec", "", "none|zip|blosc|active", "compression codec for the file or stream"},
     {"bits", "32", "32|16|8|4|N", "bit-width of floating point numbers during quantization of VDB and NanoVDB grids, i.e. 32 is full, 16, is half (defaults to 32). NanoVDB also supports 8, 4 and N which is adaptive bit-width"},// VDB: 32, 16 + for NVDB 8, 4 or N
     {"dither", "false", "1|0|true|false", "toggle dithering of quantized NanoVDB grids (disabled by default)"},
     {"absolute", "true", "1|0|true|false", "toggle absolute or relative error tolerance during quantization of NanoVDBs. Only used if bits=N. Defaults to absolute"},// absolute or relative error for N bits in NVDB
     {"tolerance", "-1", "1.0", "absolute or relative error tolerance used during quantization of NanoVDBs. Only used if bits=N."},// error tolerance for N bits in NVDB
     {"stats", "", "none|bbox|extrema|all", "specify the statistics to compute for NanoVDBs."},
     {"checksum", "", "none|partial|full", "specify the type of checksum to compute for NanoVDBs"}},
     [&](){mParser.setDefaults();}, [&](){this->write();}, 0);// anonymous options are treated as to the first option,i.e. "files"

  mParser.addAction(
     "clear", "", "Deletes geometry, VDB grids and local variables",
    {{"geo", "*", "*|0,1,...", "list of geometries to delete (defaults to all)"},
     {"vdb", "*", "*|0,1,...", "list of VDB grids to delete (defaults to all)"},
     {"variables", "0", "1|0|true|false", "clear all the local variables (defaults to off)"}},
     [](){}, [&](){this->clear();});

  mParser.addAction(
      "sphere", "", "Create a level set sphere, i.e. a narrow-band signed distance to a sphere",
    {{"dim", "", "256", "largest dimension in voxel units of the sphere (defaults to 256). If \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.0", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\". Defaults to 0.0, i.e. this option is disabled"},
     {"radius", "1.0", "1.0", "radius of sphere in world units"},
     {"center", "(0,0,0)", "(0.0,0.0,0.0)", "center of sphere in world units"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"name", "sphere", "sphere", "name assigned to the level set sphere"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetSphere();});

  mParser.addAction(
      "mesh2ls", "m2ls", "Convert a polygon mesh into a narrow-band level set, i.e. a narrow-band signed distance to a polygon mesh",
    {{"dim", "", "256", "largest dimension in voxel units of the mesh bbox (defaults to 256). If \"vdb\" or \"voxel\" is defined then \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"vdb", "-1", "0", "age (i.e. stack index) of reference grid used to define the transform. Defaults to -1, i.e. disabled. If specified this option takes precedence over \"dim\" and \"voxel\"!"},
     {"keep", "", "1|0|true|false", "toggle wether the input geometry is preserved or deleted after the conversion"},
     {"name", "", "mesh2ls_input", "specify the name of the resulting vdb (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->meshToLevelSet();});

  mParser.addAction(
      "ls2mesh", "l2m", "Convert a level set to an adaptive polygon mesh",
    {{"adapt", "0.0", "0.9", "normalized metric for the adaptive meshing. 0 is uniform and 1 is fully adaptive mesh. Defaults to 0."},
     {"iso", "0.0", "0.1", "iso-value used to define the implicit surface. Defaults to zero."},
     {"vdb", "0", "0", "age (i.e. stack index) of the level set VDB grid to be meshed. Defaults to 0, i.e. most recently inserted VDB."},
     {"mask","-1", "1", "age (i.e. stack index) of the level set VDB grid used as a surface mask during meshing. Defaults to -1, i.e. it's disabled."},
     {"invert", "false", "1|0|true|false", "boolean toggle to mesh the complement of the mask. Defaults to false and ignored if no mask is specified."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing. The mask is never removed!"},
     {"name", "", "ls2mesh_input", "specify the name of the resulting vdb (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetToMesh();});

  mParser.addAction(
      "ls2fog", "l2f", "Convert a level set VDB into a VDB with a fog volume, i.e. normalized density.",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "ls2fog_input", "specify the name of the resulting VDB (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetToFog();});

  mParser.addAction(
      "points2ls", "p2l", "Convert geometry points into a narrow-band level set",
    {{"dim", "", "256", "largest dimension in voxel units of the bbox of all the points (defaults to 256). In \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"radius", "2.0", "2.0", "radius in voxel units of the input points"},
     {"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input points are preserved or deleted after the processing"},
     {"name", "", "points2ls_input", "specify the name of the resulting VDB (by default it's derived from the input points)"}},
     [&](){mParser.setDefaults();}, [&](){this->particlesToLevelSet();});

  mParser.addAction(
      "iso2ls", "i2l", "Convert an iso-surface of a scalar field into a level set (i.e. SDF)",
    {{"vdb", "0", "0,1", "age (i.e. stack index) of the VDB grid to be processed and an optional reference grid. Defaults to 0, i.e. most recently inserted VDB."},
     {"iso", "0.0", "0.0", "value of the iso-surface from which to compute the level set"},
     {"voxel", "", "0.0", "voxel size in world units (defaults to zero, i.e the transform out the output matches the input)"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "iso2ls_input", "specify the name of the resulting VDB (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->isoToLevelSet();});

  mParser.addAction(
      "points2vdb", "p2v", "Encode geometry points into a VDB grid",
    {{"geo", "0", "0", "age (i.e. stack index) of the geometry to be processed. Defaults to 0, i.e. most recently inserted geometry."},
     {"keep", "", "1|0|true|false", "toggle wether the input points are preserved or deleted after the processing"},
     {"ppv", "8", "8", "the number of points per voxel in the output VDB grid (defaults to 8)"},
     {"bits", "16", "16|8|32", "the number of bits used to represent a single point in the VDB grid (defaults to 16, i.e. half precision)"},
     {"name", "", "points_2vdb_input", "specify the name of the resulting VDB (by default it's derived from the input geometry)"}},
     [&](){mParser.setDefaults();}, [&](){this->pointsToVdb();});

  mParser.addAction(
      "vdb2points", "v2p", "Extract points encoded in a VDB to points in a geometry format",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "vdb2points_input", "specify the name of the resulting points (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->vdbToPoints();});

  mParser.addAction(
      "scatter", "", "Scatter point into the active values of an input VDB grid",
    {{"count", "0", "0", "fixed number of points to randomly scatter (disabled by default)"},
     {"density", "0.0", "0.0", "uniform density of points per active voxel (disabled by default)"},
     {"ppv", "8", "8", "number of points per active voxel (defaults to 8)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be scatter points into. Defaults to 0, i.e. most recently inserted VDB"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"name", "", "scatter_input", "specify the name of the resulting points (by default it's derived from the input VDB)"}},
     [&](){mParser.setDefaults();}, [&](){this->scatter();});

  mParser.addAction(
      "platonic", "", "Create a level set shape with the specified number of polygon faces",
    {{"dim", "", "256", "largest dimension in voxel units of the bbox of all the shape (defaults to 256). In \"voxel\" is defined \"dim\" is ignored"},
     {"voxel", "", "0.01", "voxel size in world units (by defaults \"dim\" is used to derive \"voxel\"). If specified this option takes precedence over \"dim\""},
     {"faces", "4", "{4|6|8|12|20}", "number of polygon faces of the shape to generate the level set VDB from"},
     {"scale", "1.0", "1.0", "scale of the shape in world units. E.g. if faces=6 and scale=1.0 the result is a unit cube"},
     {"center", "(0,0,0)", "(0.0,0.0,0.0)", "center of the shape in world units. defaults to the origin"},
     {"width", "", "3.0", "half-width in voxel units of the output narrow-band level set (defaults to 3 units on either side of the zero-crossing)"},
     {"name", "", "Tetrahedron", "specify the name of the resulting VDB (by default it's derived from face count)"}},
     [&](){mParser.setDefaults();}, [&](){this->levelSetPlatonic();});

  mParser.addAction(
      "enright", "", "Performs Enright advection benchmark test on a level set",
    {{"translate", "(0,0,0)", "(0.0,0.0,0.0)", "defines the origin of the Enright velocity field"},
     {"scale", "1.0", "1.0", "defined the scale of the Enright velocity field"},
     {"dt", "0.05", "0.05", "time-step the input level set is advected"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->enright();});

  mParser.addAction(
      "dilate", "", "erode level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is dilated"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
      "erode", "", "erode level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is eroded"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
      "open", "", "morphological opening, i.e. erosion followed by dilation, of a level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is opened"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
      "close", "", "morphological closing, i.e. dilation followed by erosion, of level set surface by a fixed radius",
    {{"radius", "1.0", "1.0", "radius in voxel units by which the surface is closed"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."}},
     [&](){mParser.setDefaults();}, [&](){this->offsetLevelSet();});

  mParser.addAction(
      "gauss", "", "gaussian convolution of a level set surface",
    {{"iter",  "1", "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
      "mean", "", "mean value filtering of a level set surface",
    {{"iter",  "1",  "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
      "median", "", "median value filtering of a level set surface",
    {{"iter",  "1",  "1", "number of iterations are that the filter is applied"},
     {"space", "", "1|2|3|5", "order of the spatial discretization (defaults to 5, i.e. WENO)"},
     {"time", "", "1|2|3", "order of the temporal discretization (defaults to 1, i.e. explicit Euler)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"size", "1", "1", "size of filter in voxel units"}},
     [&](){mParser.setDefaults();}, [&](){this->filterLevelSet();});

  mParser.addAction(
      "cpt", "", "generate a vector grid with the closest-point-transform to a level set surface",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "div", "", "generate a scalar grid with the divergence of a vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "curl", "", "generate a vector grid with the curl of another vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "grad", "", "generate a vector grid with the gradient of a scalar grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "curvature", "", "generate scalar grid with the mean curvature of a level set surface",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "length", "", "generate a scalar grid with the magnitude of a vector grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->compute();});

  mParser.addAction(
      "union", "", "CSG union of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to union. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true", "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
      "intersection", "", "CSG intersection of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to intersect. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true", "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
      "difference", "", "CSG difference of two level sets surfaces",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to difference. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"prune", "true", "true", "toggle wether to prune the tree after the boolean operation (enabled by default)"},
     {"rebuild", "true",  "true", "toggle wether to re-build the level set after the boolean operation (enabled by default)"}},
     [&](){mParser.setDefaults();}, [&](){this->csg();});

  mParser.addAction(
      "min", "", "Given grids A and B, compute min(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
      "max", "", "Given grids A and B, compute max(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
      "sum", "", "Given grids A and B, compute sum(a, b) per voxel",
    {{"vdb", "0,1", "0,1", "ages (i.e. stack indices) of the two VDB grids to composit. Defaults to 0,1, i.e. two most recently inserted VDBs."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDBs is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->composite();});

  mParser.addAction(
      "multires", "", "construct a LoD sequences of VDB trees with powers of two refinements",
    {{"levels", "2", "2", "number of multi-resolution grids in the output LoD sequence"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->multires();});

  mParser.addAction(
      "resample", "", "resample one VDB grid into another VDB grid or a transformation of the input grid",
    {{"vdb", "0,1", "0,1", "pair of input and optional output grids (i.e. stack index) to be processed. Defaults to 0,1, i.e. most recent VDB is resampled to match the transform of the second to most recent VDB."},
     {"scale", "0", "0", "scale use to transform the input grid (ignored if two grids are specified with vdb)"},
     {"translate", "(0,0,0)", "(0,0,0)", "translation use to transform the input grid (ignored if two grids are specified with vdb)"},
     {"order", "1", "1", "order of the polynomial interpolation kernel used during resampling"},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->resample();});

  mParser.addAction(
      "clip", "", "Clip a VDB grid against another grid, a bbox or frustum",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"},
     {"bbox", "", "(0,0,0),(1,1,1)", "min and max of the world-space bounding-box used for clipping. Defaults to empty, i.e. disabled"},
     {"taper", "-1", "1", "taper of the frustum (requires bbox and depth to be specified). Defaults to -1, i.e. disabled"},
     {"depth", "-1", "1", "depth in world units of the frustum (requires bbox and taper to be specified). Defaults to -1, i.e. disabled"},
     {"mask", "-1", "1", "age (i.e. stack index) of a mask VDB used for clipping. Defaults to -1, i.e. disabled"}},
     [&](){mParser.setDefaults();}, [&](){this->clip();});

  mParser.addAction(
      "prune", "", "prune away inactive values in a VDB grid",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB"}},
     [](){},[&](){this->pruneLevelSet();});

  mParser.addAction(
      "flood", "", "signed-flood filling of a level set VDB",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB"}},
     [](){},[&](){this->floodLevelSet();});

  mParser.addAction(
      "expand", "", "expand narrow band of level set",
    {{"dilate", "1", "1", "number of integer voxels that the narrow band of the input SDF will be dilated"},
     {"iter", "1", "1", "number of iterations of the fast sweeping algorithm (each using 8 sweeps)"},
     {"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->expandLevelSet();});

  mParser.addAction(
      "segment", "", "segment an input VDB into a list if topologically disconnected VDB grids",
    {{"vdb", "0", "0", "age (i.e. stack index) of the VDB grid to be processed. Defaults to 0, i.e. most recently inserted VDB."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or deleted after the processing"}},
     [&](){mParser.setDefaults();}, [&](){this->segment();});

  mParser.addAction(
      "transform", "", "apply affine transformations (uniform scale -> rotation -> translation) to a VDB grids and geometry",
    {{"rotate", "(0.0,0.0,0.0)", "(0.0,0.0,0.0)", "rotation in radians around x,y,z axis"},
     {"translate", "(0.0,0.0,0.0)", "(0.0,0.0,0.0)", "translation in world units along x,y,z axis"},
     {"scale", "1.0", "1.0", "uniform scaling in world units"},
     {"vdb", "", "0,2,..", "age (i.e. stack index) of the VDB grid to be processed. Defaults to empty."},
     {"geo", "", "0,2,..", "age (i.e. stack index) of the Geometry to be processed. Defaults to empty."},
     {"keep", "", "1|0|true|false", "toggle wether the input VDB is preserved or overwritten"}},
     [&](){mParser.setDefaults();}, [&](){this->transform();});

  mParser.addAction(
      "render", "", "ray-tracing of level set surfaces and volume rendering of fog volumes",
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
      "print", "p", "prints information to the terminal about the current stack of VDB grids and Geometry",
      {{"vdb", "*", "*", "print information about VDB grids"},
       {"geo", "*", "*", "print information about geometries"},
       {"mem", "0", "0|1|false|true", "print a list of all stored variables"}},
      [](){}, [&](){this->print();});

  mParser.addAction(
      "version", "", "write timing information to the terminal", {},
      [&](){std::cerr << mCmdName << ": version " << Tool::version() << std::endl;std::exit(EXIT_SUCCESS);}, [](){});

  mParser.addAction(
      "examples", "", "print examples to the terminal and terminate", {},
      [&](){std::cerr << this->examples() << std::endl; std::exit(EXIT_SUCCESS);}, [](){});

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
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "help");
  try {
    mParser.printAction();
    const VecS actions = mParser.getVec<std::string>("actions");
    const bool stop = mParser.get<bool>("exit");
    const bool brief = mParser.get<bool>("brief");

    if (actions.empty()) {
      if (mParser.actions.size()==1) {// ./vdb_tool -help
        if (!brief) {
          std::cerr << "\nThis command-line tool can perform a use-defined, and possibly\n"
                    << "non-linear, sequence of high-level tasks available in openvdb.\n"
                    << "For instance, it can convert polygon meshes and particles to level\n"
                    << "sets, and subsequently perform a large number of operations on these\n"
                    << "level set surfaces. It can also generate adaptive polygon meshes from\n"
                    << "level sets, write them to disk and even render them to image files.\n\n"
                    << "Version: " + Tool::version() + "\n" + this->examples() + "\n";
        }
        mParser.usage_all(brief);
        if (!brief) {
          std::cerr << "\nNote that actions always start with one or more \"-\", and (except for file names)\n"
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::help()

// ==============================================================================================================

std::string Tool::examples() const
{
    const int w = 16;
    std::stringstream ss;
    ss << std::left << std::setw(w) << "Surface points:" << mCmdName << " -read points.[obj/ply/stl/pts] -points2ls d=256 r=2.0 w=3 -dilate r=2 -gauss i=1 w=1 -erode r=2 -ls2m a=0.25 -write output.[ply/obj/stl]\n";
    ss << std::left << std::setw(w) << "Convert mesh:  " << mCmdName << " -read mesh.[ply/obj] -mesh2ls d=256 -write output.vdb config.txt\n";
    ss << std::left << std::setw(w) << "Config example:" << mCmdName << " -config config.txt\n";
    return ss.str();
}

// ==============================================================================================================

void Tool::clear()
{
  OPENVDB_ASSERT(mParser.getAction().name == "clear");
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
  OPENVDB_ASSERT(mParser.getAction().name == "read");
  for (auto &fileName : mParser.getVec<std::string>("files")) {
    switch (findFileExt(fileName, {"geo,obj,ply,abc,pts,stl", "vdb", "nvdb"})) {
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
      throw std::invalid_argument("File \""+fileName+"\" has an invalid extension");
      break;
    }
  }
}

// ==============================================================================================================

void Tool::readGeo(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().name == "read");
  if (mParser.verbose>1) std::cerr << "Reading geometry from \"" << fileName << "\"\n";
  if (mParser.verbose) mTimer.start("Read geometry");
  Geometry::Ptr geom(new Geometry());
  geom->read(fileName);
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
  OPENVDB_ASSERT(mParser.getAction().name == "read");
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
  } else if (mParser.verbose>0) {
    std::cerr << "readVDB: no vdb grids in \"" << fileName << "\"";
  }
  if (mParser.verbose) {
    mTimer.stop();
    if (mGrid.size() == count) std::cerr << "readVDB: no vdb grids were loaded\n";
    if (mParser.verbose>1) for (GridBase::Ptr grid : *grids) grid->print();
  }
}// Tool::readVDB

// ==============================================================================================================

#ifdef VDB_TOOL_USE_NANO
void Tool::readNVDB(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().name == "read");
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
      if (gridNames[0]=="*" || findMatch(gridHandle.gridMetaData()->shortGridName(), gridNames)) mGrid.push_back(nanovdb::nanoToOpenVDB(gridHandle));
    }
  } else if (mParser.verbose>0) {
    std::cerr << "readVDB: no vdb grids in \"" << fileName << "\"";
  }
  if (mParser.verbose) {
    mTimer.stop();
    if (mGrid.size() == count) std::cerr << "readNVDB: no NanoVDB grids were loaded\n";
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
    OPENVDB_ASSERT(mParser.getAction().name == "config");
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
            if (mParser.verbose>1) std::cerr << "Reading configuration from \"" << fileName << "\"\n";
            if (mParser.verbose) mTimer.start("Read config");
            if (!getline (file,line)) throw std::invalid_argument("readConf: empty file \""+fileName+"\"");
            Header header(line);
            if (!header.isCompatible()) throw std::invalid_argument("readConf: incompatible version \""+line+"\"");
            std::vector<char*> args({&header.mMagic[0]});//parser is expecting first argument to the name of the executable
            while (getline(file, line)) {
                if (line.empty() || contains("#/%!", line[0])) continue;// skip empty lines and comments
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
  OPENVDB_ASSERT(mParser.getAction().name == "write");
  for (std::string &fileName : mParser.getVec<std::string>("files")) {
    switch (findFileExt(fileName, {"geo,obj,ply,stl,abc", "vdb", "nvdb", "txt"})) {
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
}// Tool::write

// ==============================================================================================================

void Tool::writeVDB(const std::string &fileName)
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "write");
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

    if (grids.empty()) throw std::invalid_argument("writeVDB: no vdb grids to write");

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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::writeVDB

// ==============================================================================================================

#ifdef VDB_TOOL_USE_NANO
void Tool::writeNVDB(const std::string &fileName)
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "write");
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

    nanovdb::StatsMode sMode = nanovdb::StatsMode::Default;
    if (stats == "none") {
      sMode = nanovdb::StatsMode::Disable;
    } else if (stats == "bbox") {
      sMode = nanovdb::StatsMode::BBox;
    } else if (stats == "extrema") {
      sMode = nanovdb::StatsMode::MinMax;
    } else if (stats == "all") {
      sMode = nanovdb::StatsMode::All;
    } else if (stats != "") {
      throw std::invalid_argument("writeNVDB: unsupported stats \""+stats+"\"");
    }

    nanovdb::ChecksumMode cMode = nanovdb::ChecksumMode::Default;
    if (checksum == "none") {
      cMode = nanovdb::ChecksumMode::Disable;
    } else if (checksum == "partial") {
      cMode = nanovdb::ChecksumMode::Partial;
    } else if (checksum == "full") {
      cMode = nanovdb::ChecksumMode::Full;
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

    if (grids.empty()) throw std::invalid_argument("writeNVDB: no vdb grids to write");

    auto openToNano = [&](const GridBase::Ptr& base) {
      if (auto floatGrid = GridBase::grid<FloatGrid>(base)) {
        using SrcGridT = openvdb::FloatGrid;
        switch (qMode){
        case nanovdb::GridType::Fp4:
          return nanovdb::createNanoGrid<SrcGridT, nanovdb::Fp4>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::Fp8:
          return nanovdb::createNanoGrid<SrcGridT, nanovdb::Fp8>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::Fp16:
          return nanovdb::createNanoGrid<SrcGridT, nanovdb::Fp16>(*floatGrid, sMode, cMode, dither, verbose);
        case nanovdb::GridType::FpN:
          if (absolute) {
            return nanovdb::createNanoGrid<SrcGridT, nanovdb::FpN>(*floatGrid, sMode, cMode, dither, verbose, nanovdb::AbsDiff(tolerance));
          } else {
            return nanovdb::createNanoGrid<SrcGridT, nanovdb::FpN>(*floatGrid, sMode, cMode, dither, verbose, nanovdb::RelDiff(tolerance));
          }
        default: break;// 32 bit float grids are handled below
        }// end of switch
      }
      return nanovdb::openToNanoVDB(base, sMode, cMode, verbose);// float and other grids
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
    throw std::invalid_argument(name+": "+e.what());
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
  OPENVDB_ASSERT(mParser.getAction().name == "write");
  const int age = mParser.get<int>("geo");
  const bool keep = mParser.get<bool>("keep");
  if (mParser.verbose>1) std::cerr << "Writing geometry to \"" << fileName << "\"\n";
  auto it = this->getGeom(age);
  const Geometry &mesh = **it;
  if (mParser.verbose) mTimer.start("Write geometry");
  mesh.write(fileName);
  if (!keep) mGeom.erase(std::next(it).base());
  if (mParser.verbose) mTimer.stop();
}// Tool::writeGeo

// ==============================================================================================================

void Tool::writeConf(const std::string &fileName)
{
  OPENVDB_ASSERT(mParser.getAction().name == "write");
  if (mParser.verbose>1) std::cerr << "Writing configuration to \"" << fileName << "\"\n";
  std::ofstream file(fileName);
  if (!file.is_open()) throw std::invalid_argument("writeConf: unable to open \""+fileName+"\"");
  if (mParser.verbose) mTimer.start("Write config");
  const Header header;
  file << header.str() << std::endl;
  for (auto &a : mParser.actions) if (a.name != "config") a.print(file);// exclude "-config" to avoid infinite loop
  file.close();
  if (mParser.verbose) mTimer.stop();
}// Tool::writeConf

// ==============================================================================================================

void Tool::vdbToPoints()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "vdb2points");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age);
    auto grid = gridPtrCast<points::PointDataGrid>(*it);
    if (!grid || grid->getGridClass() != GRID_UNKNOWN) throw std::invalid_argument("vdbToPoints: no PointDataGrid with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("VDB to points");
    const size_t count = points::pointCount(grid->tree());
    if (count==0) throw std::invalid_argument("vdbToPoints: empty PointDataGrid with age "+std::to_string(age));
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::vdbToPoints

// ==============================================================================================================

void Tool::pointsToVdb()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "points2vdb");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("geo");
    const bool keep = mParser.get<bool>("keep");
    const int pointsPerVoxel = mParser.get<int>("ppv");
    const int bits = mParser.get<int>("bits");
    std::string grid_name = mParser.get<std::string>("name");
    using GridT = points::PointDataGrid;
    if (mParser.verbose) mTimer.start("Points to VDB");
    auto it = this->getGeom(age);
    Points points((*it)->vtx());
    const float voxelSize = points::computeVoxelSize(points, pointsPerVoxel);
    auto xform = math::Transform::createLinearTransform(voxelSize);

    GridT::Ptr grid;
    switch (bits) {
    case 8:
      grid = points::createPointDataGrid<points::FixedPointCodec</*1-byte=*/true>, GridT>((*it)->vtx(), *xform);
      break;
    case 16:
      grid = points::createPointDataGrid<points::FixedPointCodec</*1-byte=*/false>, GridT>((*it)->vtx(), *xform);
      break;
    case 32:
      grid = points::createPointDataGrid<points::NullCodec, GridT>((*it)->vtx(), *xform);
      break;
    default:
      throw std::invalid_argument("pointsToVdb: unsupported bit-width: "+std::to_string(bits));
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
  const std::string &name = mParser.getAction().name;
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
        geom = (*it)->copyGeom();
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
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "ls2fog");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age);
    auto sdf = gridPtrCast<FloatGrid>(*it);
    if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("levelSetToFog: no Level Set with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("SDF to FOG");
    FloatGrid::Ptr fog = keep ? sdf->deepCopy() : sdf;
    tools::sdfToFogVolume(*fog);
    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = "ls2fog_"+sdf->getName();
    fog->setName(grid_name);
    mGrid.push_back(fog);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::levelSetToFog

// ==============================================================================================================

void Tool::isoToLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "iso2ls");
  try {
    mParser.printAction();
    const VecI age = mParser.getVec<int>("vdb");
    if (age.size()!=1 && age.size()!=2) throw std::invalid_argument(name+": expected one or two vdb grids, not "+std::to_string(age.size()));
    const float isoValue = mParser.get<float>("iso");
    const float voxel = mParser.get<float>("voxel");
    const float width = mParser.get<float>("width");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");
    auto it = this->getGrid(age[0]);
    auto grid = gridPtrCast<FloatGrid>(*it);
    if (!grid) throw std::invalid_argument(name+": no FloatGrid with age "+std::to_string(age[0]));
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::isoToLevelSet

// ==============================================================================================================

float Tool::estimateVoxelSize(int maxDim,  float halfWidth, int geo_age)
{
  auto it = this->getGeom(geo_age);
  const auto bbox = (*it)->bbox();
  if (!bbox) {
    throw std::invalid_argument("estimateVoxelSize: invalid bbox");
  } else if (maxDim <= 0) {
    throw std::invalid_argument("estimateVoxelSize: invalid maxDim");
  }
  const auto d = bbox.extents()[bbox.maxExtent()];// longest extent of bbox along any coordinate axis
  return static_cast<float>(static_cast<float>(d)/static_cast<float>(maxDim - static_cast<int>(2.f * halfWidth)));
}// Tool::estimateVoxelSize

// ==============================================================================================================

void Tool::meshToLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "mesh2ls");
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
    if (vdb_age>=0) {
      auto it = this->getGrid(vdb_age);
      xform = (*it)->transform().copy();
    } else {
      if (voxel == 0.0f) voxel = this->estimateVoxelSize(dim, width, geo_age);
      xform = math::Transform::createLinearTransform(voxel);
    }
    auto it = this->getGeom(geo_age);
    const Geometry &mesh = **it;
    if (mesh.isPoints()) this->warning("Warning: -mesh2ls was called on points, not a mesh! Hint: use -points2ls instead!");
    if (mParser.verbose) mTimer.start("Mesh -> SDF");
    auto grid  = tools::meshToLevelSet<GridT>(*xform, mesh.vtx(), mesh.tri(), mesh.quad(), width);
    if (grid_name.empty()) grid_name = "mesh2ls_" + mesh.getName();
    grid->setName(grid_name);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::meshToLevelSet

// ==============================================================================================================

void Tool::particlesToLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "points2ls");
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
    if (points.isMesh()) this->warning("Warning: -points2ls was called on a mesh, not points! Hint: use -mesh2ls instead!");
    if (mParser.verbose) mTimer.start("Points->SDF");
    GridT::Ptr grid = createLevelSet<GridT>(voxel, width);
    if (grid_name.empty()) grid_name = "points2ls_"+points.getName();
    grid->setName(grid_name);
    tools::particlesToSdf(Points(points.vtx()), *grid, voxel*radius);
    mGrid.push_back(grid);
    if (!keep) mGeom.erase(std::next(it).base());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
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
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(findMatch(name, {"dilate", "erode", "open", "close"}));
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
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("offsetLevelSet: no level set with age "+std::to_string(age));
    auto filter = this->createFilter(*grid, space, time);
    radius *= static_cast<float>((*it)->voxelSize()[0]);// voxel to world units
    if (name == "dilate") {
      if (mParser.verbose) mTimer.start("Dilate  SDF");
      filter->offset(-radius);
    } else if (name == "erode") {
      if (mParser.verbose) mTimer.start("Erode   SDF");
      filter->offset( radius);
    } else if (name == "open") {
      if (mParser.verbose) mTimer.start("Open   SDF");
      filter->offset( radius);
      filter->offset(-radius);
    } else if (name == "close") {
      if (mParser.verbose) mTimer.start("Close   SDF");
      filter->offset(-radius);
      filter->offset( radius);
    } else {
      throw std::invalid_argument("offsetLevelSet: invalid operation type");
    }
    grid->setName(name + "_" + grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::offsetLevelSet

// ==============================================================================================================

void Tool::filterLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(findMatch(name, {"gauss", "mean", "median"}));
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
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("filterLevelSet: no level set with age "+std::to_string(age));
    auto filter = this->createFilter(*grid, space, time);

    if (name == "gauss") {
      if (mParser.verbose) mTimer.start("Gauss   SDF");
      for (int i=0; i<nIter; ++i) filter->gaussian(size);
    } else if (name == "mean") {
      if (mParser.verbose) mTimer.start("Mean SDF ");
      for (int i=0; i<nIter; ++i) filter->mean(size);
    } else if (name == "median") {
      if (mParser.verbose) mTimer.start("Median SDF");
      for (int i=0; i<nIter; ++i) filter->median(size);
    } else {
      throw std::invalid_argument("filterLevelSet: invalid filter type");
    }
    grid->setName(name + "_" + grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::filterLevelSet

// ==============================================================================================================

void Tool::pruneLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "prune");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("pruneLevelSet: no level set with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("Prune   SDF");
    tools::pruneLevelSet(grid->tree());
    grid->setName("prune_"+grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::pruneLevelSet

// ==============================================================================================================

void Tool::floodLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "flood");
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("floodLevelSet: no level set with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("Flood   SDF");
    tools::signedFloodFill(grid->tree());
    grid->setName("flood_"+grid->getName());
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::floodLevelSet

// ==============================================================================================================

void Tool::compute()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(findMatch(name, {"cpt","div","curl","length","grad","curvature"}));
  try {
    mParser.printAction();
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    if (name == "cpt") {
      if (mParser.verbose) mTimer.start("CPT of SDF");
      auto sdf = gridPtrCast<FloatGrid>(*it);
      if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("cpt: no level set with age "+std::to_string(age));
      auto grid = tools::cpt(*sdf);
      grid->setName("cpt_"+sdf->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (name == "div") {
      if (mParser.verbose) mTimer.start("Divergence");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("div: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::divergence(*vec);
      grid->setName("div_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (name == "curl") {
      if (mParser.verbose) mTimer.start("Curl of Vec3");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("curl: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::curl(*vec);
      grid->setName("curl_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (name == "length") {
      if (mParser.verbose) mTimer.start("Length of Vec3");
      auto vec = gridPtrCast<Vec3fGrid>(*it);
      if (!vec) throw std::invalid_argument("length: no vec3f grid with age "+std::to_string(age));
      auto grid = tools::magnitude(*vec);
      grid->setName("length_"+vec->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (name == "grad") {
      if (mParser.verbose) mTimer.start("Gradient");
      auto scalar = gridPtrCast<FloatGrid>(*it);
      if (!scalar) throw std::invalid_argument("grad: no float grid with age "+std::to_string(age));
      auto grid = tools::gradient(*scalar);
      grid->setName("grad_"+scalar->getName());
      if (!keep) mGrid.erase(std::next(it).base());
      mGrid.push_back(grid);
    } else if (name == "curvature") {
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::compute

// ==============================================================================================================

void Tool::composite()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(findMatch(name, {"min","max","sum"}));
  try {
    mParser.printAction();
    const VecI ij = mParser.getVec<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    if (ij.size()!=2) throw std::invalid_argument(name+": expected two vdb ages, but got "+std::to_string(ij.size()));
    if (ij[0] == ij[1]) throw std::invalid_argument(name+": identical inputs: volume1=volume2="+std::to_string(ij[0]));
    auto itA = this->getGrid(ij[0]), itB = this->getGrid(ij[1]);
    GridT::Ptr gridA = gridPtrCast<GridT>(*itA);
    if (!gridA) throw std::invalid_argument(name+": no float grid with age "+std::to_string(ij[0]));
    GridT::Ptr gridB = gridPtrCast<GridT>(*itB);
    if (!gridB) throw std::invalid_argument(name+": no float grid with age "+std::to_string(ij[1]));
    if (gridA->transform() != gridB->transform()) this->warning(name+": grids have different transforms");
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
    tmpA->setName(name+"_"+tmpA->getName());
    if (mParser.verbose) mTimer.start(name);
    if (name == "min") {
      tools::compMin(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else if (name == "max") {
      tools::compMax(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else if (name == "sum") {
      tools::compSum(*tmpA, *tmpB);// Store the result in the A grid and leave the B grid empty.
    } else {
      throw std::invalid_argument(name+": invalid operation");
    }
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::composite

// ==============================================================================================================

void Tool::csg()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(findMatch(name, {"union", "intersection", "difference"}));
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
    if (!gridA || gridA->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("floodLevelSet: no level set with age "+std::to_string(ij[0]));
    GridT::Ptr gridB = gridPtrCast<GridT>(*itB);
    if (!gridB || gridB->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("floodLevelSet: no level set with age "+std::to_string(ij[1]));
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
    if (name == "union") {
      if (mParser.verbose) mTimer.start("Union");
      if (keep) {
        GridT::Ptr grid = tools::csgUnionCopy(*gridA, *gridB);
        if (rebuild) tools::sdfToSdf(*grid);
        grid->setName("union_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgUnion(*gridA, *gridB, prune);// overwrites A and cannibalizes B
        if (rebuild) tools::sdfToSdf(*gridA);
        gridA->setName("union_"+gridA->getName());
      }
    } else if (name == "intersection") {
      if (mParser.verbose) mTimer.start("Intersection");
      if (keep) {
        GridT::Ptr grid = tools::csgIntersectionCopy(*gridA, *gridB);
        if (rebuild) tools::sdfToSdf(*grid);
        grid->setName("intersection_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgIntersection(*gridA, *gridB, prune);// overwrites A and cannibalizes B
        if (rebuild) tools::sdfToSdf(*gridA);
        gridA->setName("intersection_"+gridA->getName());
      }
    } else if (name == "difference") {
      if (mParser.verbose) mTimer.start("Difference");
      if (keep) {
        GridT::Ptr grid = tools::csgDifferenceCopy(*gridA, *gridB);
        if (rebuild) tools::sdfToSdf(*grid);
        grid->setName("difference_"+gridA->getName());
        mGrid.push_back(grid);// A and B are unchanged!
      } else {
        tools::csgDifference(*gridA, *gridB, prune);// overwrites A and deletes B
        if (rebuild) tools::sdfToSdf(*gridA);
        gridA->setName("difference_"+gridA->getName());
      }
    } else {
      throw std::invalid_argument("csg: invalid type");
    }
    if (!keep) mGrid.erase(std::next(itB).base());// remove B since it was corrupted
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::csg

// ==============================================================================================================

void Tool::levelSetToMesh()
{
  const std::string &action_name = mParser.getAction().name;
  OPENVDB_ASSERT(action_name == "ls2mesh");
  try {
    mParser.printAction();
    const double adaptivity = mParser.get<float>("adapt");
    const double iso = mParser.get<float>("iso");
    const int age = mParser.get<int>("vdb");
    const int mask = mParser.get<int>("mask");
    const bool invert = mParser.get<bool>("invert");
    const bool keep = mParser.get<bool>("keep");
    std::string grid_name = mParser.get<std::string>("name");

    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("levelSetToMesh: no level set grid with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("SDF -> mesh");

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
        throw std::invalid_argument("levelSetToMesh: unsupported mask type with age "+std::to_string(mask));
      }
    }
    mesher(*grid);

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

    if (!keep) mGrid.erase(std::next(it).base());
    if (grid_name.empty()) grid_name = "ls2mesh_"+grid->getName();
    geom->setName(grid_name);
    mGeom.push_back(geom);

    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(action_name+": "+e.what());
  }
}// Tool::levelSetToMesh

// ==============================================================================================================

void Tool::levelSetSphere()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "sphere");
  try {
    mParser.printAction();
    const int dim = mParser.get<int>("dim");
    float voxel = mParser.get<float>("voxel");
    const float radius = mParser.get<float>("radius");
    const Vec3f center = mParser.getVec3<float>("center");
    const float width = mParser.get<float>("width");
    const std::string grid_name = mParser.get<std::string>("name");
    if (voxel == 0.0f) voxel = 2.0f*radius/(static_cast<float>(dim) - 2.0f*width);
    if (mParser.verbose) mTimer.start("Create sphere");
    GridT::Ptr grid = tools::createLevelSetSphere<GridT>(radius, center, voxel, width);
    if (mParser.verbose) mTimer.stop();
    grid->setName(grid_name);
    mGrid.push_back(grid);
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::levelSetSphere

// ==============================================================================================================

void Tool::levelSetPlatonic()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "platonic");
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::levelSetPlatonic

// ==============================================================================================================

void Tool::multires()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "multires");
  try {
    mParser.printAction();
    const int levels = mParser.get<int>("levels");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    GridT::Ptr grid = gridPtrCast<GridT>(*it);
    if (!grid) throw std::invalid_argument("multires: no FloatGrid with age "+std::to_string(age));
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::multires

// ==============================================================================================================

void Tool::expandLevelSet()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "expand");
  try {
    mParser.printAction();
    const int dilate = mParser.get<int>("dilate");
    const int iter = mParser.get<int>("iter");
    const int age = mParser.get<int>("vdb");
    const bool keep = mParser.get<bool>("keep");
    auto it = this->getGrid(age);
    GridT::Ptr sdf = gridPtrCast<GridT>(*it);
    if (!sdf || sdf->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("expandLevelSet: no level set with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("Expand SDF");
    auto grid = tools::dilateSdf(*sdf, dilate, tools::NN_FACE, iter);
    if (!keep) mGrid.erase(std::next(it).base());
    grid->setName("expand_"+grid->getName());
    mGrid.push_back(grid);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::expandLevelSet

// ==============================================================================================================

void Tool::segment()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "segment");
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
      throw std::invalid_argument("segment: grid with age "+std::to_string(age)+" is not a float grid");
    }
    if (!keep) mGrid.erase(std::next(it).base());
    for (auto g : grids) mGrid.push_back(g);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::segment

// ==============================================================================================================

// for simplicity we are restricting this resampler to only work on float grids!
void Tool::resample()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "resample");
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
      if (!outGrid) throw std::invalid_argument("resample: no reference grid of type float with age "+std::to_string(age[1]));
    } else {
      if (scale<=0.0f) throw std::invalid_argument("resample: invalid scale: "+std::to_string(scale));
      auto map = math::MapBase::Ptr(new math::UniformScaleTranslateMap(scale, translate));
      auto xform = math::Transform::Ptr(new math::Transform(map));
      outGrid = FloatGrid::create();
      outGrid->setTransform(xform);
    }

    if (!inGrid) throw std::invalid_argument("resample: no grid of type float with age "+std::to_string(age[0]));

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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::resample

// ==============================================================================================================

void Tool::scatter()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "scatter");
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
    if (!grid) throw std::invalid_argument("scatter: no float grid with age "+std::to_string(age));
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
    throw std::invalid_argument(name+": "+e.what());
  }
}// Tool::scatter

// ==============================================================================================================
// LeVeque, R., High-Resolution Conservative Algorithms For Advection In Incompressible Flow, SIAM J. Numer. Anal. 33, 627–665 (1996)
// https://faculty.washington.edu/rjl/pubs/hiresadv/0733033.pdf
void Tool::enright()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "enright");
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
    if (!grid || grid->getGridClass() != GRID_LEVEL_SET) throw std::invalid_argument("enright: no level set with age "+std::to_string(age));
    if (mParser.verbose) mTimer.start("Enright SDF");
    tools::LevelSetAdvection<GridT, LeVequeField> advect(*grid, field);
    advect.setSpatialScheme(math::HJWENO5_BIAS);
    advect.setTemporalScheme(math::TVD_RK2);
    advect.setTrackerSpatialScheme(math::HJWENO5_BIAS);
    advect.setTrackerTemporalScheme(math::TVD_RK1);
    advect.advect(0.0f, dt);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
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
}

// ==============================================================================================================

void Tool::clip()
{
  const std::string &name = mParser.getAction().name;
  OPENVDB_ASSERT(name == "clip");
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
      throw std::invalid_argument("clip: unsupported grid type with "+std::to_string(age));
    }
    if (!(*it)->getName().empty()) grid->setName("clip_"+(*it)->getName());
    if (!keep) mGrid.erase(std::next(it).base());
    mGrid.push_back(grid);
    if (mParser.verbose) mTimer.stop();
  } catch (const std::exception& e) {
    throw std::invalid_argument(name+": "+e.what());
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
  OPENVDB_ASSERT(mParser.getAction().name == "render");
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
  if (!grid) throw std::invalid_argument("render: no float with age "+std::to_string(age));
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

void Tool::print(std::ostream& os) const
{
  OPENVDB_ASSERT(mParser.getAction().name == "print");

  if (mParser.verbose>1) {
    os << "\n" << std::setw(40) << std::setfill('=') << "> Actions <" << std::setw(40) << "\n";
    mParser.print(os);
    os << std::setw(80) << std::setfill('=') << "\n" << std::endl;
    os << "\n" << std::setw(40) << std::setfill('=') << "> Variables <" << std::setw(40) << "\n";
    mParser.processor.memory().print(os);
    os << std::setw(80) << std::setfill('=') << "\n" << std::endl;
  }

  if (mParser.verbose>0) {
    os << "\n" << std::setw(40) << std::setfill('=') << "> Primitives <" << std::setw(39) << "\n";

    if (mParser.getStr("geo")=="*") {
      for (auto begin = mGeom.crbegin(), it = begin, end = mGeom.crend(); it != end; ++it) {
        const Geometry &geom = **it;
        os << "Geometry: age = " << std::distance(begin,it) << ", name = \"" << geom.getName() << "\", ";
        geom.print(0, os);
        os << "\n";
      }
      if (mGeom.empty()) os << "Geometry: none\n";
    } else {
      for (int age : mParser.getVec<int>("geo")) {
        auto it = this->getGeom(age);
        const Geometry &geom = **it;
        os << "Geometry: age = " << age << ", name = \"" << geom.getName() << "\", ";
        geom.print(0, os);
        os << "\n";
      }
    }

    if (mParser.getStr("vdb")=="*") {
      for (auto begin = mGrid.crbegin(), it = begin, end = mGrid.crend(); it != end; ++it) {
        const auto &grid = **it;
        const auto bbox = grid.evalActiveVoxelBoundingBox();
        os << "VDB grid: age = " << std::distance(begin,it) << ", name = \"" << grid.getName() << "\", type = \"";
        os << grid.valueType() << "\", bbox = " << bbox << ", dim = " << bbox.dim();
        os << ", voxels = " << grid.activeVoxelCount() << ", dx = " << grid.voxelSize()[0];
        os << ", size = ";
        util::printBytes(os, grid.memUsage());
      }
      if (mGrid.empty()) os << "VDB grid: none\n";
    } else {
      for (int age : mParser.getVec<int>("vdb")) {
        auto it = this->getGrid(age);
        const auto &grid = **it;
        const auto bbox = grid.evalActiveVoxelBoundingBox();
        os << "VDB grid: age = " << age << ", name = \"" << grid.getName() << "\", type = \"";
        os << grid.valueType() << "\", bbox = " << bbox << ", dim = " << bbox.dim();
        os << ", voxels = " << grid.activeVoxelCount() << ", dx = " << grid.voxelSize()[0];
        os << ", size = ";
        util::printBytes(os, grid.memUsage());
      }
    }

    if (mParser.get<bool>("mem")) {
      os << "\n" << std::setw(40) << std::setfill('=') << "> Variables <" << std::setw(40) << "\n";
      mParser.processor.memory().print(os);
    }

    os << std::setw(80) << std::setfill('=') << "\n\n" << std::endl;

  }
}// Tool::print

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_TOOL_HAS_BEEN_INCLUDED
