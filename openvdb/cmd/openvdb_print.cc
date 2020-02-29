// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <openvdb/openvdb.h>
#include <openvdb/util/logging.h>

#include <openvdb/points/PointDataGrid.h>

namespace {

using StringVec = std::vector<std::string>;

const char* INDENT = "   ";
const char* gProgName = "";

struct PointStats{
    using Index64 = openvdb::Index64;
    using Name = openvdb::Name;
    using GridBase  = openvdb::GridBase;
    using AttributeSet = openvdb::points::AttributeSet;
    using PointDataGrid = openvdb::points::PointDataGrid;

    // proxy collect attributes not covered by AttributeSet::Info::Array
    struct PointAttrib{
        AttributeSet::Info::Array array;
        Index64 index = 0;
        bool shared = false;
        bool uniform = true;
        Name codec = "";
        Name type = "";
    };

    Index64 total = 0;
    Index64 active = 0;
    Index64 inactive = 0;
    bool firstLeaf = true;
    std::map<Name, Index64> groups;
    std::map<Name, PointAttrib> attribs;


static void
inspectPoints(const openvdb::GridBase::ConstPtr grid, PointStats& pointStats)
{
    PointDataGrid::ConstPtr inputGrid = GridBase::grid<PointDataGrid>(grid);

    for (auto leafIter = inputGrid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        auto attrset = leafIter->attributeSet();
        auto dptr = attrset.descriptorPtr();
        auto attrmap = dptr->map();

        pointStats.total += leafIter->pointCount();
        pointStats.active += leafIter->onPointCount();
        pointStats.inactive += leafIter->offPointCount();

        // groups
        auto grmap = dptr->groupMap();

        // Count points in groups
        for (auto it=grmap.begin(); it!=grmap.end(); ++it){
            if(pointStats.groups.find(it->first) == pointStats.groups.end()){
                pointStats.groups[it->first] = 0;
            }
            pointStats.groups[it->first]+=leafIter->groupPointCount(it->first);
        }

        // Collect attribute info - gather attributes info from the first leaf only.
        // (Assume consistency across leaves).
        AttributeSet::Info info(dptr);
        if (!pointStats.firstLeaf) continue;
        for (auto it=attrmap.begin(); it!=attrmap.end(); ++it){
            AttributeSet::Info::Array& arrInfo = info.arrayInfo(it->first);
            auto attrArr = attrset.getConst(it->first /*name*/);
            PointAttrib pa{};
            pa.array = std::move(arrInfo);
            pa.type = dptr->valueType(pa.index);
            pa.index = it->second;
            pa.shared = attrset.isShared(pa.index);
            // pa.codec = dptr->valueType(pa.index);
            pa.codec = attrArr->codecType();
            pa.uniform = attrArr->isUniform();
            pointStats.attribs[it->first] = pa;
        }
        pointStats.firstLeaf = false;
    }
}

static void
printPointStats(const PointStats& pointStats)
{
    std::cout << "Total Point Count:\n"
              << INDENT << "total: " << pointStats.total << '\n'
              << INDENT << "active: " << pointStats.active  << '\n'
              << INDENT << "inactive: " << pointStats.inactive << '\n';

    std::cout << "Point attributes:" << '\n';
    for(auto it=pointStats.attribs.begin(); it!=pointStats.attribs.end(); ++it){
        auto attr = it->second;
        std::cout << "  name: " << it->first << '\n';
        std::cout << INDENT << "type: " << attr.type << '\n';
        std::cout << INDENT << "codec: " << attr.codec << '\n';
        std::cout << INDENT << "uniform: " << attr.uniform << '\n';
        std::cout << INDENT << "shared: " << attr.shared << '\n';
        std::cout << INDENT << "hidden: " << attr.array.hidden << '\n';
        std::cout << INDENT << "transient: " << attr.array.transient << '\n';
        std::cout << INDENT << "group: " << attr.array.group << '\n';
        std::cout << INDENT << "string: " << attr.array.string << '\n';
        std::cout << INDENT << "constantStride: " << attr.array.constantStride << '\n';
    }
    std::cout << "Point groups:" << '\n';
    for (auto it=pointStats.groups.begin(); it!=pointStats.groups.end(); ++it){
        std::cout << INDENT << it->first << ": " << it->second << '\n';
    }
}
};

void
usage [[noreturn]] (int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
"Usage: " << gProgName << " in.vdb [in.vdb ...] [options]\n" <<
"Which: prints information about OpenVDB grids\n" <<
"Options:\n" <<
"    -l, -stats     long printout, including grid statistics\n" <<
"    -m, -metadata  print per-file and per-grid metadata\n" <<
"    -version       print version information\n";
    exit(exitStatus);
}


std::string
sizeAsString(openvdb::Index64 n, const std::string& units)
{
    std::ostringstream ostr;
    ostr << std::setprecision(3);
    if (n < 1000) {
        ostr << n;
    } else if (n < 1000000) {
        ostr << (double(n) / 1.0e3) << "K";
    } else if (n < 1000000000) {
        ostr << (double(n) / 1.0e6) << "M";
    } else {
        ostr << (double(n) / 1.0e9) << "G";
    }
    ostr << units;
    return ostr.str();
}


std::string
bytesAsString(openvdb::Index64 n)
{
    std::ostringstream ostr;
    ostr << std::setprecision(3);
    if (n >> 30) {
        ostr << (double(n) / double(uint64_t(1) << 30)) << "GB";
    } else if (n >> 20) {
        ostr << (double(n) / double(uint64_t(1) << 20)) << "MB";
    } else if (n >> 10) {
        ostr << (double(n) / double(uint64_t(1) << 10)) << "KB";
    } else {
        ostr << n << "B";
    }
    return ostr.str();
}


std::string
coordAsString(const openvdb::Coord ijk, const std::string& sep,
              const std::string& start, const std::string& stop)
{
    std::ostringstream ostr;
    ostr << start << ijk[0] << sep << ijk[1] << sep << ijk[2] << stop;
    return ostr.str();
}


std::string
bkgdValueAsString(const openvdb::GridBase::ConstPtr& grid)
{
    std::ostringstream ostr;
    if (grid) {
        const openvdb::TreeBase& tree = grid->baseTree();
        ostr << "background: ";
        openvdb::Metadata::Ptr background = tree.getBackgroundValue();
        if (background) ostr << background->str();
    }
    return ostr.str();
}

/// Print detailed information about the given VDB files.
/// If @a metadata is true, include file-level metadata key, value pairs.
void
printLongListing(const StringVec& filenames)
{
    bool oneFile = (filenames.size() == 1), firstFile = true;

    for (size_t i = 0, N = filenames.size(); i < N; ++i, firstFile = false) {
        openvdb::io::File file(filenames[i]);
        std::string version;
        openvdb::GridPtrVecPtr grids;
        openvdb::MetaMap::Ptr meta;
        try {
            file.open();
            grids = file.getGrids();
            meta = file.getMetadata();
            version = file.version();
            file.close();
        } catch (openvdb::Exception& e) {
            OPENVDB_LOG_ERROR(e.what() << " (" << filenames[i] << ")");
        }
        if (!grids) continue;

        if (!oneFile) {
            if (!firstFile) {
                std::cout << "\n" << std::string(40, '-') << "\n\n";
            }
            std::cout << filenames[i] << "\n\n";
        }

        // Print file-level metadata.
        std::cout << "VDB version: " << version << "\n";
        if (meta) {
            std::string str = meta->str();
            if (!str.empty()) std::cout << str << "\n";
        }
        std::cout << "\n";

        PointStats pointStats;

        // For each grid in the file...
        bool firstGrid = true;
        for (openvdb::GridPtrVec::const_iterator it = grids->begin(); it != grids->end(); ++it) {
            const openvdb::GridBase::ConstPtr grid = *it;
            if (grid) {
                if (!firstGrid) std::cout << "\n\n";
                std::cout << "Name: " << grid->getName() << std::endl;
                grid->print(std::cout, /*verboseLevel=*/11);
                firstGrid = false;
            }

            // Inspect points if this is a point grid
            if (openvdb::GridBase::grid<openvdb::points::PointDataGrid>(grid))
                PointStats::inspectPoints(grid, pointStats);
        }

        PointStats::printPointStats(pointStats);

    }
}


/// Print condensed information about the given VDB files.
/// If @a metadata is true, include file- and grid-level metadata.
void
printShortListing(const StringVec& filenames, bool metadata)
{
    bool oneFile = (filenames.size() == 1), firstFile = true;

    for (size_t i = 0, N = filenames.size(); i < N; ++i, firstFile = false) {
        const std::string
            indent(oneFile ? "": INDENT),
            indent2(indent + INDENT);

        if (!oneFile) {
            if (metadata && !firstFile) std::cout << "\n";
            std::cout << filenames[i] << ":\n";
        }

        openvdb::GridPtrVecPtr grids;
        openvdb::MetaMap::Ptr meta;

        openvdb::io::File file(filenames[i]);
        try {
            file.open();
            grids = file.getGrids();
            meta = file.getMetadata();
            file.close();
        } catch (openvdb::Exception& e) {
            OPENVDB_LOG_ERROR(e.what() << " (" << filenames[i] << ")");
        }
        if (!grids) continue;

        if (metadata) {
            // Print file-level metadata.
            std::string str = meta->str(indent);
            if (!str.empty()) std::cout << str << "\n";
        }

        // For each grid in the file...
        for (openvdb::GridPtrVec::const_iterator it = grids->begin(); it != grids->end(); ++it) {
            const openvdb::GridBase::ConstPtr grid = *it;
            if (!grid) continue;

            // Print the grid name and its voxel value datatype.
            std::cout << indent << std::left << std::setw(11) << grid->getName()
                << " " << std::right << std::setw(6) << grid->valueType();

            // Print the grid's bounding box and dimensions.
            openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
            std::string
                boxStr = coordAsString(bbox.min(), ",", "(", ")") + "->" +
                         coordAsString(bbox.max(), ",", "(", ")"),
                dimStr = coordAsString(bbox.extents(), "x", "", "");
            boxStr += std::string(
                std::max(1, int(40 - boxStr.size() - dimStr.size())), ' ') + dimStr;
            std::cout << " " << std::left << std::setw(40) << boxStr;

            // Print the number of active voxels.
            std::cout << "  " << std::right << std::setw(8)
                << sizeAsString(grid->activeVoxelCount(), "Vox");

            // Print the grid's in-core size, in bytes.
            std::cout << " " << std::right << std::setw(6) << bytesAsString(grid->memUsage());

            std::cout << std::endl;

            // Print grid-specific metadata.
            if (metadata) {
                // Print background value.
                std::string str = bkgdValueAsString(grid);
                if (!str.empty()) {
                    std::cout << indent2 << str << "\n";
                }
                // Print local and world transforms.
                grid->transform().print(std::cout, indent2);
                // Print custom metadata.
                str = grid->str(indent2);
                if (!str.empty()) std::cout << str << "\n";
                std::cout << std::flush;
            }
        }
    }
}

} // unnamed namespace


int
main(int argc, char *argv[])
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE
    gProgName = argv[0];
    if (const char* ptr = ::strrchr(gProgName, '/')) gProgName = ptr + 1;
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE

    int exitStatus = EXIT_SUCCESS;

    if (argc == 1) usage();

    openvdb::logging::initialize(argc, argv);

    bool stats = false, metadata = false, version = false;
    StringVec filenames;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-m" || arg == "-metadata") {
                metadata = true;
            } else if (arg == "-l" || arg == "-stats") {
                stats = true;
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                usage(EXIT_SUCCESS);
            } else if (arg == "-version" || arg == "--version") {
                version = true;
            } else {
                OPENVDB_LOG_FATAL("\"" << arg << "\" is not a valid option");
                usage();
            }
        } else if (!arg.empty()) {
            filenames.push_back(arg);
        }
    }

    if (version) {
        std::cout << "OpenVDB library version: "
            << openvdb::getLibraryAbiVersionString() << "\n";
        std::cout << "OpenVDB file format version: "
            << openvdb::OPENVDB_FILE_VERSION << std::endl;
        if (filenames.empty()) return EXIT_SUCCESS;
    }

    if (filenames.empty()) {
        OPENVDB_LOG_FATAL("expected one or more OpenVDB files");
        usage();
    }

    try {
        openvdb::initialize();

        /// @todo Remove the following at some point:
        openvdb::Grid<openvdb::tree::Tree4<bool, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<float, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<double, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<int32_t, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<int64_t, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec2i, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec2s, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec2d, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec3i, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec3f, 4, 3, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<openvdb::Vec3d, 4, 3, 3>::Type>::registerGrid();

        if (stats) {
            printLongListing(filenames);
        } else {
            printShortListing(filenames, metadata);
        }
    }
    catch (const std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        exitStatus = EXIT_FAILURE;
    }
    catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
    }

    return exitStatus;
}
