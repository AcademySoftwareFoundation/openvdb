// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Count.h>
#include <openvdb/util/Formats.h>
#include <openvdb/util/logging.h>


namespace {

using StringVec = std::vector<std::string>;

const char* INDENT = "   ";
const char* gProgName = "";

/// Human-readable attribute value-type and codec (single field; avoids duplicating type vs codec).
std::string
attributeValueAndCodec(const openvdb::NamePair& typePair)
{
    std::ostringstream oss;
    oss << typePair.first;
    if (!typePair.second.empty()) {
        oss << " (" << typePair.second << ")";
    }
    return oss.str();
}


/// Human-readable attribute flags (no hex).
std::string
attributeFlagsReadable(uint8_t flags)
{
    using Attr = openvdb::points::AttributeArray;
    const auto f = flags;
    std::ostringstream oss;
    bool first = true;
    auto append = [&](const char* label) {
        if (!first) oss << ", ";
        first = false;
        oss << label;
    };
    if (f & Attr::TRANSIENT) append("transient");
    if (f & Attr::HIDDEN) append("hidden");
    if (f & Attr::CONSTANTSTRIDE) append("constantstride");
    if (f & Attr::STREAMING) append("streaming");
    if (f & Attr::PARTIALREAD) append("partialread");
    return first ? std::string("none") : oss.str();
}


void
printAttributeRows(std::ostream& os,
    const openvdb::points::PointDataGrid::TreeType::LeafNodeType& leaf,
    const std::string& rowIndent)
{
    using namespace openvdb;
    using namespace openvdb::points;

    const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();
    std::vector<std::pair<size_t, Name>> attrs;
    attrs.reserve(descriptor.map().size());
    for (const auto& nameAndPos : descriptor.map()) {
        attrs.emplace_back(nameAndPos.second, nameAndPos.first);
    }
    std::sort(attrs.begin(), attrs.end(),
        [](const std::pair<size_t, Name>& a, const std::pair<size_t, Name>& b) {
            return a.first < b.first;
        });

    for (const auto& idxAndName : attrs) {
        const size_t idx = idxAndName.first;
        const Name& attrName = idxAndName.second;
        const AttributeArray& array = leaf.constAttributeArray(attrName);
        const NamePair& typePair = descriptor.type(idx);

        os << rowIndent << attrName << ": index=" << idx
            << ", type " << attributeValueAndCodec(typePair)
            << ", uniform=" << (array.isUniform() ? "yes" : "no")
            << ", flags " << attributeFlagsReadable(array.flags())
            << "\n";
    }
}


/// Extra verbose statistics for VDB point grids, aligned with Tree::print formatting.
void
printPointDataDetails(std::ostream& os, const openvdb::points::PointDataGrid& grid)
{
    using namespace openvdb;
    using namespace openvdb::points;

    using TreeT = PointDataGrid::TreeType;
    using LeafT = TreeT::LeafNodeType;
    using Descriptor = AttributeSet::Descriptor;

    struct DescriptorGroup {
        Index64 leafCount{0};
        Coord exampleOrigin;
        bool hasExampleLeaf = false;
    };

    Index64 totalPts = 0, activePts = 0, inactivePts = 0;
    std::map<const Descriptor*, DescriptorGroup> descriptorGroups;

    for (typename TreeT::LeafCIter leafIter = grid.tree().cbeginLeaf(); leafIter; ++leafIter) {
        const LeafT& leaf = *leafIter;
        totalPts += leaf.pointCount();
        activePts += leaf.onPointCount();
        inactivePts += leaf.offPointCount();

        const Descriptor* descPtr = leaf.attributeSet().descriptorPtr().get();
        DescriptorGroup& grp = descriptorGroups[descPtr];
        ++grp.leafCount;
        if (!grp.hasExampleLeaf) {
            grp.exampleOrigin = leaf.origin();
            grp.hasExampleLeaf = true;
        }
    }

    os << "Point data:\n";
    os << "  Total points:                   " << util::formattedInt(totalPts) << "\n";
    os << "  Active points:                  " << util::formattedInt(activePts) << "\n";
    os << "  Inactive points:                " << util::formattedInt(inactivePts) << "\n";

    if (descriptorGroups.empty()) {
        os << "  Descriptor shared by all leaves: n/a\n";
        return;
    }

    const bool descriptorSharedGlobally = (descriptorGroups.size() == 1);

    os << "  Descriptor shared by all leaves: "
        << (descriptorSharedGlobally ? "yes" : "no") << "\n";

    if (!descriptorSharedGlobally) {
        return;
    }

    const Coord origin = descriptorGroups.begin()->second.exampleOrigin;
    const LeafT* sampleLeaf = grid.tree().probeConstLeaf(origin);

    os << "  Attributes:\n";
    if (sampleLeaf == nullptr) {
        os << "    (unavailable)\n";
    } else if (sampleLeaf->attributeSet().size() == 0) {
        os << "    (none)\n";
    } else {
        printAttributeRows(os, *sampleLeaf, "    ");
    }

    std::vector<Name> groupNames;
    if (sampleLeaf != nullptr) {
        for (const auto& groupEntry : sampleLeaf->attributeSet().descriptor().groupMap()) {
            groupNames.push_back(groupEntry.first);
        }
        std::sort(groupNames.begin(), groupNames.end());
    }

    os << "  Groups:\n";
    if (sampleLeaf == nullptr) {
        os << "    (unavailable)\n";
    } else if (groupNames.empty()) {
        os << "    (none)\n";
    } else {
        for (const Name& groupName : groupNames) {
            Index64 groupPts = 0;
            for (typename TreeT::LeafCIter leafIter = grid.tree().cbeginLeaf(); leafIter; ++leafIter) {
                groupPts += leafIter->groupPointCount(groupName);
            }
            os << "    " << groupName << ": " << util::formattedInt(groupPts) << "\n";
        }
    }
}

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

        // For each grid in the file...
        bool firstGrid = true;
        for (openvdb::GridPtrVec::const_iterator it = grids->begin(); it != grids->end(); ++it) {
            if (openvdb::GridBase::ConstPtr grid = *it) {
                if (!firstGrid) std::cout << "\n\n";
                std::cout << "Name: " << grid->getName() << std::endl;
                grid->print(std::cout, /*verboseLevel=*/11);
                if (auto pointsGrid =
                        openvdb::GridBase::grid<openvdb::points::PointDataGrid>(grid)) {
                    printPointDataDetails(std::cout, *pointsGrid);
                }
                firstGrid = false;
            }
        }
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

            // Print the grid's size, in bytes

            // no support for memUsageIfLoaded until ABI >= 10 for points::PointDataGrid types
            using ListT = openvdb::GridTypes;
            grid->apply<ListT>([&](const auto& typed){
                // @todo combine these methods to avoid iterating across the tree twice
                const openvdb::Index64 incore = openvdb::tools::memUsage(typed.tree());
                const openvdb::Index64 total = openvdb::tools::memUsageIfLoaded(typed.tree());

                std::cout << " " << std::right << std::setw(6) << bytesAsString(incore) << " (In Core)";
                std::cout << " " << std::right << std::setw(6) << bytesAsString(total) << " (Total)";
            });

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
        std::terminate();
    }

    return exitStatus;
}
