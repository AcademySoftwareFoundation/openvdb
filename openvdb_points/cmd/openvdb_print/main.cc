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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <openvdb/openvdb.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointCount.h>

namespace {

typedef std::vector<std::string> StringVec;

const char* INDENT = "   ";
const char* gProgName = "";


void
usage(int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
"Usage: " << gProgName << " in.vdb [in.vdb ...] [options]\n" <<
"Which: prints information about OpenVDB (and OpenVDB Points) grids\n" <<
"Options:\n" <<
"    -l, -stats     long printout, including grid statistics\n" <<
"    -m, -metadata  print per-file and per-grid metadata\n";
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
coordAsString(const openvdb::Coord ijk, const std::string& sep)
{
    std::ostringstream ostr;
    ostr << ijk[0] << sep << ijk[1] << sep << ijk[2];
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

// some common typedefs
typedef openvdb::tools::PointDataTree PointDataTree;
typedef openvdb::tools::PointDataGrid PointDataGrid;
typedef openvdb::tools::AttributeSet AttributeSet;

std::vector<std::string>
attributesAsString(const PointDataTree::LeafCIter& iter)
{
    assert(iter);

    std::vector<std::string> strs;

    std::ostringstream ostr;

    const AttributeSet& attributeSet = iter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();

    typedef AttributeSet::Descriptor::NameToPosMap NameToPosMap;

    const NameToPosMap map = descriptor.map();

    int index = 0;

    for (NameToPosMap::const_iterator it = map.begin(), itEnd = map.end();
        it != itEnd; ++it) {
        const std::string& name = it->first;
        const size_t pos = it->second;

        assert(pos != AttributeSet::INVALID_POS);

        const openvdb::NamePair& type = descriptor.type(pos);
        const openvdb::Name& valueType = type.first;
        const openvdb::Name& codec = type.second;

        ostr.str("");

        ostr << "attribute[" << index++ << "]: " << name <<
            " (value_type: " << valueType << ", compression: " << codec << ")";

        strs.push_back(ostr.str());
    }

    return strs;
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
            std::cout << indent << std::left << std::setw(15) << grid->getName()
                << " " << std::right << std::setw(13) << grid->valueType();

            // Print the grid's bounding box and dimensions.
            openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
            std::string
                boxStr = coordAsString(bbox.min()," ") + "  " + coordAsString(bbox.max()," "),
                dimStr = coordAsString(bbox.extents(), "x");
            boxStr += std::string(
                std::max(1, int(40 - boxStr.size() - dimStr.size())), ' ') + dimStr;
            std::cout << " " << std::left << std::setw(40) << boxStr;

            // Print the number of active voxels.
            std::cout << "  " << std::right << std::setw(8)
                << sizeAsString(grid->activeVoxelCount(), "Vox");

            // Print the number of points (if a PointDataGrid)
            if (grid->isType<PointDataGrid>()) {
                PointDataGrid::ConstPtr pointDataGrid = openvdb::gridConstPtrCast<PointDataGrid>(grid);

                std::cout << "  " << std::right << std::setw(8)
                    << sizeAsString(pointCount(pointDataGrid->tree()), "Pts");
            }
            else {
                std::cout << "  " << std::right << std::setw(8)
                    << " ";
            }

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

                // if PointDataGrid, print attributes
                // @todo: move this code into a print function on the LeafNode itself

                if (grid->isType<PointDataGrid>()) {
                    PointDataGrid::ConstPtr pointDataGrid = openvdb::gridConstPtrCast<PointDataGrid>(grid);

                    PointDataTree::LeafCIter iter = pointDataGrid->tree().cbeginLeaf();

                    if (iter) {
                        std::vector<std::string> strs = attributesAsString(iter);
                        for (std::vector<std::string>::const_iterator strIt = strs.begin(),
                            strItEnd = strs.end(); strIt != strItEnd; ++strIt) {
                            std::string str = *strIt;
                            if (!str.empty())   std::cout << indent2 << str << "\n";
                        }
                    }
                }

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

    bool stats = false, metadata = false;
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
            } else {
                std::cerr << gProgName << ": \"" << arg << "\" is not a valid option\n";
                usage();
            }
        } else if (!arg.empty()) {
            filenames.push_back(arg);
        }
    }
    if (filenames.empty()) {
        std::cerr << gProgName << ": expected one or more OpenVDB files\n";
        usage();
    }

    try {
        openvdb::initialize();
        openvdb::points::initialize();

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
        std::unexpected();
    }

    return exitStatus;
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
