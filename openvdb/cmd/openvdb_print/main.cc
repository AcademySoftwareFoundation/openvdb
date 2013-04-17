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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <openvdb/openvdb.h>
#ifdef DWA_OPENVDB
#include <arg.h>
#include <logging/logging.h>
#include <except/catchall.h>
#include <pdi.h>
#endif
#ifdef _WIN32
#include <openvdb/port/getopt.c>
#else
#include <unistd.h> // for getopt(), optarg
#endif


namespace {

const char* INDENT = "   ";

typedef std::vector<std::string> StringVec;

StringVec sFilenames;


void
usage(const char* progName)
{
    std::cerr <<
"Usage: " << progName << " in.vdb [in.vdb ...] [options]\n" <<
"Which: prints information about OpenVDB grids\n" <<
"Options:\n" <<
#ifdef DWA_OPENVDB
"    -l, -stats     long printout, including grid statistics\n" <<
"    -m, -metadata  print per-file and per-grid metadata\n";
#else
"    -l  long printout, including grid statistics\n" <<
"    -m  print per-file and per-grid metadata\n";
#endif
    exit(EXIT_FAILURE);
}


std::string
sizeAsString(openvdb::Index64 n, const std::string& units)
{
    std::ostringstream ostr;
    ostr << std::setprecision(3);
    if (n < 1000) {
        ostr << n;
    } else if (n < 1000000) {
        ostr << (n / 1.0e3) << "K";
    } else if (n < 1000000000) {
        ostr << (n / 1.0e6) << "M";
    } else {
        ostr << (n / 1.0e9) << "G";
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
        ostr << (n / double(uint64_t(1) << 30)) << "GB";
    } else if (n >> 20) {
        ostr << (n / double(uint64_t(1) << 20)) << "MB";
    } else if (n >> 10) {
        ostr << (n / double(uint64_t(1) << 10)) << "KB";
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


/// Return a string representation of the given metadata key, value pairs
std::string
metadataAsString(
    const openvdb::MetaMap::ConstMetaIterator& begin,
    const openvdb::MetaMap::ConstMetaIterator& end,
    const std::string& indent = "")
{
    std::ostringstream ostr;
    char sep[2] = { 0, 0 };
    for (openvdb::MetaMap::ConstMetaIterator it = begin; it != end; ++it) {
        ostr << sep << indent << it->first;
        if (it->second) {
            const std::string value = it->second->str();
            if (!value.empty()) ostr << ": " << value;
        }
        sep[0] = '\n';
    }
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
            std::string str = metadataAsString(meta->beginMeta(), meta->endMeta());
            if (!str.empty()) std::cout << str << "\n";
        }
        std::cout << "\n";

        // For each grid in the file...
        bool firstGrid = true;
        for (openvdb::GridPtrVec::const_iterator it = grids->begin(); it != grids->end(); ++it) {
            if (openvdb::GridBase::ConstPtr grid = *it) {
                if (!firstGrid) std::cout << "\n\n";
                std::cout << "Name: " << grid->getName() << std::endl;
                grid->print(std::cout, /*verboseLevel=*/3);
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
            std::string str = metadataAsString(meta->beginMeta(), meta->endMeta(), indent);
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
                boxStr = coordAsString(bbox.min()," ") + "  " + coordAsString(bbox.max()," "),
                dimStr = coordAsString(bbox.extents(), "x");
            boxStr += std::string(std::max<int>(1,
                40 - boxStr.size() - dimStr.size()), ' ') + dimStr;
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
                str = metadataAsString(grid->beginMeta(), grid->endMeta(), indent2);
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
    int retcode = EXIT_SUCCESS;

#ifdef DWA_OPENVDB
    logging::configure(argc, argv);

    const char* progName = PDI_get_program_name();
#else
    const char* progName = argv[0];
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;
#endif

    if (argc == 1) usage(progName);

    struct Local {
        static void handleFilenames(int argc, const char* argv[]) {
            for (int i = 0; i < argc; ++i) {
                if (argv[i] && argv[i][0]) sFilenames.push_back(argv[i]);
            }
        }
    };

    int stats = 0, metadata = 0;
#ifdef DWA_OPENVDB
    if (ARG_get(argc, argv,
        "", ARG_GET_SUBR(&Local::handleFilenames), "VDB files",
        "-m", ARG_GET_FLAG(&metadata), "print metadata",
        "-metadata", ARG_GET_FLAG(&metadata), "print metadata",
        "-l", ARG_GET_FLAG(&stats), "print grid statistics",
        "-stats", ARG_GET_FLAG(&stats), "print grid statistics",
        ARG_NULL) < 0)
    {
        usage(progName);
    }
#else
    int c = -1;
    while ((c = getopt(argc, argv, "lm")) != -1) {
        switch (c) {
            case 'l': stats = 1; break;
            case 'm': metadata = 1; break;
            default: usage(progName); break;
        }
    }
    Local::handleFilenames(
        argc - ::optind, const_cast<const char**>(argv) + ::optind);
#endif

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
            printLongListing(sFilenames);
        } else {
            printShortListing(sFilenames, metadata);
        }
    }
#ifdef DWA_OPENVDB
    CATCH_ALL(retcode);
#else
    catch (const std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        retcode = EXIT_FAILURE;
    }
    catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
        std::unexpected();
    }
#endif

    return retcode;
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
