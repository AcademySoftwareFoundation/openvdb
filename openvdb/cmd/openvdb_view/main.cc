///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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

#include <openvdb_viewer/Viewer.h>
#include <iostream>
#include <string>
#include <vector>
#include <exception>
#ifdef DWA_OPENVDB
#include <logging_base/logging.h>
#include <usagetrack.h>
#endif


void
usage(const char* progName, int status)
{
    (status == EXIT_SUCCESS ? std::cout : std::cerr) <<
"Usage: " << progName << " file.vdb [file.vdb ...] [options]\n" <<
"Which: displays OpenVDB grids\n" <<
"Build: library version " << openvdb::getLibraryVersionString() << "\n" << 
"Options:\n" <<
"    -i            print grid info\n" <<
"    -d            print debugging info\n" <<
"Controls:\n" <<
"    Esc                exit\n" <<
"    -> (Right)         show next grid\n" <<
"    <- (Left)          show previous grid\n" <<
"    1                  toggle tree topology view on/off\n" <<
"    2                  toggle surface view on/off\n" <<
"    3                  toggle data view on/off\n" <<
"    G                  (\"geometry\") look at center of geometry\n" <<
"    H                  (\"home\") look at origin\n" <<
"    I                  toggle on-screen grid info on/off\n" <<
"    left mouse         tumble\n" <<
"    right mouse        pan\n" <<
"    mouse wheel        zoom\n" <<
"\n" <<
"    X + wheel          move right cut plane\n" <<
"    Shift + X + wheel  move left cut plane\n" <<
"    Y + wheel          move top cut plane\n" <<
"    Shift + Y + wheel  move bottom cut plane\n" <<
"    Z + wheel          move front cut plane\n" <<
"    Shift + Z + wheel  move back cut plane\n" <<
"    Ctrl + X + wheel   move both X cut planes\n" <<
"    Ctrl + Y + wheel   move both Y cut planes\n" <<
"    Ctrl + Z + wheel   move both Z cut planes\n";
    exit(status);
}


////////////////////////////////////////


int
main(int argc, char *argv[])
{
#ifdef DWA_OPENVDB
    USAGETRACK_report_basic_tool_usage(argc, argv, /*duration=*/0);
    logging_base::configure(argc, argv);
#endif

    const char* progName = argv[0];
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;

    int status = EXIT_SUCCESS;

    try {
        openvdb::initialize();

        bool printInfo = false, printDebugInfo = false;

        // Parse the command line.
        std::vector<std::string> filenames;
        for (int n = 1; n < argc; ++n) {
            std::string str(argv[n]);
            if (str[0] != '-') {
                filenames.push_back(str);
            } else if (str == "-i") {
                printInfo = true;
            } else if (str == "-d") {
                printDebugInfo = true;
            } else if (str == "-h" || str == "--help") {
                usage(progName, EXIT_SUCCESS);
            } else {
                usage(progName, EXIT_FAILURE);
            }
        }

        openvdb_viewer::Viewer viewer = openvdb_viewer::init(progName, /*bg=*/false);

        if (printDebugInfo) {
            std::cout << viewer.getVersionString() << std::endl;
        }

        const size_t numFiles = filenames.size();
        if (numFiles == 0) usage(progName, EXIT_FAILURE);

        openvdb::GridCPtrVec allGrids;

        // Load VDB files.
        std::string indent(numFiles == 1 ? "" : "    ");
        for (size_t n = 0; n < numFiles; ++n) {
            openvdb::io::File file(filenames[n]);
            file.open();

            openvdb::GridPtrVecPtr grids = file.getGrids();
            if (grids->empty()) {
                OPENVDB_LOG_WARN(filenames[n] << " is empty");
                continue;
            }
            allGrids.insert(allGrids.end(), grids->begin(), grids->end());

            if (printInfo) {
                if (numFiles > 1) std::cout << filenames[n] << ":\n";
                for (size_t i = 0; i < grids->size(); ++i) {
                    const std::string name = (*grids)[i]->getName();
                    openvdb::Coord dim = (*grids)[i]->evalActiveVoxelDim();
                    std::cout << indent << (name.empty() ? "<unnamed>" : name)
                        << " (" << dim[0] << " x " << dim[1] << " x " << dim[2]
                        << " voxels)" << std::endl;
                }
            }
        }

        viewer.open();
        viewer.view(allGrids);

        openvdb_viewer::exit();

    } catch (const char* s) {
        OPENVDB_LOG_ERROR(progName << ": " << s);
        status = EXIT_FAILURE;
    } catch (std::exception& e) {
        OPENVDB_LOG_ERROR(progName << ": " << e.what());
        status = EXIT_FAILURE;
    }
    return status;
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
