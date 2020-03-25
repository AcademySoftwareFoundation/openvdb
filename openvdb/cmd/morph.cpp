
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/util/CpuTimer.h>

enum Mode {
    dilate_face, dilate_face_edge, dilate_face_edge_vert,
    erode_face, erode_face_edge, erode_face_edge_vert
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// When enabled, uses old methods - disable to use new methods
//#define OLD_MORPH
using GridType = openvdb::FloatGrid;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

#ifdef OLD_MORPH
#pragma message("Building with OLD Morphology ops")
#include <openvdb/tools/Morphology.h>
#else
#pragma message("Building with NEW Morphology ops")
#include "Morphology.h"
#endif

inline openvdb::tools::NearestNeighbors nnFromMode(const Mode& mode)
{
    if (mode == dilate_face || mode == erode_face) return openvdb::tools::NN_FACE;
    if (mode == dilate_face_edge)      return openvdb::tools::NN_FACE_EDGE;
    if (mode == dilate_face_edge_vert) return openvdb::tools::NN_FACE_EDGE_VERTEX;
#ifndef OLD_MORPH
    if (mode == erode_face_edge)      return openvdb::tools::NN_FACE_EDGE;
    if (mode == erode_face_edge_vert) return openvdb::tools::NN_FACE_EDGE_VERTEX;
#endif
    throw std::runtime_error("bad mode");
}

inline std::string modestr(const Mode& mode)
{
    if (mode == dilate_face)           return "DILATE NN_FACE";
    if (mode == dilate_face_edge)      return "DILATE NN_FACE_EDGE";
    if (mode == dilate_face_edge_vert) return "DILATE NN_FACE_EDGE_VERTEX";
    if (mode == erode_face)            return "ERODE NN_FACE";
    if (mode == erode_face_edge)       return "ERODE NN_FACE_EDGE";
    if (mode == erode_face_edge_vert)  return "ERODE NN_FACE_EDGE_VERTEX";
    throw std::runtime_error("bad mode");
}


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

int main(int argc,char** argv) {

    auto invalid = [](const std::string& mesg) {
        std::cerr << mesg << std::endl;
        std::cerr << "./bin <vdb_to_morph> <exp> "<< std::endl;
        std::cerr << "where <exp> is in the form [0-9]*[d|e][6|18|26]. e.g. \"5d26\"" << std::endl;
        exit(EXIT_FAILURE);
    };

    if (argc < 2) {
        invalid("no file");
    }

    const std::string file(argv[1]);
    const std::string modeStr(argc > 2 ? argv[2] : "5d6");

    const size_t iter = std::atoi(modeStr.c_str());
    if (iter == 0) invalid("bad iterations");

    const std::string tmps(std::to_string(iter));
    const char d = modeStr[tmps.length()];
    if (d != 'd' && d != 'e') invalid("bad dilation/erosion mode");
    const bool dilate = d == 'd';

    const size_t neighbours = std::atoi(modeStr.c_str() + tmps.length() + 1);
    if (neighbours != 6 && neighbours != 18 && neighbours != 26) invalid("bad neighbour string");

    Mode mode;
    if (dilate) {
        if (neighbours == 6)  mode = dilate_face;
        if (neighbours == 18) mode = dilate_face_edge;
        if (neighbours == 26) mode = dilate_face_edge_vert;
    }
    else {
        if (neighbours == 6)  mode = erode_face;
        if (neighbours == 18) mode = erode_face_edge;
        if (neighbours == 26) mode = erode_face_edge_vert;
    }

#ifdef OLD_MORPH
    std::cerr << "Running OLD Morphology " << modestr(mode) << " : " << iter << std::endl;
#else
    std::cerr << "Running NEW Morpholohy " << modestr(mode) << " : " << iter << std::endl;
#endif

    openvdb::initialize();
    openvdb::io::File in(file);
    in.open(false);
    GridType::Ptr test =
        openvdb::StaticPtrCast<GridType>(in.readGrid("surface"));
    test->tree().voxelizeActiveTiles();

    const openvdb::tools::NearestNeighbors nn = nnFromMode(mode);
    GridType::Ptr tmp;
    double t = 0;

    for (size_t i = 0; i < 100; ++i) {
        tmp = test->deepCopy();

        openvdb::util::CpuTimer timer;

#ifdef OLD_MORPH
        if (dilate) {
            openvdb::tools::dilateActiveValues(tmp->tree(), iter, nn, openvdb::tools::IGNORE_TILES);
        }
        else {
            openvdb::tools::erodeVoxels(tmp->tree(), iter, nn);
        }
#else
        openvdb::tools::Morphology<GridType::TreeType> morph(tmp->tree());
        if (dilate) morph.dilateVoxels(iter, nn);
        else        morph.erodeVoxels(iter, nn);
#endif

        t += timer.milliseconds();
    }

    std::cerr << "Total   : " << t << std::endl;
    std::cerr << "Average : " << (t/100.0) << std::endl;
    return 0;
}
