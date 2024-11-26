// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Geometry.h
///
/// @brief Class that encapsulates (explicit) geometry, i.e. vertices/points,
///        triangles and quads. It is used to represent points and polygon meshes
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_GEOMETRY_HAS_BEEN_INCLUDED
#define VDB_TOOL_GEOMETRY_HAS_BEEN_INCLUDED

#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib> // for std::malloc and std::free

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/util/Assert.h>

#ifdef VDB_TOOL_USE_NANO
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#endif

#ifdef VDB_TOOL_USE_ABC
#include <Alembic/Abc/TypedPropertyTraits.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/Util/All.h>
#endif

#ifdef VDB_TOOL_USE_PDAL
#include "pdal/pdal.hpp"
#include "pdal/PipelineManager.hpp"
#include "pdal/PipelineReaderJSON.hpp"
#include "pdal/util/FileUtils.hpp"
#include <sstream>
#endif

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "Util.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

#define MY_CLEAN_VERSION

/// @brief Class that encapsulates (explicit) geometry, i.e. vertices/points,
///        triangles and quads. It is used to represent points and polygon meshes
class Geometry
{
public:

    using PosT  = Vec3f;
    using BBoxT = math::BBox<PosT>;
    using Ptr = std::shared_ptr<Geometry>;
    struct Header;

    Geometry() = default;
    ~Geometry() = default;

    Geometry(const Geometry&) = delete;// disallow copy construction
    Geometry(Geometry&&) = delete;// disallow move construction
    Geometry& operator=(const Geometry&) = delete;// disallow assignment
    Geometry& operator=(Geometry&&) = delete;// disallow move assignment

    inline Ptr copyGeom() const;

    const std::vector<Vec3s>& vtx() const  { return mVtx; }
    const std::vector<Vec3I>& tri() const  { return mTri; }
    const std::vector<Vec4I>& quad() const { return mQuad; }
    const std::vector<Vec3s>& rgb() const  { return mRGB; }

    std::vector<Vec3s>& vtx()  { return mVtx; }
    std::vector<Vec3I>& tri()  { return mTri; }
    std::vector<Vec4I>& quad() { return mQuad; }
    std::vector<Vec3s>& rgb()  { return mRGB; }

    const BBoxT& bbox() const;

    void clear();

    // Reads all the vertices in the file and treats them as Geometry
    void write(const std::string &fileName) const;
    void writeOBJ(const std::string &fileName) const;
    void writeOFF(const std::string &fileName) const;
    void writePLY(const std::string &fileName, bool binary = true) const;
    void writeSTL(const std::string &fileName) const;
    void writeGEO(const std::string &fileName) const;
    void writeABC(const std::string &fileName) const;

    void writeOBJ(std::ostream &os) const;
    void writeOFF(std::ostream &os) const;
    void writePLY(std::ostream &os, bool binary = true) const;
    void writeSTL(std::ostream &os) const;

    void read(const std::string &fileName);
    void readOBJ(const std::string &fileName);
    void readOFF(const std::string &fileName);
    void readPLY(const std::string &fileName);
    void readSTL(const std::string &fileName);
    void readPTS(const std::string &fileName);
    void readGEO(const std::string &fileName);
    void readABC(const std::string &fileName);
    void readPDAL(const std::string &fileName);
    void readVDB(const std::string &fileName);
    void readNVDB(const std::string &fileName);

    void readOBJ(std::istream &is);
    void readOFF(std::istream &is);
    void readPLY(std::istream &is);

    size_t vtxCount() const { return mVtx.size(); }
    size_t triCount() const { return mTri.size(); }
    size_t quadCount() const { return mQuad.size(); }
    size_t polyCount() const { return mTri.size() + mQuad.size(); }

    inline void transform(const math::Transform &xform);

    bool isEmpty() const { return mVtx.empty() && mTri.empty() && mQuad.empty(); }
    bool isPoints() const { return !mVtx.empty() && mTri.empty() && mQuad.empty(); }
    bool isMesh() const { return !mVtx.empty() && (!mTri.empty() || !mQuad.empty()); }

    const std::string getName() const { return mName; }
    void setName(const std::string &name) { mName = name; }

    void print(size_t n = 0, std::ostream& os = std::cout) const;

    size_t write(std::ostream &os) const;
    size_t read(std::istream &is);

private:

    std::vector<PosT>  mVtx;
    std::vector<Vec3I> mTri;
    std::vector<Vec4I> mQuad;
    std::vector<Vec3s> mRGB;
    mutable BBoxT      mBBox;
    std::string        mName;

};// Geometry class

struct Geometry::Header
{
    const static uint64_t sMagic = 0x7664625f67656f31UL;// "vdb_geo1" in hex
    uint64_t magic, name, vtx, tri, quad;
    Header() : magic(sMagic), name(0), vtx(0), tri(0), quad(0) {}
    Header(const Geometry &g) : magic(sMagic), name(g.getName().size()), vtx(g.vtxCount()), tri(g.triCount()), quad(g.quadCount()) {}
    uint64_t size() const { return sizeof(*this) + name + sizeof(BBoxT) + sizeof(PosT)*vtx + sizeof(Vec3I)*tri + sizeof(Vec4I)*quad;}
};// Geometry::Header

size_t Geometry::write(std::ostream &os) const
{
    Header header(*this);// followed by name, bbox, vtx, tri, quad
    os.write((const char*)&header, sizeof(Header));
    os.write(&mName[0], mName.size());
    os.write((const char*)&(this->bbox()), sizeof(BBoxT));
    os.write((const char*)mVtx.data(),  sizeof(PosT)*mVtx.size());
    os.write((const char*)mTri.data(),  sizeof(Vec3I)*mTri.size());
    os.write((const char*)mQuad.data(), sizeof(Vec4I)*mQuad.size());
    return header.size();
}// Geometry::write

size_t Geometry::read(std::istream &is)
{
    Header header;
    if (!is.read((char*)&header, sizeof(Header)) || header.magic != Header::sMagic) {
        is.clear();                 // clear fail and eof bits
        is.seekg(0, std::ios::beg); // rewind to start of stream
        return 0;
    }
    mName.resize(header.name);
    mVtx.resize(header.vtx);
    mTri.resize(header.tri);
    mQuad.resize(header.quad);
    is.read(&mName[0], mName.size());
    is.read((char*)&mBBox, sizeof(BBoxT));
    is.read((char*)mVtx.data(), sizeof(PosT)*mVtx.size());
    is.read((char*)mTri.data(), sizeof(Vec3I)*mTri.size());
    is.read((char*)mQuad.data(), sizeof(Vec4I)*mQuad.size());
    return header.size();
}// Geometry::read

void Geometry::clear()
{
    mName.clear();
    mBBox = BBoxT();//invalidate BBox
    mVtx.clear();
    mTri.clear();
    mQuad.clear();
}// Geometry::clear

const math::BBox<Vec3s>& Geometry::bbox() const
{
    if (mBBox) return mBBox;// early termination if it was already computed
#if 0
    for (auto &p : mVtx) mBBox.expand(p);
#else
    using RangeT = tbb::blocked_range<std::vector<PosT>::const_iterator>;
    struct BBoxOp {
      BBoxT bbox;
      BBoxOp() : bbox() {}
      BBoxOp(BBoxOp& s, tbb::split) : bbox(s.bbox) {}
      void operator()(const RangeT& r) {for (auto p=r.begin(); p!=r.end(); ++p) bbox.expand(*p);}
      void join(BBoxOp& rhs) {bbox.expand(rhs.bbox);}
    } tmp;
    tbb::parallel_reduce(RangeT(mVtx.begin(), mVtx.end(), 1024), tmp);
    mBBox = tmp.bbox;
#endif
    return mBBox;
}// Geometry::bbox

void Geometry::write(const std::string &fileName) const
{
    switch (findFileExt(fileName, {"geo", "obj", "ply", "stl", "abc", "off"})) {
    case 1:
        this->writeGEO(fileName);
        break;
    case 2:
        this->writeOBJ(fileName);
        break;
    case 3:
        this->writePLY(fileName);
        break;
    case 4:
        this->writeSTL(fileName);
        break;
    case 5:
        this->writeABC(fileName);
        break;
    case 6:
        this->writeOFF(fileName);
        break;
    default:
        throw std::invalid_argument("Geometry file \"" + fileName + "\" has an invalid extension");
    }
}// Geometry::write

void Geometry::writePLY(const std::string &fileName, bool binary) const
{
    if (fileName == "stdout.ply") {
        //if (isatty(fileno(stdout))) throw std::invalid_argument("writePLY: stdout is not connected to the terminal!");
        this->writePLY(std::cout, binary);
    } else {
        std::ofstream outfile(fileName, std::ios_base::binary);
        if (!outfile.is_open()) throw std::invalid_argument("Error writing to ply file \""+fileName+"\"");
        this->writePLY(outfile, binary);
    }
}// Geometry::writePLY

void Geometry::writePLY(std::ostream &os, bool binary) const
{
    os << "ply\nformat ";
    if (binary) {
        os << "binary_" << (isLittleEndian() ? "little" : "big") << "_endian 1.0\n";
    } else {
        os << "ascii 1.0\n";
    }
    os << "comment created by vdb_tool" << std::endl;
    os << "element vertex " << mVtx.size() << std::endl;
    os << "property float x\n";
    os << "property float y\n";
    os << "property float z\n";
    os << "element face " << (mTri.size() + mQuad.size())<< std::endl;
    os << "property list uchar int vertex_index\n";
    os << "end_header\n";
    static_assert(sizeof(Vec3s) == 3 * sizeof(float), "Unexpected sizeof(Vec3s)");
    if (binary) {
        os.write((const char *)mVtx.data(), mVtx.size() * 3 * sizeof(float));// write x,y,z vertex coordinates
        auto writeFaces = [](std::ostream &os, const uint32_t *faces, size_t count, uint8_t n) {
            if (count==0) return;
            const int size = 1 + 4*n;
            char *buffer = (char*)std::malloc(count*size), *p = buffer;// uninitialized
            if (buffer==nullptr) throw std::invalid_argument("Geometry::writePLY: failed to allocate buffer");
            for (const uint32_t *f = faces, *e = f + n*count; f!=e; f+=n, p += size) {
                *p = (char)n;
                std::memcpy(p + 1, f, 4*n);
            }
            os.write(buffer, count*size);
            std::free(buffer);
        };
        writeFaces(os, (const uint32_t*)mTri.data(),  mTri.size(),  3);
        writeFaces(os, (const uint32_t*)mQuad.data(), mQuad.size(), 4);
    } else {// ascii
        for (auto &v : mVtx)  os << v[0] << " " << v[1] << " " << v[2] << "\n";
        for (auto &t : mTri)  os << "3 " << t[0] << " " << t[1] << " " << t[2] << "\n";
        for (auto &q : mQuad) os << "4 " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    }
}// Geometry::writePLY

void Geometry::writeOBJ(const std::string &fileName) const
{
    if (fileName=="stdout.obj") {
        //if (isatty(fileno(stdout))) throw std::invalid_argument("writeOBJ: stdout is not connected to the terminal!");
        this->writeOBJ(std::cout);
    } else {
        std::ofstream outfile(fileName);
        if (!outfile.is_open()) throw std::invalid_argument("Error writing to obj file \""+fileName+"\"");
        this->writeOBJ(outfile);
    }
}// Geometry::writeOBJ

void Geometry::writeOBJ(std::ostream &os) const
{
    os << "# obj file created by vdb_tool\n";
    for (auto &v : mVtx)  os << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    for (auto &t : mTri)  os << "f " << t[0]+1 << " " << t[1]+1 << " " << t[2]+1 << "\n";// obj is 1-based
    for (auto &q : mQuad) os << "f " << q[0]+1 << " " << q[1]+1 << " " << q[2]+1 << " " << q[3]+1 << "\n";// obj is 1-based
}// Geometry::writeOBJ

void Geometry::writeOFF(const std::string &fileName) const
{
    if (fileName=="stdout.off") {
        this->writeOFF(std::cout);
    } else {
        std::ofstream outfile(fileName);
        if (!outfile.is_open()) throw std::invalid_argument("Error writing to off file \""+fileName+"\"");
        this->writeOFF(outfile);
    }
}// Geometry::writeOFF

void Geometry::writeOFF(std::ostream &os) const
{
    os << "OFF\n# Created by vdb_tool\n";
    os << mVtx.size() << " " << (mTri.size() + mQuad.size()) << " " << 0 << "\n";
    for (auto &v : mVtx)  os << v[0] << " " << v[1] << " " << v[2] << "\n";
    for (auto &t : mTri)  os << "3 " << t[0] << " " << t[1] << " " << t[2] << "\n";
    for (auto &q : mQuad) os << "4 " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
}// Geometry::writeOFF

void Geometry::writeSTL(const std::string &fileName) const
{
    if (fileName == "stdout.stl") {
        //if (isatty(fileno(stdout))) throw std::invalid_argument("writeSTL: stdout is not connected to the terminal!");
        this->writeSTL(std::cout);
    } else {
        std::ofstream outfile(fileName, std::ios::out | std::ios_base::binary);
        if (!outfile.is_open()) throw std::invalid_argument("Error writing to stl file \""+fileName+"\"");
        this->writeSTL(outfile);
    }
}// Geometry::writeSTL

void Geometry::writeSTL(std::ostream &os) const
{
    if (!isLittleEndian()) throw std::invalid_argument("STL file only supports little endian, but this system is big endian");
    if (!mQuad.empty()) throw std::invalid_argument("STL file only supports triangles");
    uint8_t buffer[80] = {0};// fixed-sized buffer initiated with zeros!
    os.write((const char*)buffer, 80);// write header as zeros
    const uint32_t nTri = static_cast<uint32_t>(mTri.size());
    os.write((const char*)&nTri, 4);
    float *p = 3 + reinterpret_cast<float*>(buffer);// the normal will remain zero
    for (const Vec3I &tri : mTri) {
        float *q = p;
        for (int i=0; i<3; ++i) {
            const PosT &vtx = mVtx[tri[i]];
            *q++ = vtx[0];
            *q++ = vtx[1];
            *q++ = vtx[2];
        }
        os.write((const char*)buffer, 50);
    }
}// Geometry::writeSTL

void Geometry::writeGEO(const std::string &fileName) const
{
    if (fileName == "stdout.geo") {
        //if (isatty(fileno(stdout))) throw std::invalid_argument("writeGEO: stdout is not connected to the terminal!");
        this->write(std::cout);
    } else {
        std::ofstream outfile(fileName, std::ios::out | std::ios_base::binary);
        if (!outfile.is_open()) throw std::invalid_argument("Error writing to geo file \""+fileName+"\"");
        this->write(outfile);
    }
}// Geometry::writeGEO

void Geometry::read(const std::string &fileName)
{
    switch (findFileExt(fileName, {"obj", "ply", "pts", "stl", "abc", "vdb", "nvdb", "geo", "off"})) {
    case 1:
        this->readOBJ(fileName);
        break;
    case 2:
        this->readPLY(fileName);
        break;
    case 3:
        this->readPTS(fileName);
        break;
    case 4:
        this->readSTL(fileName);
        break;
    case 5:
        this->readABC(fileName);
        break;
    case 6:
        this->readVDB(fileName);
        break;
    case 7:
        this->readNVDB(fileName);
        break;
    case 8:
        this->readGEO(fileName);
        break;
    case 9:
        this->readOFF(fileName);
        break;
    default:
#if VDB_TOOL_USE_PDAL
        pdal::StageFactory factory;
        const std::string driver = factory.inferReaderDriver(fileName);
        if (driver != "") {
            this->readPDAL(fileName);
            break;
        }
#endif
        throw std::invalid_argument("Geometry::read: File \""+fileName+"\" has an invalid extension");
        break;
    }
}// Geometry::read

void Geometry::readOBJ(const std::string &fileName)
{
    if (fileName == "stdin.obj") {
        //if (isatty(fileno(stdin))) throw std::invalid_argument("readOBJ: stdin is not connected to the terminal!");
        this->readOBJ(std::cin);
    } else {
        std::ifstream infile(fileName);
        if (!infile.is_open()) throw std::invalid_argument("Error opening Geometry file \""+fileName+"\"");
        this->readOBJ(infile);
    }
}// Geometry::readOBJ

void Geometry::readOBJ(std::istream &is)
{
    Vec3f p;
    std::string line;
    while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::string str;
        iss >> str;
        if (str == "v") {
            iss >> p[0] >> p[1] >> p[2];
            mVtx.push_back(p);
        } else if (str == "f") {
            std::vector<int> v;
            while (iss >> str) {
                v.push_back(std::stoi(str.substr(0, str.find_first_of("/"))));
            }
            switch (v.size()) {
            case 3:
                mTri.emplace_back(v[0] - 1, v[1] - 1, v[2] - 1);// obj is 1-based
                break;
            case 4:
                mQuad.emplace_back(v[0] - 1, v[1] - 1, v[2] - 1, v[3] - 1);// obj is 1-based
                break;
            default:
                throw std::invalid_argument("Geometry::readOBJ: " + std::to_string(v.size()) + "-gons are not supported");
                break;
            }
        }
    }
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readOBJ

void Geometry::readPDAL(const std::string &fileName)
{
 #if VDB_TOOL_USE_PDAL
    if (!pdal::FileUtils::fileExists(fileName)) throw std::invalid_argument("Error opening file \""+fileName+"\"  - it doesn't exist!");

    pdal::StageFactory factory;
    std::string type = factory.inferReaderDriver(fileName);
    std::string pipelineJson = R"({
        "pipeline" : [
            {
                "type" : ")" + type + R"(",
                "filename" : ")" + fileName + R"("
            }
        ]
    })";

    Vec3f p;
    Vec3s rgb;
    try {
        pdal::PipelineManager manager;
        std::stringstream s(pipelineJson);
        manager.readPipeline(s);
        manager.execute(pdal::ExecMode::Standard);

        for (const std::shared_ptr<pdal::PointView>& view : manager.views()) {
            bool hasColor = false;
            if (view->hasDim(pdal::Dimension::Id::Red) && view->hasDim(pdal::Dimension::Id::Green) && view->hasDim(pdal::Dimension::Id::Blue))
                hasColor = true;
            for (const pdal::PointRef& point : *view) {
                p[0] = point.getFieldAs<float>(pdal::Dimension::Id::X);
                p[1] = point.getFieldAs<float>(pdal::Dimension::Id::Y);
                p[2] = point.getFieldAs<float>(pdal::Dimension::Id::Z);
                mVtx.push_back(p);
                if (hasColor) {
                    rgb[0] = point.getFieldAs<float>(pdal::Dimension::Id::Red);
                    rgb[1] = point.getFieldAs<float>(pdal::Dimension::Id::Green);
                    rgb[2] = point.getFieldAs<float>(pdal::Dimension::Id::Blue);
                    mRGB.push_back(rgb);
                }
            }
        }

    }
    catch (const pdal::pdal_error& e) {
        throw std::runtime_error("PDAL failed: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Reading file failed: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("Cannot read file \"" + fileName + "\".  PDAL support is not enabled in this build, please recompile with PDAL support");
#endif
    mBBox = BBoxT(); //invalidate BBox
}// Geometry::readPDAL

void Geometry::readOFF(const std::string &fileName)
{
    if (fileName == "stdin.off") {
        this->readOFF(std::cin);
    } else {
        std::ifstream infile(fileName);
        if (!infile.is_open()) throw std::invalid_argument("Error opening Geometry file \""+fileName+"\"");
        this->readOFF(infile);
    }
}// Geometry::readOFF

void Geometry::readOFF(std::istream &is)
{
    // read header
    std::string line;
    if (!std::getline(is, line) || line != "OFF") {
        throw std::invalid_argument("Geometry::readOFF: expected header \"OFF\" but read \"" + line + "\"");
    }

    // read vertex and face counts
    size_t vtxCount=0, faceCount=0, edgeCount=0, nGon=0;
    while (vtxCount == 0 && std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        iss >> vtxCount >> faceCount >> edgeCount;
    }

    // read vertices
    Vec3f p;
    vtxCount += mVtx.size();
    while (mVtx.size() < vtxCount && std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        iss >> p[0] >> p[1] >> p[2];
        mVtx.push_back(p);
    }

    // read faces
    int f[4];
    faceCount += mTri.size() + mQuad.size();
    while (mTri.size() + mQuad.size() < faceCount && std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        iss >> nGon;
        if (nGon == 3) {
            iss >> f[0] >> f[1] >> f[2];
            mTri.emplace_back(f[0],f[1],f[2]);
        } else if (nGon == 4) {
            iss >> f[0] >> f[1] >> f[2] >> f[3];
            mQuad.emplace_back(f[0],f[1],f[2],f[3]);
        } else {
            throw std::invalid_argument("Geometry::readOFF: " + std::to_string(nGon) + "-gons are not supported");
        }
    }
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readOFF

void Geometry::readPLY(const std::string &fileName)
{
    if (fileName == "stdin.ply") {
        //if (isatty(fileno(stdin))) throw std::invalid_argument("readPLY: stdin is not connected to the terminal!");
        this->readPLY(std::cin);
    } else {
        std::ifstream infile(fileName, std::ios::in | std::ios_base::binary);
        if (!infile.is_open()) throw std::invalid_argument("Error opening ply file \""+fileName+"\"");
        this->readPLY(infile);
    }
}// Geometry::readPLY

void Geometry::readPLY(std::istream &is)
{
    auto tokenize_line = [&is]() {
        std::string line, token;
        std::getline(is, line);
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        while (iss >> token) tokens.push_back(token);
        if (tokens.empty()) tokens.emplace_back("comment empty");
        return tokens;// move semantics
    };
    auto tokens = tokenize_line();
    auto test = [&tokens](int i, std::vector<std::string> str) {
        if (i >= static_cast<int>(tokens.size())) return false;
        for (auto &s : str) if (tokens[i] == s) return true;
        return false;
    };
    auto error = [&tokens](const std::string &msg){
        std::cerr << "Tokens: \"";
        for (auto &t : tokens) std::cerr << t << " ";
        std::cerr << "\"\n";
        throw std::invalid_argument(msg);
    };
    auto sizeOf = [test, error](int i){
        if ( test(i, {"float", "float32", "int", "int32"}) ) return 4;
        if ( test(i, {"double", "float64"}) ) return 8;
        if ( test(i, {"int16", "uint16"}) )   return 2;
        if ( test(i, {"uchar", "int8"}) )     return 1;
        error("vdb_tool::readPLY: unsupported type");
        return 0;
    };

    // check header
    if (!test(0, {"ply"})) error("vdb_tool::readPLY: not a ply file");

    // check file format
    int format = -1;// 0 is ascii, 1 is little endian and 2 is big endian
    tokens = tokenize_line();
    if (!(test(0, {"format"}) && test(2, {"1.0"})) ) {
        error("vdb_tool::readPLY: expected format version 1.0");
    } else if (test(1, {"ascii"})) {
        format = 0;
    } else if (test(1, {"binary_little_endian"})) {
        format = 1;
    } else if (test(1, {"binary_big_endian"})) {
        format = 2;
    } else {
        error("vdb_tool::readPLY: invalid format");
    }
    const bool reverseBytes = format && format != (isLittleEndian() ? 1 : 2);
    // header: https://www.mathworks.com/help/vision/ug/the-ply-format.html
    size_t vtxCount = 0, faceCount = 0;
    int vtxStride=0, vtxProps=0;// byte size of all vtx properties, number of vertex properties
    struct Triplet {int offset, id, size;} xyz[3];// byte offset, id#, byte size
    struct Skip {int count, bytes;} faceSkip[2]={{0,0},{0,0}};// head, {faces}, tail

    // parse header with vertex, face and property information
    tokens = tokenize_line();
    bool run = true;
    while(run) {
        if ( test(0, {"element"}) ) {
            if ( test(1, {"vertex"}) ) {
                vtxCount = std::stoll(tokens[2]);
                const std::string axis[3] = {"x", "y", "z"};
                while(true) {
                    tokens = tokenize_line();
                    if ( test(0, {"end_header"}) ) {
                        run = false;
                        break;
                    } else if ( test(0, {"element"}) ) {
                        break;
                    } else if ( test(0, {"property"}) ) {
                        Triplet t{vtxStride, vtxProps++, sizeOf(1)};
                        for (int i=0; i<3; ++i) if (test(2, {axis[i]})) xyz[i] = t;
                        vtxStride += t.size;
                    }
                }
                for (int i=0; i<3; ++i) if (xyz[i].size!=4 && xyz[i].size!=8) error("vdb_tool::readPLY: missing "+axis[i]+
                                                                                    " vertex coordinates or unsupported size "+std::to_string(xyz[i].size));
            } else if ( test(1, {"face"}) ) {
                faceCount = std::stoll(tokens[2]);
                int n = 0;// 0 is head and 1 is tail
                while (true) {
                    tokens = tokenize_line();
                    if ( test(0, {"end_header"}) ) {
                        run = false;
                        break;
                    } else if (test(0, {"element"}) ) {
                        break;
                    } else if (test(0, {"property"}) ) {// eg: "property list uchar int vertex_indices"
                        if (test(1, {"list"}) &&// list of vertex ID belonging to a polygon
                            test(2, {"uchar", "uint8"}) &&// size of polygon, e.g. 3 or 4
                            test(3, {"int", "uint", "int32"}) &&// type of vertex id
                            test(4, {"vertex_index", "vertex_indices"}) ) {
                            n = 1;// change from head to tail
                        } else if ( test(1, {"uchar", "uint8"}) ) {// eg: "property uchar intensity"
                            faceSkip[n].count += 1;
                            faceSkip[n].bytes += 1;
                        } else {
                            error("vdb_tool::readPLY: invalid face properties");
                        }
                    }
                }
            } else if ( test(1, {"edge", "material"}) ) {
                while(true) {
                    tokens = tokenize_line();
                    if (test(0, {"end_header"}) ) {
                        run = false;
                        break;
                    } else if (tokens[0] == "element") {
                        break;
                    }
                }
            } else {
                error("vdb_tool::readPLY: invalid element");
            }
        } else if ( test(0, {"comment", "obj_info"}) ) {// eq: "obj_info 3D colored patch boundaries" and "comment author: Paraform"
            tokens = tokenize_line();
        } else {
            error("vdb_tool::readPLY: unexpected entry in header");
        }
    }

    // read vertex coordinates
    mVtx.resize(vtxCount);
    if (format) {// binary
        if (xyz[0].offset==0 && xyz[1].offset==4 && xyz[2].offset==8 && vtxStride==12) {// most common case
            is.read((char *)(mVtx.data()), vtxCount * 3 * sizeof(float));
            if (reverseBytes) for (Vec3f &v : mVtx) swapBytes(&v[0], 3);
        } else {
            char *buffer = static_cast<char*>(std::malloc(vtxCount*vtxStride)), *p = buffer;// uninitialized
            if (buffer==nullptr) throw std::invalid_argument("Geometry::readPLY: failed to allocate buffer");
            is.read(buffer, vtxCount*vtxStride);
            for (Vec3f &vtx : mVtx) {
                for (int i=0; i<3; ++i) {
                    if (xyz[i].size == 4) {
                        float v = *(float*)(p + xyz[i].offset);
                        vtx[i] = reverseBytes ? swapBytes(v) : v;
                    } else {
                        double v = *(double*)(p + xyz[i].offset);
                        vtx[i] = float(reverseBytes ? swapBytes(v) : v);
                    }
                }
                p += vtxStride;
            }
            std::free(buffer);
        }

    } else {// ascii vertices
        for (auto &v : mVtx) {
            tokens = tokenize_line();
            if (int(tokens.size()) != vtxProps) error("vdb_tool::readPLY: error reading ascii vertex coordinates");
            for (int i = 0; i<3; ++i) v[i] = std::stof(tokens[xyz[0].id]);
        }// loop over vertices
    }

    // read polygon vertex lists
    uint32_t vtx[4];
    if (format) {// binary
        char *buffer = static_cast<char*>(std::malloc(faceSkip[0].bytes + 1));// uninitialized
        if (buffer==nullptr) throw std::invalid_argument("Geometry::readPLY: failed to allocate buffer");
        for (size_t i=0; i<faceCount; ++i) {
            is.read(buffer, faceSkip[0].bytes + 1);// polygon size is encoded as a single char
            const unsigned int n = (unsigned int)buffer[faceSkip[0].bytes];// char -> unsigned int
            switch (n) {
            case 3:
                is.read((char*)vtx, 3*sizeof(uint32_t));
                if (reverseBytes) swapBytes(vtx, 3);
                mTri.emplace_back(vtx);
                break;
            case 4:
                is.read((char*)vtx, 4*sizeof(uint32_t));
                if (reverseBytes) swapBytes(vtx, 4);
                mQuad.emplace_back(vtx);
                break;
            default:
                throw std::invalid_argument("Geometry::readPLY: binary " + std::to_string(n) + "-gons are not supported");
                break;
            }
            is.ignore(faceSkip[1].bytes);
        }// loop over polygons
        std::free(buffer);
    } else {// ascii format faces
        for (size_t i=0; i<faceCount; ++i) {
            tokens = tokenize_line();
            const std::string polySize = tokens[faceSkip[0].count];
            const int n = std::stoi(polySize);
            if (n!=3 && n!=4) throw std::invalid_argument("Geometry::readPLY: ascii " + polySize + "-gons are not supported");
            for (int i = 0, j=1+faceSkip[0].count; i<n; ++i, ++j) vtx[i] = static_cast<uint32_t>(std::stoll(tokens[j]));
            if (n==3) {
                mTri.emplace_back(vtx);
            } else {
                mQuad.emplace_back(vtx);
            }
        }// loop over polygons
    }
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readPLY

void Geometry::readGEO(const std::string &fileName)
{
    if (fileName == "stdin.geo") {
        //if (isatty(fileno(stdin))) throw std::invalid_argument("readGEO: stdin is not connected to the terminal!");
        this->read(std::cin);
    } else {
        std::ifstream infile(fileName, std::ios::in | std::ios_base::binary);
        if (!infile.is_open()) throw std::invalid_argument("Error opening geo file \""+fileName+"\"");
        this->read(infile);
    }
}//  Geometry::readGEO

// Read vertices from all PointDataGrids in the specified file
void Geometry::readVDB(const std::string &fileName)
{
    initialize();
    io::File file(fileName);
    file.open();// enables delayed loading by default
    GridPtrVecPtr meta = file.readAllGridMetadata();
    for (auto m : *meta) {
        if (m->isType<points::PointDataGrid>()) {
            auto grid = gridPtrCast<points::PointDataGrid>(file.readGrid(m->getName()));
            OPENVDB_ASSERT(grid);
            size_t n = mVtx.size();
            const auto m = points::pointCount(grid->tree());
            mVtx.resize(n + m);
            for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
                const points::AttributeArray& array = leafIter->constAttributeArray("P");
                points::AttributeHandle<Vec3f> positionHandle(array);
                for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                    Vec3f voxelPosition = positionHandle.get(*indexIter);
                    const Vec3d xyz = indexIter.getCoord().asVec3d();
                    Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);
                    mVtx[n++] = worldPosition;
                }// loop over points in leaf node
            }// loop over leaf nodes
        }// is a PointDataGrid
    }// loop over gids in file
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readVDB

/*
see http://paulbourke.net/dataformats/pts

pts is a very simple ASCII file format for a collection of one or more unordered scans.
Each scan consists of:

n
x1 y1 z1 i1 [r1 g1 b1]
...
xn yn zn in [rn gn bn]

and chunks like the above can be repeated in a file for multiple scans

n - number of points (that immediately follow the line with n)
x,y,z are coords in meters
i is intensity, in range -2048 to +2047
[r g b] is optional color, and if present each part is in the range 0-255
*/
void Geometry::readPTS(const std::string &fileName)
{
    std::ifstream infile(fileName, std::ios::in);
    if (!infile.is_open()) throw std::runtime_error("Error opening particle file \""+fileName+"\"");
    std::string line;
    std::istringstream iss;
    bool readColor = false;
    Vec3s rgb;
    while(std::getline(infile, line)) {
        const size_t n = mVtx.size(), m = std::stoi(line);
        mVtx.resize(n + m);
        for (size_t i=n; i<mVtx.size(); ++i) {
            PosT &p = mVtx[i];
            std::getline(infile, line);
            iss.clear();
            iss.str(line);
            if (!(iss >> p[0] >> p[1] >> p[2])) {;//ignore intensity, r, g, b
                throw std::invalid_argument("Geometry::readPTS: error parsing line: \""+line+"\"");
            }
            if (readColor) {
                if (!(iss >> i) ) { // converting intensity to a multiplier on rgb might be appropriate, but i can't find a good spec for it
                    readColor = false;
                    continue;
                }
                if (!(iss >> rgb[0] >> rgb[1] >> rgb[2])) {
                    readColor = false;
                    continue;
                }
                mRGB.push_back(rgb/255.0);
            }

        }// loop over points
    }// loop over scans
    mBBox = BBoxT();//invalidate BBox
}// readPTS

// Reading ASCII or binary STL file
void Geometry::readSTL(const std::string &fileName)
{
    std::ifstream infile(fileName, std::ios::in | std::ios::binary);
    if (!infile.is_open()) throw std::runtime_error("Geometry::readSTL: Error opening STL file \""+fileName+"\"");
    PosT xyz;
    char buffer[80] = "";// small fixed stack allocated buffer
    if (!infile.read(buffer, 5)) throw std::invalid_argument("Geometry::readSTL: Failed to head header");
    if (strcmp(buffer, "solid") == 0) {//ASCII file
        std::string line;
        std::getline(infile, line);// read rest of the first line, which completes the header
        std::istringstream iss;
        while(std::getline(infile, line)) {
            std::string tmp = trim(line, " ");// remove leading (and trailing) white spaces
            if (tmp.compare(0, 5, "facet")==0) {
                while (std::getline(infile, line) && trim(line, " ").compare(0, 10, "outer loop"));
                int nGone = 0;
                while(std::getline(infile, line)) {// loop over vertices of the facet
                    tmp = trim(line, " ");
                    if (tmp.compare(0, 7, "endloop")==0) break;
                    OPENVDB_ASSERT(tmp.compare(0, 6, "vertex")==0);
                    iss.clear();
                    iss.str(tmp.substr(6));
                    if (iss >> xyz[0] >> xyz[1] >> xyz[2]) {
                        mVtx.push_back(xyz);
                        ++nGone;
                    } else {
                        throw std::invalid_argument("Geometry::readSTL ASCII: error parsing line: \""+line+"\"");
                    }
                }// endloop
                const int vtx = static_cast<int>(mVtx.size()) - 1;
                switch (nGone){
                case 3:
                    mTri.emplace_back(vtx - 2, vtx - 1, vtx);
                    break;
                case 4:
                    mQuad.emplace_back(vtx - 3, vtx - 2, vtx - 1, vtx);
                    break;
                default:
                    throw std::invalid_argument("Geometry::readSTL ASCII: " + std::to_string(nGone)+"-gons are not supported");
                }
            }
        }// loop over lines in file
    } else {// binary file
        if (!isLittleEndian()) throw std::invalid_argument("Geometry::readSTL binary: STL file only supports little endian, but this system is big endian");
        if (!infile.read(buffer, 80 - 5)) throw std::invalid_argument("Geometry::readSTL binary: Failed to head header");
        uint32_t numTri;
        if (!infile.read((char*)&numTri, sizeof(numTri))) throw std::invalid_argument("Geometry::readSTL binary: Failed to read triangle count");
        infile.seekg (0, infile.end);
        if (infile.tellg() != 80 + 4 + 50*numTri) throw std::invalid_argument("Geometry::readSTL binary: Unexpected file size");
        infile.seekg(80 + 4, infile.beg);
        uint32_t vtxBegin = static_cast<uint32_t>(mVtx.size()), triBegin = static_cast<uint32_t>(mTri.size());
        mVtx.resize(vtxBegin + 3*numTri);
        mTri.resize(triBegin +   numTri);
        Vec3f *pV = mVtx.data() + vtxBegin;
        Vec3I *pT = mTri.data() + triBegin;
        for (uint32_t i = 0; i < numTri; ++i) {// loop over triangles
            if (!infile.read(buffer, 50)) throw std::invalid_argument("Geometry::readSTL binary: error reading triangle #"+std::to_string(i));
            const float *p = 3 + reinterpret_cast<const float*>(buffer);// ignore 3 vector components of normal
            for (int j=0; j<3; ++j) {// loop over vertices of triangle
                for (int k=0; k<3; ++k) xyz[k] = *p++;//loop over coordinates of vertex
                *pV++ = xyz;
            }
            *pT++ = Vec3I(vtxBegin, vtxBegin + 1, vtxBegin + 2);
            vtxBegin += 3;
        }
    }// end binary
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readSTL

#ifdef VDB_TOOL_USE_NANO
void Geometry::readNVDB(const std::string &fileName)
{
    auto handle = nanovdb::io::readGrid(fileName);
    auto grid = handle.grid<uint32_t>();
    if (grid == nullptr || !grid->isPointData()) return;
    nanovdb::PointAccessor<nanovdb::Vec3f> acc(*grid);
    const nanovdb::Vec3f *begin = nullptr, *end = nullptr; // iterators over points in a given voxel
    const size_t count = acc.gridPoints(begin, end);
    auto *p = reinterpret_cast<const Vec3s*>(begin);
    size_t n = mVtx.size();
    mVtx.resize(n + count);
    for (size_t i=n; i<mVtx.size(); ++i) mVtx[i] = *p++;// loop over points
    mBBox = BBoxT();//invalidate BBox
}// Geometry::readNVDB
#else
void Geometry::readNVDB(const std::string&)
{
    throw std::runtime_error("NanoVDB support was disabled during compilation!");
}// Geometry::readNVDB
#endif

void Geometry::print(size_t n, std::ostream& os) const
{
    os << "vtx = " << mVtx.size() << ", tri = " << mTri.size() << ", quad = " << mQuad.size()<< ", bbox=" << this->bbox();
    if (size_t m = std::min(n, mVtx.size())) {
        os << std::endl;
        for (size_t i=0; i<m; ++i) {
            os << "vtx[" << i << "] = " << mVtx[i] << std::endl;
        }
    }
    if (size_t m = std::min(n, mTri.size())) {
        os << std::endl;
        for (size_t i=0; i<m; ++i) {
            os << "Tri[" << i << "] = " << mTri[i] << std::endl;
        }
    }
    if (size_t m = std::min(n, mQuad.size())) {
        os << std::endl;
        for (size_t i=0; i<m; ++i) {
            os << "Quad[" << i << "] = " << mQuad[i] << std::endl;
        }
    }
}// Geometry::print

#ifdef VDB_TOOL_USE_ABC

class AlembicReader
{
public:

    struct Context {
        std::string full_name;
        std::vector<std::string> path;
        Alembic::AbcGeom::M44d accumulated_transform;
        std::vector<Alembic::AbcGeom::M44d> transform_stack;
    };

    template <typename PtrT>
    struct Span {
        size_t count = 0;
        PtrT pointer = nullptr;
    };

private:

    //-----------------------------------------------------------------------------
    Context make_context_append_name(const Context &parent_context,
                                     const std::string &name)
    {
        Context context{parent_context};
        context.full_name += "/" + name;
        context.path.push_back(name);
        return context;
    }

    //-----------------------------------------------------------------------------
    void visit_children(Alembic::AbcGeom::IObject parent_object,
                        const Context &parent_context);

    //-----------------------------------------------------------------------------
    void visit_object(Alembic::AbcGeom::IObject object,
                      const Context &parent_context)
    {
        auto context = make_context_append_name(parent_context, object.getName());
        context.transform_stack.push_back(Alembic::AbcGeom::M44d{});
        mObjectVisitor(context);
        visit_children(object, context);
    }

    //-----------------------------------------------------------------------------
    void visit_xform(Alembic::AbcGeom::IXform xform,
                                const Context &parent_context)
    {
        auto context = make_context_append_name(parent_context, xform.getName());
        const auto &schema = xform.getSchema();
        const auto sample = schema.getValue();
        const auto transform = sample.getMatrix();
        // Imath is transposed, uses row_vector * matrix * matrix.
        context.accumulated_transform = transform * context.accumulated_transform;
        context.transform_stack.push_back(transform);
        mXformVisitor(context);
        visit_children(xform, context);
    }

    //-----------------------------------------------------------------------------
    void visit_mesh(Alembic::AbcGeom::IPolyMesh mesh,
                    const Context &parent_context)
    {
        auto context = make_context_append_name(parent_context, mesh.getName());
        context.transform_stack.push_back(Alembic::AbcGeom::M44d{});
        const auto &schema = mesh.getSchema();
        const auto sample = schema.getValue();
        Span<const int32_t*> face_counts;
        Span<const int32_t*> face_indices;
        Span<const Alembic::AbcGeom::V3f*> positions;

        face_counts.count = sample.getFaceCounts()->size();
        face_counts.pointer = sample.getFaceCounts()->get();
        face_indices.count = sample.getFaceIndices()->size();
        face_indices.pointer = sample.getFaceIndices()->get();
        positions.count = sample.getPositions()->size();
        positions.pointer = sample.getPositions()->get();

        mMeshVisitor(context, face_counts, face_indices, positions);

        visit_children(mesh, context);
    }

    std::function<void(const Context&)> mObjectVisitor;// = [](const Context&){};
    std::function<void(const Context&)> mXformVisitor;//  = [](const Context&){};
    std::function<void(const Context&,
                       const Span<const int32_t*>,
                       const Span<const int32_t*>,
                       const Span<const Alembic::AbcGeom::V3f*>)> mMeshVisitor;

public:

    AlembicReader(decltype(mMeshVisitor) meshVisitor,
                  decltype(mObjectVisitor) objectVisitor = [](const Context&){},
                  decltype(mXformVisitor) xformVisitor = [](const Context&){})
    : mObjectVisitor(objectVisitor), mXformVisitor(xformVisitor), mMeshVisitor(meshVisitor)
    {
    }

    void visit(const std::string &filename)
    {
        Alembic::AbcCoreFactory::IFactory factory;
        auto archive = factory.getArchive(filename);

        Context context;
        visit_children(archive.getTop(), context);
    }
};// AlembicReader

void AlembicReader::visit_children(Alembic::AbcGeom::IObject parent_object,
                                   const Context &parent_context)
{
    for (size_t i = 0; i < parent_object.getNumChildren(); i++) {
        const auto &child_header = parent_object.getChildHeader(i);
        if (Alembic::AbcGeom::IXform::matches(child_header)) {
            visit_xform(Alembic::AbcGeom::IXform{parent_object, child_header.getName()},
                        parent_context);
        } else if (Alembic::AbcGeom::IPolyMesh::matches(child_header)) {
            visit_mesh(Alembic::AbcGeom::IPolyMesh{parent_object, child_header.getName()},
                       parent_context);
        } else {
            visit_object(Alembic::AbcGeom::IObject{parent_object, child_header.getName()},
                         parent_context);
        }
    }
}

void Geometry::readABC(const std::string &fileName)
{
    auto meshVisitor = [&](const AlembicReader::Context &context,
                           AlembicReader::Span<const int32_t*> face_counts,
                           AlembicReader::Span<const int32_t*> face_indices,
                           AlembicReader::Span<const Alembic::AbcGeom::V3f*> positions)
        {
            const int32_t N = mVtx.size(), *f = face_indices.pointer;
            for (int i=0; i<face_counts.count; ++i) {
                switch (face_counts.pointer[i]) {
                case 3:
                    mTri.emplace_back(N + f[0], N + f[1], N + f[2]);
                    f += 3;
                    break;
                case 4:
                    mQuad.emplace_back(N + f[0], N + f[1], N + f[2], N + f[3]);
                    f += 4;
                    break;
                default:
                    throw std::invalid_argument("AlembicReader: only supports triangles and quads");
                }
            }
            mVtx.resize(N + positions.count);
            const Alembic::AbcGeom::V3f *v = positions.pointer;
            for (size_t i = N; i<mVtx.size(); ++i, ++v) {
                const float *p = v->getValue();
                mVtx[i].init(p[0], p[1], p[2]);
            }
        };// meshVisitor lambda function

        AlembicReader tmp(meshVisitor);
        tmp.visit(fileName);
        mBBox = BBoxT();//invalidate BBox
}// Geometry::readABC

void Geometry::writeABC(const std::string &fileName) const
{
    std::vector<int32_t> abcCounts;
    std::vector<int32_t> abcIndices;

    abcCounts.reserve(mTri.size() + mQuad.size());
    abcIndices.reserve(3 * mTri.size() + 4 * mQuad.size());

    for (const auto &tri : mTri) {
        abcCounts.push_back(3);
        abcIndices.push_back(tri.x());
        abcIndices.push_back(tri.y());
        abcIndices.push_back(tri.z());
    }

    for (const auto &quad : mQuad) {
        abcCounts.push_back(4);
        abcIndices.push_back(quad.x());
        abcIndices.push_back(quad.y());
        abcIndices.push_back(quad.z());
        abcIndices.push_back(quad.w());
    }

    {
        using namespace Alembic::AbcGeom;

        P3fArraySample pointsArraySample{reinterpret_cast<const Alembic::Abc::V3f*>(mVtx.data()), mVtx.size()};
        Int32ArraySample indicesArraySample{reinterpret_cast<const int32_t*>(abcIndices.data()), abcIndices.size()};
        Int32ArraySample countsArraySample{reinterpret_cast<const int32_t*>(abcCounts.data()), abcCounts.size()};
        OPolyMeshSchema::Sample meshSample{pointsArraySample, indicesArraySample, countsArraySample};

        OArchive archive{Alembic::AbcCoreOgawa::WriteArchive(), fileName.c_str()};
        OObject topObject{archive, kTop};
        OPolyMesh meshObject{topObject, "vdb_mesh"};
        auto &mesh = meshObject.getSchema();

        mesh.set(meshSample);
    }
} // Geometry::writeABC

#else

void Geometry::readABC(const std::string&)
{
    throw std::runtime_error("Alembic read support was disabled during compilation!");
}

void Geometry::writeABC(const std::string&) const
{
    throw std::runtime_error("Alembic write support was disabled during compilation!");
}

#endif// VDB_TOOL_USE_ABC

Geometry::Ptr Geometry::copyGeom() const
{
    Ptr other = std::make_shared<Geometry>();
    other->mVtx = mVtx;
    other->mTri = mTri;
    other->mQuad = mQuad;
    other->mBBox = mBBox;
    other->mName = mName;
    return other;
}

void Geometry::transform(const math::Transform &xform)
{
    using RangeT = tbb::blocked_range<size_t>;
    tbb::parallel_for(RangeT(0, mVtx.size()), [&](RangeT r){
        for (size_t i=r.begin(); i<r.end(); ++i){
            Vec3d xyz(mVtx[i]);
            mVtx[i] = static_cast<Vec3s>(xform.baseMap()->applyMap(xyz));
        }
    });
    mBBox = BBoxT();//invalidate BBox
}// Geometry::transform

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif// VDB_TOOL_GEOMETRY_HAS_BEEN_INCLUDED
