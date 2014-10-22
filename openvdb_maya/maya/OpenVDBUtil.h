///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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

/// @author FX R&D OpenVDB team


#ifndef OPENVDB_MAYA_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_MAYA_UTIL_HAS_BEEN_INCLUDED

#include "OpenVDBData.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/util/Formats.h> // printBytes

#include <tbb/tick_count.h>

#include <maya/M3dView.h>
#include <maya/MString.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnPluginData.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <iostream>
#include <sstream>
#include <limits>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


////////////////////////////////////////


namespace openvdb_maya {


typedef openvdb::GridBase           Grid;
typedef openvdb::GridBase::Ptr      GridPtr;
typedef openvdb::GridBase::ConstPtr GridCPtr;
typedef openvdb::GridBase&          GridRef;
typedef const openvdb::GridBase&    GridCRef;

typedef openvdb::GridPtrVec         GridPtrVec;
typedef GridPtrVec::iterator        GridPtrVecIter;
typedef GridPtrVec::const_iterator  GridPtrVecCIter;

typedef openvdb::GridCPtrVec        GridCPtrVec;
typedef GridCPtrVec::iterator       GridCPtrVecIter;
typedef GridCPtrVec::const_iterator GridCPtrVecCIter;


////////////////////////////////////////


/// @brief  returns a pointer to the input VDB data object or NULL if this fails.
const OpenVDBData* getInputVDB(const MObject& vdb, MDataBlock& data);


void getGrids(std::vector<openvdb::GridBase::ConstPtr>& grids,
    const OpenVDBData& vdb, const std::string& names);

std::string getGridNames(const OpenVDBData& vdb);

bool containsGrid(const std::vector<std::string>& selectionList,
    const std::string& gridName, size_t gridIndex);

/// @brief  Constructs a list of selected grids @c grids from
///         the @c inputVdb and passes through unselected grids
///         to the @c outputVdb.
///
/// @return @c false if no matching grids were found.
bool getSelectedGrids(GridCPtrVec& grids, const std::string& selection,
    const OpenVDBData& inputVdb, OpenVDBData& outputVdb);


/// @brief  Constructs a list of selected grids @c grids from
///         the @c inputVdb.
///
/// @return @c false if no matching grids were found.
bool getSelectedGrids(GridCPtrVec& grids, const std::string& selection,
    const OpenVDBData& inputVdb);


////////////////////////////////////////

// Statistics and grid info


struct Timer
{
    Timer() : mStamp(tbb::tick_count::now()) { }

    void reset() { mStamp = tbb::tick_count::now(); }

    double seconds() const { return (tbb::tick_count::now() - mStamp).seconds(); }

    std::string elapsedTime() const {
        double sec = seconds();
        return sec < 1.0 ? boost::lexical_cast<std::string>(sec * 1000.0) + " ms" :
             boost::lexical_cast<std::string>(sec) + " s";
    }

private:
    tbb::tick_count mStamp;
};


void printGridInfo(std::ostream& os, const OpenVDBData& vdb);

void updateNodeInfo(std::stringstream& stream, MDataBlock& data, MObject& strAttr);


////////////////////////////////////////

// OpenGL helper objects


class BufferObject
{
public:
    BufferObject();
    ~BufferObject();

    void render() const;

    /// @note accepted @c primType: GL_POINTS, GL_LINE_STRIP, GL_LINE_LOOP,
    /// GL_LINES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES,
    /// GL_QUAD_STRIP, GL_QUADS and GL_POLYGON
    void genIndexBuffer(const std::vector<GLuint>&, GLenum primType);

    void genVertexBuffer(const std::vector<GLfloat>&);
    void genNormalBuffer(const std::vector<GLfloat>&);
    void genColorBuffer(const std::vector<GLfloat>&);

    void clear();

private:
    GLuint mVertexBuffer, mNormalBuffer, mIndexBuffer, mColorBuffer;
    GLenum mPrimType;
    GLsizei mPrimNum;
};


class ShaderProgram
{
public:
    ShaderProgram();
    ~ShaderProgram();

    void setVertShader(const std::string&);
    void setFragShader(const std::string&);

    void build();
    void build(const std::vector<GLchar*>& attributes);

    void startShading() const;
    void stopShading() const;

    void clear();

private:
    GLuint mProgram, mVertShader, mFragShader;
};

////////////////////////////////////////

///@todo Move this into an graphics library.
// Should be shared with the stand alone viewer.

class WireBoxBuilder
{
public:
    WireBoxBuilder(const openvdb::math::Transform& xform,
        std::vector<GLuint>& indices, std::vector<GLfloat>& points, std::vector<GLfloat>& colors);

    void add(GLuint boxIndex, const openvdb::CoordBBox& bbox, const openvdb::Vec3s& color);

private:
    const openvdb::math::Transform *mXForm;
    std::vector<GLuint> *mIndices;
    std::vector<GLfloat> *mPoints;
    std::vector<GLfloat> *mColors;
}; // WireBoxBuilder


class BoundingBoxGeo
{
public:
    BoundingBoxGeo(BufferObject& buffer)
        : mBuffer(&buffer)
        , mMin(-1.0)
        , mMax(1.0)
    {
    }

    void operator()(openvdb::GridBase::ConstPtr grid)
    {
        const size_t N = 8 * 3;

        std::vector<GLuint> indices(N);
        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);

        WireBoxBuilder boxBuilder(grid->constTransform(), indices, points, colors);

        boxBuilder.add(0, grid->evalActiveVoxelBoundingBox(), openvdb::Vec3s(0.045, 0.045, 0.045));

        // store the sorted min/max points.
        mMin[0] = std::numeric_limits<float>::max();
        mMin[1] = mMin[0];
        mMin[2] = mMin[0];
        mMax[0] = -mMin[0];
        mMax[1] = -mMin[0];
        mMax[2] = -mMin[0];

        for (int i = 0; i < 8; ++i) {
            int p = i * 3;
            mMin[0] = std::min(mMin[0], points[p]);
            mMin[1] = std::min(mMin[1], points[p+1]);
            mMin[2] = std::min(mMin[2], points[p+2]);

            mMax[0] = std::max(mMax[0], points[p]);
            mMax[1] = std::max(mMax[1], points[p+1]);
            mMax[2] = std::max(mMax[2], points[p+2]);
        }

        // gen buffers and upload data to GPU (ignoring color array)
        mBuffer->genVertexBuffer(points);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

    const openvdb::Vec3s& min() const { return mMin; }
    const openvdb::Vec3s& max() const { return mMax; }

private:
    BufferObject *mBuffer;
    openvdb::Vec3s mMin, mMax;
}; // BoundingBoxGeo


class InternalNodesGeo
{
public:
    InternalNodesGeo(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        size_t nodeCount = grid->tree().nonLeafCount();
        const size_t N = nodeCount * 8 * 3;

        std::vector<GLuint> indices(N);
        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);

        WireBoxBuilder boxBuilder(grid->constTransform(), indices, points, colors);

        openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(10));
        size_t boxIndex = 0;

        typename GridType::TreeType::NodeCIter iter = grid->tree().cbeginNode();
        iter.setMaxDepth(GridType::TreeType::NodeCIter::LEAF_DEPTH - 1);

        const openvdb::Vec3s nodeColor[2] = {
            openvdb::Vec3s(0.0432, 0.33, 0.0411023), // first internal node level
            openvdb::Vec3s(0.871, 0.394, 0.01916) // intermediate internal node levels
        };

        for ( ; iter; ++iter) {
            iter.getBoundingBox(bbox);
            boxBuilder.add(boxIndex++, bbox, nodeColor[(iter.getLevel() == 1)]);

        } // end node iteration


        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(points);
        mBuffer->genColorBuffer(colors);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mBuffer;
}; // InternalNodesGeo


class LeafNodesGeo
{
public:
    LeafNodesGeo(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        typedef typename GridType::TreeType TreeType;

        openvdb::tree::LeafManager<const TreeType> leafs(grid->tree());

        const size_t N = leafs.leafCount() * 8 * 3;
        std::vector<GLuint> indices(N);
        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);

        WireBoxBuilder boxBuilder(grid->constTransform(), indices, points, colors);
        const openvdb::Vec3s color(0.00608299, 0.279541, 0.625); // leaf node color

        tbb::parallel_for(leafs.getRange(), LeafOp<TreeType>(leafs, boxBuilder, color));

        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(points);
        mBuffer->genColorBuffer(colors);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

protected:
    template<typename TreeType>
    struct LeafOp
    {
        typedef openvdb::tree::LeafManager<const TreeType> LeafManagerType;

        LeafOp(const LeafManagerType& leafs, WireBoxBuilder& boxBuilder, const openvdb::Vec3s& color)
            : mLeafs(&leafs), mBoxBuilder(&boxBuilder), mColor(color) {}

        void operator()(const tbb::blocked_range<size_t>& range) const
        {
            openvdb::CoordBBox bbox;
            openvdb::Coord& min = bbox.min();
            openvdb::Coord& max = bbox.max();
            const int offset = int(TreeType::LeafNodeType::DIM) - 1;

            for (size_t n = range.begin(); n < range.end(); ++n) {
                min = mLeafs->leaf(n).origin();
                max[0] = min[0] + offset;
                max[1] = min[1] + offset;
                max[2] = min[2] + offset;
                mBoxBuilder->add(n, bbox, mColor);
            }
        }

    private:
        const LeafManagerType *mLeafs;
        WireBoxBuilder *mBoxBuilder;
        const openvdb::Vec3s mColor;
    }; // LeafOp

private:
    BufferObject *mBuffer;
}; // LeafNodesGeo


class ActiveTileGeo
{
public:
    ActiveTileGeo(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        typedef typename GridType::TreeType TreeType;
        const openvdb::Index maxDepth = TreeType::ValueAllIter::LEAF_DEPTH - 1;
        size_t tileCount = 0;

        {
            typename TreeType::ValueOnCIter iter(grid->tree());
            iter.setMaxDepth(maxDepth);
            for ( ; iter; ++iter) { ++tileCount; }
        }

        const size_t N = tileCount * 8 * 3;

        std::vector<GLuint> indices(N);
        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);

        WireBoxBuilder boxBuilder(grid->constTransform(), indices, points, colors);

        const openvdb::Vec3s color(0.9, 0.3, 0.3);
        openvdb::CoordBBox bbox;
        size_t boxIndex = 0;



        typename TreeType::ValueOnCIter iter(grid->tree());
        iter.setMaxDepth(maxDepth);

        for ( ; iter; ++iter) {
            iter.getBoundingBox(bbox);
            boxBuilder.add(boxIndex++, bbox, color);
        } // end tile iteration


        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(points);
        mBuffer->genColorBuffer(colors);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mBuffer;
}; // ActiveTileGeo


class SurfaceGeo
{
public:
    SurfaceGeo(BufferObject& buffer, float iso) : mBuffer(&buffer), mIso(iso) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        openvdb::tools::VolumeToMesh mesher(mIso);
        mesher(*grid);

        // Copy points and generate point normals.
        std::vector<GLfloat> points(mesher.pointListSize() * 3);
        std::vector<GLfloat> normals(mesher.pointListSize() * 3);

        openvdb::tree::ValueAccessor<const typename GridType::TreeType> acc(grid->tree());
        typedef openvdb::math::Gradient<openvdb::math::GenericMap, openvdb::math::CD_2ND> Gradient;
        openvdb::math::GenericMap map(grid->transform());
        openvdb::Coord ijk;

        for (size_t n = 0, i = 0,  N = mesher.pointListSize(); n < N; ++n) {
            const openvdb::Vec3s& p = mesher.pointList()[n];
            points[i++] = p[0];
            points[i++] = p[1];
            points[i++] = p[2];
        }

        // Copy primitives
        openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();
        size_t numQuads = 0;
        for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            numQuads += polygonPoolList[n].numQuads();
        }

        std::vector<GLuint> indices;
        indices.reserve(numQuads * 4);
        openvdb::Vec3d normal, e1, e2;

        for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
            for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
                const openvdb::Vec4I& quad = polygons.quad(i);
                indices.push_back(quad[0]);
                indices.push_back(quad[1]);
                indices.push_back(quad[2]);
                indices.push_back(quad[3]);

                e1 = mesher.pointList()[quad[1]];
                e1 -= mesher.pointList()[quad[0]];
                e2 = mesher.pointList()[quad[2]];
                e2 -= mesher.pointList()[quad[1]];
                normal = e1.cross(e2);

                const double length = normal.length();
                if (length > 1.0e-7) normal *= (1.0 / length);

                for (size_t v = 0; v < 4; ++v) {
                    normals[quad[v]*3]    = -normal[0];
                    normals[quad[v]*3+1]  = -normal[1];
                    normals[quad[v]*3+2]  = -normal[2];
                }
            }
        }

        // Construct and transfer GPU buffers.
        mBuffer->genVertexBuffer(points);
        mBuffer->genNormalBuffer(normals);
        mBuffer->genIndexBuffer(indices, GL_QUADS);
    }

private:
    BufferObject *mBuffer;
    float mIso;
}; // SurfaceGeo


template<typename TreeType>
class PointGenerator
{
public:
    typedef openvdb::tree::LeafManager<TreeType> LeafManagerType;

    PointGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLuint>& indices,
        LeafManagerType& leafs,
        std::vector<unsigned>& indexMap,
        const openvdb::math::Transform& transform,
        size_t voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS)
        : mPoints(&points)
        , mIndices(&indices)
        , mLeafs(&leafs)
        , mIndexMap(&indexMap)
        , mTransform(&transform)
        , mVoxelsPerLeaf(voxelsPerLeaf)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(mLeafs->getRange(), *this);
    }


    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        typedef typename TreeType::LeafNodeType::ValueOnCIter ValueOnCIter;

        openvdb::Vec3d pos;
        unsigned index = 0;
        size_t activeVoxels = 0;

        for (size_t n = range.begin(); n < range.end(); ++n) {

            index = (*mIndexMap)[n];
            ValueOnCIter it = mLeafs->leaf(n).cbeginValueOn();

            activeVoxels = mLeafs->leaf(n).onVoxelCount();

            if (activeVoxels <= mVoxelsPerLeaf) {

                for ( ; it; ++it) {
                    pos = mTransform->indexToWorld(it.getCoord());
                    insertPoint(pos, index);
                    ++index;
                }

            } else if (1 == mVoxelsPerLeaf) {

                 pos = mTransform->indexToWorld(it.getCoord());
                 insertPoint(pos, index);

            } else {

                std::vector<openvdb::Coord> coords;
                coords.reserve(activeVoxels);
                for ( ; it; ++it) { coords.push_back(it.getCoord()); }

                pos = mTransform->indexToWorld(coords[0]);
                insertPoint(pos, index);
                ++index;

                pos = mTransform->indexToWorld(coords[activeVoxels-1]);
                insertPoint(pos, index);
                ++index;

                int r = int(std::floor(mVoxelsPerLeaf / activeVoxels));
                for (int i = 1, I = mVoxelsPerLeaf - 2; i < I; ++i) {
                    pos = mTransform->indexToWorld(coords[i * r]);
                    insertPoint(pos, index);
                    ++index;
                }
            }
        }
    }

private:
    void insertPoint(const openvdb::Vec3d& pos, unsigned index) const
    {
        (*mIndices)[index] = index;
        const unsigned element = index * 3;
        (*mPoints)[element    ] = pos[0];
        (*mPoints)[element + 1] = pos[1];
        (*mPoints)[element + 2] = pos[2];
    }

    std::vector<GLfloat> *mPoints;
    std::vector<GLuint> *mIndices;
    LeafManagerType *mLeafs;
    std::vector<unsigned> *mIndexMap;
    const openvdb::math::Transform *mTransform;
    const size_t mVoxelsPerLeaf;
}; // PointGenerator


template<typename GridType>
class PointAttributeGenerator
{
public:
    typedef typename GridType::ValueType ValueType;

    PointAttributeGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLfloat>& colors,
        const GridType& grid,
        ValueType minValue,
        ValueType maxValue,
        openvdb::Vec3s (&colorMap)[4],
        bool isLevelSet = false)
        : mPoints(&points)
        , mColors(&colors)
        , mNormals(NULL)
        , mGrid(&grid)
        , mAccessor(grid.tree())
        , mMinValue(minValue)
        , mMaxValue(maxValue)
        , mColorMap(colorMap)
        , mIsLevelSet(isLevelSet)
        , mZeroValue(openvdb::zeroVal<ValueType>())
    {
        init();
    }

    PointAttributeGenerator(
        std::vector<GLfloat>& points,
        std::vector<GLfloat>& colors,
        std::vector<GLfloat>& normals,
        const GridType& grid,
        ValueType minValue,
        ValueType maxValue,
        openvdb::Vec3s (&colorMap)[4],
        bool isLevelSet = false)
        : mPoints(&points)
        , mColors(&colors)
        , mNormals(&normals)
        , mGrid(&grid)
        , mAccessor(grid.tree())
        , mMinValue(minValue)
        , mMaxValue(maxValue)
        , mColorMap(colorMap)
        , mIsLevelSet(isLevelSet)
        , mZeroValue(openvdb::zeroVal<ValueType>())
    {
        init();
    }

    void runParallel()
    {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, (mPoints->size() / 3)), *this);
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        openvdb::Coord ijk;
        openvdb::Vec3d pos, tmpNormal, normal(0.0, -1.0, 0.0);
        openvdb::Vec3s color(0.9, 0.3, 0.3);
        float w = 0.0;

        size_t e1, e2, e3, voxelNum = 0;
        for (size_t n = range.begin(); n < range.end(); ++n) {

            e1 = 3 * n;
            e2 = e1 + 1;
            e3 = e2 + 1;

            pos[0] = (*mPoints)[e1];
            pos[1] = (*mPoints)[e2];
            pos[2] = (*mPoints)[e3];

            pos = mGrid->worldToIndex(pos);
            ijk[0] = int(pos[0]);
            ijk[1] = int(pos[1]);
            ijk[2] = int(pos[2]);

            const ValueType& value = mAccessor.getValue(ijk);

            if (value < mZeroValue) { // is negative
                if (mIsLevelSet) {
                    color = mColorMap[1];
                } else {
                    w = (float(value) - mOffset[1]) * mScale[1];
                    color = w * mColorMap[0] + (1.0 - w) * mColorMap[1];
                }
            } else {
                if (mIsLevelSet) {
                    color = mColorMap[2];
                } else {
                    w = (float(value) - mOffset[0]) * mScale[0];
                    color = w * mColorMap[2] + (1.0 - w) * mColorMap[3];
                }
            }

            (*mColors)[e1] = color[0];
            (*mColors)[e2] = color[1];
            (*mColors)[e3] = color[2];

            if (mNormals) {

                if ((voxelNum % 2) == 0) {
                    tmpNormal = openvdb::math::ISGradient<
                        openvdb::math::CD_2ND>::result(mAccessor, ijk);

                    double length = tmpNormal.length();
                    if (length > 1.0e-7) {
                        tmpNormal *= 1.0 / length;
                        normal = tmpNormal;
                    }
                }
                ++voxelNum;

                (*mNormals)[e1] = normal[0];
                (*mNormals)[e2] = normal[1];
                (*mNormals)[e3] = normal[2];
            }
        }
    }

private:

    void init()
    {
        mOffset[0] = float(std::min(mZeroValue, mMinValue));
        mScale[0] = 1.0 / float(std::abs(std::max(mZeroValue, mMaxValue) - mOffset[0]));
        mOffset[1] = float(std::min(mZeroValue, mMinValue));
        mScale[1] = 1.0 / float(std::abs(std::max(mZeroValue, mMaxValue) - mOffset[1]));
    }

    std::vector<GLfloat> *mPoints;
    std::vector<GLfloat> *mColors;
    std::vector<GLfloat> *mNormals;

    const GridType *mGrid;
    openvdb::tree::ValueAccessor<const typename GridType::TreeType> mAccessor;

    ValueType mMinValue, mMaxValue;
    openvdb::Vec3s (&mColorMap)[4];
    const bool mIsLevelSet;

    ValueType mZeroValue;
    float mOffset[2], mScale[2];
}; // PointAttributeGenerator


class ActiveVoxelGeo
{
public:

    ActiveVoxelGeo(BufferObject& pointBuffer)
        : mPointBuffer(&pointBuffer)
        , mColorMinPosValue(0.3, 0.9, 0.3) // green
        , mColorMaxPosValue(0.9, 0.3, 0.3) // red
        , mColorMinNegValue(0.9, 0.9, 0.3) // yellow
        , mColorMaxNegValue(0.3, 0.3, 0.9) // blue
    { }

    void setColorMinPosValue(const openvdb::Vec3s& c) { mColorMinPosValue = c; }
    void setColorMaxPosValue(const openvdb::Vec3s& c) { mColorMaxPosValue = c; }
    void setColorMinNegValue(const openvdb::Vec3s& c) { mColorMinNegValue = c; }
    void setColorMaxNegValue(const openvdb::Vec3s& c) { mColorMaxNegValue = c; }

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        const size_t maxVoxelPoints = 26000000;

        openvdb::Vec3s colorMap[4];
        colorMap[0] = mColorMinPosValue;
        colorMap[1] = mColorMaxPosValue;
        colorMap[2] = mColorMinNegValue;
        colorMap[3] = mColorMaxNegValue;

        //////////

        typedef typename GridType::ValueType ValueType;
        typedef typename GridType::TreeType TreeType;
        typedef typename TreeType::template ValueConverter<bool>::Type BoolTreeT;

        const TreeType& tree = grid->tree();
        const bool isLevelSetGrid = grid->getGridClass() == openvdb::GRID_LEVEL_SET;

        ValueType minValue, maxValue;
        openvdb::tree::LeafManager<const TreeType> leafs(tree);

        {
            openvdb::tools::MinMaxVoxel<const TreeType> minmax(leafs);
            minmax.runParallel();
            minValue = minmax.minVoxel();
            maxValue = minmax.maxVoxel();
        }

        size_t voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS;

        if (tree.activeLeafVoxelCount() > maxVoxelPoints) {
            voxelsPerLeaf = std::max((maxVoxelPoints / tree.leafCount()), size_t(1));
        }

        std::vector<unsigned> indexMap(leafs.leafCount());
        unsigned voxelCount = 0;
        for (size_t l = 0, L = leafs.leafCount(); l < L; ++l) {
            indexMap[l] = voxelCount;
            voxelCount += std::min(leafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
        }

        std::vector<GLfloat>
            points(voxelCount * 3),
            colors(voxelCount * 3),
            normals(voxelCount * 3);
        std::vector<GLuint> indices(voxelCount);

        PointGenerator<const TreeType> pointGen(
            points, indices, leafs, indexMap, grid->transform(), voxelsPerLeaf);
        pointGen.runParallel();

        PointAttributeGenerator<GridType> attributeGen(
            points, colors, normals, *grid, minValue, maxValue, colorMap, isLevelSetGrid);
        attributeGen.runParallel();

        mPointBuffer->genVertexBuffer(points);
        mPointBuffer->genColorBuffer(colors);
        mPointBuffer->genNormalBuffer(normals);
        mPointBuffer->genIndexBuffer(indices, GL_POINTS);
    }

private:
    BufferObject *mPointBuffer;
    openvdb::Vec3s mColorMinPosValue, mColorMaxPosValue, mColorMinNegValue, mColorMaxNegValue;
}; // ActiveVoxelGeo


////////////////////////////////////////


/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType, bool IsConst/*=false*/>
struct GridProcessor {
    static inline void call(OpType& op, openvdb::GridBase::Ptr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
#endif
    }
};

/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType>
struct GridProcessor<GridType, OpType, /*IsConst=*/true> {
    static inline void call(OpType& op, openvdb::GridBase::ConstPtr grid) {
#ifdef _MSC_VER
        op.operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#else
        op.template operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
#endif
    }
};


/// Helper function used internally by processTypedGrid()
template<typename GridType, typename OpType, typename GridPtrType>
inline void
doProcessTypedGrid(GridPtrType grid, OpType& op)
{
    GridProcessor<GridType, OpType,
        boost::is_const<typename GridPtrType::element_type>::value>::call(op, grid);
}


////////////////////////////////////////


/// @brief Utility function that, given a generic grid pointer,
/// calls a functor on the fully-resolved grid
///
/// Usage:
/// @code
/// struct PruneOp {
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const { grid->tree()->prune(); }
/// };
///
/// processTypedGrid(myGridPtr, PruneOp());
/// @endcode
///
/// @return @c false if the grid type is unknown or unhandled.
template<typename GridPtrType, typename OpType>
bool
processTypedGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<BoolGrid>())        doProcessTypedGrid<BoolGrid>(grid, op);
    else if (grid->template isType<FloatGrid>())  doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else if (grid->template isType<Vec3IGrid>())  doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>())  doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else return false;
    return true;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are scalars
///
/// Usage:
/// @code
/// struct PruneOp {
///     template<typename GridT>
///     void operator()(typename GridT::Ptr grid) const { grid->tree()->prune(); }
/// };
///
/// processTypedScalarGrid(myGridPtr, PruneOp());
/// @endcode
///
/// @return @c false if the grid type is unknown or non-scalar.
template<typename GridPtrType, typename OpType>
bool
processTypedScalarGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<FloatGrid>())       doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>()) doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())  doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())  doProcessTypedGrid<Int64Grid>(grid, op);
    else return false;
    return true;
}


/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are vectors
template<typename GridPtrType, typename OpType>
bool
processTypedVectorGrid(GridPtrType grid, OpType& op)
{
    using namespace openvdb;
    if (grid->template isType<Vec3IGrid>())       doProcessTypedGrid<Vec3IGrid>(grid, op);
    else if (grid->template isType<Vec3SGrid>()) doProcessTypedGrid<Vec3SGrid>(grid, op);
    else if (grid->template isType<Vec3DGrid>())  doProcessTypedGrid<Vec3DGrid>(grid, op);
    else return false;
    return true;
}

} // namespace util

////////////////////////////////////////


#endif // OPENVDB_MAYA_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
