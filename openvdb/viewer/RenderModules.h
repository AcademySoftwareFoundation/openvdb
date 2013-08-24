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

#ifndef OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED
#define OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/math/Operators.h>

#include <boost/random/mersenne_twister.hpp>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif


namespace openvdb_viewer {

// OpenGL helper objects.

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


/// @brief interface class.
class RenderModule
{
public:
    virtual void render() = 0;
    virtual ~RenderModule() {}
    virtual bool& visible() { return mIsVisible; }
protected:
    RenderModule() : mIsVisible(true) {}
    bool mIsVisible;
};


////////////////////////////////////////


/// @brief Basic render module, axis gnomon and ground plane.
class ViewportModule: public RenderModule
{
public:
    ViewportModule();
    void render();

private:
    float mAxisGnomonScale, mGroundPlaneScale;
};


////////////////////////////////////////


/// @brief Tree topology render module
class TreeTopologyModule: public RenderModule
{
public:
    TreeTopologyModule(const openvdb::GridBase::ConstPtr&);
    ~TreeTopologyModule() {}

    void render();

private:
    void init();
    const openvdb::GridBase::ConstPtr& mGrid;
    BufferObject mBufferObject;
    bool mIsInitialized;
    ShaderProgram mShader;
};


////////////////////////////////////////


/// @brief Tree topology render module
class ActiveValueModule: public RenderModule
{
public:
    ActiveValueModule(const openvdb::GridBase::ConstPtr&);
    ~ActiveValueModule() {}

    void render();

private:
    void init();
    const openvdb::GridBase::ConstPtr& mGrid;

    BufferObject mInteriorBuffer, mSurfaceBuffer, mVectorBuffer;
    bool mIsInitialized;
    ShaderProgram mFlatShader, mSurfaceShader;
};


////////////////////////////////////////


/// @brief Surfacing render module
class MeshModule: public RenderModule
{
public:
    MeshModule(const openvdb::GridBase::ConstPtr&);
    ~MeshModule() {}

    void render();
private:
    void init();
    const openvdb::GridBase::ConstPtr& mGrid;

    BufferObject mBufferObject;
    bool mIsInitialized;
    ShaderProgram mShader;
};


////////////////////////////////////////


class TreeTopologyOp
{
public:
    TreeTopologyOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        Index64 nodeCount = grid->tree().leafCount() + grid->tree().nonLeafCount();
        const Index64 N = nodeCount * 8 * 3;

        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);
        std::vector<GLuint> indices(N);


        openvdb::Vec3d ptn;
        openvdb::Vec3s color;
        openvdb::CoordBBox bbox;
        Index64 pOffset = 0, iOffset = 0,  cOffset = 0, idx = 0;

        for (typename GridType::TreeType::NodeCIter iter = grid->tree().cbeginNode(); iter; ++iter)
        {
            iter.getBoundingBox(bbox);

            // Nodes are rendered as cell-centered
            const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
            const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);

            // corner 1
            ptn = grid->indexToWorld(min);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 2
            ptn = openvdb::Vec3d(min.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 3
            ptn = openvdb::Vec3d(max.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 4
            ptn = openvdb::Vec3d(max.x(), min.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 5
            ptn = openvdb::Vec3d(min.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 6
            ptn = openvdb::Vec3d(min.x(), max.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 7
            ptn = grid->indexToWorld(max);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];

            // corner 8
            ptn = openvdb::Vec3d(max.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            points[pOffset++] = ptn[0];
            points[pOffset++] = ptn[1];
            points[pOffset++] = ptn[2];


            // edge 1
            indices[iOffset++] = idx;
            indices[iOffset++] = idx + 1;
            // edge 2
            indices[iOffset++] = idx + 1;
            indices[iOffset++] = idx + 2;
            // edge 3
            indices[iOffset++] = idx + 2;
            indices[iOffset++] = idx + 3;
            // edge 4
            indices[iOffset++] = idx + 3;
            indices[iOffset++] = idx;
            // edge 5
            indices[iOffset++] = idx + 4;
            indices[iOffset++] = idx + 5;
            // edge 6
            indices[iOffset++] = idx + 5;
            indices[iOffset++] = idx + 6;
            // edge 7
            indices[iOffset++] = idx + 6;
            indices[iOffset++] = idx + 7;
            // edge 8
            indices[iOffset++] = idx + 7;
            indices[iOffset++] = idx + 4;
            // edge 9
            indices[iOffset++] = idx;
            indices[iOffset++] = idx + 4;
            // edge 10
            indices[iOffset++] = idx + 1;
            indices[iOffset++] = idx + 5;
            // edge 11
            indices[iOffset++] = idx + 2;
            indices[iOffset++] = idx + 6;
            // edge 12
            indices[iOffset++] = idx + 3;
            indices[iOffset++] = idx + 7;


            // node vertex color
            const int level = iter.getLevel();
            color = sNodeColors[(level == 0) ? 3 : (level == 1) ? 2 : 1];

            for (Index64 n = 0; n < 8; ++n) {
                colors[cOffset++] = color[0];
                colors[cOffset++] = color[1];
                colors[cOffset++] = color[2];
            }

            idx += 8;
        } // end node iteration

        // gen buffers and upload data to GPU
        mBuffer->genVertexBuffer(points);
        mBuffer->genColorBuffer(colors);
        mBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mBuffer;

    static openvdb::Vec3s sNodeColors[];

}; // TreeTopologyOp


////////////////////////////////////////

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
        openvdb::Index64 voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS)
        : mPoints(points)
        , mIndices(indices)
        , mLeafs(leafs)
        , mIndexMap(indexMap)
        , mTransform(transform)
        , mVoxelsPerLeaf(voxelsPerLeaf)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(mLeafs.getRange(), *this);
    }


    inline void operator()(const tbb::blocked_range<openvdb::Index64>& range) const
    {
        using openvdb::Index64;

        typedef typename TreeType::LeafNodeType::ValueOnCIter ValueOnCIter;

        openvdb::Vec3d pos;
        unsigned index = 0;
        Index64 activeVoxels = 0;

        for (Index64 n = range.begin(); n < range.end(); ++n) {

            index = mIndexMap[n];
            ValueOnCIter it = mLeafs.leaf(n).cbeginValueOn();

            activeVoxels = mLeafs.leaf(n).onVoxelCount();

            if (activeVoxels <= mVoxelsPerLeaf) {

                for ( ; it; ++it) {
                    pos = mTransform.indexToWorld(it.getCoord());
                    insertPoint(pos, index);
                    ++index;
                }

            } else if (1 == mVoxelsPerLeaf) {

                 pos = mTransform.indexToWorld(it.getCoord());
                 insertPoint(pos, index);

            } else {

                std::vector<openvdb::Coord> coords;
                coords.reserve(activeVoxels);
                for ( ; it; ++it) { coords.push_back(it.getCoord()); }

                pos = mTransform.indexToWorld(coords[0]);
                insertPoint(pos, index);
                ++index;

                pos = mTransform.indexToWorld(coords[activeVoxels-1]);
                insertPoint(pos, index);
                ++index;

                int r = int(std::floor(mVoxelsPerLeaf / activeVoxels));
                for (int i = 1, I = mVoxelsPerLeaf - 2; i < I; ++i) {
                    pos = mTransform.indexToWorld(coords[i * r]);
                    insertPoint(pos, index);
                    ++index;
                }
            }
        }
    }

private:
    void insertPoint(const openvdb::Vec3d& pos, unsigned index) const
    {
        mIndices[index] = index;
        const unsigned element = index * 3;
        mPoints[element    ] = pos[0];
        mPoints[element + 1] = pos[1];
        mPoints[element + 2] = pos[2];
    }

    std::vector<GLfloat>& mPoints;
    std::vector<GLuint>& mIndices;
    LeafManagerType& mLeafs;
    std::vector<unsigned>& mIndexMap;
    const openvdb::math::Transform& mTransform;
    const openvdb::Index64 mVoxelsPerLeaf;
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
        : mPoints(points)
        , mColors(colors)
        , mNormals(NULL)
        , mGrid(grid)
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
        : mPoints(points)
        , mColors(colors)
        , mNormals(&normals)
        , mGrid(grid)
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
            tbb::blocked_range<openvdb::Index64>(0, (mPoints.size() / 3)), *this);
    }

    inline void operator()(const tbb::blocked_range<openvdb::Index64>& range) const
    {
        using openvdb::Index64;

        openvdb::Coord ijk;
        openvdb::Vec3d pos, tmpNormal, normal(0.0, -1.0, 0.0);
        openvdb::Vec3s color(0.9, 0.3, 0.3);
        float w = 0.0;

        Index64 e1, e2, e3, voxelNum = 0;
        for (Index64 n = range.begin(); n < range.end(); ++n) {

            e1 = 3 * n;
            e2 = e1 + 1;
            e3 = e2 + 1;

            pos[0] = mPoints[e1];
            pos[1] = mPoints[e2];
            pos[2] = mPoints[e3];

            pos = mGrid.worldToIndex(pos);
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

            mColors[e1] = color[0];
            mColors[e2] = color[1];
            mColors[e3] = color[2];

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

    std::vector<GLfloat>& mPoints;
    std::vector<GLfloat>& mColors;
    std::vector<GLfloat>* mNormals;

    const GridType& mGrid;
    openvdb::tree::ValueAccessor<const typename GridType::TreeType> mAccessor;

    ValueType mMinValue, mMaxValue;
    openvdb::Vec3s (&mColorMap)[4];
    const bool mIsLevelSet;

    ValueType mZeroValue;
    float mOffset[2], mScale[2];
}; // PointAttributeGenerator


////////////////////////////////////////


class ActiveScalarValuesOp
{
public:

    ActiveScalarValuesOp(
        BufferObject& interiorBuffer, BufferObject& surfaceBuffer)
        : mInteriorBuffer(&interiorBuffer)
        , mSurfaceBuffer(&surfaceBuffer)
    {
    }

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        const Index64 maxVoxelPoints = 26000000;

        openvdb::Vec3s colorMap[4];
        colorMap[0] = openvdb::Vec3s(0.3, 0.9, 0.3); // green
        colorMap[1] = openvdb::Vec3s(0.9, 0.3, 0.3); // red
        colorMap[2] = openvdb::Vec3s(0.9, 0.9, 0.3); // yellow
        colorMap[3] = openvdb::Vec3s(0.3, 0.3, 0.9); // blue

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

        openvdb::Index64 voxelsPerLeaf = TreeType::LeafNodeType::NUM_VOXELS;

        if (!isLevelSetGrid) {

            typename BoolTreeT::Ptr interiorMask(new BoolTreeT(false));

            { // Generate Interior Points
                interiorMask->topologyUnion(tree);
                interiorMask->voxelizeActiveTiles();

                if (interiorMask->activeLeafVoxelCount() > maxVoxelPoints) {
                    voxelsPerLeaf = std::max<Index64>(1,
                        (maxVoxelPoints / interiorMask->leafCount()));
                }

                openvdb::tools::erodeVoxels(*interiorMask, 2);

                openvdb::tree::LeafManager<BoolTreeT> maskleafs(*interiorMask);
                std::vector<unsigned> indexMap(maskleafs.leafCount());
                unsigned voxelCount = 0;
                for (Index64 l = 0, L = maskleafs.leafCount(); l < L; ++l) {
                    indexMap[l] = voxelCount;
                    voxelCount += std::min(maskleafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
                }

                std::vector<GLfloat> points(voxelCount * 3), colors(voxelCount * 3);
                std::vector<GLuint> indices(voxelCount);

                PointGenerator<BoolTreeT> pointGen(
                    points, indices, maskleafs, indexMap, grid->transform(), voxelsPerLeaf);
                pointGen.runParallel();


                PointAttributeGenerator<GridType> attributeGen(
                    points, colors, *grid, minValue, maxValue, colorMap);
                attributeGen.runParallel();


                // gen buffers and upload data to GPU
                mInteriorBuffer->genVertexBuffer(points);
                mInteriorBuffer->genColorBuffer(colors);
                mInteriorBuffer->genIndexBuffer(indices, GL_POINTS);
            }

            { // Generate Surface Points
                typename BoolTreeT::Ptr surfaceMask(new BoolTreeT(false));
                surfaceMask->topologyUnion(tree);
                surfaceMask->voxelizeActiveTiles();

                openvdb::tree::ValueAccessor<BoolTreeT> interiorAcc(*interiorMask);
                for (typename BoolTreeT::LeafIter leafIt = surfaceMask->beginLeaf();
                    leafIt; ++leafIt)
                {
                    const typename BoolTreeT::LeafNodeType* leaf =
                        interiorAcc.probeConstLeaf(leafIt->origin());
                    if (leaf) leafIt->topologyDifference(*leaf, false);
                }
                surfaceMask->pruneInactive();

                openvdb::tree::LeafManager<BoolTreeT> maskleafs(*surfaceMask);
                std::vector<unsigned> indexMap(maskleafs.leafCount());
                unsigned voxelCount = 0;
                for (Index64 l = 0, L = maskleafs.leafCount(); l < L; ++l) {
                    indexMap[l] = voxelCount;
                    voxelCount += std::min(maskleafs.leaf(l).onVoxelCount(), voxelsPerLeaf);
                }

                std::vector<GLfloat>
                    points(voxelCount * 3),
                    colors(voxelCount * 3),
                    normals(voxelCount * 3);
                std::vector<GLuint> indices(voxelCount);

                PointGenerator<BoolTreeT> pointGen(
                    points, indices, maskleafs, indexMap, grid->transform(), voxelsPerLeaf);
                pointGen.runParallel();

                PointAttributeGenerator<GridType> attributeGen(
                    points, colors, normals, *grid, minValue, maxValue, colorMap);
                attributeGen.runParallel();

                mSurfaceBuffer->genVertexBuffer(points);
                mSurfaceBuffer->genColorBuffer(colors);
                mSurfaceBuffer->genNormalBuffer(normals);
                mSurfaceBuffer->genIndexBuffer(indices, GL_POINTS);
            }

            return;
        }

        // Level set rendering
        if (tree.activeLeafVoxelCount() > maxVoxelPoints) {
            voxelsPerLeaf = std::max<Index64>(1, (maxVoxelPoints / tree.leafCount()));
        }

        std::vector<unsigned> indexMap(leafs.leafCount());
        unsigned voxelCount = 0;
        for (Index64 l = 0, L = leafs.leafCount(); l < L; ++l) {
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

        mSurfaceBuffer->genVertexBuffer(points);
        mSurfaceBuffer->genColorBuffer(colors);
        mSurfaceBuffer->genNormalBuffer(normals);
        mSurfaceBuffer->genIndexBuffer(indices, GL_POINTS);
    }

private:
    BufferObject *mInteriorBuffer;
    BufferObject *mSurfaceBuffer;
}; // ActiveScalarValuesOp


class ActiveVectorValuesOp
{
public:

    ActiveVectorValuesOp(BufferObject& vectorBuffer)
        : mVectorBuffer(&vectorBuffer)
    {
    }

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        typedef typename GridType::ValueType ValueType;
        typedef typename GridType::TreeType TreeType;
        typedef typename TreeType::template ValueConverter<bool>::Type BoolTreeT;


        const TreeType& tree = grid->tree();

        double length = 0.0;
        {
            ValueType minVal, maxVal;
            tree.evalMinMax(minVal, maxVal);
            length = maxVal.length();
        }

        typename BoolTreeT::Ptr mask(new BoolTreeT(false));
        mask->topologyUnion(tree);
        mask->voxelizeActiveTiles();

        ///@todo thread and restructure.

        const Index64 voxelCount = mask->activeLeafVoxelCount();

        const Index64 pointCount = voxelCount * 2;
        std::vector<GLfloat> points(pointCount*3), colors(pointCount*3);
        std::vector<GLuint> indices(pointCount);

        openvdb::Coord ijk;
        openvdb::Vec3d pos, color, normal;
        openvdb::tree::LeafManager<BoolTreeT> leafs(*mask);

        openvdb::tree::ValueAccessor<const TreeType> acc(tree);

        Index64 idx = 0, pt = 0, cc = 0;
        for (Index64 l = 0, L = leafs.leafCount(); l < L; ++l) {
            typename BoolTreeT::LeafNodeType::ValueOnIter iter = leafs.leaf(l).beginValueOn();
            for (; iter; ++iter) {
                ijk = iter.getCoord();
                ValueType vec = acc.getValue(ijk);

                pos = grid->indexToWorld(ijk);

                points[idx++] = pos[0];
                points[idx++] = pos[1];
                points[idx++] = pos[2];

                indices[pt] = pt;
                ++pt;
                indices[pt] = pt;

                ++pt;
                double w = vec.length() / length;
                vec.normalize();
                pos += grid->voxelSize()[0] * 0.9 * vec;

                points[idx++] = pos[0];
                points[idx++] = pos[1];
                points[idx++] = pos[2];


                color = w * openvdb::Vec3d(0.9, 0.3, 0.3)
                    + (1.0 - w) * openvdb::Vec3d(0.3, 0.3, 0.9);

                colors[cc++] = color[0]  * 0.3;
                colors[cc++] = color[1]  * 0.3;
                colors[cc++] = color[2]  * 0.3;

                colors[cc++] = color[0];
                colors[cc++] = color[1];
                colors[cc++] = color[2];
            }
        }

        mVectorBuffer->genVertexBuffer(points);
        mVectorBuffer->genColorBuffer(colors);
        mVectorBuffer->genIndexBuffer(indices, GL_LINES);
    }

private:
    BufferObject *mVectorBuffer;

}; // ActiveVectorValuesOp


////////////////////////////////////////


class MeshOp
{
public:
    MeshOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        using openvdb::Index64;

        openvdb::tools::VolumeToMesh mesher(
            grid->getGridClass() == openvdb::GRID_LEVEL_SET ? 0.0 : 0.01);
        mesher(*grid);

        // Copy points and generate point normals.
        std::vector<GLfloat> points(mesher.pointListSize() * 3);
        std::vector<GLfloat> normals(mesher.pointListSize() * 3);

        openvdb::tree::ValueAccessor<const typename GridType::TreeType> acc(grid->tree());
        typedef openvdb::math::Gradient<openvdb::math::GenericMap, openvdb::math::CD_2ND> Gradient;
        openvdb::math::GenericMap map(grid->transform());
        openvdb::Coord ijk;

        for (Index64 n = 0, i = 0,  N = mesher.pointListSize(); n < N; ++n) {
            const openvdb::Vec3s& p = mesher.pointList()[n];
            points[i++] = p[0];
            points[i++] = p[1];
            points[i++] = p[2];
        }

        // Copy primitives
        openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();
        Index64 numQuads = 0;
        for (Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            numQuads += polygonPoolList[n].numQuads();
        }

        std::vector<GLuint> indices;
        indices.reserve(numQuads * 4);
        openvdb::Vec3d normal, e1, e2;

        for (Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
            for (Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
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

                for (Index64 v = 0; v < 4; ++v) {
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

    static openvdb::Vec3s sNodeColors[];

}; // MeshOp


////////////////////////////////////////


namespace util {

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

} // namespace openvdb_viewer

#endif // OPENVDB_VIEWER_RENDERMODULES_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
