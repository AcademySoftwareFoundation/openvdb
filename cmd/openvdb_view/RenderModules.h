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

#ifndef OPENVDB_VIEWER_MODULE_HAS_BEEN_INCLUDED
#define OPENVDB_VIEWER_MODULE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

////////////////////////////////////////

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

protected:
    RenderModule() {}
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
    TreeTopologyModule(const openvdb::GridCPtrVec&);
    ~TreeTopologyModule() {}

    void render();

private:
    void init();
    const openvdb::GridCPtrVec mGrids;

    std::vector<BufferObject> mBufferObjects;
    bool mIsInitialized;
    ShaderProgram mShader;
};


////////////////////////////////////////


/// @brief Surfacing render module
class MeshModule: public RenderModule
{
public:
    MeshModule(const openvdb::GridCPtrVec&);
    ~MeshModule() {}

    void render();
private:
    void init();
    const openvdb::GridCPtrVec mGrids;

    std::vector<BufferObject> mBufferObjects;
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
        size_t nodeCount = grid->tree().leafCount() + grid->tree().nonLeafCount();
        const size_t N = nodeCount * 8 * 3;

        std::vector<GLfloat> points(N);
        std::vector<GLfloat> colors(N);
        std::vector<GLuint> indices(N);


        openvdb::Vec3d ptn;
        openvdb::Vec3s color;
        openvdb::CoordBBox bbox;
        size_t pOffset = 0, iOffset = 0,  cOffset = 0, idx = 0;

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

            for (size_t n = 0; n < 8; ++n) {
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


class MeshOp
{
public:
    MeshOp(BufferObject& buffer) : mBuffer(&buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid)
    {
        openvdb::tools::VolumeToMesh mesher(
            grid->getGridClass() == openvdb::GRID_LEVEL_SET ? 0.0 : 0.01);
        mesher(*grid);

        // Copy points and generate point normals.
        std::vector<GLfloat> points, normals(mesher.pointListSize() * 3);
        points.reserve(mesher.pointListSize() * 3);
        //normals.reserve(mesher.pointListSize() * 3);

        openvdb::tree::ValueAccessor<const typename GridType::TreeType> acc(grid->tree());
        typedef openvdb::math::Gradient<openvdb::math::GenericMap, openvdb::math::CD_2ND> Gradient;
        openvdb::math::GenericMap map(grid->transform());
        openvdb::Coord ijk;

        for (size_t n = 0, N = mesher.pointListSize(); n < N; ++n) {
            const openvdb::Vec3s& p = mesher.pointList()[n];
            points.push_back(p[0]);
            points.push_back(p[1]);
            points.push_back(p[2]);
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
    else return false; ///< @todo throw exception ("unknown grid type")
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
    else return false; ///< @todo throw exception ("grid type is not scalar")
    return true;
}

} // namespace util

#endif // OPENVDB_VIEWER_MODULE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
