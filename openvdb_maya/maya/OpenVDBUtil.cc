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


#include "OpenVDBUtil.h"

#include <maya/MGlobal.h>


namespace openvdb_maya {


////////////////////////////////////////


const OpenVDBData*
getInputVDB(const MObject& vdb, MDataBlock& data)
{
    MStatus status;
    MDataHandle inputVdbHandle = data.inputValue(vdb, &status);

    if (status != MS::kSuccess) {
        MGlobal::displayError("Invalid VDB input");
        return NULL;
    }

    MFnPluginData inputPluginData(inputVdbHandle.data());
    const MPxData * inputPxData = inputPluginData.data();

    if (!inputPxData) {
        MGlobal::displayError("Invalid VDB input");
        return NULL;
    }

    return dynamic_cast<const OpenVDBData*>(inputPxData);
}

void getGrids(std::vector<openvdb::GridBase::ConstPtr>& grids,
    const OpenVDBData& vdb, const std::string& names)
{
    grids.clear();

    if (names.empty() || names == "*") {
        for (size_t n = 0, N = vdb.numberOfGrids(); n < N; ++n) {
            grids.push_back(vdb.gridPtr(n));
        }
    } else {
        for (size_t n = 0, N = vdb.numberOfGrids(); n < N; ++n) {
            if (vdb.grid(n).getName() == names) grids.push_back(vdb.gridPtr(n));
        }
    }
}


std::string getGridNames(const OpenVDBData& vdb)
{
    std::vector<std::string> names;
    for (size_t n = 0, N = vdb.numberOfGrids(); n < N; ++n) {
        names.push_back(vdb.grid(n).getName());
    }

    return boost::algorithm::join(names, " ");
}


bool containsGrid(const std::vector<std::string>& selectionList,
    const std::string& gridName, size_t gridIndex)
{
    for (size_t n = 0, N = selectionList.size(); n < N; ++n) {

        const std::string& word = selectionList[n];

        try {

            return boost::lexical_cast<size_t>(word) == gridIndex;

        } catch (const boost::bad_lexical_cast&) {

            bool match = true;
            for (size_t i = 0, I = std::min(word.length(), gridName.length()); i < I; ++i) {

                if (word[i] == '*') {
                    return true;
                } else if (word[i] != gridName[i]) {
                    match = false;
                    break;
                }
            }

            if (match && (word.length() == gridName.length())) return true;
        }
    }

    return selectionList.empty();
}


bool
getSelectedGrids(GridCPtrVec& grids, const std::string& selection,
    const OpenVDBData& inputVdb, OpenVDBData& outputVdb)
{
    grids.clear();

    std::vector<std::string> selectionList;
    boost::split(selectionList, selection, boost::is_any_of(" "));

    for (size_t n = 0, N = inputVdb.numberOfGrids(); n < N; ++n) {

        GridCRef grid = inputVdb.grid(n);

        if (containsGrid(selectionList, grid.getName(), n)) {
            grids.push_back(inputVdb.gridPtr(n));
        } else {
            outputVdb.insert(grid);
        }
    }

    return !grids.empty();
}


bool
getSelectedGrids(GridCPtrVec& grids, const std::string& selection,
    const OpenVDBData& inputVdb)
{
    grids.clear();

    std::vector<std::string> selectionList;
    boost::split(selectionList, selection, boost::is_any_of(" "));

    for (size_t n = 0, N = inputVdb.numberOfGrids(); n < N; ++n) {

        GridCRef grid = inputVdb.grid(n);

        if (containsGrid(selectionList, grid.getName(), n)) {
            grids.push_back(inputVdb.gridPtr(n));
        }
    }

    return !grids.empty();
}


////////////////////////////////////////


void
printGridInfo(std::ostream& os, const OpenVDBData& vdb)
{
    os << "\nOutput " << vdb.numberOfGrids() << " VDB(s)\n";
    openvdb::GridPtrVec::const_iterator it;

    size_t memUsage = 0, idx = 0;
    for (size_t n = 0, N = vdb.numberOfGrids(); n < N; ++n) {

        const openvdb::GridBase& grid = vdb.grid(n);

        memUsage += grid.memUsage();
        openvdb::Coord dim = grid.evalActiveVoxelDim();

        os << "[" << idx++ << "]";

        if (!grid.getName().empty()) os << " '" << grid.getName() << "'";

        os << " voxel size: " << grid.voxelSize()[0] << ", type: "
            << grid.valueType() << ", dim: "
            << dim[0] << "x" << dim[1] << "x" << dim[2] <<"\n";
    }

    openvdb::util::printBytes(os, memUsage, "\nApproximate Memory Usage:");
}


void
updateNodeInfo(std::stringstream& stream, MDataBlock& data, MObject& strAttr)
{
    MString str = stream.str().c_str();
    MDataHandle strHandle = data.outputValue(strAttr);
    strHandle.set(str);
    data.setClean(strAttr);
}


////////////////////////////////////////


BufferObject::BufferObject():
    mVertexBuffer(0),
    mNormalBuffer(0),
    mIndexBuffer(0),
    mColorBuffer(0),
    mPrimType(GL_POINTS),
    mPrimNum(0)
{
}

BufferObject::~BufferObject() { clear(); }

void
BufferObject::render() const
{
    if (mPrimNum == 0 || !glIsBuffer(mIndexBuffer) || !glIsBuffer(mVertexBuffer)) {
        OPENVDB_LOG_DEBUG_RUNTIME("request to render empty or uninitialized buffer");
        return;
    }

    const bool usesColorBuffer = glIsBuffer(mColorBuffer);
    const bool usesNormalBuffer = glIsBuffer(mNormalBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    if (usesColorBuffer) {
        glBindBuffer(GL_ARRAY_BUFFER, mColorBuffer);
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, 0);
    }

    if (usesNormalBuffer) {
        glEnableClientState(GL_NORMAL_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
        glNormalPointer(GL_FLOAT, 0, 0);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    glDrawElements(mPrimType, mPrimNum, GL_UNSIGNED_INT, 0);

    // disable client-side capabilities
    if (usesColorBuffer) glDisableClientState(GL_COLOR_ARRAY);
    if (usesNormalBuffer) glDisableClientState(GL_NORMAL_ARRAY);

    // release vbo's
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void
BufferObject::genIndexBuffer(const std::vector<GLuint>& v, GLenum primType)
{
    // clear old buffer
    if (glIsBuffer(mIndexBuffer) == GL_TRUE) glDeleteBuffers(1, &mIndexBuffer);

    // gen new buffer
    glGenBuffers(1, &mIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
    if (glIsBuffer(mIndexBuffer) == GL_FALSE) throw "Error: Unable to create index buffer";

    // upload data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        sizeof(GLuint) * v.size(), &v[0], GL_STATIC_DRAW); // upload data
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload index buffer data";

    // release buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    mPrimNum = v.size();
    mPrimType = primType;
}

void
BufferObject::genVertexBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mVertexBuffer) == GL_TRUE) glDeleteBuffers(1, &mVertexBuffer);

    glGenBuffers(1, &mVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
    if (glIsBuffer(mVertexBuffer) == GL_FALSE) throw "Error: Unable to create vertex buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload vertex buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::genNormalBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mNormalBuffer) == GL_TRUE) glDeleteBuffers(1, &mNormalBuffer);

    glGenBuffers(1, &mNormalBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mNormalBuffer);
    if (glIsBuffer(mNormalBuffer) == GL_FALSE) throw "Error: Unable to create normal buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload normal buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::genColorBuffer(const std::vector<GLfloat>& v)
{
    if (glIsBuffer(mColorBuffer) == GL_TRUE) glDeleteBuffers(1, &mColorBuffer);

    glGenBuffers(1, &mColorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mColorBuffer);
    if (glIsBuffer(mColorBuffer) == GL_FALSE) throw "Error: Unable to create color buffer";

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * v.size(), &v[0], GL_STATIC_DRAW);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to upload color buffer data";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
BufferObject::clear()
{
    if (glIsBuffer(mIndexBuffer) == GL_TRUE) glDeleteBuffers(1, &mIndexBuffer);
    if (glIsBuffer(mVertexBuffer) == GL_TRUE) glDeleteBuffers(1, &mVertexBuffer);
    if (glIsBuffer(mColorBuffer) == GL_TRUE) glDeleteBuffers(1, &mColorBuffer);
    if (glIsBuffer(mNormalBuffer) == GL_TRUE) glDeleteBuffers(1, &mNormalBuffer);

    mPrimType = GL_POINTS;
    mPrimNum = 0;
}

////////////////////////////////////////

ShaderProgram::ShaderProgram():
    mProgram(0),
    mVertShader(0),
    mFragShader(0)
{
}

ShaderProgram::~ShaderProgram() { clear(); }

void
ShaderProgram::setVertShader(const std::string& s)
{
    mVertShader = glCreateShader(GL_VERTEX_SHADER);
    if (glIsShader(mVertShader) == GL_FALSE) throw "Error: Unable to create shader program.";

    GLint length = s.length();
    const char *str = s.c_str();
    glShaderSource(mVertShader, 1, &str, &length);

    glCompileShader(mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to compile vertex shader.";
}

void
ShaderProgram::setFragShader(const std::string& s)
{
    mFragShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (glIsShader(mFragShader) == GL_FALSE) throw "Error: Unable to create shader program.";

    GLint length = s.length();
    const char *str = s.c_str();
    glShaderSource(mFragShader, 1, &str, &length);

    glCompileShader(mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to compile fragment shader.";
}

void
ShaderProgram::build()
{
    mProgram = glCreateProgram();
    if (glIsProgram(mProgram) == GL_FALSE) throw "Error: Unable to create shader program.";

    if (glIsShader(mVertShader) == GL_TRUE) glAttachShader(mProgram, mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach vertex shader.";

    if (glIsShader(mFragShader) == GL_TRUE) glAttachShader(mProgram, mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach fragment shader.";


    glLinkProgram(mProgram);

    GLint linked;
    glGetProgramiv(mProgram, GL_LINK_STATUS, &linked);

    if (!linked) throw "Error: Unable to link shader program.";
}

void
ShaderProgram::build(const std::vector<GLchar*>& attributes)
{
    mProgram = glCreateProgram();
    if (glIsProgram(mProgram) == GL_FALSE) throw "Error: Unable to create shader program.";


    for (GLuint n = 0, N = attributes.size(); n < N; ++n) {
        glBindAttribLocation(mProgram, n, attributes[n]);
    }

    if (glIsShader(mVertShader) == GL_TRUE) glAttachShader(mProgram, mVertShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach vertex shader.";

    if (glIsShader(mFragShader) == GL_TRUE) glAttachShader(mProgram, mFragShader);
    if (GL_NO_ERROR != glGetError()) throw "Error: Unable to attach fragment shader.";


    glLinkProgram(mProgram);

    GLint linked;
    glGetProgramiv(mProgram, GL_LINK_STATUS, &linked);

    if (!linked) throw "Error: Unable to link shader program.";
}

void
ShaderProgram::startShading() const
{
    if (glIsProgram(mProgram) == GL_FALSE)
        throw "Error: called startShading() on uncompiled shader program.";

    glUseProgram(mProgram);
}

void
ShaderProgram::stopShading() const
{
    glUseProgram(0);
}

void
ShaderProgram::clear()
{
    GLsizei numShaders;
    GLuint shaders[2];

    glGetAttachedShaders(mProgram, 2, &numShaders, shaders);

    // detach and remove shaders
    for (GLsizei n = 0; n < numShaders; ++n) {

        glDetachShader(mProgram, shaders[n]);

        if (glIsShader(shaders[n]) == GL_TRUE) glDeleteShader(shaders[n]);
    }

    // remove program
    if (glIsProgram(mProgram)) glDeleteProgram(mProgram);
}


////////////////////////////////////////

WireBoxBuilder::WireBoxBuilder(
    const openvdb::math::Transform& xform,
    std::vector<GLuint>& indices,
    std::vector<GLfloat>& points,
    std::vector<GLfloat>& colors)
    : mXForm(&xform)
    , mIndices(&indices)
    , mPoints(&points)
    , mColors(&colors)
{
}

void WireBoxBuilder::add(GLuint boxIndex, const openvdb::CoordBBox& bbox, const openvdb::Vec3s& color)
{
    GLuint ptnCount = boxIndex * 8;

    // Generate corner points

    GLuint ptnOffset = ptnCount * 3;
    GLuint colorOffset = ptnOffset;

    // Nodes are rendered as cell-centered
    const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
    const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);

    // corner 1
    openvdb::Vec3d ptn = mXForm->indexToWorld(min);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 2
    ptn.x() = min.x();
    ptn.y() = min.y();
    ptn.z() = max.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 3
    ptn.x() = max.x();
    ptn.y() = min.y();
    ptn.z() = max.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 4
    ptn.x() = max.x();
    ptn.y() = min.y();
    ptn.z() = min.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 5
    ptn.x() = min.x();
    ptn.y() = max.y();
    ptn.z() = min.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 6
    ptn.x() = min.x();
    ptn.y() = max.y();
    ptn.z() = max.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 7
    ptn = mXForm->indexToWorld(max);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset++] = ptn[2];

    // corner 8
    ptn.x() = max.x();
    ptn.y() = max.y();
    ptn.z() = min.z();
    ptn = mXForm->indexToWorld(ptn);
    (*mPoints)[ptnOffset++] = ptn[0];
    (*mPoints)[ptnOffset++] = ptn[1];
    (*mPoints)[ptnOffset] = ptn[2];

    for (int n = 0; n < 8; ++n) {
        (*mColors)[colorOffset++] = color[0];
        (*mColors)[colorOffset++] = color[1];
        (*mColors)[colorOffset++] = color[2];
    }

    // Generate edges

    GLuint edgeOffset = boxIndex * 24;

    // edge 1
    (*mIndices)[edgeOffset++] = ptnCount;
    (*mIndices)[edgeOffset++] = ptnCount + 1;
    // edge 2
    (*mIndices)[edgeOffset++] = ptnCount + 1;
    (*mIndices)[edgeOffset++] = ptnCount + 2;
    // edge 3
    (*mIndices)[edgeOffset++] = ptnCount + 2;
    (*mIndices)[edgeOffset++] = ptnCount + 3;
    // edge 4
    (*mIndices)[edgeOffset++] = ptnCount + 3;
    (*mIndices)[edgeOffset++] = ptnCount;
    // edge 5
    (*mIndices)[edgeOffset++] = ptnCount + 4;
    (*mIndices)[edgeOffset++] = ptnCount + 5;
    // edge 6
    (*mIndices)[edgeOffset++] = ptnCount + 5;
    (*mIndices)[edgeOffset++] = ptnCount + 6;
    // edge 7
    (*mIndices)[edgeOffset++] = ptnCount + 6;
    (*mIndices)[edgeOffset++] = ptnCount + 7;
    // edge 8
    (*mIndices)[edgeOffset++] = ptnCount + 7;
    (*mIndices)[edgeOffset++] = ptnCount + 4;
    // edge 9
    (*mIndices)[edgeOffset++] = ptnCount;
    (*mIndices)[edgeOffset++] = ptnCount + 4;
    // edge 10
    (*mIndices)[edgeOffset++] = ptnCount + 1;
    (*mIndices)[edgeOffset++] = ptnCount + 5;
    // edge 11
    (*mIndices)[edgeOffset++] = ptnCount + 2;
    (*mIndices)[edgeOffset++] = ptnCount + 6;
    // edge 12
    (*mIndices)[edgeOffset++] = ptnCount + 3;
    (*mIndices)[edgeOffset]   = ptnCount + 7;
}


} // namespace util


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
