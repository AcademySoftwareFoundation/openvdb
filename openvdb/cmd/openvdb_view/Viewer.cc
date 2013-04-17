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

#include "Viewer.h"
#include <iomanip> // for std::setprecision()
#include <iostream>
#include <sstream>
#include <math.h>
#include <limits>


////////////////////////////////////////

class BitmapFont13
{
public:
    BitmapFont13() {}

    static void initialize();

    static void enableFontRendering();
    static void disableFontRendering();

    static void print(GLint px, GLint py, const std::string&);

private:
    static GLuint sOffset;
    static GLubyte sCharacters[95][13];
};

GLuint BitmapFont13::sOffset = 0;

GLubyte BitmapFont13::sCharacters[95][13] = {
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X18, 0X18, 0X00, 0X00, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X36, 0X36, 0X36, 0X36}, 
    {0X00, 0X00, 0X00, 0X66, 0X66, 0XFF, 0X66, 0X66, 0XFF, 0X66, 0X66, 0X00, 0X00}, 
    {0X00, 0X00, 0X18, 0X7E, 0XFF, 0X1B, 0X1F, 0X7E, 0XF8, 0XD8, 0XFF, 0X7E, 0X18}, 
    {0X00, 0X00, 0X0E, 0X1B, 0XDB, 0X6E, 0X30, 0X18, 0X0C, 0X76, 0XDB, 0XD8, 0X70}, 
    {0X00, 0X00, 0X7F, 0XC6, 0XCF, 0XD8, 0X70, 0X70, 0XD8, 0XCC, 0XCC, 0X6C, 0X38}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X18, 0X1C, 0X0C, 0X0E}, 
    {0X00, 0X00, 0X0C, 0X18, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X18, 0X0C}, 
    {0X00, 0X00, 0X30, 0X18, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X18, 0X30}, 
    {0X00, 0X00, 0X00, 0X00, 0X99, 0X5A, 0X3C, 0XFF, 0X3C, 0X5A, 0X99, 0X00, 0X00}, 
    {0X00, 0X00, 0X00, 0X18, 0X18, 0X18, 0XFF, 0XFF, 0X18, 0X18, 0X18, 0X00, 0X00}, 
    {0X00, 0X00, 0X30, 0X18, 0X1C, 0X1C, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0XFF, 0XFF, 0X00, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X00, 0X38, 0X38, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X60, 0X60, 0X30, 0X30, 0X18, 0X18, 0X0C, 0X0C, 0X06, 0X06, 0X03, 0X03}, 
    {0X00, 0X00, 0X3C, 0X66, 0XC3, 0XE3, 0XF3, 0XDB, 0XCF, 0XC7, 0XC3, 0X66, 0X3C}, 
    {0X00, 0X00, 0X7E, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X78, 0X38, 0X18}, 
    {0X00, 0X00, 0XFF, 0XC0, 0XC0, 0X60, 0X30, 0X18, 0X0C, 0X06, 0X03, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X7E, 0XE7, 0X03, 0X03, 0X07, 0X7E, 0X07, 0X03, 0X03, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0XFF, 0XCC, 0X6C, 0X3C, 0X1C, 0X0C}, 
    {0X00, 0X00, 0X7E, 0XE7, 0X03, 0X03, 0X07, 0XFE, 0XC0, 0XC0, 0XC0, 0XC0, 0XFF}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC3, 0XC3, 0XC7, 0XFE, 0XC0, 0XC0, 0XC0, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X30, 0X30, 0X30, 0X30, 0X18, 0X0C, 0X06, 0X03, 0X03, 0X03, 0XFF}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC3, 0XC3, 0XE7, 0X7E, 0XE7, 0XC3, 0XC3, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X7E, 0XE7, 0X03, 0X03, 0X03, 0X7F, 0XE7, 0XC3, 0XC3, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X00, 0X38, 0X38, 0X00, 0X00, 0X38, 0X38, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X30, 0X18, 0X1C, 0X1C, 0X00, 0X00, 0X1C, 0X1C, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X06, 0X0C, 0X18, 0X30, 0X60, 0XC0, 0X60, 0X30, 0X18, 0X0C, 0X06}, 
    {0X00, 0X00, 0X00, 0X00, 0XFF, 0XFF, 0X00, 0XFF, 0XFF, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X60, 0X30, 0X18, 0X0C, 0X06, 0X03, 0X06, 0X0C, 0X18, 0X30, 0X60}, 
    {0X00, 0X00, 0X18, 0X00, 0X00, 0X18, 0X18, 0X0C, 0X06, 0X03, 0XC3, 0XC3, 0X7E}, 
    {0X00, 0X00, 0X3F, 0X60, 0XCF, 0XDB, 0XD3, 0XDD, 0XC3, 0X7E, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC3, 0XC3, 0XC3, 0XC3, 0XFF, 0XC3, 0XC3, 0XC3, 0X66, 0X3C, 0X18}, 
    {0X00, 0X00, 0XFE, 0XC7, 0XC3, 0XC3, 0XC7, 0XFE, 0XC7, 0XC3, 0XC3, 0XC7, 0XFE}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XE7, 0X7E}, 
    {0X00, 0X00, 0XFC, 0XCE, 0XC7, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC7, 0XCE, 0XFC}, 
    {0X00, 0X00, 0XFF, 0XC0, 0XC0, 0XC0, 0XC0, 0XFC, 0XC0, 0XC0, 0XC0, 0XC0, 0XFF}, 
    {0X00, 0X00, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XFC, 0XC0, 0XC0, 0XC0, 0XFF}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC3, 0XC3, 0XCF, 0XC0, 0XC0, 0XC0, 0XC0, 0XE7, 0X7E}, 
    {0X00, 0X00, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XFF, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3}, 
    {0X00, 0X00, 0X7E, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X7E}, 
    {0X00, 0X00, 0X7C, 0XEE, 0XC6, 0X06, 0X06, 0X06, 0X06, 0X06, 0X06, 0X06, 0X06}, 
    {0X00, 0X00, 0XC3, 0XC6, 0XCC, 0XD8, 0XF0, 0XE0, 0XF0, 0XD8, 0XCC, 0XC6, 0XC3}, 
    {0X00, 0X00, 0XFF, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0}, 
    {0X00, 0X00, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XDB, 0XFF, 0XFF, 0XE7, 0XC3}, 
    {0X00, 0X00, 0XC7, 0XC7, 0XCF, 0XCF, 0XDF, 0XDB, 0XFB, 0XF3, 0XF3, 0XE3, 0XE3}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XE7, 0X7E}, 
    {0X00, 0X00, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XFE, 0XC7, 0XC3, 0XC3, 0XC7, 0XFE}, 
    {0X00, 0X00, 0X3F, 0X6E, 0XDF, 0XDB, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0X66, 0X3C}, 
    {0X00, 0X00, 0XC3, 0XC6, 0XCC, 0XD8, 0XF0, 0XFE, 0XC7, 0XC3, 0XC3, 0XC7, 0XFE}, 
    {0X00, 0X00, 0X7E, 0XE7, 0X03, 0X03, 0X07, 0X7E, 0XE0, 0XC0, 0XC0, 0XE7, 0X7E}, 
    {0X00, 0X00, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0XFF}, 
    {0X00, 0X00, 0X7E, 0XE7, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3}, 
    {0X00, 0X00, 0X18, 0X3C, 0X3C, 0X66, 0X66, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3}, 
    {0X00, 0X00, 0XC3, 0XE7, 0XFF, 0XFF, 0XDB, 0XDB, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3}, 
    {0X00, 0X00, 0XC3, 0X66, 0X66, 0X3C, 0X3C, 0X18, 0X3C, 0X3C, 0X66, 0X66, 0XC3}, 
    {0X00, 0X00, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X3C, 0X3C, 0X66, 0X66, 0XC3}, 
    {0X00, 0X00, 0XFF, 0XC0, 0XC0, 0X60, 0X30, 0X7E, 0X0C, 0X06, 0X03, 0X03, 0XFF}, 
    {0X00, 0X00, 0X3C, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X30, 0X3C}, 
    {0X00, 0X03, 0X03, 0X06, 0X06, 0X0C, 0X0C, 0X18, 0X18, 0X30, 0X30, 0X60, 0X60}, 
    {0X00, 0X00, 0X3C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X3C}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0XC3, 0X66, 0X3C, 0X18}, 
    {0XFF, 0XFF, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X18, 0X38, 0X30, 0X70}, 
    {0X00, 0X00, 0X7F, 0XC3, 0XC3, 0X7F, 0X03, 0XC3, 0X7E, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XFE, 0XC3, 0XC3, 0XC3, 0XC3, 0XFE, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0}, 
    {0X00, 0X00, 0X7E, 0XC3, 0XC0, 0XC0, 0XC0, 0XC3, 0X7E, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X7F, 0XC3, 0XC3, 0XC3, 0XC3, 0X7F, 0X03, 0X03, 0X03, 0X03, 0X03}, 
    {0X00, 0X00, 0X7F, 0XC0, 0XC0, 0XFE, 0XC3, 0XC3, 0X7E, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X30, 0X30, 0X30, 0X30, 0X30, 0XFC, 0X30, 0X30, 0X30, 0X33, 0X1E}, 
    {0X7E, 0XC3, 0X03, 0X03, 0X7F, 0XC3, 0XC3, 0XC3, 0X7E, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XC3, 0XFE, 0XC0, 0XC0, 0XC0, 0XC0}, 
    {0X00, 0X00, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X00, 0X00, 0X18, 0X00}, 
    {0X38, 0X6C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X0C, 0X00, 0X00, 0X0C, 0X00}, 
    {0X00, 0X00, 0XC6, 0XCC, 0XF8, 0XF0, 0XD8, 0XCC, 0XC6, 0XC0, 0XC0, 0XC0, 0XC0}, 
    {0X00, 0X00, 0X7E, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X78}, 
    {0X00, 0X00, 0XDB, 0XDB, 0XDB, 0XDB, 0XDB, 0XDB, 0XFE, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC6, 0XC6, 0XC6, 0XC6, 0XC6, 0XC6, 0XFC, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X7C, 0XC6, 0XC6, 0XC6, 0XC6, 0XC6, 0X7C, 0X00, 0X00, 0X00, 0X00}, 
    {0XC0, 0XC0, 0XC0, 0XFE, 0XC3, 0XC3, 0XC3, 0XC3, 0XFE, 0X00, 0X00, 0X00, 0X00}, 
    {0X03, 0X03, 0X03, 0X7F, 0XC3, 0XC3, 0XC3, 0XC3, 0X7F, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC0, 0XC0, 0XC0, 0XC0, 0XC0, 0XE0, 0XFE, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XFE, 0X03, 0X03, 0X7E, 0XC0, 0XC0, 0X7F, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X1C, 0X36, 0X30, 0X30, 0X30, 0X30, 0XFC, 0X30, 0X30, 0X30, 0X00}, 
    {0X00, 0X00, 0X7E, 0XC6, 0XC6, 0XC6, 0XC6, 0XC6, 0XC6, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X18, 0X3C, 0X3C, 0X66, 0X66, 0XC3, 0XC3, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC3, 0XE7, 0XFF, 0XDB, 0XC3, 0XC3, 0XC3, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XC3, 0X66, 0X3C, 0X18, 0X3C, 0X66, 0XC3, 0X00, 0X00, 0X00, 0X00}, 
    {0XC0, 0X60, 0X60, 0X30, 0X18, 0X3C, 0X66, 0X66, 0XC3, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0XFF, 0X60, 0X30, 0X18, 0X0C, 0X06, 0XFF, 0X00, 0X00, 0X00, 0X00}, 
    {0X00, 0X00, 0X0F, 0X18, 0X18, 0X18, 0X38, 0XF0, 0X38, 0X18, 0X18, 0X18, 0X0F}, 
    {0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18, 0X18}, 
    {0X00, 0X00, 0XF0, 0X18, 0X18, 0X18, 0X1C, 0X0F, 0X1C, 0X18, 0X18, 0X18, 0XF0}, 
    {0X00, 0X00, 0X00, 0X00, 0X00, 0X00, 0X06, 0X8F, 0XF1, 0X60, 0X00, 0X00, 0X00}
}; // sCharacters

void 
BitmapFont13::initialize()
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE

    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    BitmapFont13::sOffset = glGenLists(128);

    for (GLuint c = 32; c < 127; ++c) {
        glNewList(c + BitmapFont13::sOffset, GL_COMPILE);
        glBitmap(8, 13, 0.0, 2.0, 10.0, 0.0, BitmapFont13::sCharacters[c-32]);
        glEndList();
    }
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
}


void
BitmapFont13::enableFontRendering()
{
    glPushMatrix();
    int width, height;
    glfwGetWindowSize(&width, &height);
    height = height < 1 ? 1 : height;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho (0, width, 0, height, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void
BitmapFont13::disableFontRendering()
{
    glFlush();
    glPopMatrix();
}

void
BitmapFont13::print(GLint px, GLint py, const std::string& str)
{
    glRasterPos2i(px, py);
    glPushAttrib(GL_LIST_BIT);
    glListBase(BitmapFont13::sOffset);
    glCallLists(str.length(), GL_UNSIGNED_BYTE, reinterpret_cast<const GLubyte*>(str.c_str()));
    glPopAttrib();
}


////////////////////////////////////////

// Basic camera class

struct Viewer::Camera
{
    Camera();

    void aim();

    void lookAt(const openvdb::Vec3d& p, double dist = 1.0);
    void lookAtTarget();

    void setTarget(const openvdb::Vec3d& p, double dist = 1.0);

    void setNearFarPlanes(double n, double f) { mNearPlane = n; mFarPlane = f; }
    void setFieldOfView(double degrees) { mFov = degrees; }
    void setSpeed(double zoomSpeed, double strafeSpeed, double tumblingSpeed);

    void keyCallback(int key, int action);
    void mouseButtonCallback(int button, int action);
    void mousePosCallback(int x, int y);
    void mouseWheelCallback(int pos, int prevPos);

    bool needsDisplay() const { return mNeedsDisplay; }

private:
    // Camera parameters
    double mFov, mNearPlane, mFarPlane;
    openvdb::Vec3d mTarget, mLookAt, mUp, mForward, mRight, mEye;
    double mTumblingSpeed, mZoomSpeed, mStrafeSpeed;
    double mHead, mPitch, mTargetDistance, mDistance;

    // Input states
    bool mMouseDown, mStartTumbling, mZoomMode, mChanged, mNeedsDisplay;
    double mMouseXPos, mMouseYPos;
    int mWheelPos;

    static double sDeg2rad;
};

double Viewer::Camera::sDeg2rad = M_PI / 180.0;

Viewer::Camera::Camera()
    : mFov(65.0)
    , mNearPlane(0.1)
    , mFarPlane(10000.0)
    , mTarget(openvdb::Vec3d(0.0))
    , mLookAt(mTarget)
    , mUp(openvdb::Vec3d(0.0, 1.0, 0.0))
    , mForward(openvdb::Vec3d(0.0, 0.0, 1.0))
    , mRight(openvdb::Vec3d(1.0, 0.0, 0.0))
    , mEye(openvdb::Vec3d(0.0, 0.0, -1.0))
    , mTumblingSpeed(0.5)
    , mZoomSpeed(0.2)
    , mStrafeSpeed(0.05)
    , mHead(30.0)
    , mPitch(45.0)
    , mTargetDistance(25.0)
    , mDistance(mTargetDistance)
    , mMouseDown(false)
    , mStartTumbling(false)
    , mZoomMode(false)
    , mChanged(true)
    , mNeedsDisplay(true)
    , mMouseXPos(0.0)
    , mMouseYPos(0.0)
    , mWheelPos(0)
{
}

void
Viewer::Camera::lookAt(const openvdb::Vec3d& p, double dist)
{
    mLookAt = p;
    mDistance = dist;
    mNeedsDisplay = true;
}

void
Viewer::Camera::lookAtTarget()
{
    mLookAt = mTarget;
    mDistance = mTargetDistance;
    mNeedsDisplay = true;
}


void
Viewer::Camera::setSpeed(double zoomSpeed, double strafeSpeed, double tumblingSpeed)
{
    mZoomSpeed = std::max(0.0001, zoomSpeed);
    mStrafeSpeed = std::max(0.0001, strafeSpeed);
    mTumblingSpeed = std::max(0.2, tumblingSpeed);
    mTumblingSpeed = std::min(1.0, tumblingSpeed);
}

void
Viewer::Camera::setTarget(const openvdb::Vec3d& p, double dist)
{
    mTarget = p;
    mTargetDistance = dist;
}

void
Viewer::Camera::aim()
{
    // Get the window size
    int width, height;
    glfwGetWindowSize(&width, &height);

    // Make sure that height is non-zero to avoid division by zero
    height = height < 1 ? 1 : height;

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Window aspect (assumes square pixels)
    double aspectRatio = (double)width / (double)height;

    // Set perspective view (fov is in degrees in the y direction.)
    gluPerspective(mFov, aspectRatio, mNearPlane, mFarPlane);

    if (mChanged) {

        mChanged = false;

        mEye[0] = mLookAt[0] + mDistance * std::cos(mHead * sDeg2rad) * std::cos(mPitch * sDeg2rad);
        mEye[1] = mLookAt[1] + mDistance * std::sin(mHead * sDeg2rad);
        mEye[2] = mLookAt[2] + mDistance * std::cos(mHead * sDeg2rad) * std::sin(mPitch * sDeg2rad);

        mForward = mLookAt - mEye;
        mForward.normalize();

        mUp[1] = std::cos(mHead * sDeg2rad) > 0 ? 1.0 : -1.0;
        mRight = mForward.cross(mUp);
    }

    // Set up modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
        
    gluLookAt(mEye[0], mEye[1], mEye[2],
              mLookAt[0], mLookAt[1], mLookAt[2],
              mUp[0], mUp[1], mUp[2]);
    mNeedsDisplay = false;
}


void
Viewer::Camera::keyCallback(int key, int )
{
    if (glfwGetKey(key) == GLFW_PRESS) {
        switch(key) {
            case GLFW_KEY_SPACE:
                mZoomMode = true;
                break;
        }
    } else if (glfwGetKey(key) == GLFW_RELEASE) {
        switch(key) {
            case GLFW_KEY_SPACE:
                mZoomMode = false;
                break;
        }
    }

    mChanged = true;
}


void
Viewer::Camera::mouseButtonCallback(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) mMouseDown = true;
        else if (action == GLFW_RELEASE) mMouseDown = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mMouseDown = true;
            mZoomMode = true;
        } else if (action == GLFW_RELEASE) {
            mMouseDown = false;
            mZoomMode = false;
        }
    }
    if (action == GLFW_RELEASE) mMouseDown = false;

    mStartTumbling = true;
    mChanged = true;
}


void
Viewer::Camera::mousePosCallback(int x, int y)
{
    if (mStartTumbling) {
        mMouseXPos = x;
        mMouseYPos = y;
        mStartTumbling = false;
    }

    double dx, dy;
    dx = x - mMouseXPos;
    dy = y - mMouseYPos;

    if (mMouseDown && !mZoomMode) {
        mNeedsDisplay = true;
        mHead += dy * mTumblingSpeed;
        mPitch += dx * mTumblingSpeed;
    } else if (mMouseDown && mZoomMode) {
        mNeedsDisplay = true;
        mLookAt += (dy * mUp - dx * mRight) * mStrafeSpeed;
    }

    mMouseXPos = x;
    mMouseYPos = y;
    mChanged = true;
}


void
Viewer::Camera::mouseWheelCallback(int pos, int prevPos)
{
    double speed = std::abs(prevPos - pos);

    if (prevPos < pos) {
        mDistance += speed * mZoomSpeed;
        setSpeed(mDistance * 0.1, mDistance * 0.002, mDistance * 0.02);
    } else {
        double temp = mDistance - speed * mZoomSpeed;
        mDistance = std::max(0.0, temp);
        setSpeed(mDistance * 0.1, mDistance * 0.002, mDistance * 0.02);
    }

    mChanged = true;
    mNeedsDisplay = true;
}


////////////////////////////////////////

// Clip Box class

struct Viewer::ClipBox
{
    ClipBox();

    void enableClipping() const;
    void disableClipping() const;

    void setBBox(const openvdb::BBoxd&);
    void setStepSize(const openvdb::Vec3d& s) { mSeptSize = s; }


    void render();

    void update(double steps);
    void reset();

    bool isActive() const { return (mXIsActive || mYIsActive ||mZIsActive); }

    bool& activateXPlanes() { return mXIsActive;  }
    bool& activateYPlanes() { return mYIsActive;  }
    bool& activateZPlanes() { return mZIsActive;  }

    bool& shiftIsDown() { return mShiftIsDown; }
    bool& ctrlIsDown() { return mCtrlIsDown; }

private:
    
    void update() const;

    openvdb::Vec3d mSeptSize;
    openvdb::BBoxd mBBox;
    bool mXIsActive, mYIsActive, mZIsActive, mShiftIsDown, mCtrlIsDown;
    GLdouble mFrontPlane[4], mBackPlane[4], mLeftPlane[4], mRightPlane[4], mTopPlane[4], mBottomPlane[4];
};


Viewer::ClipBox::ClipBox()
    : mSeptSize(1.0)
    , mBBox()
    , mXIsActive(false)
    , mYIsActive(false)
    , mZIsActive(false)
    , mShiftIsDown(false)
    , mCtrlIsDown(false)
{
    GLdouble front [] = { 0.0, 0.0, 1.0, 0.0};
    std::copy(front, front + 4, mFrontPlane);

    GLdouble back [] = { 0.0, 0.0,-1.0, 0.0};
    std::copy(back, back + 4, mBackPlane);

    GLdouble left [] = { 1.0, 0.0, 0.0, 0.0};
    std::copy(left, left + 4, mLeftPlane);

    GLdouble right [] = {-1.0, 0.0, 0.0, 0.0};
    std::copy(right, right + 4, mRightPlane);

    GLdouble top [] = { 0.0, 1.0, 0.0, 0.0};
    std::copy(top, top + 4, mTopPlane);

    GLdouble bottom [] = { 0.0,-1.0, 0.0, 0.0};
    std::copy(bottom, bottom + 4, mBottomPlane);
}

void
Viewer::ClipBox::setBBox(const openvdb::BBoxd& bbox)
{
    mBBox = bbox;
    reset();
}

void
Viewer::ClipBox::update(double steps)
{
    if (mXIsActive) {
        GLdouble s = steps * mSeptSize.x() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mLeftPlane[3] -= s;
            mLeftPlane[3] = -std::min(-mLeftPlane[3], (mRightPlane[3] - mSeptSize.x()));
            mLeftPlane[3] = -std::max(-mLeftPlane[3], mBBox.min().x());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mRightPlane[3] += s;
            mRightPlane[3] = std::min(mRightPlane[3], mBBox.max().x());
            mRightPlane[3] = std::max(mRightPlane[3], (-mLeftPlane[3] + mSeptSize.x()));
        }

    }

     if (mYIsActive) {
        GLdouble s = steps * mSeptSize.y() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mTopPlane[3] -= s;
            mTopPlane[3] = -std::min(-mTopPlane[3], (mBottomPlane[3] - mSeptSize.y()));
            mTopPlane[3] = -std::max(-mTopPlane[3], mBBox.min().y());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mBottomPlane[3] += s;
            mBottomPlane[3] = std::min(mBottomPlane[3], mBBox.max().y());
            mBottomPlane[3] = std::max(mBottomPlane[3], (-mTopPlane[3] + mSeptSize.y()));
        }

    }

     if (mZIsActive) {
        GLdouble s = steps * mSeptSize.z() * 4.0;

        if (mShiftIsDown || mCtrlIsDown) {
            mFrontPlane[3] -= s;
            mFrontPlane[3] = -std::min(-mFrontPlane[3], (mBackPlane[3] - mSeptSize.z()));
            mFrontPlane[3] = -std::max(-mFrontPlane[3], mBBox.min().z());
        }

        if (!mShiftIsDown || mCtrlIsDown) {
            mBackPlane[3] += s;
            mBackPlane[3] = std::min(mBackPlane[3], mBBox.max().z());
            mBackPlane[3] = std::max(mBackPlane[3], (-mFrontPlane[3] + mSeptSize.z()));
        }

    }

}

void
Viewer::ClipBox::reset()
{
    mFrontPlane[3] = std::abs(mBBox.min().z());
    mBackPlane[3] = mBBox.max().z();

    mLeftPlane[3] = std::abs(mBBox.min().x());
    mRightPlane[3] = mBBox.max().x();

    mTopPlane[3] = std::abs(mBBox.min().y());
    mBottomPlane[3] = mBBox.max().y();
}

void
Viewer::ClipBox::update() const
{
    glClipPlane(GL_CLIP_PLANE0, mFrontPlane);
    glClipPlane(GL_CLIP_PLANE1, mBackPlane);
    glClipPlane(GL_CLIP_PLANE2, mLeftPlane);
    glClipPlane(GL_CLIP_PLANE3, mRightPlane);
    glClipPlane(GL_CLIP_PLANE4, mTopPlane);
    glClipPlane(GL_CLIP_PLANE5, mBottomPlane);
}

void
Viewer::ClipBox::enableClipping() const
{
    update();
    if (-mFrontPlane[3] > mBBox.min().z())   glEnable(GL_CLIP_PLANE0);
    if (mBackPlane[3] < mBBox.max().z())    glEnable(GL_CLIP_PLANE1);
    if (-mLeftPlane[3] > mBBox.min().x())    glEnable(GL_CLIP_PLANE2);
    if (mRightPlane[3] < mBBox.max().x())   glEnable(GL_CLIP_PLANE3);
    if (-mTopPlane[3] > mBBox.min().y())     glEnable(GL_CLIP_PLANE4);
    if (mBottomPlane[3] < mBBox.max().y())  glEnable(GL_CLIP_PLANE5);
}

void
Viewer::ClipBox::disableClipping() const
{
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
}

void
Viewer::ClipBox::render()
{
    glColor3f(0.1, 0.1, 0.9);
    bool drawBbox = false;
    if (-mFrontPlane[3] > mBBox.min().z()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), mBBox.min().y(), -mFrontPlane[3]);
        glVertex3f(mBBox.min().x(), mBBox.max().y(), -mFrontPlane[3]);
        glVertex3f(mBBox.max().x(), mBBox.max().y(), -mFrontPlane[3]);
        glVertex3f(mBBox.max().x(), mBBox.min().y(), -mFrontPlane[3]);
        glEnd();
        drawBbox = true;
    }

    if (mBackPlane[3] < mBBox.max().z()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), mBBox.min().y(), mBackPlane[3]);
        glVertex3f(mBBox.min().x(), mBBox.max().y(), mBackPlane[3]);
        glVertex3f(mBBox.max().x(), mBBox.max().y(), mBackPlane[3]);
        glVertex3f(mBBox.max().x(), mBBox.min().y(), mBackPlane[3]);
        glEnd();
        drawBbox = true;
    }

    glColor3f(0.9, 0.1, 0.1);
    if (-mLeftPlane[3] > mBBox.min().x()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(-mLeftPlane[3], mBBox.min().y(), mBBox.min().z());
        glVertex3f(-mLeftPlane[3], mBBox.max().y(), mBBox.min().z());
        glVertex3f(-mLeftPlane[3], mBBox.max().y(), mBBox.max().z());
        glVertex3f(-mLeftPlane[3], mBBox.min().y(), mBBox.max().z());
        glEnd();
        drawBbox = true;
    }

    if (mRightPlane[3] < mBBox.max().x()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(mRightPlane[3], mBBox.min().y(), mBBox.min().z());
        glVertex3f(mRightPlane[3], mBBox.max().y(), mBBox.min().z());
        glVertex3f(mRightPlane[3], mBBox.max().y(), mBBox.max().z());
        glVertex3f(mRightPlane[3], mBBox.min().y(), mBBox.max().z());
        glEnd();
        drawBbox = true;
    }

    glColor3f(0.1, 0.9, 0.1);
    if (-mTopPlane[3] > mBBox.min().y()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), -mTopPlane[3], mBBox.min().z());
        glVertex3f(mBBox.min().x(), -mTopPlane[3], mBBox.max().z());
        glVertex3f(mBBox.max().x(), -mTopPlane[3], mBBox.max().z());
        glVertex3f(mBBox.max().x(), -mTopPlane[3], mBBox.min().z());
        glEnd();
        drawBbox = true;
    }

    if (mBottomPlane[3] < mBBox.max().y()) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), mBottomPlane[3], mBBox.min().z());
        glVertex3f(mBBox.min().x(), mBottomPlane[3], mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBottomPlane[3], mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBottomPlane[3], mBBox.min().z());
        glEnd();
        drawBbox = true;
    }

    if (drawBbox) {
        glColor3f(0.5, 0.5, 0.5);
        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3f(mBBox.min().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.min().y(), mBBox.min().z());
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(mBBox.min().x(), mBBox.max().y(), mBBox.min().z());
        glVertex3f(mBBox.min().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.max().y(), mBBox.min().z());
        glEnd();

        glBegin(GL_LINES);
        glVertex3f(mBBox.min().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3f(mBBox.min().x(), mBBox.max().y(), mBBox.min().z());
        glVertex3f(mBBox.min().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3f(mBBox.min().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.min().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.max().y(), mBBox.max().z());
        glVertex3f(mBBox.max().x(), mBBox.min().y(), mBBox.min().z());
        glVertex3f(mBBox.max().x(), mBBox.max().y(), mBBox.min().z());
        glEnd();
    }
}

////////////////////////////////////////


Viewer::Viewer()
    : mCamera(new Camera)
    , mClipBox(new ClipBox)
    , mRenderModules(0)
    , mGrids()
    , mGridIdx(0)
    , mUpdates(0)
    , mGridName("")
    , mProgName("")
    , mGridInfo("")
    , mTransformInfo("")
    , mTreeInfo("")
    , mWheelPos(0)
    , mShiftIsDown(false)
    , mCtrlIsDown(false)
    , mShowInfo(true)
{
}


Viewer* Viewer::sViewer = NULL;
tbb::mutex sLock;


////////////////////////////////////////


Viewer&
Viewer::init(const std::string& progName, bool verbose)
{
    tbb::mutex::scoped_lock(sLock);

    if (sViewer == NULL) {
        OPENVDB_START_THREADSAFE_STATIC_WRITE
        sViewer = new Viewer;
        OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
    }

    sViewer->mProgName = progName;

    if (glfwInit() != GL_TRUE) {
        OPENVDB_LOG_ERROR("GLFW Initialization Failed.");
    }

    if (verbose) {
        if (glfwOpenWindow(100, 100, 8, 8, 8, 8, 24, 0, GLFW_WINDOW)) {
            int major, minor, rev;
            glfwGetVersion(&major, &minor, &rev);
            std::cout << "GLFW: " << major << "." << minor << "." << rev << "\n"
                << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
            glfwCloseWindow();
        }
    }

    return *sViewer;
}


////////////////////////////////////////


void
Viewer::setWindowTitle(double fps)
{
    std::ostringstream ss;
    ss  << mProgName << ": "
        << (mGridName.empty() ? std::string("OpenVDB") : mGridName)
        << " (grid " << (mGridIdx + 1) << " of " << mGrids.size() << ") @ "
        << std::setprecision(1) << std::fixed << fps << " fps";
    glfwSetWindowTitle(ss.str().c_str());
}


////////////////////////////////////////


void
Viewer::render()
{
    mCamera->aim();

    // draw scene
    mRenderModules[0]->render(); // ground plane.

    mClipBox->render();
    mClipBox->enableClipping();
    
    for (size_t n = 1, N = mRenderModules.size(); n < N; ++n) {
        mRenderModules[n]->render();
    }

    mClipBox->disableClipping();

    // Render text

    if (mShowInfo) {

    BitmapFont13::enableFontRendering();

    glColor3f (0.2, 0.2, 0.2);

    BitmapFont13::print(10, 50, mGridInfo);
    BitmapFont13::print(10, 30, mTransformInfo);
    BitmapFont13::print(10, 10, mTreeInfo);

    BitmapFont13::disableFontRendering();
    }
}


////////////////////////////////////////


void
Viewer::view(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    sViewer->viewGrids(gridList, width, height);
}

openvdb::BBoxd
worldSpaceBBox(const openvdb::math::Transform& xform, const openvdb::CoordBBox& bbox)
{
    openvdb::Vec3d pMin = openvdb::Vec3d(std::numeric_limits<double>::max());
    openvdb::Vec3d pMax = -pMin;

    const openvdb::Coord& min = bbox.min();
    const openvdb::Coord& max = bbox.max();
    openvdb::Coord ijk;

    // corner 1
    openvdb::Vec3d ptn = xform.indexToWorld(min);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 2
    ijk[0] = min.x();
    ijk[1] = min.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 3
    ijk[0] = max.x();
    ijk[1] = min.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }
    
    // corner 4
    ijk[0] = max.x();
    ijk[1] = min.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 5
    ijk[0] = min.x();
    ijk[1] = max.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 6
    ijk[0] = min.x();
    ijk[1] = max.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }


    // corner 7
    ptn = xform.indexToWorld(max);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 8
    ijk[0] = max.x();
    ijk[1] = max.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    return openvdb::BBoxd(pMin, pMax);
}

void
Viewer::viewGrids(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    mGrids = gridList;
    mGridIdx = size_t(-1);
    mGridName.clear();


    //////////

    // Create window

    if (!glfwOpenWindow(width, height,  // Window size
                       8, 8, 8, 8,      // # of R,G,B, & A bits
                       32, 0,           // # of depth & stencil buffer bits
                       GLFW_WINDOW))    // Window mode
    {
        glfwTerminate();
        return;
    }

    glfwSetWindowTitle(mProgName.c_str());
    glfwSwapBuffers();

    BitmapFont13::initialize();

    //////////

    // Eval grid bbox

    openvdb::BBoxd bbox(openvdb::Vec3d(0.0), openvdb::Vec3d(0.0));

    if (!gridList.empty()) {
        bbox = worldSpaceBBox(gridList[0]->transform(), gridList[0]->evalActiveVoxelBoundingBox());
        openvdb::Vec3d voxelSize = gridList[0]->voxelSize();

        for (size_t n = 1; n < gridList.size(); ++n) {
            bbox.expand(
                worldSpaceBBox(gridList[n]->transform(), gridList[n]->evalActiveVoxelBoundingBox()));

            voxelSize = minComponent(voxelSize, gridList[n]->voxelSize());
        }

        mClipBox->setStepSize(voxelSize);
    }

    mClipBox->setBBox(bbox);


    // setup camera

    openvdb::Vec3d extents = bbox.extents();
    double max_extent = std::max(extents[0], std::max(extents[1], extents[2]));

    mCamera->setTarget(bbox.getCenter(), max_extent);
    mCamera->lookAtTarget();
    mCamera->setSpeed(/*zoom=*/0.1, /*strafe=*/0.002, /*tumbling=*/0.02);

    //////////

    // register callback functions

    glfwSetKeyCallback(Viewer::keyCallback);
    glfwSetMouseButtonCallback(Viewer::mouseButtonCallback);
    glfwSetMousePosCallback(Viewer::mousePosCallback);
    glfwSetMouseWheelCallback(Viewer::mouseWheelCallback);
    glfwSetWindowSizeCallback(Viewer::windowSizeCallback);


    //////////

    // Screen color
    glClearColor(0.85, 0.85, 0.85, 0.0f);
    
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    glPointSize(4);
    glLineWidth(2);
    //////////

    // construct render modules
    showNthGrid(/*n=*/0);


    // main loop

    size_t frame = 0;
    double time = glfwGetTime();

    glfwSwapInterval(1);

    do {
        if (needsDisplay()) render();

        // eval fps
        ++frame;
        double elapsed = glfwGetTime() - time;
        if (elapsed > 1.0) {
            time = glfwGetTime();
            setWindowTitle(/*fps=*/double(frame) / elapsed);
            frame = 0;
        }

        // Swap front and back buffers
        glfwSwapBuffers();

    // exit if the esc key is pressed or the window is closed.
    } while (!glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED));

    glfwTerminate();
}


////////////////////////////////////////

void
Viewer::updateCutPlanes(int wheelPos)
{
    double speed = std::abs(mWheelPos - wheelPos);
    if (mWheelPos < wheelPos) mClipBox->update(speed);
    else mClipBox->update(-speed);
    setNeedsDisplay();
}


////////////////////////////////////////


void
Viewer::showPrevGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = ((numGrids + mGridIdx) - 1) % numGrids;
    showNthGrid(idx);
}


void
Viewer::showNextGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = (mGridIdx + 1) % numGrids;
    showNthGrid(idx);
}


void
Viewer::showNthGrid(size_t n)
{
    n = n % mGrids.size();
    if (n == mGridIdx) return;

    mGridName = mGrids[n]->getName();
    mGridIdx = n;

    // save render settings
    std::vector<bool> active(mRenderModules.size());
    for (size_t i = 0, I = active.size(); i < I; ++i) {
        active[i] = mRenderModules[i]->visible();
    }

    mRenderModules.clear();
    mRenderModules.push_back(RenderModulePtr(new ViewportModule));
    mRenderModules.push_back(RenderModulePtr(new TreeTopologyModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new MeshModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new ActiveValueModule(mGrids[n])));

    if (active.empty()) {
        for (size_t i = 2, I = mRenderModules.size(); i < I; ++i) {
            mRenderModules[i]->visible() = false;
        }
    } else {
        for (size_t i = 0, I = active.size(); i < I; ++i) {
            mRenderModules[i]->visible() = active[i];
        }
    }



    

    // Collect info
    {
        std::stringstream stream;
        stream << "Gid name: " << mGrids[n]->getName() << " Grid class: ";
        stream << openvdb::GridBase::gridClassToMenuName(mGrids[n]->getGridClass());
        stream << " Value type: " << mGrids[n]->valueType();
        mGridInfo = stream.str();
    }
    {
        openvdb::Coord dim = mGrids[n]->evalActiveVoxelDim();
        std::stringstream stream;
        stream << "Voxel resolution: " << dim[0] << "x" << dim[1] << "x" << dim[2];
        stream << " Voxel size: " << mGrids[n]->voxelSize()[0];
        stream << " Transform: " << mGrids[n]->transform().mapType();
        mTransformInfo = stream.str();
    }

    {
        std::stringstream stream;
        stream << "Active voxels: " << mGrids[n]->activeVoxelCount();
        mTreeInfo = stream.str();
    }

    setWindowTitle();
}


////////////////////////////////////////


void
Viewer::keyCallback(int key, int action)
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE

    sViewer->mCamera->keyCallback(key, action);
    const bool keyPress = glfwGetKey(key) == GLFW_PRESS;
    sViewer->mShiftIsDown = glfwGetKey(GLFW_KEY_LSHIFT);
    sViewer->mCtrlIsDown = glfwGetKey(GLFW_KEY_LCTRL);

    if (keyPress) {
        switch (key) {
        case '1':
            sViewer->toggleRenderModule(1);
            break;
        case '2':
            sViewer->toggleRenderModule(2);
            break;
        case '3':
            sViewer->toggleRenderModule(3);
            break;
        case 'c': case 'C':
            sViewer->mClipBox->reset();
            break;
        case 'h': case 'H': // center home
            sViewer->mCamera->lookAt(openvdb::Vec3d(0.0), 10.0);
            break;
        case 'g': case 'G': // center geometry
            sViewer->mCamera->lookAtTarget();
            break;
        case 'i': case 'I':
            sViewer->toggleInfoText();
            break;
        case GLFW_KEY_LEFT:
            sViewer->showPrevGrid();
            break;
        case GLFW_KEY_RIGHT:
            sViewer->showNextGrid();
            break;
        }
    }
    
    switch (key) {
    case 'x': case 'X':
        sViewer->mClipBox->activateXPlanes() = keyPress;
        break;
    case 'y': case 'Y':
        sViewer->mClipBox->activateYPlanes() = keyPress;
        break;
    case 'z': case 'Z':
        sViewer->mClipBox->activateZPlanes() = keyPress;
        break;
    }

    sViewer->mClipBox->shiftIsDown() = sViewer->mShiftIsDown;
    sViewer->mClipBox->ctrlIsDown() = sViewer->mCtrlIsDown;

    sViewer->setNeedsDisplay();

    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
}

void
Viewer::mouseButtonCallback(int button, int action)
{
    sViewer->mCamera->mouseButtonCallback(button, action);
    if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
}

void
Viewer::mousePosCallback(int x, int y)
{
    sViewer->mCamera->mousePosCallback(x, y);
    if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
}

void
Viewer::mouseWheelCallback(int pos)
{
    if (sViewer->mClipBox->isActive()) {
        sViewer->updateCutPlanes(pos);
    } else {
        sViewer->mCamera->mouseWheelCallback(pos, sViewer->mWheelPos);
        if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
    }
    
    sViewer->mWheelPos = pos;
}

void
Viewer::windowSizeCallback(int, int)
{
    sViewer->setNeedsDisplay();
}


////////////////////////////////////////


bool
Viewer::needsDisplay()
{
    if (sViewer->mUpdates < 2) {
        sViewer->mUpdates += 1;
        return true;
    }
    return false;
}


void
Viewer::setNeedsDisplay()
{
    sViewer->mUpdates = 0;
}

void
Viewer::toggleRenderModule(size_t n)
{
    mRenderModules[n]->visible() = !mRenderModules[n]->visible();
}

void
Viewer::toggleInfoText()
{
    mShowInfo = !mShowInfo;
}


// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
