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
//
/// @file geometry.cc
/// @author FX R&D OpenVDB team

#include "geometry.h"

#include <GU/GU_PrimPoly.h>
#include <UT/UT_Version.h>

namespace houdini_utils {


void
createBox(GU_Detail& gdp, UT_Vector3 corners[8],
    const UT_Vector3* color, bool shaded, float alpha)
{
    // Create points
    GA_Offset ptoff[8];
    for (size_t i = 0; i < 8; ++i) {
        ptoff[i] = gdp.appendPointOffset();
        gdp.setPos3(ptoff[i], corners[i].x(), corners[i].y(), corners[i].z());
    }

    if (color != NULL) {
#if (UT_VERSION_INT >= 0x0e0000b4) // 14.0.180 or later
        GA_RWHandleV3 cd(gdp.addDiffuseAttribute(GA_ATTRIB_POINT));
#else
        GA_RWHandleV3 cd(gdp.addDiffuseAttribute(GA_ATTRIB_POINT).getAttribute());
#endif
        for (size_t i = 0; i < 8; ++i) {
            cd.set(ptoff[i], *color);
        }
    }

    if (alpha < 0.99) {
#if (UT_VERSION_INT >= 0x0e0000b4) // 14.0.180 or later
        GA_RWHandleF A(gdp.addAlphaAttribute(GA_ATTRIB_POINT));
#else
        GA_RWHandleF A(gdp.addAlphaAttribute(GA_ATTRIB_POINT).getAttribute());
#endif
        for (size_t i = 0; i < 8; ++i) {
            A.set(ptoff[i], alpha);
        }
    }

    GEO_PrimPoly *poly;
    if (shaded) {
        // Bottom
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[0]);
        poly->appendVertex(ptoff[1]);
        poly->appendVertex(ptoff[2]);
        poly->appendVertex(ptoff[3]);
        poly->close();

        // Top
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[7]);
        poly->appendVertex(ptoff[6]);
        poly->appendVertex(ptoff[5]);
        poly->appendVertex(ptoff[4]);
        poly->close();

        // Front
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[4]);
        poly->appendVertex(ptoff[5]);
        poly->appendVertex(ptoff[1]);
        poly->appendVertex(ptoff[0]);
        poly->close();

        // Back
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[6]);
        poly->appendVertex(ptoff[7]);
        poly->appendVertex(ptoff[3]);
        poly->appendVertex(ptoff[2]);
        poly->close();

        // Left
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[0]);
        poly->appendVertex(ptoff[3]);
        poly->appendVertex(ptoff[7]);
        poly->appendVertex(ptoff[4]);
        poly->close();

        // Right
        poly = GU_PrimPoly::build(&gdp, 0);
        poly->appendVertex(ptoff[1]);
        poly->appendVertex(ptoff[5]);
        poly->appendVertex(ptoff[6]);
        poly->appendVertex(ptoff[2]);
        poly->close();

    } else {

        // 12 Edges as one line
        poly = GU_PrimPoly::build(&gdp, 0, GU_POLY_OPEN);
        poly->appendVertex(ptoff[0]);
        poly->appendVertex(ptoff[1]);
        poly->appendVertex(ptoff[2]);
        poly->appendVertex(ptoff[3]);
        poly->appendVertex(ptoff[0]);
        poly->appendVertex(ptoff[4]);
        poly->appendVertex(ptoff[5]);
        poly->appendVertex(ptoff[6]);
        poly->appendVertex(ptoff[7]);
        poly->appendVertex(ptoff[4]);
        poly->appendVertex(ptoff[5]);
        poly->appendVertex(ptoff[1]);
        poly->appendVertex(ptoff[2]);
        poly->appendVertex(ptoff[6]);
        poly->appendVertex(ptoff[7]);
        poly->appendVertex(ptoff[3]);
    }
} // createBox

} // namespace houdini_utils

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
