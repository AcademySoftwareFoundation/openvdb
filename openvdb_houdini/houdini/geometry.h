///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file geoemetry.h
/// @author FX R&D OpenVDB team
///
/// @brief A collection of Houdini geometry related methods and helper functions.

#ifndef HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED
#define HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED

#include <UT/UT_VectorTypes.h>
#include <GU/GU_Detail.h>
#include <GA/GA_ElementGroup.h>

#if defined(PRODDEV_BUILD) || defined(DWREAL_IS_DOUBLE) || defined(SESI_OPENVDB)
  // OPENVDB_HOUDINI_API, which has no meaning in a DWA build environment but
  // must at least exist, is normally defined by including openvdb/Platform.h.
  // For DWA builds (i.e., if either PRODDEV_BUILD or DWREAL_IS_DOUBLE exists),
  // that introduces an unwanted and unnecessary library dependency.
  #ifndef OPENVDB_HOUDINI_API
    #define OPENVDB_HOUDINI_API
  #endif
#else
  #include <openvdb/Platform.h>
#endif

namespace houdini_utils {


/// @brief Add geometry to the given GU_Detail to create a box with the given corners.
/// @param corners  the eight corners of the box
/// @param color    an optional color for the added geometry
/// @param shaded   if false, generate a wireframe box; otherwise, generate a solid box
/// @param alpha    an optional opacity for the added geometry
OPENVDB_HOUDINI_API void createBox(GU_Detail&, UT_Vector3 corners[8],
    const UT_Vector3* color = NULL, bool shaded = false, float alpha = 1.0);

} // namespace houdini_utils

#endif // HOUDINI_UTILS_GEOMETRY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
