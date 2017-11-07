///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

/// @file PointUtils.h
///
/// @authors Dan Bailey, Nick Avramoussis, Richard Kwok
///
/// @brief Utility classes and functions for OpenVDB Points Houdini plugins

#ifndef OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED

#include <openvdb/math/Vec3.h>
#include <openvdb/Types.h>
#include <openvdb/points/PointDataGrid.h>

#include <GA/GA_Attribute.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_ChoiceList.h>

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>


#ifdef SESI_OPENVDB
    #ifdef OPENVDB_HOUDINI_API
        #undef OPENVDB_HOUDINI_API
        #define OPENVDB_HOUDINI_API
    #endif
#endif


namespace openvdb_houdini {

using OffsetList = std::vector<GA_Offset>;
using OffsetListPtr = std::shared_ptr<OffsetList>;

using OffsetPair = std::pair<GA_Offset, GA_Offset>;
using OffsetPairList = std::vector<OffsetPair>;
using OffsetPairListPtr = std::shared_ptr<OffsetPairList>;

using AttributeInfoMap = std::map<openvdb::Name, std::pair<int, bool>>;


/// Metadata name for viewport groups
const std::string META_GROUP_VIEWPORT = "group_viewport";


/// Enum to store available compression types for point grids
enum POINT_COMPRESSION_TYPE
{
    COMPRESSION_NONE = 0,
    COMPRESSION_TRUNCATE,
    COMPRESSION_UNIT_VECTOR,
    COMPRESSION_UNIT_FIXED_POINT_8,
    COMPRESSION_UNIT_FIXED_POINT_16,
};


// forward declaration
class Interrupter;


/// @brief Compute a voxel size from a Houdini detail
///
/// @param  detail           GU_Detail to compute the voxel size from
/// @param  pointsPerVoxel   the target number of points per voxel, must be positive and non-zero
/// @param  matrix           voxel size will be computed using this transform
/// @param  decimalPlaces    for readability, truncate voxel size to this number of decimals
/// @param  interrupter      a Houdini interrupter
OPENVDB_HOUDINI_API
float
computeVoxelSizeFromHoudini(
    const GU_Detail& detail,
    const openvdb::Index pointsPerVoxel,
    const openvdb::math::Mat4d& matrix,
    const openvdb::Index decimalPlaces,
    Interrupter& interrupter);


/// @brief Convert a Houdini detail into a VDB Points grid
///
/// @param  detail         GU_Detail to convert the points and attributes from
/// @param  compression    position compression to use
/// @param  attributes     a vector of VDB Points attributes to be included
///                        (empty vector defaults to all)
/// @param  transform      transform to use for the new point grid
OPENVDB_HOUDINI_API
openvdb::points::PointDataGrid::Ptr
convertHoudiniToPointDataGrid(
    const GU_Detail& detail,
    const int compression,
    const AttributeInfoMap& attributes,
    const openvdb::math::Transform& transform);


/// @brief Convert a VDB Points grid into Houdini points and append them to a Houdini Detail
///
/// @param  detail         GU_Detail to append the converted points and attributes to
/// @param  grid           grid containing the points that will be converted
/// @param  attributes     a vector of VDB Points attributes to be included
///                        (empty vector defaults to all)
/// @param  includeGroups  a vector of VDB Points groups to be included
///                        (empty vector defaults to all)
/// @param  excludeGroups  a vector of VDB Points groups to be excluded
///                        (empty vector defaults to none)
/// @param inCoreOnly      true if out-of-core leaf nodes are to be ignored
OPENVDB_HOUDINI_API
void
convertPointDataGridToHoudini(
    GU_Detail& detail,
    const openvdb::points::PointDataGrid& grid,
    const std::vector<std::string>& attributes = {},
    const std::vector<std::string>& includeGroups = {},
    const std::vector<std::string>& excludeGroups = {},
    const bool inCoreOnly = false);


/// @brief Populate VDB Points grid metadata from Houdini detail attributes
///
/// @param  grid           grid to be populated with metadata
/// @param  warnings       list of warnings to be added to the SOP
/// @param  detail         GU_Detail to extract the detail attributes from
OPENVDB_HOUDINI_API
void
populateMetadataFromHoudini(
    openvdb::points::PointDataGrid& grid,
    std::vector<std::string>& warnings,
    const GU_Detail& detail);


/// @brief Convert VDB Points grid metadata into Houdini detail attributes
///
/// @param  detail         GU_Detail to add the Houdini detail attributes
/// @param  metaMap        the metamap to create the Houdini detail attributes from
/// @param  warnings       list of warnings to be added to the SOP
OPENVDB_HOUDINI_API
void
convertMetadataToHoudini(
    GU_Detail& detail,
    const openvdb::MetaMap& metaMap,
    std::vector<std::string>& warnings);


/// @brief Returns supported tuple sizes for conversion from GA_Attribute
OPENVDB_HOUDINI_API
int16_t
attributeTupleSize(const GA_Attribute* const attribute);


/// @brief Returns supported Storage types for conversion from GA_Attribute
OPENVDB_HOUDINI_API
GA_Storage
attributeStorageType(const GA_Attribute* const attribute);


///////////////////////////////////////


/// @brief If the given grid is a PointDataGrid, add node specific info text to the stream provided
OPENVDB_HOUDINI_API
void
pointDataGridSpecificInfoText(std::ostream&, const openvdb::GridBase&);


///////////////////////////////////////


// VDB Points group name drop-down menu

OPENVDB_HOUDINI_API extern const PRM_ChoiceList VDBPointsGroupMenuInput1;
OPENVDB_HOUDINI_API extern const PRM_ChoiceList VDBPointsGroupMenuInput2;
OPENVDB_HOUDINI_API extern const PRM_ChoiceList VDBPointsGroupMenuInput3;
OPENVDB_HOUDINI_API extern const PRM_ChoiceList VDBPointsGroupMenuInput4;

/// @note   Use this if you have more than 4 inputs, otherwise use
///         the input specific menus instead which automatically
///         handle the appropriate spare data settings.
OPENVDB_HOUDINI_API extern const PRM_ChoiceList VDBPointsGroupMenu;

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_POINT_UTILS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
