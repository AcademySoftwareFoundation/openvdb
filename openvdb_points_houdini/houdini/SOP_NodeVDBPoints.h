///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file SOP_NodeVDB.h
///
/// @author Dan Bailey
///
/// @brief Base class for OpenVDB Points plugins


#ifndef OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Platform.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointCount.h>

#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <houdini_utils/ParmFactory.h>

#include <boost/algorithm/string/predicate.hpp> // for starts_with

#include <OP/OP_NodeInfoParms.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_Cache.h> // for dynamic cast

#ifndef SESI_OPENVDB
#include <UT/UT_DSOVersion.h>
#endif


class GU_Detail;

namespace openvdb_houdini {

/// @brief Base class from which to derive OpenVDB Points-related Houdini SOPs
class OPENVDB_HOUDINI_API SOP_NodeVDBPoints: public SOP_NodeVDB
{
public:
    SOP_NodeVDBPoints(OP_Network* network, const char* name, OP_Operator* op)
        : SOP_NodeVDB(network, name, op) { }
    virtual ~SOP_NodeVDBPoints() { }

    void
    getNodeSpecificInfoText(OP_Context &context, OP_NodeInfoParms &parms)
    {
        SOP_Node::getNodeSpecificInfoText(context, parms);

    #ifdef SESI_OPENVDB
        // Nothing needed since we will report it as part of native prim info
    #else
        // Get a handle to the geometry.
        GU_DetailHandle gd_handle = getCookedGeoHandle(context);

       // Check if we have a valid detail handle.
        if (gd_handle.isNull()) return;

        // Lock it for reading.
        GU_DetailHandleAutoReadLock gd_lock(gd_handle);
        // Finally, get at the actual GU_Detail.
        const GU_Detail* tmp_gdp = gd_lock.getGdp();

        std::ostringstream infoStr;

        unsigned gridn = 0;
        for (VdbPrimCIterator it(tmp_gdp); it; ++it) {

            const openvdb::GridBase& grid = it->getGrid();

            // ignore openvdb point grids
            if (dynamic_cast<const openvdb::tools::PointDataGrid*>(&grid))  continue;

            openvdb::Coord dim = grid.evalActiveVoxelDim();
            const UT_String gridName = it.getPrimitiveName();

            infoStr << "    ";
            infoStr << "(" << it.getIndex() << ")";
            if (gridName.isstring()) infoStr << " name: '" << gridName << "',";
            infoStr << " voxel size: " << grid.transform().voxelSize()[0] << ",";
            infoStr << " type: "<< grid.valueType() << ",";

            if (grid.activeVoxelCount() != 0) {
                infoStr << " dim: " << dim[0] << "x" << dim[1] << "x" << dim[2];
            } else {
                infoStr <<" <empty>";
            }

            infoStr<<"\n";

            ++gridn;
        }

        if (gridn > 0) {
            std::ostringstream headStr;
            headStr << gridn << " VDB grid" << (gridn == 1 ? "" : "s") << "\n";

            parms.append(headStr.str().c_str());
            parms.append(infoStr.str().c_str());
        }
    #endif


        // VDB points info (to be included in native VDB primitive)

        // Get a handle to the geometry.
        GU_DetailHandle gd_handle_pts = getCookedGeoHandle(context);

       // Check if we have a valid detail handle.
        if (gd_handle_pts.isNull()) return;

        // Lock it for reading.
        GU_DetailHandleAutoReadLock gd_lock_pts(gd_handle_pts);
        // Finally, get at the actual GU_Detail.
        const GU_Detail* tmp_gdp_pts = gd_lock_pts.getGdp();

        std::ostringstream infoStr_pts;

        unsigned gridn_pts = 0;
        for (VdbPrimCIterator it(tmp_gdp_pts); it; ++it) {

            const openvdb::GridBase& grid = it->getGrid();
            const UT_String gridName = it.getPrimitiveName();

            infoStr_pts << "    ";
            infoStr_pts << "(" << it.getIndex() << ")";
            if (gridName.isstring()) infoStr_pts << " name: '" << gridName << "',";

            typedef openvdb::tools::PointDataGrid PointDataGrid;
            typedef openvdb::tools::PointDataTree PointDataTree;
            typedef openvdb::tools::AttributeSet AttributeSet;

            const PointDataGrid* pointDataGrid = dynamic_cast<const PointDataGrid*>(&grid);

            if (!pointDataGrid)     continue;

            const PointDataTree& pointDataTree = pointDataGrid->tree();

            PointDataTree::LeafCIter iter = pointDataTree.cbeginLeaf();

            const openvdb::Index64 count = pointCount(pointDataTree);

            infoStr_pts << " count: " << count << ",";

            if (!iter.getLeaf()) {
                infoStr_pts << " attributes: ";
                infoStr_pts << "<none>";
            }
            else {
                const AttributeSet::DescriptorPtr& descriptor = iter->attributeSet().descriptorPtr();

                infoStr_pts << " groups: ";

                const AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor->groupMap();

                bool first = true;
                for (AttributeSet::Descriptor::ConstIterator it = groupMap.begin(), it_end = groupMap.end();
                        it != it_end; ++it) {
                    if (first) {
                        first = false;
                    }
                    else {
                        infoStr_pts << ", ";
                    }

                    infoStr_pts << it->first << "(";

                    infoStr_pts << groupPointCount(pointDataTree, it->first);

                    infoStr_pts << ")";
                }

                if (first)  infoStr_pts << "<none>";

                infoStr_pts << ",";

                infoStr_pts << " attributes: ";

                const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor->map();

                first = true;
                for (AttributeSet::Descriptor::ConstIterator it = nameToPosMap.begin(), it_end = nameToPosMap.end();
                        it != it_end; ++it) {
                    const openvdb::tools::AttributeArray& array = iter->attributeArray(it->second);
                    if (openvdb::tools::GroupAttributeArray::isGroup(array))    continue;

                    if (first) {
                        first = false;
                    }
                    else {
                        infoStr_pts << ", ";
                    }
                    const openvdb::NamePair& type = descriptor->type(it->second);

                    // if no value compression, hide the codec from the middle-click output

                    if (boost::starts_with(type.second, "null_") &&
                        boost::ends_with(type.second, type.first)) {
                        infoStr_pts << it->first << "[" << type.first << "]";
                    }
                    else {
                        infoStr_pts << it->first << "[" << type.first << "_" << type.second << "]";
                    }
                }

                if (first)  infoStr_pts << "<none>";
            }

            infoStr_pts << "\n";

            ++gridn_pts;
        }

        if (gridn_pts > 0) {
            std::ostringstream headStr;
            headStr << gridn_pts << " VDB points" << "\n";

            parms.append(headStr.str().c_str());
            parms.append(infoStr_pts.str().c_str());
        }
    }
};

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
