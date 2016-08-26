///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
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
/// @file SOP_NodeVDBPoints.h
///
/// @author Dan Bailey
///
/// @brief Base class for OpenVDB Points plugins


#ifndef OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Platform.h>
#include <openvdb_points/tools/AttributeArrayString.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointCount.h>

#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <houdini_utils/ParmFactory.h>

#include <boost/algorithm/string/predicate.hpp> // for starts_with

#include <OP/OP_NodeInfoParms.h>
#include <SOP/SOP_Node.h>

#ifndef SESI_OPENVDB
#include <UT/UT_DSOVersion.h>
#endif

class GU_Detail;

namespace {

// turn 10000 into 10,000 when output (comma-based locale)

template<class T>
std::string addDigitSeparators(T value)
{
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

} // namespace

namespace openvdb_houdini {

/// @brief Base class from which to derive OpenVDB Points-related Houdini SOPs
class OPENVDB_HOUDINI_API SOP_NodeVDBPoints: public SOP_NodeVDB
{
public:
    SOP_NodeVDBPoints(OP_Network* network, const char* name, OP_Operator* op)
        : SOP_NodeVDB(network, name, op)
    {
// the ability to register specific info text callbacks was only introduced in OpenVDB 3.2
#if (OPENVDB_LIBRARY_VERSION_NUMBER < 0x03020000) // earlier than OpenVDB 3.2
        // do nothing
#else
        node_info_text::registerGridSpecificInfoText<openvdb::tools::PointDataGrid>(&pointDataGridSpecificInfoText);
#endif // OPENVDB_LIBRARY_VERSION_NUMBER
    }

    virtual ~SOP_NodeVDBPoints() { }

// prior to OpenVDB 3.2, there was no way of associating a specific VDB primitive
// to a function that generated output for this primitive type, so accessing the
// GDP and generating output for all OpenVDB primitives was required
#if (OPENVDB_LIBRARY_VERSION_NUMBER < 0x03020000) // earlier than OpenVDB 3.2
    void
    getNodeSpecificInfoText(OP_Context &context, OP_NodeInfoParms &parms)
    {
        SOP_Node::getNodeSpecificInfoText(context, parms);

        // Get a handle to the geometry.
        GU_DetailHandle gridHandle = getCookedGeoHandle(context);

       // Check if we have a valid detail handle.
        if (gridHandle.isNull()) return;

        // Lock it for reading.
        GU_DetailHandleAutoReadLock gridLock(gridHandle);
        // Finally, get at the actual GU_Detail.
        const GU_Detail* tmpGdp = gridLock.getGdp();

        std::ostringstream infoStr;

        unsigned gridn = 0;

#ifdef SESI_OPENVDB
        // Nothing needed since we will report it as part of native prim info
#else
        for (VdbPrimCIterator it(tmpGdp); it; ++it) {

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
                infoStr << " <empty>";
            }

            infoStr << "\n";

            ++gridn;
        }

        if (gridn > 0) {
            std::ostringstream headStr;
            headStr << gridn << " VDB grid" << (gridn == 1 ? "" : "s") << "\n";

            parms.append(headStr.str().c_str());
            parms.append(infoStr.str().c_str());
        }
#endif // SESI_OPENVDB

        // clear the output string stream and grid counter
        infoStr.str("");
        gridn = 0;

        for (VdbPrimCIterator it(tmpGdp); it; ++it) {

            const openvdb::GridBase& grid = it->getGrid();

            // ignore non openvdb point grids
            if (!dynamic_cast<const openvdb::tools::PointDataGrid*>(&grid))  continue;

            const UT_String gridName = it.getPrimitiveName();

            infoStr << "    ";
            infoStr << "(" << it.getIndex() << ")";
            if (gridName.isstring()) infoStr << " name: '" << gridName << "',";

            pointDataGridSpecificInfoText(infoStr, grid);

            infoStr << "\n";

            gridn++;
        }

        if (gridn > 0) {
            std::ostringstream headStr;
            headStr << gridn << " VDB point" << (gridn == 1 ? "" : "s") << "\n";

            parms.append(headStr.str().c_str());
            parms.append(infoStr.str().c_str());
        }
    }

#endif // OPENVDB_LIBRARY_VERSION_NUMBER

    static void
    pointDataGridSpecificInfoText(std::ostream& infoStr, const openvdb::GridBase& grid)
    {
        typedef openvdb::tools::PointDataGrid PointDataGrid;
        typedef openvdb::tools::PointDataTree PointDataTree;
        typedef openvdb::tools::AttributeSet AttributeSet;

        const PointDataGrid* pointDataGrid = dynamic_cast<const PointDataGrid*>(&grid);

        if (!pointDataGrid) return;

        // match native OpenVDB convention as much as possible

        infoStr << " voxel size: " << pointDataGrid->transform().voxelSize()[0] << ",";
        infoStr << " type: points,";

        if (pointDataGrid->activeVoxelCount() != 0) {
            const openvdb::Coord dim = grid.evalActiveVoxelDim();
            infoStr << " dim: " << dim[0] << "x" << dim[1] << "x" << dim[2] << ",";
        } else {
            infoStr <<" <empty>,";
        }

        std::string viewportGroupName = "";
        if (openvdb::StringMetadata::ConstPtr stringMeta = grid.getMetadata<openvdb::StringMetadata>(openvdb::META_GROUP_VIEWPORT)) {
            viewportGroupName = stringMeta->value();
        }

        const PointDataTree& pointDataTree = pointDataGrid->tree();

        PointDataTree::LeafCIter iter = pointDataTree.cbeginLeaf();

        const openvdb::Index64 totalPointCount = pointCount(pointDataTree);

        infoStr << " count: " << addDigitSeparators(totalPointCount) << ",";

        if (!iter.getLeaf()) {
            infoStr << " attributes: <none>";
        }
        else {
            const AttributeSet::DescriptorPtr& descriptor = iter->attributeSet().descriptorPtr();

            infoStr << " groups: ";

            const AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor->groupMap();

            bool first = true;
            for (AttributeSet::Descriptor::ConstIterator it = groupMap.begin(), it_end = groupMap.end();
                    it != it_end; ++it) {
                if (first) {
                    first = false;
                }
                else {
                    infoStr << ", ";
                }

                // add an asterisk as a viewport group indicator
                if (it->first == viewportGroupName)     infoStr << "*";

                infoStr << it->first << "(";

                infoStr << addDigitSeparators(groupPointCount(pointDataTree, it->first));

                infoStr << ")";
            }

            if (first)  infoStr << "<none>";

            infoStr << ",";

            infoStr << " attributes: ";

            const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor->map();

            first = true;
            for (AttributeSet::Descriptor::ConstIterator it = nameToPosMap.begin(), it_end = nameToPosMap.end();
                    it != it_end; ++it) {
                const openvdb::tools::AttributeArray& array = iter->attributeArray(it->second);
                if (isGroup(array))    continue;

                if (first) {
                    first = false;
                }
                else {
                    infoStr << ", ";
                }
                const openvdb::NamePair& type = descriptor->type(it->second);

                const openvdb::Name valueType = type.first;
                const openvdb::Name codecType = type.second;

                infoStr << it->first << "[";

                // if no value compression, hide the codec from the middle-click output

                if (codecType == "null") {
                    infoStr << valueType;
                }
                else if (isString(array)) {
                    infoStr << "str";
                }
                else {
                    infoStr << valueType << "_" << codecType;
                }

                if (array.isStrided()) {
                    infoStr << "[" << array.stride() << "]";
                }

                infoStr << "]";
            }

            if (first)  infoStr << "<none>";
        }
    }
};

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_SOP_NODEVDB_POINTS_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
