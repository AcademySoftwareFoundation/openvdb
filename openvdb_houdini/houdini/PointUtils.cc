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

/// @file PointUtils.cc
/// @authors Dan Bailey, Nick Avramoussis, Richard Kwok

#include "PointUtils.h"
#include <openvdb/openvdb.h>
#include <openvdb/util/Formats.h>
#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointDataGrid.h>
#include <ostream>
#include <sstream>
#include <string>


void
openvdb_houdini::convertPointDataGridToHoudini(
    GU_Detail& detail,
    const openvdb::points::PointDataGrid& grid,
    const std::vector<std::string>& attributes,
    const std::vector<std::string>& includeGroups,
    const std::vector<std::string>& excludeGroups,
    const bool inCoreOnly)
{
    using openvdb_houdini::HoudiniWriteAttribute;

    const openvdb::points::PointDataTree& tree = grid.tree();

    auto leafIter = tree.cbeginLeaf();
    if (!leafIter) return;

    // position attribute is mandatory
    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const openvdb::points::AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
    const bool hasPosition = descriptor.find("P") != openvdb::points::AttributeSet::INVALID_POS;
    if (!hasPosition)   return;

    // sort for binary search
    std::vector<std::string> sortedAttributes(attributes);
    std::sort(sortedAttributes.begin(), sortedAttributes.end());

    // obtain cumulative point offsets and total points
    std::vector<openvdb::Index64> pointOffsets;
    const openvdb::Index64 total = getPointOffsets(pointOffsets, tree, includeGroups, excludeGroups, inCoreOnly);

    // a block's global offset is needed to transform its point offsets to global offsets
    const openvdb::Index64 startOffset = detail.appendPointBlock(total);

    HoudiniWriteAttribute<openvdb::Vec3f> positionAttribute(*detail.getP());
    convertPointDataGridPosition(positionAttribute, grid, pointOffsets, startOffset, includeGroups, excludeGroups, inCoreOnly);

    // add other point attributes to the hdk detail
    const openvdb::points::AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

    for (const auto& namePos : nameToPosMap) {

        const openvdb::Name& name = namePos.first;
        // position handled explicitly
        if (name == "P")    continue;

        // filter attributes
        if (!sortedAttributes.empty() && !std::binary_search(sortedAttributes.begin(), sortedAttributes.end(), name))   continue;

        // don't convert group attributes
        if (descriptor.hasGroup(name))  continue;

        GA_RWAttributeRef attributeRef = detail.findPointAttribute(name.c_str());

        const auto index = static_cast<unsigned>(namePos.second);

        const openvdb::points::AttributeArray& array = leafIter->constAttributeArray(index);
        const unsigned stride = array.stride();

        const openvdb::NamePair& type = descriptor.type(index);
        const openvdb::Name valueType(openvdb::points::isString(array) ? "string" : type.first);

        // create the attribute if it doesn't already exist in the detail
        if (attributeRef.isInvalid()) {

            const bool truncate(type.second == openvdb::points::TruncateCodec::name());

            GA_Storage storage(gaStorageFromAttrString(valueType));
            if (storage == GA_STORE_INVALID) continue;
            if (storage == GA_STORE_REAL32 && truncate) {
                storage = GA_STORE_REAL16;
            }

            unsigned width = stride;
            const bool isVector = valueType.compare(0, 4, "vec3") == 0;
            const bool isQuaternion = valueType.compare(0, 4, "quat") == 0;
            const bool isMatrix4 = valueType.compare(0, 4, "mat4") == 0;

            if (isVector)               width = 3;
            else if (isQuaternion)      width = 4;
            else if (isMatrix4)         width = 16;

            const GA_Defaults defaults = gaDefaultsFromDescriptor(descriptor, name);

            attributeRef = detail.addTuple(storage, GA_ATTRIB_POINT, name.c_str(), width, defaults);

            // apply type info to some recognised types
            if (isVector) {
                if (name == "Cd")       attributeRef->getOptions().setTypeInfo(GA_TYPE_COLOR);
                else if (name == "N")   attributeRef->getOptions().setTypeInfo(GA_TYPE_NORMAL);
                else                    attributeRef->getOptions().setTypeInfo(GA_TYPE_VECTOR);
            }

            if (isQuaternion) {
                attributeRef->getOptions().setTypeInfo(GA_TYPE_QUATERNION);
            }

            if (isMatrix4) {
                attributeRef->getOptions().setTypeInfo(GA_TYPE_TRANSFORM);
            }

            // '|' and ':' characters are valid in OpenVDB Points names but will make Houdini Attribute names invalid
            if (attributeRef.isInvalid()) {
                OPENVDB_THROW(  openvdb::RuntimeError,
                                "Unable to create Houdini Points Attribute with name '" + name +
                                "'. '|' and ':' characters are not supported by Houdini.");
            }
        }

        if (valueType == "string") {
            HoudiniWriteAttribute<openvdb::Name> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "bool") {
            HoudiniWriteAttribute<bool> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int16") {
            HoudiniWriteAttribute<int16_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int32") {
            HoudiniWriteAttribute<int32_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "int64") {
            HoudiniWriteAttribute<int64_t> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "float") {
            HoudiniWriteAttribute<float> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "double") {
            HoudiniWriteAttribute<double> attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3i") {
            HoudiniWriteAttribute<openvdb::math::Vec3<int> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3s") {
            HoudiniWriteAttribute<openvdb::math::Vec3<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "vec3d") {
            HoudiniWriteAttribute<openvdb::math::Vec3<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "quats") {
            HoudiniWriteAttribute<openvdb::math::Quat<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "quatd") {
            HoudiniWriteAttribute<openvdb::math::Quat<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "mat4s") {
            HoudiniWriteAttribute<openvdb::math::Mat4<float> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else if (valueType == "mat4d") {
            HoudiniWriteAttribute<openvdb::math::Mat4<double> > attribute(*attributeRef.getAttribute());
            convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset, index, stride, includeGroups, excludeGroups, inCoreOnly);
        }
        else {
            throw std::runtime_error("Unknown Attribute Type for Conversion: " + valueType);
        }
    }

    // add point groups to the hdk detail
    const openvdb::points::AttributeSet::Descriptor::NameToPosMap& groupMap = descriptor.groupMap();

    for (const auto& namePos : groupMap) {
        const openvdb::Name& name = namePos.first;

        assert(!name.empty());

        GA_PointGroup* pointGroup = detail.findPointGroup(name.c_str());
        if (!pointGroup) pointGroup = detail.newPointGroup(name.c_str());

        const openvdb::points::AttributeSet::Descriptor::GroupIndex index =
            attributeSet.groupIndex(name);

        HoudiniGroup group(*pointGroup, startOffset, total);
        convertPointDataGridGroup(group, tree, pointOffsets, startOffset, index,
            includeGroups, excludeGroups, inCoreOnly);
    }
}


////////////////////////////////////////


void
openvdb_houdini::pointDataGridSpecificInfoText(std::ostream& infoStr, const openvdb::GridBase& grid)
{
    typedef openvdb::points::PointDataGrid PointDataGrid;
    typedef openvdb::points::PointDataTree PointDataTree;
    typedef openvdb::points::AttributeSet AttributeSet;

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
    if (openvdb::StringMetadata::ConstPtr stringMeta =
        grid.getMetadata<openvdb::StringMetadata>(openvdb_houdini::META_GROUP_VIEWPORT))
    {
        viewportGroupName = stringMeta->value();
    }

    const PointDataTree& pointDataTree = pointDataGrid->tree();

    PointDataTree::LeafCIter iter = pointDataTree.cbeginLeaf();

    // iterate through all leaf nodes to find out if all are out-of-core
    bool allOutOfCore = true;
    for (; iter; ++iter) {
        if (!iter->buffer().isOutOfCore()) {
            allOutOfCore = false;
            break;
        }
    }

    openvdb::Index64 totalPointCount = 0;

    // it is more technically correct to rely on the voxel count as this may be out of
    // sync with the attribute size, however for faster node preview when the voxel buffers
    // are all out-of-core, count up the sizes of the first attribute array instead
    if (allOutOfCore) {
        iter = pointDataTree.cbeginLeaf();
        for (; iter; ++iter) {
            if (iter->attributeSet().size() > 0) {
                totalPointCount += iter->constAttributeArray(0).size();
            }
        }
    }
    else {
        totalPointCount = pointCount(pointDataTree);
    }

    infoStr << " count: " << openvdb::util::formattedInt(totalPointCount) << ",";

    iter = pointDataTree.cbeginLeaf();

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

            // for faster node preview when all the voxel buffers are out-of-core, don't load the
            // group arrays to display the group sizes, just print "out-of-core" instead
            // @todo - put the group sizes into the grid metadata on write for this use case

            if (allOutOfCore) {
                infoStr << "out-of-core";
            }
            else {
                infoStr << openvdb::util::formattedInt(groupPointCount(pointDataTree, it->first));
            }

            infoStr << ")";
        }

        if (first)  infoStr << "<none>";

        infoStr << ",";

        infoStr << " attributes: ";

        const AttributeSet::Descriptor::NameToPosMap& nameToPosMap = descriptor->map();

        first = true;
        for (AttributeSet::Descriptor::ConstIterator it = nameToPosMap.begin(), it_end = nameToPosMap.end();
                it != it_end; ++it) {
            const openvdb::points::AttributeArray& array = iter->constAttributeArray(it->second);
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

            if (!array.hasConstantStride()) {
                infoStr << "[dynamic]";
            }
            else if (array.stride() > 1) {
                infoStr << "[" << array.stride() << "]";
            }

            infoStr << "]";
        }

        if (first)  infoStr << "<none>";
    }
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
