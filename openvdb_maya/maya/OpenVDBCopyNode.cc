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

/// @author FX R&D OpenVDB team

#include "OpenVDBPlugin.h"
#include <openvdb_maya/OpenVDBData.h>
#include <openvdb_maya/OpenVDBUtil.h>

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MFnPluginData.h>
#include <maya/MGlobal.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFloatVector.h>

#include <boost/math/constants/constants.hpp> // boost::math::constants::pi


namespace mvdb = openvdb_maya;


////////////////////////////////////////


struct OpenVDBCopyNode : public MPxNode
{
    OpenVDBCopyNode() {}
    virtual ~OpenVDBCopyNode() {}

    virtual MStatus compute(const MPlug& plug, MDataBlock& data);

    static void* creator();
    static MStatus initialize();

    static MTypeId id;
    static MObject aVdbInputA;
    static MObject aVdbInputB;
    static MObject aVdbOutput;
    static MObject aVdbSelectedGridNamesA;
    static MObject aVdbSelectedGridNamesB;
};


MTypeId OpenVDBCopyNode::id(0x00108A58);
MObject OpenVDBCopyNode::aVdbOutput;
MObject OpenVDBCopyNode::aVdbInputA;
MObject OpenVDBCopyNode::aVdbInputB;
MObject OpenVDBCopyNode::aVdbSelectedGridNamesA;
MObject OpenVDBCopyNode::aVdbSelectedGridNamesB;


namespace {
    mvdb::NodeRegistry registerNode("OpenVDBCopy", OpenVDBCopyNode::id,
        OpenVDBCopyNode::creator, OpenVDBCopyNode::initialize);
}


////////////////////////////////////////


void* OpenVDBCopyNode::creator()
{
    return new OpenVDBCopyNode();
}


MStatus OpenVDBCopyNode::initialize()
{
    MStatus stat;

    // attributes

    MFnTypedAttribute tAttr;
    MFnStringData strData;

    aVdbSelectedGridNamesA = tAttr.create("GridsFromA", "anames", MFnData::kString, strData.create("*"), &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(false);
    stat = addAttribute(aVdbSelectedGridNamesA);
    if (stat != MS::kSuccess) return stat;


    aVdbSelectedGridNamesB = tAttr.create("GridsFromB", "bnames", MFnData::kString, strData.create("*"), &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(false);
    stat = addAttribute(aVdbSelectedGridNamesB);
    if (stat != MS::kSuccess) return stat;


    // input / output

    aVdbInputA = tAttr.create("VdbInputA", "a", OpenVDBData::id, MObject::kNullObj, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(true);
    stat = addAttribute(aVdbInputA);
    if (stat != MS::kSuccess) return stat;


    aVdbInputB = tAttr.create("VdbInputB", "b", OpenVDBData::id, MObject::kNullObj, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(true);
    stat = addAttribute(aVdbInputB);
    if (stat != MS::kSuccess) return stat;


    aVdbOutput = tAttr.create("VdbOutput", "vdboutput", OpenVDBData::id, MObject::kNullObj, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setWritable(false);
    tAttr.setStorable(false);
    stat = addAttribute(aVdbOutput);
    if (stat != MS::kSuccess) return stat;


    // attribute dependencies

    stat = attributeAffects(aVdbInputA, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    stat = attributeAffects(aVdbInputB, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    stat = attributeAffects(aVdbSelectedGridNamesA, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    stat = attributeAffects(aVdbSelectedGridNamesB, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    return MS::kSuccess;
}


////////////////////////////////////////


MStatus OpenVDBCopyNode::compute(const MPlug& plug, MDataBlock& data)
{

    if (plug == aVdbOutput) {

        const OpenVDBData* inputVdbA = mvdb::getInputVDB(aVdbInputA, data);
        const OpenVDBData* inputVdbB = mvdb::getInputVDB(aVdbInputB, data);


        MStatus status;
        MFnPluginData pluginData;
        pluginData.create(OpenVDBData::id, &status);

        if (status != MS::kSuccess) {
            MGlobal::displayError("Failed to create a new OpenVDBData object.");
            return MS::kFailure;
        }

        OpenVDBData* outputVdb = static_cast<OpenVDBData*>(pluginData.data(&status));
        MDataHandle output = data.outputValue(aVdbOutput);

        if (inputVdbA && outputVdb) {

            const std::string selectionStr =
                data.inputValue(aVdbSelectedGridNamesA, &status).asString().asChar();

            mvdb::GridCPtrVec grids;
            mvdb::getSelectedGrids(grids, selectionStr, *inputVdbA);

            for (mvdb::GridCPtrVecIter it = grids.begin(); it != grids.end(); ++it) {
                outputVdb->insert((*it)->copyGrid()); // shallow copy
            }
        }

        if (inputVdbB && outputVdb) {

            const std::string selectionStr =
                data.inputValue(aVdbSelectedGridNamesB, &status).asString().asChar();

            mvdb::GridCPtrVec grids;
            mvdb::getSelectedGrids(grids, selectionStr, *inputVdbB);

            for (mvdb::GridCPtrVecIter it = grids.begin(); it != grids.end(); ++it) {
                outputVdb->insert((*it)->copyGrid()); // shallow copy
            }
        }

        output.set(outputVdb);
        return data.setClean(plug);

    }

    return MS::kUnknownParameter;
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
