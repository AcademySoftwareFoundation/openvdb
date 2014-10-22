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

#include "OpenVDBPlugin.h"
#include <openvdb_maya/OpenVDBData.h>
#include <openvdb/io/Stream.h>

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MFnPluginData.h>
#include <maya/MFnNumericAttribute.h>


namespace mvdb = openvdb_maya;


////////////////////////////////////////


struct OpenVDBWriteNode : public MPxNode
{
    OpenVDBWriteNode() {}
    virtual ~OpenVDBWriteNode() {}

    virtual MStatus compute(const MPlug& plug, MDataBlock& data);

    static void* creator();
    static MStatus initialize();
    static MObject aVdbFilePath;
    static MObject aVdbInput;
    static MObject aVdbOutput;
    static MTypeId id;
};


MTypeId OpenVDBWriteNode::id(0x00108A52);
MObject OpenVDBWriteNode::aVdbFilePath;
MObject OpenVDBWriteNode::aVdbOutput;
MObject OpenVDBWriteNode::aVdbInput;


namespace {
    mvdb::NodeRegistry registerNode("OpenVDBWrite", OpenVDBWriteNode::id,
        OpenVDBWriteNode::creator, OpenVDBWriteNode::initialize);
}


////////////////////////////////////////


void* OpenVDBWriteNode::creator()
{
        return new OpenVDBWriteNode();
}


MStatus OpenVDBWriteNode::initialize()
{
    MStatus stat;
    MFnTypedAttribute tAttr;

    MFnStringData fnStringData;
    MObject defaultStringData = fnStringData.create("volume.vdb");

    // Setup the input attributes

    aVdbFilePath = tAttr.create("VdbFilePath", "file", MFnData::kString, defaultStringData, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(false);
    stat = addAttribute(aVdbFilePath);
    if (stat != MS::kSuccess) return stat;

    aVdbInput = tAttr.create("VdbInput", "vdbinput", OpenVDBData::id, MObject::kNullObj, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setConnectable(true);
    stat = addAttribute(aVdbInput);
    if (stat != MS::kSuccess) return stat;

    // Setup the output attributes

    aVdbOutput = tAttr.create("VdbOutput", "vdboutput", OpenVDBData::id, MObject::kNullObj, &stat);
    if (stat != MS::kSuccess) return stat;

    tAttr.setWritable(false);
    tAttr.setStorable(false);
    stat = addAttribute(aVdbOutput);
    if (stat != MS::kSuccess) return stat;


    // Set the attribute dependencies

    stat = attributeAffects(aVdbFilePath, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    stat = attributeAffects(aVdbInput, aVdbOutput);
    if (stat != MS::kSuccess) return stat;

    return MS::kSuccess;
}


////////////////////////////////////////


MStatus OpenVDBWriteNode::compute(const MPlug& plug, MDataBlock& data)
{

    if (plug == aVdbOutput) {

        MStatus status;
        MDataHandle filePathHandle = data.inputValue(aVdbFilePath, &status);
        if (status != MS::kSuccess) return status;


        const std::string filename = filePathHandle.asString().asChar();
        if (filename.empty()) {
            return MS::kUnknownParameter;
        }

        MDataHandle inputVdbHandle = data.inputValue(aVdbInput, &status);
        if (status != MS::kSuccess) return status;

        MFnPluginData fnData(inputVdbHandle.data());
        MPxData * pxData = fnData.data();

        if (pxData) {
            OpenVDBData* vdb = dynamic_cast<OpenVDBData*>(pxData);

            if (vdb) {

                // Add file-level metadata.
                openvdb::MetaMap outMeta;
                outMeta.insertMeta("creator",
                    openvdb::StringMetadata("Maya/OpenVDB_Write_Node"));

                // Create a VDB file object.
                openvdb::io::File file(filename);

                vdb->write(file, outMeta);

                //file.write(vdb->grids(), outMeta);
                file.close();

                // Output
                MFnPluginData outputDataCreators;
                outputDataCreators.create(OpenVDBData::id, &status);
                if (status != MS::kSuccess) return status;

                OpenVDBData* outputVdb = static_cast<OpenVDBData*>(outputDataCreators.data(&status));
                if (status != MS::kSuccess) return status;

                outputVdb = vdb;

                MDataHandle outHandle = data.outputValue(aVdbOutput);
                outHandle.set(outputVdb);
            }

            return data.setClean(plug);
        }
    }

    return MS::kUnknownParameter;
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
