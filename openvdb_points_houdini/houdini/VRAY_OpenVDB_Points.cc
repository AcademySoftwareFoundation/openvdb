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
/// @file VRAY_OpenVDB_Points.cc
///
/// @authors Dan Bailey, Richard Kwok
///
/// @brief The Delayed Load Mantra Procedural for OpenVDB Points.


#include <UT/UT_DSOVersion.h>
#include <GU/GU_Detail.h>
#include <OP/OP_OperatorTable.h>
#include <UT/UT_BoundingBox.h>
#include <VRAY/VRAY_Procedural.h>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointGroup.h>

#include "Utils.h"

using namespace openvdb;

namespace hvdbp = openvdb_points_houdini;

// mantra renders points with a world-space radius of 0.05 by default
static const float DEFAULT_PSCALE = 0.05f;

class VRAY_OpenVDB_Points : public VRAY_Procedural {
public:
    VRAY_OpenVDB_Points();
    virtual ~VRAY_OpenVDB_Points();

    virtual const char  *className() const;

    virtual int      initialize(const UT_BoundingBox *);
    virtual void     getBoundingBox(UT_BoundingBox &box);
    virtual void     render();

private:
    UT_BoundingBox                              mBox;
    UT_String                                   mFilename;
    UT_String                                   mGroupStr;
    UT_String                                   mAttrStr;
    std::vector<tools::PointDataGrid::Ptr>      mGridPtrs;

}; // class VRAY_OpenVDB_Points

////////////////////////////////////////

// TODO: this bbox could be optimized further by considering group masks,
//       currently it assumes all groups are included

template <typename PointDataTreeT>
struct GenerateBBoxOp {

    typedef typename PointDataTreeT::LeafNodeType                          PointDataLeaf;
    typedef typename tree::LeafManager<const PointDataTreeT>::LeafRange    LeafRangeT;

    GenerateBBoxOp(const math::Transform& transform)
        : mTransform(transform)
        , mBbox() { }

    GenerateBBoxOp(const GenerateBBoxOp& parent, tbb::split)
        : mTransform(parent.mTransform)
        , mBbox(parent.mBbox) { }

    void operator()(const LeafRangeT& range) {

        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {

            const PointDataLeaf& leaf = *leafIter;

            tools::AttributeHandle<Vec3f>::Ptr positionHandle =
                tools::AttributeHandle<Vec3f>::create(leaf.constAttributeArray("P"));

            tools::AttributeHandle<float>::Ptr pscaleHandle;
            if (leaf.attributeSet().find("pscale") != tools::AttributeSet::INVALID_POS) {
                pscaleHandle = tools::AttributeHandle<float>::create(leaf.constAttributeArray("pscale"));
            }

            bool pscaleIsUniform = true;
            float uniformPscale = DEFAULT_PSCALE;
            if (pscaleHandle)
            {
                pscaleIsUniform = pscaleHandle->isUniform();
                uniformPscale = pscaleHandle->get(0);
            }

            // combine the bounds of every point on this leaf into an index-space bbox
            for (typename PointDataLeaf::IndexOnIter iter = leaf.beginIndexOn(); iter; ++iter) {

                double pscale = double(pscaleIsUniform ? uniformPscale : pscaleHandle->get(*iter));

                // the pscale attribute is converted from world space to index space
                Vec3d radius = mTransform.worldToIndex(Vec3d(pscale));
                Vec3d position = iter.getCoord().asVec3d() + Vec3d(positionHandle->get(*iter));

                mBbox.expand(position - radius);
                mBbox.expand(position + radius);
            }
        }
    }

    void join(GenerateBBoxOp& rhs) {
        mBbox.expand(rhs.mBbox);
    }

    /////////////

    const math::Transform&      mTransform;
    BBoxd                       mBbox;

}; // GenerateBBoxOp

namespace {

template <typename PointDataGridT>
inline BBoxd
getBoundingBox(const std::vector<typename PointDataGridT::Ptr>& gridPtrs)
{
    typedef typename PointDataGridT::TreeType                       PointDataTree;
    typedef typename PointDataGridT::Ptr                            PointDataGridPtr;
    typedef typename std::vector<PointDataGridPtr>::const_iterator  PointDataGridPtrVecCIter;

    BBoxd worldBounds;

    for (PointDataGridPtrVecCIter   iter = gridPtrs.begin(),
                                    endIter = gridPtrs.end(); iter != endIter; ++iter) {

        const PointDataGridPtr grid = *iter;

        tree::LeafManager<const PointDataTree> leafManager(grid->tree());

        // size and combine the boxes for each leaf in the tree via a reduction
        GenerateBBoxOp<PointDataTree> generateBbox(grid->transform());
        tbb::parallel_reduce(leafManager.leafRange(), generateBbox);

        // all the bounds must be unioned in world space
        BBoxd gridBounds = grid->transform().indexToWorld(generateBbox.mBbox);
        worldBounds.expand(gridBounds);
    }

    return worldBounds;
}

} // namespace

static VRAY_ProceduralArg   theArgs[] = {
    VRAY_ProceduralArg("file", "string", ""),
    VRAY_ProceduralArg("groupmask", "string", ""),
    VRAY_ProceduralArg("attrmask", "string", ""),
    VRAY_ProceduralArg()
};

VRAY_Procedural *
allocProcedural(const char *)
{
    return new VRAY_OpenVDB_Points();
}

const VRAY_ProceduralArg *
getProceduralArgs(const char *)
{
    return theArgs;
}

VRAY_OpenVDB_Points::VRAY_OpenVDB_Points()
{
    openvdb::initialize();
    openvdb::points::initialize();
}

VRAY_OpenVDB_Points::~VRAY_OpenVDB_Points()
{
}

const char *
VRAY_OpenVDB_Points::className() const
{
    return "VRAY_OpenVDB_Points";
}

int
VRAY_OpenVDB_Points::initialize(const UT_BoundingBox *)
{

    import("file", mFilename);
    import("groupmask", mGroupStr);
    import("attrmask", mAttrStr);

    // save the grids so that we only read the file once
    try
    {
        io::File file(mFilename.toStdString());
        file.open();

        for (io::File::NameIterator     iter = file.beginName(),
                                        endIter = file.endName(); iter != endIter; ++iter) {

            GridBase::Ptr baseGrid = file.readGridMetadata(*iter);
            if (baseGrid->isType<tools::PointDataGrid>()) {
                tools::PointDataGrid::Ptr grid = boost::static_pointer_cast<tools::PointDataGrid>(file.readGrid(*iter));
                assert(grid);
                mGridPtrs.push_back(grid);
            }
        }

        file.close();
    }
    catch (IoError& e)
    {
        OPENVDB_LOG_ERROR(e.what() << " (" << mFilename << ")");
        return 0;
    }

    // get openvdb bounds and convert to houdini bounds
    BBoxd vdbBox = ::getBoundingBox<tools::PointDataGrid>(mGridPtrs);
    mBox.setBounds(vdbBox.min().x(), vdbBox.min().y(), vdbBox.min().z(),
                   vdbBox.max().x(), vdbBox.max().y(), vdbBox.max().z());

    return 1;
}

void
VRAY_OpenVDB_Points::getBoundingBox(UT_BoundingBox &box)
{
    box = mBox;
}

void
VRAY_OpenVDB_Points::render()
{
    typedef std::vector<tools::PointDataGrid::Ptr>::const_iterator  PointDataGridPtrVecCIter;
    typedef openvdb::tools::AttributeSet                            AttributeSet;
    typedef AttributeSet::Descriptor                                Descriptor;

    /// Allocate geometry and extract the GU_Detail
    VRAY_ProceduralGeo  geo = createGeometry();

    GU_Detail* gdp = geo.get();

    // extract which groups to include and exclude
    std::vector<Name> includeGroups;
    std::vector<Name> excludeGroups;
    tools::AttributeSet::Descriptor::parseNames(includeGroups, excludeGroups, mGroupStr.toStdString());

    // extract which attributes to include and exclude
    std::vector<Name> includeAttributes;
    std::vector<Name> excludeAttributes;
    tools::AttributeSet::Descriptor::parseNames(includeAttributes, excludeAttributes, mAttrStr.toStdString());

    // if nothing was explicitly included or excluded: "all attributes" is implied with an empty vector
    // if nothing was explicitly included but something was explicitly excluded: add all attributes but then remove the excluded
    // if something was explicitly included: add only explicitly included attributes and then removed any excluded

    if (includeAttributes.empty() && !excludeAttributes.empty()) {

        // add all attributes
        for (PointDataGridPtrVecCIter   iter = mGridPtrs.begin(),
                                        endIter = mGridPtrs.end(); iter != endIter; ++iter) {

            const tools::PointDataGrid::Ptr grid = *iter;

            tools::PointDataTree::LeafCIter leafIter = grid->tree().cbeginLeaf();
            if (!leafIter) continue;

            const AttributeSet& attributeSet = leafIter->attributeSet();
            const Descriptor& descriptor = attributeSet.descriptor();
            const Descriptor::NameToPosMap& nameToPosMap = descriptor.map();

            for (Descriptor::ConstIterator  nameIter = nameToPosMap.begin(),
                                            nameIterEnd = nameToPosMap.end(); nameIter != nameIterEnd; ++nameIter) {

                includeAttributes.push_back(nameIter->first);
            }
        }
    }

    // sort, and then remove any duplicates
    std::sort(includeAttributes.begin(), includeAttributes.end());
    std::sort(excludeAttributes.begin(), excludeAttributes.end());
    includeAttributes.erase(std::unique(includeAttributes.begin(), includeAttributes.end()), includeAttributes.end());
    excludeAttributes.erase(std::unique(excludeAttributes.begin(), excludeAttributes.end()), excludeAttributes.end());

    // make a vector (validAttributes) of all elements that are in includeAttributes but are NOT in excludeAttributes
    std::vector<Name> validAttributes(includeAttributes.size());
    std::vector<Name>::iterator pastEndIter = std::set_difference(includeAttributes.begin(), includeAttributes.end(),
        excludeAttributes.begin(), excludeAttributes.end(), validAttributes.begin());
    validAttributes.resize(pastEndIter - validAttributes.begin());

    // if any of the grids are going to add a pscale, set the default here
    if (std::binary_search(validAttributes.begin(), validAttributes.end(), "pscale")) {
        gdp->addTuple(GA_STORE_REAL32, GA_ATTRIB_POINT, "pscale", 1, GA_Defaults(DEFAULT_PSCALE));
    }

    for (PointDataGridPtrVecCIter   iter = mGridPtrs.begin(),
                                    endIter = mGridPtrs.end(); iter != endIter; ++iter) {

        const tools::PointDataGrid::Ptr grid = *iter;
        hvdbp::convertPointDataGridToHoudini(*gdp, *grid, validAttributes, includeGroups, excludeGroups);
    }

    // Create a geometry object in mantra
    VRAY_ProceduralChildPtr obj = createChild();
    obj->addGeometry(geo);

    // Override the renderpoints setting to always enable points only rendering
    obj->changeSetting("renderpoints", "true");
}

////////////////////////////////////////

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
