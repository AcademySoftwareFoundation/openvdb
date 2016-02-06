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
/// @file ResourceData_OpenVDBPoints.cc
///
/// @author Dan Bailey


#include <app_base.h>
#include <app_progress_bar.h>

#include <openvdb/tools/Prune.h>
#include <openvdb_points/tools/AttributeArray.h>

#include "ResourceData_OpenVDBPoints.h"

using namespace openvdb;
using namespace openvdb::tools;


////////////////////////////////////////


namespace openvdb_points
{

PointDataGrid::Ptr
load(   const std::string& filename,
        const std::string& gridname,
        const bool doPrune)
{
    // early exit if filename or gridname are blank

    if (filename == "" || gridname == "")   return PointDataGrid::Ptr();

    GridBase::Ptr gridBase;
    io::File file(filename);
    file.open();

    gridBase = file.readGrid(gridname);

    file.close();

    PointDataGrid::Ptr grid = gridPtrCast<PointDataGrid>(gridBase);

    // @todo: re-introduce pruning
    //if (doPrune)  prune(grid);

    return grid;
}


////////////////////////////////////////


namespace resource_data_internal
{

template <typename VelocityType>
struct ConvertVelocityToIndexSpaceOp {

    typedef tree::LeafManager<PointDataTree> LeafManager;
    typedef PointDataTree::LeafNodeType PointDataLeafNode;

    ConvertVelocityToIndexSpaceOp(  const size_t velocityIndex,
                                    const math::Transform& transform)
        : mIndex(velocityIndex)
        , mTransform(transform) { }

    void operator()(const LeafManager::LeafRange& range) const {

        for (LeafManager::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            typename AttributeWriteHandle<VelocityType>::Ptr velocityWriteHandle =
                AttributeWriteHandle<VelocityType>::create(leaf->attributeArray(mIndex));

            if (velocityWriteHandle->isUniform()) {
                const VelocityType velocity = mTransform.worldToIndex(velocityWriteHandle->get(Index64(0)));
                velocityWriteHandle->collapse(velocity);
            }
            else {
                for (PointDataLeafNode::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
                {
                    Coord ijk = value.getCoord();

                    for (IndexIter iter = leaf->beginIndex(ijk); iter; ++iter) {
                        const VelocityType velocity = mTransform.worldToIndex(velocityWriteHandle->get(*iter));
                        velocityWriteHandle->set(*iter, velocity);
                    }
                }
            }
        }
    }

    //////////

    const unsigned                      mIndex;
    const math::Transform&              mTransform;
}; // ConvertVelocityToIndexSpaceOp

template <typename ScalarType>
struct ConvertScalarToIndexSpaceOp {

    typedef tree::LeafManager<PointDataTree> LeafManagerT;
    typedef PointDataTree::LeafNodeType PointDataLeafNode;

    ConvertScalarToIndexSpaceOp(const size_t index,
                                const math::Transform& transform)
        : mIndex(index)
        , mTransform(transform) { }

    void operator()(const LeafManagerT::LeafRange& range) const {

        for (LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            typename AttributeWriteHandle<ScalarType>::Ptr scalarWriteHandle =
                AttributeWriteHandle<ScalarType>::create(leaf->attributeArray(mIndex));

            if (scalarWriteHandle->isUniform()) {
                const ScalarType transformedScalar = scalarWriteHandle->get(Index64(0)) / mTransform.voxelSize()[0];
                scalarWriteHandle->collapse(transformedScalar);
            }
            else {
                for (PointDataLeafNode::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
                {
                    Coord ijk = value.getCoord();

                    for (IndexIter iter = leaf->beginIndex(ijk); iter; ++iter) {
                        const ScalarType scalar = scalarWriteHandle->get(*iter);
                        const ScalarType transformedScalar = scalar / mTransform.voxelSize()[0];
                        scalarWriteHandle->set(*iter, transformedScalar);
                    }
                }
            }
        }
    }

    //////////

    const unsigned                      mIndex;
    const math::Transform&              mTransform;
}; // ConvertScalarToIndexSpaceOp

} // resource_data_internal


////////////////////////////////////////


void localise(PointDataGrid::Ptr& grid)
{
    PointDataTree& tree = grid->tree();
    PointDataTree::LeafCIter iter = tree.cbeginLeaf();
    math::Transform& transform = grid->transform();

    // early exit if no points

    if (!iter)  return;

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    // obtain leaf range

    typedef tree::LeafManager<PointDataTree> LeafManager;
    typedef LeafManager::LeafRange LeafRange;

    LeafManager manager(tree);
    LeafRange range = manager.leafRange();

    using resource_data_internal::ConvertVelocityToIndexSpaceOp;
    using resource_data_internal::ConvertScalarToIndexSpaceOp;

    // localise velocity

    if (iter->hasAttribute("v"))
    {
        const size_t velocityIndex = descriptor.find("v");
        const NamePair& type = descriptor.type(velocityIndex);

        if (type.first == "vec3s") {
            tbb::parallel_for(range, ConvertVelocityToIndexSpaceOp<Vec3f>(velocityIndex, transform));
        }
        else if (type.first == "vec3h") {
            tbb::parallel_for(range, ConvertVelocityToIndexSpaceOp<math::Vec3<half> >(velocityIndex, transform));
        }
        else {
            std::cerr << "Unsupported type for velocity - " << type.first << std::endl;
        }
    }

    // localise radius

    if (iter->hasAttribute("pscale"))
    {
        const size_t pscaleIndex = descriptor.find("pscale");
        const NamePair& type = descriptor.type(pscaleIndex);

        if (type.first == "float") {
            tbb::parallel_for(range, ConvertScalarToIndexSpaceOp<float>(pscaleIndex, transform));
        }
        else if (type.first == "half") {
            tbb::parallel_for(range, ConvertScalarToIndexSpaceOp<half>(pscaleIndex, transform));
        }
        else {
            std::cerr << "Unsupported type for pscale - " << type.first << std::endl;
        }
    }
}

} // namespace openvdb_points


////////////////////////////////////////


ResourceData_OpenVDBPoints*
ResourceData_OpenVDBPoints::create(const PointDataGrid::Ptr& grid)
{
    return new ResourceData_OpenVDBPoints(grid);
}


ResourceData_OpenVDBPoints::ResourceData_OpenVDBPoints(const PointDataGrid::Ptr& grid)
    : m_grid(grid)
    , m_descriptor(grid->tree().cbeginLeaf()->attributeSet().descriptorPtr())
{
    PointDataTree& tree = m_grid->tree();

    // cache the grid leaves in an array

    PointDataTree::LeafIter iter = tree.beginLeaf();

    for (; iter; ++iter)
    {
        PointDataLeaf& leaf = *iter;

        m_leaves.push_back(&leaf);
    }
}


const PointDataGrid::Ptr
ResourceData_OpenVDBPoints::grid() const
{
    return m_grid;
}


const PointDataTree::LeafNodeType*
ResourceData_OpenVDBPoints::leaf(const unsigned int id) const
{
    return (id < m_leaves.size()) ? m_leaves[id] : 0;
}


size_t
ResourceData_OpenVDBPoints::get_memory_size() const
{
    return sizeof(*this) + m_grid->memUsage() + m_leaves.size();
}


std::string
ResourceData_OpenVDBPoints::attribute_type(const std::string& name) const
{
    if (!m_descriptor)   return "";

    const size_t pos = m_descriptor->find(name);

    if (pos == AttributeSet::INVALID_POS)   return "";

    return m_descriptor->type(pos).first;
}
