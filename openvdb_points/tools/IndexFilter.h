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
/// @file IndexFilter.h
///
/// @authors Dan Bailey
///
/// @brief  Index filters primarily designed to be used with a FilterIndexIter.
///


#ifndef OPENVDB_TOOLS_INDEX_FILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_INDEX_FILTER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>

#include <openvdb/math/Transform.h>

#include <openvdb_points/tools/IndexIterator.h>
#include <openvdb_points/tools/AttributeArray.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


// Random index filtering per leaf
template <typename RandGenT>
class RandomLeafFilter
{
public:
    typedef std::map<openvdb::Coord, Index64> LeafSeedMap;
    typedef boost::uniform_01<RandGenT> Distribution;
    typedef typename Distribution::result_type ResultT;

    struct Data
    {
        Data(const ResultT _factor, const LeafSeedMap& _leafSeedMap)
            : factor(_factor), leafSeedMap(_leafSeedMap) { }
        const ResultT factor;
        const LeafSeedMap& leafSeedMap;
    };

    RandomLeafFilter(const Data& data, const unsigned int seed)
        : mData(data)
        , mDistribution(RandGenT(seed)) { }

    inline ResultT next() const {
        return const_cast<boost::uniform_01<boost::mt11213b>&>(mDistribution)();
    }

    template <typename LeafT>
    static RandomLeafFilter create(const LeafT& leaf, const Data& data) {
        const LeafSeedMap::const_iterator it = data.leafSeedMap.find(leaf.origin());
        if (it == data.leafSeedMap.end()) {
            OPENVDB_THROW(openvdb::KeyError, "Cannot find leaf origin in offset map for random filter");
        }
        return RandomLeafFilter(data, (unsigned int) it->second);
    }

    template <typename IterT>
    bool valid(const IterT&) const {
        return next() < mData.factor;
    }

private:
    const Data mData;
    Distribution mDistribution;
}; // class RandomLeafFilter


// BBox index filtering
class BBoxFilter
{
public:
    struct Data
    {
        Data(const openvdb::math::Transform& _transform,
             const openvdb::BBoxd& _bboxWS)
            : transform(_transform)
            , bbox(transform.worldToIndex(_bboxWS)) { }
        const openvdb::math::Transform transform;
        const openvdb::BBoxd bbox;
    };

    BBoxFilter( const Data& data,
                const AttributeHandle<openvdb::Vec3f>::Ptr& positionHandle)
        : mData(data)
        , mPositionHandle(positionHandle) { }

    template <typename LeafT>
    static BBoxFilter create(const LeafT& leaf, const Data& data) {
        return BBoxFilter(data, AttributeHandle<openvdb::Vec3f>::create(leaf.constAttributeArray("P")));
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        const openvdb::Coord ijk = iter.getCoord();
        const openvdb::Vec3f voxelIndexSpace = ijk.asVec3d();

        // Retrieve point position in voxel space
        const openvdb::Vec3f& pointVoxelSpace = mPositionHandle->get(*iter);

        // Compute point position in index space
        const openvdb::Vec3f pointIndexSpace = pointVoxelSpace + voxelIndexSpace;

        return mData.bbox.isInside(pointIndexSpace);
    }

private:
    const Data mData;
    const AttributeHandle<openvdb::Vec3f>::Ptr mPositionHandle;
}; // class BBoxFilter


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_INDEX_FILTER_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
