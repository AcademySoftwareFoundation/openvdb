// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_METADATA_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_METADATA_HAS_BEEN_INCLUDED


/* openvdbmma::metadata members

 class GridMetadata

 public members are

 getMetadata
 setMetadata

*/

namespace openvdbmma {
namespace metadata {

//////////// metadata class

template<typename GridT>
class GridMetadata
{
public:

    using GridPtr = typename GridT::Ptr;

    GridMetadata(GridPtr grid) : mGrid(grid)
    {
    }

    GridMetadata() {}

    template<typename T>
    T getMetadata(std::string key) const
    {
        using MetaIter = openvdb::MetaMap::MetaIterator;

        openvdb::Metadata::Ptr metadata;
        bool seen = false;
        T val;

        for (MetaIter iter = mGrid->beginMeta(); iter != mGrid->endMeta(); ++iter) {
            if(iter->first == key){
                openvdb::Metadata::Ptr metadata = iter->second;
                val = static_cast<TypedMetadata<T>&>(*metadata).value();
                seen = true;
                break;
            }
        }

        if(!seen)
            throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

        return val;
    }

    template<typename T>
    void setMetadata(std::string key, T val)
    {
        mGrid->insertMeta(key, TypedMetadata<T>(val));
    }

private:

    GridPtr mGrid;

}; // end of GridMetadata class

} // namespace metadata
} // namespace openvdbmesh

#endif // OPENVDBLINK_UTILITIES_METADATA_HAS_BEEN_INCLUDED
