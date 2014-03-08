///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
//

#ifndef DENSESPARSETOOLS_HAS_BEEN_INCLUDED
#define DENSESPARSETOOLS_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range3d.h>
#include <tbb/blocked_range.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/openvdb.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {



/// @brief Selectively extract and transform data from a dense grid, producing a 
/// sparse tree with leaf nodes only (e.g. create a tree from the square
/// of values greater than a cutoff.)
/// @param dense       A dense grid that acts as a data source
/// @param functor     A functor that selects and transforms data for output
/// @param background  The background value of the resulting sparse grid
/// @param threaded    Option to use threaded or serial code path
/// @return @c Ptr to tree with the valuetype and configuration defined 
/// by typedefs in the @c functor.
/// @note To achieve optimal sparsity  consider calling the prune() 
/// method on the result.
/// @note To simply copy the all the data from a Dense grid to a 
/// OpenVDB Grid, use tools::copyFromDense() for better performance.
///
/// The type of the sparse tree is determined by the specified OtpType 
/// functor by means of the typedef OptType::ResultTreeType  
///
/// The OptType function is responsible for the the transformation of 
/// dense grid data to sparse grid data on a per-voxel basis.
/// 
/// Only leaf nodes with active values will be added to the sparse grid. 
///
/// The OpType must struct that defines a the minimal form
/// @code
/// struct ExampleOp 
/// { 
///     typedef DesiredTreeType   ResultTreeType;
///
///     template<typename IndexOrCoord>
///      void OpType::operator() (const DenseValueType a, const IndexOrCoord& ijk,
///                    ResultTreeType::LeafNodeType* leaf);
/// };
/// @endcode
///
/// For example, to generate a <ValueType, 5, 4, 3> tree with valuesOn 
/// at locations greater than a given maskvalue 
/// @code    
/// template <typename ValueType>
/// class Rule
/// {
/// public:
///     // Standard tree type (e.g. BoolTree or FloatTree in openvdb.h)
///     typedef typename openvdb::tree::Tree4<ValueType, 5, 4, 3>::Type  ResultTreeType;
///    
///     typedef typename ResultTreeType::LeafNodeType  ResultLeafNodeType;
///     typedef typename ResultTreeType::ValueType     ResultValueType;
///    
///     typedef float                         DenseValueType;
///    
///     typedef vdbmath::Coord::ValueType     Index;
///    
///     Rule(const DenseValueType& value): mMaskValue(value){};
///    
///     template <typename IndexOrCoord>
///     void operator()(const DenseValueType& a, const IndexOrCoord& offset,
///                 ResultLeafNodeType* leaf) const 
///     {
///             if (a > mMaskValue) {
///                 leaf->setValueOn(offset, a);
///             }
///     }
///    
/// private:
///     const DenseValueType mMaskValue;
/// };
/// @endcode
template<typename OpType, typename DenseType>
typename OpType::ResultTreeType::Ptr
extractSparseTree(const DenseType& dense, const OpType& functor, 
                  const typename OpType::ResultValueType& background,
                  bool threaded = true);

/// This struct that aids template resoluion of a new tree type 
/// has the same configuration at TreeType, but the ValueType from
/// DenseType. 
template <typename DenseType, typename TreeType> struct DSConverter {
    typedef typename DenseType::ValueType  ValueType;
    
    typedef typename TreeType::template ValueConverter<ValueType>::Type Type;
};


/// @brief Copy data from the intersection of a sparse tree and a dense input grid.
/// The resulting tree has the same configuration as the sparse tree, but holds
/// the data type specified by the dense input.
/// @param dense       A dense grid that acts as a data source
/// @param mask        The active voxels and tiles intersected with dense define iteration mask 
/// @param background  The background value of the resulting sparse grid
/// @param threaded    Option to use threaded or serial code path
/// @return @c Ptr to tree with the same configuration as @c mask but of value type
/// defined by @c dense.  
template <typename DenseType, typename MaskTreeType>
typename DSConverter<DenseType, MaskTreeType>::Type::Ptr
extractSparseTreeWithMask(const DenseType& dense, 
                          const MaskTreeType& mask, 
                          const typename DenseType::ValueType& background,
                          bool threaded = true);



/// @brief Functor-based class used to extract data that satisfies some
/// criteria defined by the embedded @c OpType functor. The @c extractSparseTree
/// function wraps this class. 
template <typename OpType, typename DenseType>
class SparseExtractor
{

public:

    typedef openvdb::math::Coord::ValueType              Index;
        
    typedef typename DenseType::ValueType                 DenseValueType;
    typedef typename OpType::ResultTreeType               ResultTreeType;
    typedef typename ResultTreeType::ValueType            ResultValueType;
    typedef typename ResultTreeType::LeafNodeType         ResultLeafNodeType;
    typedef typename ResultTreeType::template ValueConverter<bool>::Type BoolTree;
        
    typedef tbb::blocked_range3d<Index, Index, Index>     Range3d;
        

private:
       
    const DenseType&                     mDense;
    const OpType&                        mFunctor;
    const ResultValueType                mBackground;
    const openvdb::math::CoordBBox       mBBox; 
    const Index                          mWidth;
    typename ResultTreeType::Ptr         mMask;
    openvdb::math::Coord                 mMin;
            
  
public:
        
    SparseExtractor(const DenseType& dense, const OpType& functor,
                    const ResultValueType background) : 
        mDense(dense), mFunctor(functor), 
        mBackground(background),
        mBBox(dense.bbox()),
        mWidth(ResultLeafNodeType::DIM),
        mMask( new ResultTreeType(mBackground))
    {}
        

    SparseExtractor(const DenseType& dense, 
                    const openvdb::math::CoordBBox& bbox,
                    const OpType& functor,
                    const ResultValueType background) : 
        mDense(dense), mFunctor(functor), 
        mBackground(background),
        mBBox(bbox),
        mWidth(ResultLeafNodeType::DIM),
        mMask( new ResultTreeType(mBackground))
    {
        // mBBox must be inside the coordinate rage of the dense grid
        if (!dense.bbox().isInside(mBBox)) {
            OPENVDB_THROW(ValueError, "Data extraction window out of bound");
        }
    }
            
        
    SparseExtractor(SparseExtractor& other, tbb::split):
        mDense(other.mDense), mFunctor(other.mFunctor),
        mBackground(other.mBackground), mBBox(other.mBBox),
        mWidth(other.mWidth),  
        mMask(new ResultTreeType(mBackground)),
        mMin(other.mMin)
    {}
                
    typename ResultTreeType::Ptr extract(bool threaded = true) {
            
    
        // Construct 3D range of leaf nodes that 
        // intersect mBBox.

        // Snap the bbox to nearest leaf nodes min and max
                  
        openvdb::math::Coord padded_min = mBBox.min();
        openvdb::math::Coord padded_max = mBBox.max();
            
    
        padded_min &= ~(mWidth - 1);
        padded_max &= ~(mWidth - 1);
            
        padded_max[0] += mWidth - 1;
        padded_max[1] += mWidth - 1;
        padded_max[2] += mWidth - 1;
            
            
        // number of leaf nodes in each direction 
        // division by leaf width, e.g. 8 in most cases

        const Index xleafCount = ( padded_max.x() - padded_min.x() + 1 ) / mWidth;
        const Index yleafCount = ( padded_max.y() - padded_min.y() + 1 ) / mWidth;
        const Index zleafCount = ( padded_max.z() - padded_min.z() + 1 ) / mWidth;
            
        mMin = padded_min;


        Range3d  leafRange(0, xleafCount, 1, 
                           0, yleafCount, 1,
                           0, zleafCount, 1);
            

        // Iterate over the leafnodes applying *this as a functor.
        if (threaded) {
            tbb::parallel_reduce(leafRange, *this);
        } else {
            (*this)(leafRange);
        }

        return mMask;
    }
            
            
    void operator()(const Range3d& range) {
                        
        ResultLeafNodeType* leaf = NULL;
            
        // Unpack the range3d item.
        const Index imin = range.pages().begin();
        const Index imax = range.pages().end();
            
        const Index jmin = range.rows().begin();
        const Index jmax = range.rows().end();
            
        const Index kmin = range.cols().begin();
        const Index kmax = range.cols().end();

            
        // loop over all the canidate leafs. Adding only those with 'true' values
        // to the tree

        for (Index i = imin; i < imax; ++i) {
            for (Index j = jmin; j < jmax; ++j) {
                for (Index k = kmin; k < kmax; ++k) {

                    // Calculate the origin of canidate leaf
                    const openvdb::math::Coord origin = 
                        mMin + openvdb::math::Coord(mWidth * i, 
                                                    mWidth * j, 
                                                    mWidth * k );
    
                    if (leaf == NULL) {
                        leaf = new ResultLeafNodeType(origin, mBackground);
                    } else {
                        leaf->setOrigin(origin);
                        leaf->fill(mBackground);
                        leaf->setValuesOff();
                    }
                
                    // The bouding box for this leaf
                        
                    openvdb::math::CoordBBox localBBox = leaf->getNodeBoundingBox();
                
                    // Shrink to the intersection with mBBox (i.e. the dense
                    // volume)
                        
                    localBBox.intersect(mBBox);
                                                
                    // Early out for non-intersecting leafs
                        
                    if (localBBox.empty()) continue;
                        
                        
                    const openvdb::math::Coord start = localBBox.getStart();
                    const openvdb::math::Coord end   = localBBox.getEnd();
                        
                    // Order the looping to respect the memory layout in 
                    // the Dense source

                    if (mDense.memoryLayout() == openvdb::tools::LayoutZYX) {
                            
                        openvdb::math::Coord ijk;
                        Index offset;
                        const DenseValueType* dp;
                        for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                            for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                                for (ijk[2] = start.z(), 
                                         offset = ResultLeafNodeType::coordToOffset(ijk),
                                         dp = &mDense.getValue(ijk); 
                                     ijk[2] < end.z(); ++ijk[2], ++offset, ++dp) {
                                        
                                    mFunctor(*dp, offset, leaf);
                                }
                            }
                        }
                            
                    } else {
                            
                        openvdb::math::Coord ijk;
                        const DenseValueType* dp;
                        for (ijk[2] = start.z(); ijk[2] < end.z(); ++ijk[2]) {
                            for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1]) {
                                for (ijk[0] = start.x(), 
                                         dp = &mDense.getValue(ijk); 
                                     ijk[0] < end.x(); ++ijk[0], ++dp) {
                                        
                                    mFunctor(*dp, ijk, leaf);
                                        
                                }
                            }
                        }
                    }
                        
                    // Only add non-empty leafs (empty is defined as all inactive)
                        
                    if (!leaf->isEmpty()) {
                        mMask->addLeaf(*leaf);
                        leaf = NULL;
                    }
                        
                }
            }
        }
            
        // Clean up an unused leaf.

        if (leaf != NULL) delete leaf;
    };
        
    void join(SparseExtractor& rhs) {
        mMask->merge(*rhs.mMask);
    }
};



template<typename OpType, typename DenseType>
typename OpType::ResultTreeType::Ptr
extractSparseTree(const DenseType& dense, const OpType& functor, 
                  const typename OpType::ResultValueType& background,
                  bool threaded)
{

    // Construct the mask using a parallel reduce patern. 
    // Each thread computes disjoint mask-trees.  The join merges
    // into a single tree.
    
    SparseExtractor<OpType, DenseType> extractor(dense, functor, background);
   
    return extractor.extract(threaded);
                                 
}

/// @brief Functor-based class used to extract data from a dense grid, at 
/// the index-space intersection with a suppiled maks in the form of a sparse tree.
/// The @c extractSparseTreeWithMask function wraps this class. 
template <typename DenseType, typename MaskTreeType>
class SparseMaskedExtractor
{
public:
    
    typedef typename DSConverter<DenseType, MaskTreeType>::Type  _ResultTreeType;
    typedef _ResultTreeType                                      ResultTreeType;
    typedef typename ResultTreeType::LeafNodeType                ResultLeafNodeType;
    typedef typename ResultTreeType::ValueType                   ResultValueType;
    typedef ResultValueType                                      DenseValueType;

    typedef typename ResultTreeType::template ValueConverter<bool>::Type  BoolTree;
    typedef typename BoolTree::LeafCIter                         BoolLeafCIter;
    typedef std::vector<const typename BoolTree::LeafNodeType*>  BoolLeafVec;


    SparseMaskedExtractor(const DenseType& dense,
                  const ResultValueType& background,
                  const BoolLeafVec& leafVec
                  ):
        mDense(dense), mBackground(background), mBBox(dense.bbox()),
        mLeafVec(leafVec),
        mResult(new ResultTreeType(mBackground))
    {}


        
    SparseMaskedExtractor(const SparseMaskedExtractor& other, tbb::split):
        mDense(other.mDense), mBackground(other.mBackground), mBBox(other.mBBox),
        mLeafVec(other.mLeafVec), mResult( new ResultTreeType(mBackground))
    {}
        
    typename ResultTreeType::Ptr extract(bool threaded = true) {
            
        tbb::blocked_range<size_t> range(0, mLeafVec.size());

        if (threaded) {
            tbb::parallel_reduce(range, *this);
        } else {
            (*this)(range);
        }

        return mResult;
    }

            
    // Used in looping over leaf nodes in the masked grid
    // and using the active mask to select data to 
    void operator()(const tbb::blocked_range<size_t>& range) {
                        
        ResultLeafNodeType* leaf = NULL;

            
        // loop over all the canidate leafs. Adding only those with 'true' values
        // to the tree
    
        for (size_t idx = range.begin(); idx < range.end(); ++ idx) {

            const typename BoolTree::LeafNodeType* boolLeaf = mLeafVec[idx]; 
                
            // The bouding box for this leaf
                
            openvdb::math::CoordBBox localBBox = boolLeaf->getNodeBoundingBox();
                
            // Shrink to the intersection with the dense volume
                
            localBBox.intersect(mBBox);
                                
            // Early out if there was no intersection
                
            if (localBBox.empty()) continue;
                
            // Reset or allocate the target leaf
                
            if (leaf == NULL) {
                leaf = new ResultLeafNodeType(boolLeaf->origin(), mBackground);
            } else {
                leaf->setOrigin(boolLeaf->origin());
                leaf->fill(mBackground);
                leaf->setValuesOff();
            }
                
                
            // Iterate over the intersecting bounding box
            // copying active values to the result tree

            const openvdb::math::Coord start = localBBox.getStart();
            const openvdb::math::Coord end   = localBBox.getEnd();
                
                
            openvdb::math::Coord ijk;

            if (mDense.memoryLayout() == openvdb::tools::LayoutZYX 
                  && boolLeaf->isDense()) {
                    
                Index offset;
                const DenseValueType* src;
                for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                    for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                        for (ijk[2] = start.z(),
                                 offset = ResultLeafNodeType::coordToOffset(ijk), 
                                 src  = &mDense.getValue(ijk);
                             ijk[2] < end.z(); ++ijk[2], ++offset, ++src) {
                                
                            // copy into leaf
                            leaf->setValueOn(offset, *src);
                        }
                                                       
                    }
                }
                    
            } else {

                Index offset;
                for (ijk[0] = start.x(); ijk[0] < end.x(); ++ijk[0] ) {
                    for (ijk[1] = start.y(); ijk[1] < end.y(); ++ijk[1] ) {
                        for (ijk[2] = start.z(),
                                 offset = ResultLeafNodeType::coordToOffset(ijk); 
                             ijk[2] < end.z(); ++ijk[2], ++offset) {
                            
                            if (boolLeaf->isValueOn(offset)) {
                                const ResultValueType denseValue =  mDense.getValue(ijk);
                                leaf->setValueOn(offset, denseValue);
                            }
                        }
                    }
                }
            }
            // Only add non-empty leafs (empty is defined as all inactive)
                
            if (!leaf->isEmpty()) {
                mResult->addLeaf(*leaf);
                leaf = NULL;
            }
        }
    
        // Clean up an unused leaf.

        if (leaf != NULL) delete leaf;
    };
        
    void join(SparseMaskedExtractor& rhs) {
        mResult->merge(*rhs.mResult);
    }

     
private:
    const DenseType&                   mDense;
    const ResultValueType              mBackground;
    const openvdb::math::CoordBBox&          mBBox;                 
    const BoolLeafVec&                 mLeafVec;

    typename ResultTreeType::Ptr       mResult;

};


/// @brief a simple utility class used by @c extractSparseTreeWithMask
template<typename _ResultTreeType, typename DenseValueType>
struct ExtractAll {
    typedef  _ResultTreeType                       ResultTreeType;
    typedef typename ResultTreeType::LeafNodeType  ResultLeafNodeType;
    
    template<typename CoordOrIndex>
    inline void operator()(const DenseValueType& a, const CoordOrIndex& offset, 
                                  ResultLeafNodeType* leaf) const {
        leaf->setValueOn(offset, a);
    }
};
    

template <typename DenseType, typename MaskTreeType>
typename DSConverter<DenseType, MaskTreeType>::Type::Ptr
extractSparseTreeWithMask(const DenseType& dense, 
                          const MaskTreeType& mask, 
                          const typename DenseType::ValueType& background,
                          bool threaded)
{
    
 
    typedef SparseMaskedExtractor<DenseType, MaskTreeType>       LeafExtractor;
    typedef typename LeafExtractor::DenseValueType               DenseValueType;
    typedef typename LeafExtractor::ResultTreeType               ResultTreeType;
    typedef typename LeafExtractor::BoolLeafVec                  BoolLeafVec;
    typedef typename LeafExtractor::BoolTree                     BoolTree;
    typedef typename LeafExtractor::BoolLeafCIter                BoolLeafCIter;
    typedef ExtractAll<ResultTreeType, DenseValueType>           ExtractionRule;
         
    // Use Bool tree to hold the topology
    
    BoolTree boolTree(mask, false, TopologyCopy());
   
    // Construct an array of pointers to the mask leafs.
    
    const size_t leafCount = boolTree.leafCount();
    BoolLeafVec leafarray(leafCount);
    BoolLeafCIter leafiter = boolTree.cbeginLeaf();
    for (size_t n = 0; n != leafCount; ++n, ++leafiter) {
        leafarray[n] = leafiter.getLeaf();
    }    


    // Extract the data that is masked leaf nodes in the mask.

    LeafExtractor leafextractor(dense, background, leafarray); 
    typename ResultTreeType::Ptr resultTree = leafextractor.extract(threaded); 


    // Extract data that is masked by tiles in the mask.


    // Loop over the mask tiles, extracting the data into new trees.
    // These trees will be leaf-orthogonal to the leafTree (i.e. no leaf
    // nodes will overlap).  Merge these trees into the result.

    typename MaskTreeType::ValueOnCIter tileIter(mask);
    tileIter.setMaxDepth(MaskTreeType::ValueOnCIter::LEAF_DEPTH - 1);
    
    // Return the leaf tree if the mask had no tiles

    if (!tileIter) return resultTree;

    ExtractionRule allrule;
        
    // Loop over the tiles in series, but the actual data extraction 
    // is in parallel.

    CoordBBox bbox;
    for ( ; tileIter; ++tileIter) {

        // Find the intersection of the tile with the dense grid.

        tileIter.getBoundingBox(bbox);
        bbox.intersect(dense.bbox());
        
        if (bbox.empty()) continue;
        
        SparseExtractor<ExtractionRule, DenseType> copyData(dense, bbox, allrule, background);
        typename ResultTreeType::Ptr fromTileTree = copyData.extract(threaded);
        resultTree->merge(*fromTileTree);
    }
    
    return resultTree;

} 


    
// close namespaces
}
}
}
#endif //DENSESPARSETOOLS_HAS_BEEN_INCLUDED


// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
