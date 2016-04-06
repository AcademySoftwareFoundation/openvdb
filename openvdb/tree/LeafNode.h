///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#ifndef OPENVDB_TREE_LEAFNODE_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAFNODE_HAS_BEEN_INCLUDED

#include <iostream>
#include <algorithm> // for std::swap
#include <cstring> // for std::memcpy()
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/bind.hpp>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>
#include <openvdb/Types.h>
#include <openvdb/util/NodeMasks.h>
#include <openvdb/io/Compression.h> // for io::readData(), etc.
#include "Iterator.h"


class TestLeaf;
template<typename> class TestLeafIO;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

template<Index, typename> struct SameLeafConfig; // forward declaration


/// @brief Templated block class to hold specific data types and a fixed
/// number of values determined by Log2Dim. The actual coordinate
/// dimension of the block is 2^Log2Dim, i.e. Log2Dim=3 corresponds to
/// a LeafNode that spans a 8^3 block.
template<typename T, Index Log2Dim>
class LeafNode
{
public:
    typedef T                            BuildType; 
    typedef T                            ValueType;
    typedef LeafNode<ValueType, Log2Dim> LeafNodeType;
    typedef boost::shared_ptr<LeafNode>  Ptr;
    typedef util::NodeMask<Log2Dim>      NodeMaskType;

    static const Index
        LOG2DIM     = Log2Dim,      // needed by parent nodes
        TOTAL       = Log2Dim,      // needed by parent nodes
        DIM         = 1 << TOTAL,   // dimension along one coordinate direction
        NUM_VALUES  = 1 << 3 * Log2Dim,
        NUM_VOXELS  = NUM_VALUES,   // total number of voxels represented by this node
        SIZE        = NUM_VALUES,
        LEVEL       = 0;            // level 0 = leaf

    /// @brief ValueConverter<T>::Type is the type of a LeafNode having the same
    /// dimensions as this node but a different value type, T.
    template<typename OtherValueType>
    struct ValueConverter {
        typedef LeafNode<OtherValueType, Log2Dim> Type;
    };

    /// @brief SameConfiguration<OtherNodeType>::value is @c true if and only if
    /// OtherNodeType is the type of a LeafNode with the same dimensions as this node.
    template<typename OtherNodeType>
    struct SameConfiguration {
        static const bool value = SameLeafConfig<LOG2DIM, OtherNodeType>::value;
    };

#ifndef OPENVDB_2_ABI_COMPATIBLE
    struct FileInfo
    {
        FileInfo(): bufpos(0) , maskpos(0) {}
        std::streamoff bufpos;
        std::streamoff maskpos;
        io::MappedFile::Ptr mapping;
        boost::shared_ptr<io::StreamMetadata> meta;
    };
#endif

    /// @brief Array of fixed size @f$2^{3 \times {\rm Log2Dim}}@f$ that stores
    /// the voxel values of a LeafNode
    class Buffer
    {
    public:
#ifdef OPENVDB_2_ABI_COMPATIBLE
        /// Default constructor
        Buffer(): mData(new ValueType[SIZE]) {}
        /// Construct a buffer populated with the specified value.
        explicit Buffer(const ValueType& val): mData(new ValueType[SIZE]) { this->fill(val); }
        /// Copy constructor
        Buffer(const Buffer& other): mData(new ValueType[SIZE]) { *this = other; }
        /// Destructor
        ~Buffer() { delete[] mData; }

        /// Return @c true if this buffer's values have not yet been read from disk.
        bool isOutOfCore() const { return false; }
        /// Return @c true if memory for this buffer has not yet been allocated.
        bool empty() const { return (mData == NULL); }
#else
        typedef ValueType WordType;
        static const Index WORD_COUNT = SIZE;
        /// Default constructor
        Buffer(): mData(new ValueType[SIZE]), mOutOfCore(0) {}
        /// Construct a buffer populated with the specified value.
        explicit Buffer(const ValueType& val): mData(new ValueType[SIZE]), mOutOfCore(0)
        {
            this->fill(val);
        }
        /// Copy constructor
        Buffer(const Buffer& other): mData(NULL), mOutOfCore(other.mOutOfCore)
        {
            if (other.isOutOfCore()) {
                mFileInfo = new FileInfo(*other.mFileInfo);
            } else {
                this->allocate();
                ValueType* target = mData;
                const ValueType* source = other.mData;
                Index n = SIZE;
                while (n--) *target++ = *source++;
            }
        }
        /// Construct a buffer but don't allocate memory for the full array of values.
        Buffer(PartialCreate, const ValueType&): mData(NULL), mOutOfCore(0) {}
        /// Destructor
        ~Buffer()
        {
            if (this->isOutOfCore()) {
                this->detachFromFile();
            } else {
                this->deallocate();
            }
        }

        /// Return @c true if this buffer's values have not yet been read from disk.
        bool isOutOfCore() const { return bool(mOutOfCore); }
        /// Return @c true if memory for this buffer has not yet been allocated.
        bool empty() const { return !mData || this->isOutOfCore(); }
#endif
        /// Allocate memory for this buffer if it has not already been allocated.
        bool allocate() { if (mData == NULL) mData = new ValueType[SIZE]; return !this->empty(); }

        /// Populate this buffer with a constant value.
        void fill(const ValueType& val)
        {
            this->detachFromFile();
            if (mData != NULL) {
                ValueType* target = mData;
                Index n = SIZE;
                while (n--) *target++ = val;
            }
        }

        /// Return a const reference to the i'th element of this buffer.
        const ValueType& getValue(Index i) const { return this->at(i); }
        /// Return a const reference to the i'th element of this buffer.
        const ValueType& operator[](Index i) const { return this->at(i); }
        /// Set the i'th value of this buffer to the specified value.
        void setValue(Index i, const ValueType& val)
        {
            assert(i < SIZE);
#ifdef OPENVDB_2_ABI_COMPATIBLE
            mData[i] = val;
#else
            this->loadValues();
            if (mData) mData[i] = val;
#endif
        }

        /// Copy the other buffer's values into this buffer.
        Buffer& operator=(const Buffer& other)
        {
            if (&other != this) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
                if (this->isOutOfCore()) {
                    this->detachFromFile();
                } else {
                    if (other.isOutOfCore()) this->deallocate();
                }
                if (other.isOutOfCore()) {
                    mOutOfCore = other.mOutOfCore;
                    mFileInfo = new FileInfo(*other.mFileInfo);
                } else {
#endif
                    this->allocate();
                    ValueType* target = mData;
                    const ValueType* source = other.mData;
                    Index n = SIZE;
                    while (n--) *target++ = *source++;
#ifndef OPENVDB_2_ABI_COMPATIBLE
                }
#endif
            }
            return *this;
        }

        /// @brief Return @c true if the contents of the other buffer
        /// exactly equal the contents of this buffer.
        bool operator==(const Buffer& other) const
        {
            this->loadValues();
            other.loadValues();
            const ValueType *target = mData, *source = other.mData;
            if (!target && !source) return true;
            if (!target || !source) return false;
            Index n = SIZE;
            while (n && math::isExactlyEqual(*target++, *source++)) --n;
            return n == 0;
        }
        /// @brief Return @c true if the contents of the other buffer
        /// are not exactly equal to the contents of this buffer.
        bool operator!=(const Buffer& other) const { return !(other == *this); }

        /// Exchange this buffer's values with the other buffer's values.
        void swap(Buffer& other)
        {
            std::swap(mData, other.mData);
#ifndef OPENVDB_2_ABI_COMPATIBLE
            std::swap(mOutOfCore, other.mOutOfCore);
#endif
        }

        /// Return the memory footprint of this buffer in bytes.
        Index memUsage() const
        {
            size_t n = sizeof(*this);
#ifdef OPENVDB_2_ABI_COMPATIBLE
            if (mData) n += SIZE * sizeof(ValueType);
#else
            if (this->isOutOfCore()) n += sizeof(FileInfo);
            else if (mData) n += SIZE * sizeof(ValueType);
#endif
            return static_cast<Index>(n);
        }
        /// Return the number of values contained in this buffer.
        static Index size() { return SIZE; }

        /// @brief Return a const pointer to the array of voxel values.
        /// @details This method guarantees that the buffer is allocated and loaded.
        /// @warning This method should only be used by experts seeking low-level optimizations.
        const ValueType* data() const
        {
#ifndef OPENVDB_2_ABI_COMPATIBLE
            this->loadValues();
            if (mData == NULL) {
                Buffer* self = const_cast<Buffer*>(this);
                // This lock will be contended at most once.
                tbb::spin_mutex::scoped_lock lock(self->mMutex);
                if (mData == NULL) self->mData = new ValueType[SIZE];
            }
#endif
            return mData;
        }

        /// @brief Return a pointer to the array of voxel values.
        /// @details This method guarantees that the buffer is allocated and loaded.
        /// @warning This method should only be used by experts seeking low-level optimizations.
        ValueType* data()
        {
#ifndef OPENVDB_2_ABI_COMPATIBLE
            this->loadValues();
            if (mData == NULL) {
                // This lock will be contended at most once.
                tbb::spin_mutex::scoped_lock lock(mMutex);
                if (mData == NULL) mData = new ValueType[SIZE];
            }
#endif
            return mData;
        }

    private:
        /// If this buffer is empty, return zero, otherwise return the value at index @ i.
        const ValueType& at(Index i) const
        {
            assert(i < SIZE);
#ifdef OPENVDB_2_ABI_COMPATIBLE
            return mData[i];
#else
            this->loadValues();
            // We can't use the ternary operator here, otherwise Visual C++ returns
            // a reference to a temporary.
            if (mData) return mData[i]; else return sZero;
#endif
        }

        /// @brief Return a non-const reference to the value at index @a i.
        /// @details This method is private since it makes assumptions about the
        /// buffer's memory layout.  Buffers associated with custom leaf node types
        /// (e.g., a bool buffer implemented as a bitmask) might not be able to
        /// return non-const references to their values.
        ValueType& operator[](Index i) { return const_cast<ValueType&>(this->at(i)); }

        bool deallocate()
        {
            if (mData != NULL && !this->isOutOfCore()) {
                delete[] mData;
                mData = NULL;
                return true;
            }
            return false;
        }

#ifdef OPENVDB_2_ABI_COMPATIBLE
        void setOutOfCore(bool) {}
        void loadValues() const {}
        void doLoad() const {}
        bool detachFromFile() { return false; }
#else
        inline void setOutOfCore(bool b) { mOutOfCore = b; }
        // To facilitate inlining in the common case in which the buffer is in-core,
        // the loading logic is split into a separate function, doLoad().
        inline void loadValues() const { if (this->isOutOfCore()) this->doLoad(); }
        inline void doLoad() const;
        inline bool detachFromFile()
        {
            if (this->isOutOfCore()) {
                delete mFileInfo;
                mFileInfo = NULL;
                this->setOutOfCore(false);
                return true;
            }
            return false;
        }
#endif

        friend class ::TestLeaf;
        // Allow the parent LeafNode to access this buffer's data pointer.
        friend class LeafNode;

#ifdef OPENVDB_2_ABI_COMPATIBLE
        ValueType* mData;
#else
        union {
            ValueType* mData;
            FileInfo*  mFileInfo;
        };
        Index32 mOutOfCore; // currently interpreted as bool; extra bits reserved for future use
        tbb::spin_mutex mMutex; // 1 byte
        //int8_t mReserved[3]; // padding for alignment

        static const ValueType sZero;
#endif
    }; // class Buffer


    /// Default constructor
    LeafNode();

    /// @brief Constructor
    /// @param coords  the grid index coordinates of a voxel
    /// @param value   a value with which to fill the buffer
    /// @param active  the active state to which to initialize all voxels
    explicit LeafNode(const Coord& coords,
                      const ValueType& value = zeroVal<ValueType>(),
                      bool active = false);


#ifndef OPENVDB_2_ABI_COMPATIBLE
    /// @brief "Partial creation" constructor used during file input
    /// @param coords  the grid index coordinates of a voxel
    /// @param value   a value with which to fill the buffer
    /// @param active  the active state to which to initialize all voxels
    /// @details This constructor does not allocate memory for voxel values.
    LeafNode(PartialCreate,
             const Coord& coords,
             const ValueType& value = zeroVal<ValueType>(),
             bool active = false);
#endif

    /// Deep copy constructor
    LeafNode(const LeafNode&);

    /// Value conversion copy constructor
    template<typename OtherValueType>
    explicit LeafNode(const LeafNode<OtherValueType, Log2Dim>& other);

    /// Topology copy constructor
    template<typename OtherValueType>
    LeafNode(const LeafNode<OtherValueType, Log2Dim>& other,
             const ValueType& offValue, const ValueType& onValue, TopologyCopy);

    /// Topology copy constructor
    template<typename OtherValueType>
    LeafNode(const LeafNode<OtherValueType, Log2Dim>& other,
             const ValueType& background, TopologyCopy);

    /// Destructor.
    ~LeafNode();

    //
    // Statistics
    //
    /// Return log2 of the dimension of this LeafNode, e.g. 3 if dimensions are 8^3
    static Index log2dim() { return Log2Dim; }
    /// Return the number of voxels in each coordinate dimension.
    static Index dim() { return DIM; }
    /// Return the total number of voxels represented by this LeafNode
    static Index size() { return SIZE; }
    /// Return the total number of voxels represented by this LeafNode
    static Index numValues() { return SIZE; }
    /// Return the level of this node, which by definition is zero for LeafNodes
    static Index getLevel() { return LEVEL; }
    /// Append the Log2Dim of this LeafNode to the specified vector
    static void getNodeLog2Dims(std::vector<Index>& dims) { dims.push_back(Log2Dim); }
    /// Return the dimension of child nodes of this LeafNode, which is one for voxels.
    static Index getChildDim() { return 1; }
    /// Return the leaf count for this node, which is one.
    static Index32 leafCount() { return 1; }
    /// Return the non-leaf count for this node, which is zero.
    static Index32 nonLeafCount() { return 0; }

    /// Return the number of voxels marked On.
    Index64 onVoxelCount() const { return mValueMask.countOn(); }
    /// Return the number of voxels marked Off.
    Index64 offVoxelCount() const { return mValueMask.countOff(); }
    Index64 onLeafVoxelCount() const { return onVoxelCount(); }
    Index64 offLeafVoxelCount() const { return offVoxelCount(); }
    static Index64 onTileCount()  { return 0; }
    static Index64 offTileCount() { return 0; }
    /// Return @c true if this node has no active voxels.
    bool isEmpty() const { return mValueMask.isOff(); }
    /// Return @c true if this node contains only active voxels.
    bool isDense() const { return mValueMask.isOn(); }

#ifndef OPENVDB_2_ABI_COMPATIBLE
    /// Return @c true if memory for this node's buffer has been allocated.
    bool isAllocated() const { return !mBuffer.isOutOfCore() && !mBuffer.empty(); }
    /// Allocate memory for this node's buffer if it has not already been allocated.
    bool allocate() { return mBuffer.allocate(); }
#endif

    /// Return the memory in bytes occupied by this node.
    Index64 memUsage() const;

    /// Expand the given bounding box so that it includes this leaf node's active voxels.
    /// If visitVoxels is false this LeafNode will be approximated as dense, i.e. with all
    /// voxels active. Else the individual active voxels are visited to produce a tight bbox.
    void evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels = true) const;

    /// @brief Return the bounding box of this node, i.e., the full index space
    /// spanned by this leaf node.
    CoordBBox getNodeBoundingBox() const { return CoordBBox::createCube(mOrigin, DIM); }

    /// Set the grid index coordinates of this node's local origin.
    void setOrigin(const Coord& origin) { mOrigin = origin; }
    //@{
    /// Return the grid index coordinates of this node's local origin.
    const Coord& origin() const { return mOrigin; }
    void getOrigin(Coord& origin) const { origin = mOrigin; }
    void getOrigin(Int32& x, Int32& y, Int32& z) const { mOrigin.asXYZ(x, y, z); }
    //@}

    /// Return the linear table offset of the given global or local coordinates.
    static Index coordToOffset(const Coord& xyz);
    /// @brief Return the local coordinates for a linear table offset,
    /// where offset 0 has coordinates (0, 0, 0).
    static Coord offsetToLocalCoord(Index n);
    /// Return the global coordinates for a linear table offset.
    Coord offsetToGlobalCoord(Index n) const;

    /// Return a string representation of this node.
    std::string str() const;

    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const LeafNode<OtherType, OtherLog2Dim>* other) const;

    /// Check for buffer, state and origin equivalence.
    bool operator==(const LeafNode& other) const;
    bool operator!=(const LeafNode& other) const { return !(other == *this); }

protected:
    typedef typename NodeMaskType::OnIterator    MaskOnIterator;
    typedef typename NodeMaskType::OffIterator   MaskOffIterator;
    typedef typename NodeMaskType::DenseIterator MaskDenseIterator;

    // Type tags to disambiguate template instantiations
    struct ValueOn {}; struct ValueOff {}; struct ValueAll {};
    struct ChildOn {}; struct ChildOff {}; struct ChildAll {};

    template<typename MaskIterT, typename NodeT, typename ValueT, typename TagT>
    struct ValueIter:
        // Derives from SparseIteratorBase, but can also be used as a dense iterator,
        // if MaskIterT is a dense mask iterator type.
        public SparseIteratorBase<
            MaskIterT, ValueIter<MaskIterT, NodeT, ValueT, TagT>, NodeT, ValueT>
    {
        typedef SparseIteratorBase<MaskIterT, ValueIter, NodeT, ValueT> BaseT;

        ValueIter() {}
        ValueIter(const MaskIterT& iter, NodeT* parent): BaseT(iter, parent) {}

        ValueT& getItem(Index pos) const { return this->parent().getValue(pos); }
        ValueT& getValue() const { return this->parent().getValue(this->pos()); }

        // Note: setItem() can't be called on const iterators.
        void setItem(Index pos, const ValueT& value) const
        {
            this->parent().setValueOnly(pos, value);
        }
        // Note: setValue() can't be called on const iterators.
        void setValue(const ValueT& value) const
        {
            this->parent().setValueOnly(this->pos(), value);
        }

        // Note: modifyItem() can't be called on const iterators.
        template<typename ModifyOp>
        void modifyItem(Index n, const ModifyOp& op) const { this->parent().modifyValue(n, op); }
        // Note: modifyValue() can't be called on const iterators.
        template<typename ModifyOp>
        void modifyValue(const ModifyOp& op) const { this->parent().modifyValue(this->pos(), op); }
    };

    /// Leaf nodes have no children, so their child iterators have no get/set accessors.
    template<typename MaskIterT, typename NodeT, typename TagT>
    struct ChildIter:
        public SparseIteratorBase<MaskIterT, ChildIter<MaskIterT, NodeT, TagT>, NodeT, ValueType>
    {
        ChildIter() {}
        ChildIter(const MaskIterT& iter, NodeT* parent): SparseIteratorBase<
            MaskIterT, ChildIter<MaskIterT, NodeT, TagT>, NodeT, ValueType>(iter, parent) {}
    };

    template<typename NodeT, typename ValueT, typename TagT>
    struct DenseIter: public DenseIteratorBase<
        MaskDenseIterator, DenseIter<NodeT, ValueT, TagT>, NodeT, /*ChildT=*/void, ValueT>
    {
        typedef DenseIteratorBase<MaskDenseIterator, DenseIter, NodeT, void, ValueT> BaseT;
        typedef typename BaseT::NonConstValueType NonConstValueT;

        DenseIter() {}
        DenseIter(const MaskDenseIterator& iter, NodeT* parent): BaseT(iter, parent) {}

        bool getItem(Index pos, void*& child, NonConstValueT& value) const
        {
            value = this->parent().getValue(pos);
            child = NULL;
            return false; // no child
        }

        // Note: setItem() can't be called on const iterators.
        //void setItem(Index pos, void* child) const {}

        // Note: unsetItem() can't be called on const iterators.
        void unsetItem(Index pos, const ValueT& value) const
        {
            this->parent().setValueOnly(pos, value);
        }
    };

public:
    typedef ValueIter<MaskOnIterator, LeafNode, const ValueType, ValueOn>        ValueOnIter;
    typedef ValueIter<MaskOnIterator, const LeafNode, const ValueType, ValueOn>  ValueOnCIter;
    typedef ValueIter<MaskOffIterator, LeafNode, const ValueType, ValueOff>      ValueOffIter;
    typedef ValueIter<MaskOffIterator,const LeafNode,const ValueType,ValueOff>   ValueOffCIter;
    typedef ValueIter<MaskDenseIterator, LeafNode, const ValueType, ValueAll>    ValueAllIter;
    typedef ValueIter<MaskDenseIterator,const LeafNode,const ValueType,ValueAll> ValueAllCIter;
    typedef ChildIter<MaskOnIterator, LeafNode, ChildOn>                         ChildOnIter;
    typedef ChildIter<MaskOnIterator, const LeafNode, ChildOn>                   ChildOnCIter;
    typedef ChildIter<MaskOffIterator, LeafNode, ChildOff>                       ChildOffIter;
    typedef ChildIter<MaskOffIterator, const LeafNode, ChildOff>                 ChildOffCIter;
    typedef DenseIter<LeafNode, ValueType, ChildAll>                             ChildAllIter;
    typedef DenseIter<const LeafNode, const ValueType, ChildAll>                 ChildAllCIter;

    ValueOnCIter  cbeginValueOn() const { return ValueOnCIter(mValueMask.beginOn(), this); }
    ValueOnCIter   beginValueOn() const { return ValueOnCIter(mValueMask.beginOn(), this); }
    ValueOnIter    beginValueOn() { return ValueOnIter(mValueMask.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(mValueMask.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(mValueMask.beginOff(), this); }
    ValueOffIter   beginValueOff() { return ValueOffIter(mValueMask.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(mValueMask.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(mValueMask.beginDense(), this); }
    ValueAllIter   beginValueAll() { return ValueAllIter(mValueMask.beginDense(), this); }

    ValueOnCIter  cendValueOn() const { return ValueOnCIter(mValueMask.endOn(), this); }
    ValueOnCIter   endValueOn() const { return ValueOnCIter(mValueMask.endOn(), this); }
    ValueOnIter    endValueOn() { return ValueOnIter(mValueMask.endOn(), this); }
    ValueOffCIter cendValueOff() const { return ValueOffCIter(mValueMask.endOff(), this); }
    ValueOffCIter  endValueOff() const { return ValueOffCIter(mValueMask.endOff(), this); }
    ValueOffIter   endValueOff() { return ValueOffIter(mValueMask.endOff(), this); }
    ValueAllCIter cendValueAll() const { return ValueAllCIter(mValueMask.endDense(), this); }
    ValueAllCIter  endValueAll() const { return ValueAllCIter(mValueMask.endDense(), this); }
    ValueAllIter   endValueAll() { return ValueAllIter(mValueMask.endDense(), this); }

    // Note that [c]beginChildOn() and [c]beginChildOff() actually return end iterators,
    // because leaf nodes have no children.
    ChildOnCIter  cbeginChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnCIter   beginChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnIter    beginChildOn() { return ChildOnIter(mValueMask.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffIter   beginChildOff() { return ChildOffIter(mValueMask.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(mValueMask.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(mValueMask.beginDense(), this); }
    ChildAllIter   beginChildAll() { return ChildAllIter(mValueMask.beginDense(), this); }

    ChildOnCIter  cendChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnCIter   endChildOn() const { return ChildOnCIter(mValueMask.endOn(), this); }
    ChildOnIter    endChildOn() { return ChildOnIter(mValueMask.endOn(), this); }
    ChildOffCIter cendChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffCIter  endChildOff() const { return ChildOffCIter(mValueMask.endOff(), this); }
    ChildOffIter   endChildOff() { return ChildOffIter(mValueMask.endOff(), this); }
    ChildAllCIter cendChildAll() const { return ChildAllCIter(mValueMask.endDense(), this); }
    ChildAllCIter  endChildAll() const { return ChildAllCIter(mValueMask.endDense(), this); }
    ChildAllIter   endChildAll() { return ChildAllIter(mValueMask.endDense(), this); }

    //
    // Buffer management
    //
    /// @brief Exchange this node's data buffer with the given data buffer
    /// without changing the active states of the values.
    void swap(Buffer& other) { mBuffer.swap(other); }
    const Buffer& buffer() const { return mBuffer; }
    Buffer& buffer() { return mBuffer; }

    //
    // I/O methods
    //
    /// @brief Read in just the topology.
    /// @param is        the stream from which to read
    /// @param fromHalf  if true, floating-point input values are assumed to be 16-bit
    void readTopology(std::istream& is, bool fromHalf = false);
    /// @brief Write out just the topology.
    /// @param os      the stream to which to write
    /// @param toHalf  if true, output floating-point values as 16-bit half floats
    void writeTopology(std::ostream& os, bool toHalf = false) const;

    /// @brief Read buffers from a stream.
    /// @param is        the stream from which to read
    /// @param fromHalf  if true, floating-point input values are assumed to be 16-bit
    void readBuffers(std::istream& is, bool fromHalf = false);
    /// @brief Read buffers that intersect the given bounding box.
    /// @param is        the stream from which to read
    /// @param bbox      an index-space bounding box
    /// @param fromHalf  if true, floating-point input values are assumed to be 16-bit
    void readBuffers(std::istream& is, const CoordBBox& bbox, bool fromHalf = false);
    /// @brief Write buffers to a stream.
    /// @param os      the stream to which to write
    /// @param toHalf  if true, output floating-point values as 16-bit half floats
    void writeBuffers(std::ostream& os, bool toHalf = false) const;

    size_t streamingSize(bool toHalf = false) const;

    //
    // Accessor methods
    //
    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const;
    /// Return the value of the voxel at the given linear offset.
    const ValueType& getValue(Index offset) const;

    /// @brief Return @c true if the voxel at the given coordinates is active.
    /// @param xyz       the coordinates of the voxel to be probed
    /// @param[out] val  the value of the voxel at the given coordinates
    bool probeValue(const Coord& xyz, ValueType& val) const;
    /// @brief Return @c true if the voxel at the given offset is active.
    /// @param offset    the linear offset of the voxel to be probed
    /// @param[out] val  the value of the voxel at the given coordinates
    bool probeValue(Index offset, ValueType& val) const;

    /// Return the level (i.e., 0) at which leaf node values reside.
    static Index getValueLevel(const Coord&) { return LEVEL; }

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on);
    /// Set the active state of the voxel at the given offset but don't change its value.
    void setActiveState(Index offset, bool on) { assert(offset<SIZE); mValueMask.set(offset, on); }

    /// Set the value of the voxel at the given coordinates but don't change its active state.
    void setValueOnly(const Coord& xyz, const ValueType& val);
    /// Set the value of the voxel at the given offset but don't change its active state.
    void setValueOnly(Index offset, const ValueType& val);

    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz) { mValueMask.setOff(LeafNode::coordToOffset(xyz)); }
    /// Mark the voxel at the given offset as inactive but don't change its value.
    void setValueOff(Index offset) { assert(offset < SIZE); mValueMask.setOff(offset); }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& val);
    /// Set the value of the voxel at the given offset and mark the voxel as inactive.
    void setValueOff(Index offset, const ValueType& val);

    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz) { mValueMask.setOn(LeafNode::coordToOffset(xyz)); }
    /// Mark the voxel at the given offset as active but don't change its value.
    void setValueOn(Index offset) { assert(offset < SIZE); mValueMask.setOn(offset); }
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValueOn(const Coord& xyz, const ValueType& val) {
        this->setValueOn(LeafNode::coordToOffset(xyz), val);
    }
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& val) { this->setValueOn(xyz, val); }
    /// Set the value of the voxel at the given offset and mark the voxel as active.
    void setValueOn(Index offset, const ValueType& val) {
        mBuffer.setValue(offset, val);
        mValueMask.setOn(offset);
    }

    /// @brief Apply a functor to the value of the voxel at the given offset
    /// and mark the voxel as active.
    template<typename ModifyOp>
    void modifyValue(Index offset, const ModifyOp& op)
    {
        ValueType val = mBuffer[offset];
        op(val);
        mBuffer.setValue(offset, val);
        mValueMask.setOn(offset);
    }
    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        this->modifyValue(this->coordToOffset(xyz), op);
    }

    /// Apply a functor to the voxel at the given coordinates.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        const Index offset = this->coordToOffset(xyz);
        bool state = mValueMask.isOn(offset);
        ValueType val = mBuffer[offset];
        op(val, state);
        mBuffer.setValue(offset, val);
        mValueMask.set(offset, state);
    }

    /// Mark all voxels as active but don't change their values.
    void setValuesOn() { mValueMask.setOn(); }
    /// Mark all voxels as inactive but don't change their values.
    void setValuesOff() { mValueMask.setOff(); }

    /// Return @c true if the voxel at the given coordinates is active.
    bool isValueOn(const Coord& xyz) const {return this->isValueOn(LeafNode::coordToOffset(xyz));}
    /// Return @c true if the voxel at the given offset is active.
    bool isValueOn(Index offset) const { return mValueMask.isOn(offset); }

    /// Return @c false since leaf nodes never contain tiles.
    static bool hasActiveTiles() { return false; }

    /// Set all voxels that lie outside the given axis-aligned box to the background.
    void clip(const CoordBBox&, const ValueType& background);

    /// Set all voxels within an axis-aligned box to the specified value and active state.
    void fill(const CoordBBox& bbox, const ValueType&, bool active = true);

    /// Set all voxels to the specified value but don't change their active states.
    void fill(const ValueType& value);
    /// Set all voxels to the specified value and active state.
    void fill(const ValueType& value, bool active);

    /// @brief Copy into a dense grid the values of the voxels that lie within
    /// a given bounding box.
    ///
    /// @param bbox   inclusive bounding box of the voxels to be copied into the dense grid
    /// @param dense  dense grid with a stride in @e z of one (see tools::Dense
    ///               in tools/Dense.h for the required API)
    ///
    /// @note @a bbox is assumed to be identical to or contained in the coordinate domains
    /// of both the dense grid and this node, i.e., no bounds checking is performed.
    /// @note Consider using tools::CopyToDense in tools/Dense.h
    /// instead of calling this method directly.
    template<typename DenseT>
    void copyToDense(const CoordBBox& bbox, DenseT& dense) const;

    /// @brief Copy from a dense grid into this node the values of the voxels
    /// that lie within a given bounding box.
    /// @details Only values that are different (by more than the given tolerance)
    /// from the background value will be active.  Other values are inactive
    /// and truncated to the background value.
    ///
    /// @param bbox        inclusive bounding box of the voxels to be copied into this node
    /// @param dense       dense grid with a stride in @e z of one (see tools::Dense
    ///                    in tools/Dense.h for the required API)
    /// @param background  background value of the tree that this node belongs to
    /// @param tolerance   tolerance within which a value equals the background value
    ///
    /// @note @a bbox is assumed to be identical to or contained in the coordinate domains
    /// of both the dense grid and this node, i.e., no bounds checking is performed.
    /// @note Consider using tools::CopyFromDense in tools/Dense.h
    /// instead of calling this method directly.
    template<typename DenseT>
    void copyFromDense(const CoordBBox& bbox, const DenseT& dense,
                       const ValueType& background, const ValueType& tolerance);

    /// @brief Return the value of the voxel at the given coordinates.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    const ValueType& getValueAndCache(const Coord& xyz, AccessorT&) const
    {
        return this->getValue(xyz);
    }

    /// @brief Return @c true if the voxel at the given coordinates is active.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    bool isValueOnAndCache(const Coord& xyz, AccessorT&) const { return this->isValueOn(xyz); }

    /// @brief Change the value of the voxel at the given coordinates and mark it as active.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueAndCache(const Coord& xyz, const ValueType& val, AccessorT&)
    {
        this->setValueOn(xyz, val);
    }

    /// @brief Change the value of the voxel at the given coordinates
    /// but preserve its state.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord& xyz, const ValueType& val, AccessorT&)
    {
        this->setValueOnly(xyz, val);
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @note Used internally by ValueAccessor.
    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndCache(const Coord& xyz, const ModifyOp& op, AccessorT&)
    {
        this->modifyValue(xyz, op);
    }

    /// Apply a functor to the voxel at the given coordinates.
    /// @note Used internally by ValueAccessor.
    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord& xyz, const ModifyOp& op, AccessorT&)
    {
        this->modifyValueAndActiveState(xyz, op);
    }

    /// @brief Change the value of the voxel at the given coordinates and mark it as inactive.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setValueOffAndCache(const Coord& xyz, const ValueType& value, AccessorT&)
    {
        this->setValueOff(xyz, value);
    }

    /// @brief Set the active state of the voxel at the given coordinates
    /// without changing its value.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    void setActiveStateAndCache(const Coord& xyz, bool on, AccessorT&)
    {
        this->setActiveState(xyz, on);
    }

    /// @brief Return @c true if the voxel at the given coordinates is active
    /// and return the voxel value in @a val.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    bool probeValueAndCache(const Coord& xyz, ValueType& val, AccessorT&) const
    {
        return this->probeValue(xyz, val);
    }

    /// @brief Return the value of the voxel at the given coordinates and return
    /// its active state and level (i.e., 0) in @a state and @a level.
    /// @note Used internally by ValueAccessor.
    template<typename AccessorT>
    const ValueType& getValue(const Coord& xyz, bool& state, int& level, AccessorT&) const
    {
        const Index offset = this->coordToOffset(xyz);
        state = mValueMask.isOn(offset);
        level = LEVEL;
        return mBuffer[offset];
    }

    /// @brief Return the LEVEL (=0) at which leaf node values reside.
    /// @note Used internally by ValueAccessor (note last argument is a dummy).
    template<typename AccessorT>
    static Index getValueLevelAndCache(const Coord&, AccessorT&) { return LEVEL; }

    /// @brief Return a const reference to the first value in the buffer.
    /// @note Though it is potentially risky you can convert this
    /// to a non-const pointer by means of const_case<ValueType*>&.
    const ValueType& getFirstValue() const { return mBuffer[0]; }
    /// Return a const reference to the last value in the buffer.
    const ValueType& getLastValue() const { return mBuffer[SIZE - 1]; }

    /// @brief Replace inactive occurrences of @a oldBackground with @a newBackground,
    /// and inactive occurrences of @a -oldBackground with @a -newBackground.
    void resetBackground(const ValueType& oldBackground, const ValueType& newBackground);

    void negate();

    /// @brief No-op
    /// @details This function exists only to enable template instantiation.
    void voxelizeActiveTiles(bool = true) {}

    template<MergePolicy Policy> void merge(const LeafNode&);
    template<MergePolicy Policy> void merge(const ValueType& tileValue, bool tileActive);
    template<MergePolicy Policy>
    void merge(const LeafNode& other, const ValueType& /*bg*/, const ValueType& /*otherBG*/);

    /// @brief Union this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active if either of the original voxels
    /// were active.
    ///
    /// @note This operation modifies only active states, not values.
    template<typename OtherType>
    void topologyUnion(const LeafNode<OtherType, Log2Dim>& other);

    /// @brief Intersect this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if both of the original voxels
    /// were active.
    ///
    /// @details The last dummy argument is required to match the signature
    /// for InternalNode::topologyIntersection.
    ///
    /// @note This operation modifies only active states, not
    /// values. Also note that this operation can result in all voxels
    /// being inactive so consider subsequnetly calling prune.
    template<typename OtherType>
    void topologyIntersection(const LeafNode<OtherType, Log2Dim>& other, const ValueType&);

    /// @brief Difference this node's set of active values with the active values
    /// of the other node, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if the original voxel is
    /// active in this LeafNode and inactive in the other LeafNode.
    ///
    /// @details The last dummy argument is required to match the signature
    /// for InternalNode::topologyDifference.
    ///
    /// @note This operation modifies only active states, not values.
    /// Also, because it can deactivate all of this node's voxels,
    /// consider subsequently calling prune.
    template<typename OtherType>
    void topologyDifference(const LeafNode<OtherType, Log2Dim>& other, const ValueType&);

    template<typename CombineOp>
    void combine(const LeafNode& other, CombineOp& op);
    template<typename CombineOp>
    void combine(const ValueType& value, bool valueIsActive, CombineOp& op);

    template<typename CombineOp, typename OtherType /*= ValueType*/>
    void combine2(const LeafNode& other, const OtherType&, bool valueIsActive, CombineOp&);
    template<typename CombineOp, typename OtherNodeT /*= LeafNode*/>
    void combine2(const ValueType&, const OtherNodeT& other, bool valueIsActive, CombineOp&);
    template<typename CombineOp, typename OtherNodeT /*= LeafNode*/>
    void combine2(const LeafNode& b0, const OtherNodeT& b1, CombineOp&);

    /// @brief Calls the templated functor BBoxOp with bounding box
    /// information. An additional level argument is provided to the
    /// callback.
    ///
    /// @note The bounding boxes are guarenteed to be non-overlapping.
    template<typename BBoxOp> void visitActiveBBox(BBoxOp&) const;

    template<typename VisitorOp> void visit(VisitorOp&);
    template<typename VisitorOp> void visit(VisitorOp&) const;

    template<typename OtherLeafNodeType, typename VisitorOp>
    void visit2Node(OtherLeafNodeType& other, VisitorOp&);
    template<typename OtherLeafNodeType, typename VisitorOp>
    void visit2Node(OtherLeafNodeType& other, VisitorOp&) const;
    template<typename IterT, typename VisitorOp>
    void visit2(IterT& otherIter, VisitorOp&, bool otherIsLHS = false);
    template<typename IterT, typename VisitorOp>
    void visit2(IterT& otherIter, VisitorOp&, bool otherIsLHS = false) const;

    //@{
    /// This function exists only to enable template instantiation.
    void prune(const ValueType& /*tolerance*/ = zeroVal<ValueType>()) {}
    void addLeaf(LeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(LeafNode*, AccessorT&) {}
    template<typename NodeT>
    NodeT* stealNode(const Coord&, const ValueType&, bool) { return NULL; }
    template<typename NodeT>
    NodeT* probeNode(const Coord&) { return NULL; }
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord&) const { return NULL; }
    template<typename ArrayT> void getNodes(ArrayT&) const {}
    template<typename ArrayT> void stealNodes(ArrayT&, const ValueType&, bool) {}
    //@}

    void addTile(Index level, const Coord&, const ValueType&, bool);
    void addTile(Index offset, const ValueType&, bool);
    template<typename AccessorT>
    void addTileAndCache(Index, const Coord&, const ValueType&, bool, AccessorT&);

    //@{
    /// @brief Return a pointer to this node.
    LeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    LeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }
    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,LeafNode>::value)) return NULL;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    LeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    LeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    //@}
    //@{
    /// @brief Return a @const pointer to this node.
    const LeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const LeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const { return this; }
    template<typename AccessorT>
    const LeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const LeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,LeafNode>::value)) return NULL;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}

    /// Return @c true if all of this node's values have the same active state
    /// and are in the range this->getFirstValue() +/- @a tolerance.
    ///
    ///
    /// @param constValue  Is updated with the first value of this leaf node.
    /// @param state       Is updated with the state of all values IF method
    ///                    returns @c true. Else the value is undefined!
    /// @param tolerance   The tolerance used to determine if values are
    ///                    approximatly equal to the for value.
    bool isConstant(ValueType& constValue, bool& state,
                    const ValueType& tolerance = zeroVal<ValueType>()) const;

    /// Return @c true if all of this node's values have the same active state
    /// and are in the range (@a maxValue + @a minValue)/2 +/- @a tolerance.
    ///
    /// @param minValue  Is updated with the minimum of all values IF method
    ///                  returns @c true. Else the value is undefined!
    /// @param maxValue  Is updated with the maximum of all values IF method
    ///                  returns @c true. Else the value is undefined!
    /// @param state     Is updated with the state of all values IF method
    ///                  returns @c true. Else the value is undefined!
    /// @param tolerance The tolerance used to determine if values are
    ///                  approximatly constant.
    bool isConstant(ValueType& minValue, ValueType& maxValue,
                    bool& state, const ValueType& tolerance = zeroVal<ValueType>()) const;
    
    /// Return @c true if all of this node's values are inactive.
    bool isInactive() const { return mValueMask.isOff(); }

protected:
    friend class ::TestLeaf;
    template<typename> friend class ::TestLeafIO;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend class LeafNode;

    friend struct ValueIter<MaskOnIterator, LeafNode, ValueType, ValueOn>;
    friend struct ValueIter<MaskOffIterator, LeafNode, ValueType, ValueOff>;
    friend struct ValueIter<MaskDenseIterator, LeafNode, ValueType, ValueAll>;
    friend struct ValueIter<MaskOnIterator, const LeafNode, ValueType, ValueOn>;
    friend struct ValueIter<MaskOffIterator, const LeafNode, ValueType, ValueOff>;
    friend struct ValueIter<MaskDenseIterator, const LeafNode, ValueType, ValueAll>;

    // Allow iterators to call mask accessor methods (see below).
    /// @todo Make mask accessors public?
    friend class IteratorBase<MaskOnIterator, LeafNode>;
    friend class IteratorBase<MaskOffIterator, LeafNode>;
    friend class IteratorBase<MaskDenseIterator, LeafNode>;

    // Mask accessors
public:
    bool isValueMaskOn(Index n) const { return mValueMask.isOn(n); }
    bool isValueMaskOn() const { return mValueMask.isOn(); }
    bool isValueMaskOff(Index n) const { return mValueMask.isOff(n); }
    bool isValueMaskOff() const { return mValueMask.isOff(); }
    const NodeMaskType& getValueMask() const { return mValueMask; }
    NodeMaskType& getValueMask() { return mValueMask; }
    const NodeMaskType& valueMask() const { return mValueMask; }
    void setValueMask(const NodeMaskType& mask) { mValueMask = mask; }
    bool isChildMaskOn(Index) const { return false; } // leaf nodes have no children
    bool isChildMaskOff(Index) const { return true; }
    bool isChildMaskOff() const { return true; }
protected:
    void setValueMask(Index n, bool on) { mValueMask.set(n, on); }
    void setValueMaskOn(Index n)  { mValueMask.setOn(n); }
    void setValueMaskOff(Index n) { mValueMask.setOff(n); }

    /// Compute the origin of the leaf node that contains the voxel with the given coordinates.
    static void evalNodeOrigin(Coord& xyz) { xyz &= ~(DIM - 1); }

    template<typename NodeT, typename VisitorOp, typename ChildAllIterT>
    static inline void doVisit(NodeT&, VisitorOp&);

    template<typename NodeT, typename OtherNodeT, typename VisitorOp,
             typename ChildAllIterT, typename OtherChildAllIterT>
    static inline void doVisit2Node(NodeT& self, OtherNodeT& other, VisitorOp&);

    template<typename NodeT, typename VisitorOp,
             typename ChildAllIterT, typename OtherChildAllIterT>
    static inline void doVisit2(NodeT& self, OtherChildAllIterT&, VisitorOp&, bool otherIsLHS);

private:
    /// Buffer containing the actual data values
    Buffer mBuffer;
    /// Bitmask that determines which voxels are active
    NodeMaskType mValueMask;
    /// Global grid index coordinates (x,y,z) of the local origin of this node
    Coord mOrigin;
}; // end of LeafNode class


#ifndef OPENVDB_2_ABI_COMPATIBLE
template<typename T, Index Log2Dim>
const T LeafNode<T, Log2Dim>::Buffer::sZero = zeroVal<T>();
#endif


////////////////////////////////////////


//@{
/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename NodeT2>
struct SameLeafConfig { static const bool value = false; };

template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, LeafNode<T2, Dim1> > { static const bool value = true; };
//@}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline
LeafNode<T, Log2Dim>::LeafNode():
    mValueMask(),//default is off!
    mOrigin(0, 0, 0)
{
}


template<typename T, Index Log2Dim>
inline
LeafNode<T, Log2Dim>::LeafNode(const Coord& xyz, const ValueType& val, bool active):
    mBuffer(val),
    mValueMask(active),
    mOrigin(xyz & (~(DIM - 1)))
{
}


#ifndef OPENVDB_2_ABI_COMPATIBLE
template<typename T, Index Log2Dim>
inline
LeafNode<T, Log2Dim>::LeafNode(PartialCreate, const Coord& xyz, const ValueType& val, bool active):
    mBuffer(PartialCreate(), val),
    mValueMask(active),
    mOrigin(xyz & (~(DIM - 1)))
{
}
#endif


template<typename T, Index Log2Dim>
inline
LeafNode<T, Log2Dim>::LeafNode(const LeafNode &other):
    mBuffer(other.mBuffer),
    mValueMask(other.valueMask()),
    mOrigin(other.mOrigin)
{
}


// Copy-construct from a leaf node with the same configuration but a different ValueType.
template<typename T, Index Log2Dim>
template<typename OtherValueType>
inline
LeafNode<T, Log2Dim>::LeafNode(const LeafNode<OtherValueType, Log2Dim>& other):
    mValueMask(other.valueMask()),
    mOrigin(other.mOrigin)
{
    struct Local {
        /// @todo Consider using a value conversion functor passed as an argument instead.
        static inline ValueType convertValue(const OtherValueType& val) { return ValueType(val); }
    };

    for (Index i = 0; i < SIZE; ++i) {
        mBuffer[i] = Local::convertValue(other.mBuffer[i]);
    }
}


template<typename T, Index Log2Dim>
template<typename OtherValueType>
inline
LeafNode<T, Log2Dim>::LeafNode(const LeafNode<OtherValueType, Log2Dim>& other,
                               const ValueType& background, TopologyCopy):
    mBuffer(background),
    mValueMask(other.valueMask()),
    mOrigin(other.mOrigin)
{
}


template<typename T, Index Log2Dim>
template<typename OtherValueType>
inline
LeafNode<T, Log2Dim>::LeafNode(const LeafNode<OtherValueType, Log2Dim>& other,
    const ValueType& offValue, const ValueType& onValue, TopologyCopy):
    mValueMask(other.valueMask()),
    mOrigin(other.mOrigin)
{
    for (Index i = 0; i < SIZE; ++i) {
        mBuffer[i] = (mValueMask.isOn(i) ? onValue : offValue);
    }
}


template<typename T, Index Log2Dim>
inline
LeafNode<T, Log2Dim>::~LeafNode()
{
}


template<typename T, Index Log2Dim>
inline std::string
LeafNode<T, Log2Dim>::str() const
{
    std::ostringstream ostr;
    ostr << "LeafNode @" << mOrigin << ": " << mBuffer;
    return ostr.str();
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline Index
LeafNode<T, Log2Dim>::coordToOffset(const Coord& xyz)
{
    assert ((xyz[0] & (DIM-1u)) < DIM && (xyz[1] & (DIM-1u)) < DIM && (xyz[2] & (DIM-1u)) < DIM);
    return ((xyz[0] & (DIM-1u)) << 2*Log2Dim)
        +  ((xyz[1] & (DIM-1u)) <<  Log2Dim)
        +   (xyz[2] & (DIM-1u));
}

template<typename T, Index Log2Dim>
inline Coord
LeafNode<T, Log2Dim>::offsetToLocalCoord(Index n)
{
    assert(n<(1<< 3*Log2Dim));
    Coord xyz;
    xyz.setX(n >> 2*Log2Dim);
    n &= ((1<<2*Log2Dim)-1);
    xyz.setY(n >> Log2Dim);
    xyz.setZ(n & ((1<<Log2Dim)-1));
    return xyz;
}


template<typename T, Index Log2Dim>
inline Coord
LeafNode<T, Log2Dim>::offsetToGlobalCoord(Index n) const
{
    return (this->offsetToLocalCoord(n) + this->origin());
}


////////////////////////////////////////


template<typename ValueT, Index Log2Dim>
inline const ValueT&
LeafNode<ValueT, Log2Dim>::getValue(const Coord& xyz) const
{
    return this->getValue(LeafNode::coordToOffset(xyz));
}

template<typename ValueT, Index Log2Dim>
inline const ValueT&
LeafNode<ValueT, Log2Dim>::getValue(Index offset) const
{
    assert(offset < SIZE);
    return mBuffer[offset];
}


template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::probeValue(const Coord& xyz, ValueType& val) const
{
    return this->probeValue(LeafNode::coordToOffset(xyz), val);
}

template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::probeValue(Index offset, ValueType& val) const
{
    assert(offset < SIZE);
    val = mBuffer[offset];
    return mValueMask.isOn(offset);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::setValueOff(const Coord& xyz, const ValueType& val)
{
    this->setValueOff(LeafNode::coordToOffset(xyz), val);
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::setValueOff(Index offset, const ValueType& val)
{
    assert(offset < SIZE);
    mBuffer.setValue(offset, val);
    mValueMask.setOff(offset);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::setActiveState(const Coord& xyz, bool on)
{
    mValueMask.set(this->coordToOffset(xyz), on);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::setValueOnly(const Coord& xyz, const ValueType& val)
{
    this->setValueOnly(LeafNode::coordToOffset(xyz), val);
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::setValueOnly(Index offset, const ValueType& val)
{
    assert(offset<SIZE); mBuffer.setValue(offset, val);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::clip(const CoordBBox& clipBBox, const T& background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with the background.
        this->fill(background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Set any voxels that lie outside the region to the background value.

    // Construct a boolean mask that is on inside the clipping region and off outside it.
    NodeMaskType mask;
    nodeBBox.intersect(clipBBox);
    Coord xyz;
    int &x = xyz.x(), &y = xyz.y(), &z = xyz.z();
    for (x = nodeBBox.min().x(); x <= nodeBBox.max().x(); ++x) {
        for (y = nodeBBox.min().y(); y <= nodeBBox.max().y(); ++y) {
            for (z = nodeBBox.min().z(); z <= nodeBBox.max().z(); ++z) {
                mask.setOn(static_cast<Index32>(this->coordToOffset(xyz)));
            }
        }
    }

    // Set voxels that lie in the inactive region of the mask (i.e., outside
    // the clipping region) to the background value.
    for (MaskOffIterator maskIter = mask.beginOff(); maskIter; ++maskIter) {
        this->setValueOff(maskIter.pos(), background);
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif

    for (Int32 x = bbox.min().x(); x <= bbox.max().x(); ++x) {
        const Index offsetX = (x & (DIM-1u)) << 2*Log2Dim;
        for (Int32 y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            const Index offsetXY = offsetX + ((y & (DIM-1u)) << Log2Dim);
            for (Int32 z = bbox.min().z(); z <= bbox.max().z(); ++z) {
                const Index offset = offsetXY + (z & (DIM-1u));
                mBuffer[offset] = value;
                mValueMask.set(offset, active);
            }
        }
    }
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const ValueType& value)
{
    mBuffer.fill(value);
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::fill(const ValueType& value, bool active)
{
    mBuffer.fill(value);
    mValueMask.set(active);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename DenseT>
inline void
LeafNode<T, Log2Dim>::copyToDense(const CoordBBox& bbox, DenseT& dense) const
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->isAllocated()) return;
#endif

    typedef typename DenseT::ValueType DenseValueType;

    const size_t xStride = dense.xStride(), yStride = dense.yStride(), zStride = dense.zStride();
    const Coord& min = dense.bbox().min();
    DenseValueType* t0 = dense.data() + zStride * (bbox.min()[2] - min[2]); // target array
    const T* s0 = &mBuffer[bbox.min()[2] & (DIM-1u)]; // source array
    for (Int32 x = bbox.min()[0], ex = bbox.max()[0] + 1; x < ex; ++x) {
        DenseValueType* t1 = t0 + xStride * (x - min[0]);
        const T* s1 = s0 + ((x & (DIM-1u)) << 2*Log2Dim);
        for (Int32 y = bbox.min()[1], ey = bbox.max()[1] + 1; y < ey; ++y) {
            DenseValueType* t2 = t1 + yStride * (y - min[1]);
            const T* s2 = s1 + ((y & (DIM-1u)) << Log2Dim);
            for (Int32 z = bbox.min()[2], ez = bbox.max()[2] + 1; z < ez; ++z, t2 += zStride) {
                *t2 = DenseValueType(*s2++);
            }
        }
    }
}


template<typename T, Index Log2Dim>
template<typename DenseT>
inline void
LeafNode<T, Log2Dim>::copyFromDense(const CoordBBox& bbox, const DenseT& dense,
                                    const ValueType& background, const ValueType& tolerance)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif

    typedef typename DenseT::ValueType DenseValueType;

    const size_t xStride = dense.xStride(), yStride = dense.yStride(), zStride = dense.zStride();
    const Coord& min = dense.bbox().min();

    const DenseValueType* s0 = dense.data() + zStride * (bbox.min()[2] - min[2]); // source
    const Int32 n0 = bbox.min()[2] & (DIM-1u);
    for (Int32 x = bbox.min()[0], ex = bbox.max()[0]+1; x < ex; ++x) {
        const DenseValueType* s1 = s0 + xStride * (x - min[0]);
        const Int32 n1 = n0 + ((x & (DIM-1u)) << 2*LOG2DIM);
        for (Int32 y = bbox.min()[1], ey = bbox.max()[1]+1; y < ey; ++y) {
            const DenseValueType* s2 = s1 + yStride * (y - min[1]);
            Int32 n2 = n1 + ((y & (DIM-1u)) << LOG2DIM);
            for (Int32 z = bbox.min()[2], ez = bbox.max()[2]+1; z < ez; ++z, ++n2, s2 += zStride) {
                if (math::isApproxEqual(background, ValueType(*s2), tolerance)) {
                    mValueMask.setOff(n2);
                    mBuffer[n2] = background;
                } else {
                    mValueMask.setOn(n2);
                    mBuffer[n2] = ValueType(*s2);
                }
            }
        }
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::readTopology(std::istream& is, bool /*fromHalf*/)
{
    mValueMask.load(is);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::writeTopology(std::ostream& os, bool /*toHalf*/) const
{
    mValueMask.save(os);
}


////////////////////////////////////////


#ifndef OPENVDB_2_ABI_COMPATIBLE
template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::Buffer::doLoad() const
{
    if (!this->isOutOfCore()) return;

    Buffer* self = const_cast<Buffer*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    if (!this->isOutOfCore()) return;

    boost::scoped_ptr<FileInfo> info(self->mFileInfo);
    assert(info.get() != NULL);
    assert(info->mapping.get() != NULL);
    assert(info->meta.get() != NULL);

    /// @todo For now, we have to clear the mData pointer in order for allocate() to take effect.
    self->mData = NULL;
    self->allocate();

    boost::shared_ptr<std::streambuf> buf = info->mapping->createBuffer();
    std::istream is(buf.get());

    io::setStreamMetadataPtr(is, info->meta, /*transfer=*/true);

    NodeMaskType mask;
    is.seekg(info->maskpos);
    mask.load(is);

    is.seekg(info->bufpos);
    io::readCompressedValues(is, self->mData, SIZE, mask, io::getHalfFloat(is));

    self->setOutOfCore(false);
}
#endif


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T,Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    this->readBuffers(is, CoordBBox::inf(), fromHalf);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T,Log2Dim>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    std::streamoff maskpos = is.tellg();
#endif

    // Read in the value mask.
    mValueMask.load(is);

    int8_t numBuffers = 1;
    if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        // Read in the origin.
        is.read(reinterpret_cast<char*>(&mOrigin), sizeof(Coord::ValueType) * 3);

        // Read in the number of buffers, which should now always be one.
        is.read(reinterpret_cast<char*>(&numBuffers), sizeof(int8_t));
    }

    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.
        // Read and discard its voxel values.
        Buffer temp;
        io::readCompressedValues(is, temp.mData, SIZE, mValueMask, fromHalf);
        mValueMask.setOff();
        mBuffer.setOutOfCore(false);
    } else {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        // If this node lies completely inside the clipping region and it is being read
        // from a memory-mapped file, delay loading of its buffer until the buffer
        // is actually accessed.  (If this node requires clipping, its buffer
        // must be accessed and therefore must be loaded.)
        io::MappedFile::Ptr mappedFile = io::getMappedFilePtr(is);
        const bool delayLoad = ((mappedFile.get() != NULL) && clipBBox.isInside(nodeBBox));

        if (delayLoad) {
            mBuffer.setOutOfCore(true);
            mBuffer.mFileInfo = new FileInfo;
            mBuffer.mFileInfo->bufpos = is.tellg();
            mBuffer.mFileInfo->mapping = mappedFile;
            // Save the offset to the value mask, because the in-memory copy
            // might change before the value buffer gets read.
            mBuffer.mFileInfo->maskpos = maskpos;

            mBuffer.mFileInfo->meta = io::getStreamMetadataPtr(is);

            // Read and discard voxel values.
            Buffer temp;
            io::readCompressedValues(is, temp.mData, SIZE, mValueMask, fromHalf);
        } else {
#endif
            mBuffer.allocate();
            io::readCompressedValues(is, mBuffer.mData, SIZE, mValueMask, fromHalf);
            mBuffer.setOutOfCore(false);

            // Get this tree's background value.
            T background = zeroVal<T>();
            if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
                background = *static_cast<const T*>(bgPtr);
            }
            this->clip(clipBBox, background);
#ifndef OPENVDB_2_ABI_COMPATIBLE
        }
#endif
    }

    if (numBuffers > 1) {
        // Read in and discard auxiliary buffers that were created with earlier
        // versions of the library.  (Auxiliary buffers are not mask compressed.)
        const bool zipped = io::getDataCompression(is) & io::COMPRESS_ZIP;
        Buffer temp;
        for (int i = 1; i < numBuffers; ++i) {
            if (fromHalf) {
                io::HalfReader<io::RealToHalf<T>::isReal, T>::read(is, temp.mData, SIZE, zipped);
            } else {
                io::readData<T>(is, temp.mData, SIZE, zipped);
            }
        }
    }
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    // Write out the value mask.
    mValueMask.save(os);

    mBuffer.loadValues();

    io::writeCompressedValues(os, mBuffer.mData, SIZE,
        mValueMask, /*childMask=*/NodeMaskType(), toHalf);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::operator==(const LeafNode& other) const
{
    return mOrigin == other.mOrigin &&
           mValueMask == other.valueMask() &&
           mBuffer == other.mBuffer;
}


template<typename T, Index Log2Dim>
inline Index64
LeafNode<T, Log2Dim>::memUsage() const
{
    // Use sizeof(*this) to capture alignment-related padding
    // (but note that sizeof(*this) includes sizeof(mBuffer)).
    return sizeof(*this) + mBuffer.memUsage() - sizeof(mBuffer);
}


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    CoordBBox this_bbox = this->getNodeBoundingBox();
    if (bbox.isInside(this_bbox)) return;//this LeafNode is already enclosed in the bbox
    if (ValueOnCIter iter = this->cbeginValueOn()) {//any active values?
        if (visitVoxels) {//use voxel granularity?
            this_bbox.reset();
            for(; iter; ++iter) this_bbox.expand(this->offsetToLocalCoord(iter.pos()));
            this_bbox.translate(this->origin());
        }
        bbox.expand(this_bbox);
    }
}


template<typename T, Index Log2Dim>
template<typename OtherType, Index OtherLog2Dim>
inline bool
LeafNode<T, Log2Dim>::hasSameTopology(const LeafNode<OtherType, OtherLog2Dim>* other) const
{
    assert(other);
    return (Log2Dim == OtherLog2Dim && mValueMask == other->getValueMask());
}


template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::isConstant(ValueType& value, bool& state,
                                 const ValueType& tolerance) const
{
    state = mValueMask.isOn();
    if (!(state || mValueMask.isOff())) return false;// Are values neither active nor inactive?
    
    value = mBuffer[0];
    for (Index i = 1; i < SIZE; ++i) {
        if ( !math::isApproxEqual(mBuffer[i], value, tolerance) ) return false;
    }
    return true;
}

template<typename T, Index Log2Dim>
inline bool
LeafNode<T, Log2Dim>::isConstant(ValueType& minValue, ValueType& maxValue,
                                 bool& state, const ValueType& tolerance) const
{
    state = mValueMask.isOn();
    if (!(state || mValueMask.isOff())) return false;// Are values neither active nor inactive?
    
    const T range = 2 * tolerance;
    minValue = maxValue = mBuffer[0];
    for (Index i = 1; i < SIZE; ++i) {// early termination
        const T& v = mBuffer[i];
        if (v < minValue) {
            if ((maxValue - v) > range) return false;
            minValue = v;
        } else if (v > maxValue) {
            if ((v - minValue) > range) return false;
            maxValue = v;
        }
    }
    return true;
}

////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::addTile(Index /*level*/, const Coord& xyz, const ValueType& val, bool active)
{
    this->addTile(this->coordToOffset(xyz), val, active);
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::addTile(Index offset, const ValueType& val, bool active)
{
    assert(offset < SIZE);
    setValueOnly(offset, val);
    setActiveState(offset, active);
}

template<typename T, Index Log2Dim>
template<typename AccessorT>
inline void
LeafNode<T, Log2Dim>::addTileAndCache(Index level, const Coord& xyz,
    const ValueType& val, bool active, AccessorT&)
{
    this->addTile(level, xyz, val, active);
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::resetBackground(const ValueType& oldBackground,
                                      const ValueType& newBackground)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif

    typename NodeMaskType::OffIterator iter;
    // For all inactive values...
    for (iter = this->mValueMask.beginOff(); iter; ++iter) {
        ValueType &inactiveValue = mBuffer[iter.pos()];
        if (math::isApproxEqual(inactiveValue, oldBackground)) {
            inactiveValue = newBackground;
        } else if (math::isApproxEqual(inactiveValue, math::negative(oldBackground))) {
            inactiveValue = math::negative(newBackground);
        }
    }
}


template<typename T, Index Log2Dim>
template<MergePolicy Policy>
inline void
LeafNode<T, Log2Dim>::merge(const LeafNode& other)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif

    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    if (Policy == MERGE_NODES) return;
    typename NodeMaskType::OnIterator iter = other.valueMask().beginOn();
    for (; iter; ++iter) {
        const Index n = iter.pos();
        if (mValueMask.isOff(n)) {
            mBuffer[n] = other.mBuffer[n];
            mValueMask.setOn(n);
        }
    }
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}

template<typename T, Index Log2Dim>
template<MergePolicy Policy>
inline void
LeafNode<T, Log2Dim>::merge(const LeafNode& other,
    const ValueType& /*bg*/, const ValueType& /*otherBG*/)
{
    this->template merge<Policy>(other);
}

template<typename T, Index Log2Dim>
template<MergePolicy Policy>
inline void
LeafNode<T, Log2Dim>::merge(const ValueType& tileValue, bool tileActive)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif

    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    if (Policy != MERGE_ACTIVE_STATES_AND_NODES) return;
    if (!tileActive) return;
    // Replace all inactive values with the active tile value.
    for (typename NodeMaskType::OffIterator iter = mValueMask.beginOff(); iter; ++iter) {
        const Index n = iter.pos();
        mBuffer[n] = tileValue;
        mValueMask.setOn(n);
    }
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


template<typename T, Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<T, Log2Dim>::topologyUnion(const LeafNode<OtherType, Log2Dim>& other)
{
    mValueMask |= other.valueMask();
}

template<typename T, Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<T, Log2Dim>::topologyIntersection(const LeafNode<OtherType, Log2Dim>& other,
                                           const ValueType&)
{
    mValueMask &= other.valueMask();
}

template<typename T, Index Log2Dim>
template<typename OtherType>
inline void
LeafNode<T, Log2Dim>::topologyDifference(const LeafNode<OtherType, Log2Dim>& other,
                                         const ValueType&)
{
    mValueMask &= !other.valueMask();
}

template<typename T, Index Log2Dim>
inline void
LeafNode<T, Log2Dim>::negate()
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    for (Index i = 0; i < SIZE; ++i) {
        mBuffer[i] = -mBuffer[i];
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename CombineOp>
inline void
LeafNode<T, Log2Dim>::combine(const LeafNode& other, CombineOp& op)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    CombineArgs<T> args;
    for (Index i = 0; i < SIZE; ++i) {
        op(args.setARef(mBuffer[i])
            .setAIsActive(mValueMask.isOn(i))
            .setBRef(other.mBuffer[i])
            .setBIsActive(other.valueMask().isOn(i))
            .setResultRef(mBuffer[i]));
        mValueMask.set(i, args.resultIsActive());
    }
}


template<typename T, Index Log2Dim>
template<typename CombineOp>
inline void
LeafNode<T, Log2Dim>::combine(const ValueType& value, bool valueIsActive, CombineOp& op)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    CombineArgs<T> args;
    args.setBRef(value).setBIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        op(args.setARef(mBuffer[i])
            .setAIsActive(mValueMask.isOn(i))
            .setResultRef(mBuffer[i]));
        mValueMask.set(i, args.resultIsActive());
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename CombineOp, typename OtherType>
inline void
LeafNode<T, Log2Dim>::combine2(const LeafNode& other, const OtherType& value,
    bool valueIsActive, CombineOp& op)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    CombineArgs<T, OtherType> args;
    args.setBRef(value).setBIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        op(args.setARef(other.mBuffer[i])
            .setAIsActive(other.valueMask().isOn(i))
            .setResultRef(mBuffer[i]));
        mValueMask.set(i, args.resultIsActive());
    }
}


template<typename T, Index Log2Dim>
template<typename CombineOp, typename OtherNodeT>
inline void
LeafNode<T, Log2Dim>::combine2(const ValueType& value, const OtherNodeT& other,
    bool valueIsActive, CombineOp& op)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    CombineArgs<T, typename OtherNodeT::ValueType> args;
    args.setARef(value).setAIsActive(valueIsActive);
    for (Index i = 0; i < SIZE; ++i) {
        op(args.setBRef(other.mBuffer[i])
            .setBIsActive(other.valueMask().isOn(i))
            .setResultRef(mBuffer[i]));
        mValueMask.set(i, args.resultIsActive());
    }
}


template<typename T, Index Log2Dim>
template<typename CombineOp, typename OtherNodeT>
inline void
LeafNode<T, Log2Dim>::combine2(const LeafNode& b0, const OtherNodeT& b1, CombineOp& op)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!this->allocate()) return;
#endif
    CombineArgs<T, typename OtherNodeT::ValueType> args;
    for (Index i = 0; i < SIZE; ++i) {
        mValueMask.set(i, b0.valueMask().isOn(i) || b1.valueMask().isOn(i));
        op(args.setARef(b0.mBuffer[i])
            .setAIsActive(b0.valueMask().isOn(i))
            .setBRef(b1.mBuffer[i])
            .setBIsActive(b1.valueMask().isOn(i))
            .setResultRef(mBuffer[i]));
        mValueMask.set(i, args.resultIsActive());
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename BBoxOp>
inline void
LeafNode<T, Log2Dim>::visitActiveBBox(BBoxOp& op) const
{
    if (op.template descent<LEVEL>()) {
        for (ValueOnCIter i=this->cbeginValueOn(); i; ++i) {
#ifdef _MSC_VER
            op.operator()<LEVEL>(CoordBBox::createCube(i.getCoord(), 1));
#else
            op.template operator()<LEVEL>(CoordBBox::createCube(i.getCoord(), 1));
#endif
        }
    } else {
#ifdef _MSC_VER
        op.operator()<LEVEL>(this->getNodeBoundingBox());
#else
        op.template operator()<LEVEL>(this->getNodeBoundingBox());
#endif
    }
}


template<typename T, Index Log2Dim>
template<typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit(VisitorOp& op)
{
    doVisit<LeafNode, VisitorOp, ChildAllIter>(*this, op);
}


template<typename T, Index Log2Dim>
template<typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit(VisitorOp& op) const
{
    doVisit<const LeafNode, VisitorOp, ChildAllCIter>(*this, op);
}


template<typename T, Index Log2Dim>
template<typename NodeT, typename VisitorOp, typename ChildAllIterT>
inline void
LeafNode<T, Log2Dim>::doVisit(NodeT& self, VisitorOp& op)
{
    for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
        op(iter);
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename OtherLeafNodeType, typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit2Node(OtherLeafNodeType& other, VisitorOp& op)
{
    doVisit2Node<LeafNode, OtherLeafNodeType, VisitorOp, ChildAllIter,
        typename OtherLeafNodeType::ChildAllIter>(*this, other, op);
}


template<typename T, Index Log2Dim>
template<typename OtherLeafNodeType, typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit2Node(OtherLeafNodeType& other, VisitorOp& op) const
{
    doVisit2Node<const LeafNode, OtherLeafNodeType, VisitorOp, ChildAllCIter,
        typename OtherLeafNodeType::ChildAllCIter>(*this, other, op);
}


template<typename T, Index Log2Dim>
template<
    typename NodeT,
    typename OtherNodeT,
    typename VisitorOp,
    typename ChildAllIterT,
    typename OtherChildAllIterT>
inline void
LeafNode<T, Log2Dim>::doVisit2Node(NodeT& self, OtherNodeT& other, VisitorOp& op)
{
    // Allow the two nodes to have different ValueTypes, but not different dimensions.
    BOOST_STATIC_ASSERT(OtherNodeT::SIZE == NodeT::SIZE);
    BOOST_STATIC_ASSERT(OtherNodeT::LEVEL == NodeT::LEVEL);

    ChildAllIterT iter = self.beginChildAll();
    OtherChildAllIterT otherIter = other.beginChildAll();

    for ( ; iter && otherIter; ++iter, ++otherIter) {
        op(iter, otherIter);
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
template<typename IterT, typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit2(IterT& otherIter, VisitorOp& op, bool otherIsLHS)
{
    doVisit2<LeafNode, VisitorOp, ChildAllIter, IterT>(
        *this, otherIter, op, otherIsLHS);
}


template<typename T, Index Log2Dim>
template<typename IterT, typename VisitorOp>
inline void
LeafNode<T, Log2Dim>::visit2(IterT& otherIter, VisitorOp& op, bool otherIsLHS) const
{
    doVisit2<const LeafNode, VisitorOp, ChildAllCIter, IterT>(
        *this, otherIter, op, otherIsLHS);
}


template<typename T, Index Log2Dim>
template<
    typename NodeT,
    typename VisitorOp,
    typename ChildAllIterT,
    typename OtherChildAllIterT>
inline void
LeafNode<T, Log2Dim>::doVisit2(NodeT& self, OtherChildAllIterT& otherIter,
    VisitorOp& op, bool otherIsLHS)
{
    if (!otherIter) return;

    if (otherIsLHS) {
        for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
            op(otherIter, iter);
        }
    } else {
        for (ChildAllIterT iter = self.beginChildAll(); iter; ++iter) {
            op(iter, otherIter);
        }
    }
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline std::ostream&
operator<<(std::ostream& os, const typename LeafNode<T, Log2Dim>::Buffer& buf)
{
    for (Index32 i = 0, N = buf.size(); i < N; ++i) os << buf.mData[i] << ", ";
    return os;
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


////////////////////////////////////////


// Specialization for LeafNodes of type bool
#include "LeafNodeBool.h"

// Specialization for LeafNodes with mask information only
#include "LeafNodeMask.h"

#endif // OPENVDB_TREE_LEAFNODE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
