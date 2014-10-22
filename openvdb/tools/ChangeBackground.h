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
//
/// @file ChangeBackground.h
///
/// @brief Efficient multi-threaded replacement of the background
/// values in tree.
///
/// @author Ken Museth

#ifndef OPENVDB_TOOLS_ChangeBACKGROUND_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ChangeBACKGROUND_HAS_BEEN_INCLUDED

#include <boost/utility/enable_if.hpp>
#include <openvdb/math/Math.h> // for isNegative and negative
#include <openvdb/Types.h> // for Index typedef
#include <boost/static_assert.hpp>
#include <openvdb/Types.h>
#include <openvdb/tree/NodeManager.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Replaces the background value in all the nodes of a
/// tree. The sign of the background value is perserved and only
/// inactive values equal to the old background value are replaced.
///
/// @param tree          Tree that will have its background value changed
/// @param background    The new background value
/// @param threaded      Enable or disable threading.  (Threading is enabled by default.)
/// @param grainSize     Used to control the granularity of the multithreaded. (default is 32).    
template<typename TreeT>
inline void
changeBackground(TreeT& tree,
                 const typename TreeT::ValueType& background,
                 bool threaded = true,
                 size_t grainSize = 32);

/// @brief Replaces the background values in all the nodes of a
/// tree that is assume to contain a symetric narrow-band level set, i.e. the
/// value type is floating-point, and all inactive values will be +| @a
/// halfWidth | if outside and -@| a halfWidth | if inside, where @a
/// halfWidth is half the width of the symetric narrow-band.
///
/// @note This method is faster then changeBackground since it does not
/// perform tests to see if inactive values are equal to the old
/// background value.    
///
/// @param tree          Tree that will have its background value changed
/// @param halfWidth     Half of the width of the symmetric narrow band
/// @param threaded      Enable or disable threading.  (Threading is enabled by default.)
/// @param grainSize     Used to control the granularity of the multithreaded. (default is 32).    
///
/// @throw ValueError if @a halfWidth is negative (as defined by math::isNegative).    
template<typename TreeT>
inline void
changeLevelSetBackground(TreeT& tree,
                         const typename TreeT::ValueType& halfWidth,
                         bool threaded = true,
                         size_t grainSize = 32);

/// @brief Replaces the background values in all the nodes of a
/// tree that is assume to contain a (possibly asymetric) narrow-band
/// level set, i.e. the value type is floating-point, and all inactive
/// values will be +| @a outsideWidth | if outside and -| @a
/// insideWidth | if inside, where @a outsideWidth is outside
/// the width of the narrow-band and @a insideWidth is the inside
/// width ot the narrow band.
///
/// @note This method is faster then changeBackground since it does not
/// perform tests to see if inactive values are equal to the old
/// background value.  
///
/// @param tree          Tree that will have its background value changed
/// @param outsideWidth  The width of the outside of the narrow band
/// @param insideWidth   The width of the inside of the narrow band
/// @param threaded      Enable or disable threading.  (Threading is enabled by default.)
/// @param grainSize     Used to control the granularity of the multithreaded. (default is 32).    
/// 
/// @throw ValueError if @a outsideWidth is negative or @a insideWidth is
/// not negative (as defined by math::isNegative).    
template<typename TreeT>
inline void
changeLevelSetBackground(TreeT& tree,
                         const typename TreeT::ValueType& outsideWidth,
                         const typename TreeT::ValueType& insideWidth,
                         bool threaded = true,
                         size_t grainSize = 32);

//////////////////////////////////////////////////////    


// Replaces the background value in a Tree of any type.    
template<typename TreeT>
class ChangeBackgroundOp
{
public:
    typedef typename TreeT::ValueType    ValueT;
    typedef typename TreeT::RootNodeType RootT;
    typedef typename TreeT::LeafNodeType LeafT;

    
    ChangeBackgroundOp(const TreeT& tree, const ValueT& newValue)
        : mOldValue(tree.background())
        , mNewValue(newValue)
    {
    }
    void operator()(RootT& root) const
    {
        for (typename RootT::ValueOffIter it = root.beginValueOff(); it; ++it) this->set(it);
        root.setBackground(mNewValue, false);
    }
    void operator()(LeafT& node) const
    {
        for (typename LeafT::ValueOffIter it = node.beginValueOff(); it; ++it) this->set(it);
    }
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        typename NodeT::NodeMaskType mask = node.getValueOffMask();
        for (typename NodeT::ValueOnIter it(mask.beginOn(), &node); it; ++it) this->set(it);
    }
private:

    template<typename IterT>
    inline void set(IterT& iter) const
    {
        if (math::isApproxEqual(*iter, mOldValue)) {
            iter.setValue(mNewValue);
        } else if (math::isApproxEqual(*iter, math::negative(mOldValue))) {
            iter.setValue(math::negative(mNewValue));
        }
    }
    const ValueT mOldValue, mNewValue;
};// ChangeBackgroundOp

// Replaces the background value in a Tree assumed to represent a  
// level set. It is generally faster then ChangeBackgroundOp.
// Note that is follows the sign-convension that outside is positive
// and inside is negative!    
template<typename TreeT>
class ChangeLevelSetBackgroundOp
{
public:
    typedef typename TreeT::ValueType    ValueT;
    typedef typename TreeT::RootNodeType RootT;
    typedef typename TreeT::LeafNodeType LeafT;
    
    /// @brief Constructor for asymetric narrow-bands
    ChangeLevelSetBackgroundOp(const ValueT& outside, const ValueT& inside)
        : mOutside(outside)
        , mInside(inside)
    {
        if (math::isNegative(mOutside)) {
            OPENVDB_THROW(ValueError,
                          "ChangeLevelSetBackgroundOp: the outside value cannot be negative!");
        }
        if (!math::isNegative(mInside)) {
            OPENVDB_THROW(ValueError,
                          "ChangeLevelSetBackgroundOp: the inside value must be negative!");
        }
    }
    void operator()(RootT& root) const
    {
        for (typename RootT::ValueOffIter it = root.beginValueOff(); it; ++it) this->set(it);
        root.setBackground(mOutside, false);
    }
    void operator()(LeafT& node) const
    {
        for(typename LeafT::ValueOffIter it = node.beginValueOff(); it; ++it) this->set(it);
    }
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        typedef typename NodeT::ValueOffIter IterT;
        for (IterT it(node.getChildMask().beginOff(), &node); it; ++it) this->set(it);
    }
private:

    template<typename IterT>
    inline void set(IterT& iter) const
    {
        //this is safe since we know ValueType is_floating_point 
        ValueT& v = const_cast<ValueT&>(*iter);
        v = v < 0 ? mInside : mOutside;
    }
    const ValueT mOutside, mInside;
};// ChangeLevelSetBackgroundOp

    
template<typename TreeT>
void changeBackground(TreeT& tree,
                      const typename TreeT::ValueType& background,
                      bool threaded,
                      size_t grainSize)
{
    tree::NodeManager<TreeT> linearTree(tree);
    ChangeBackgroundOp<TreeT> op(tree, background);
    linearTree.processTopDown(op, threaded, grainSize);
}

template <typename TreeT>    
inline void
changeLevelSetBackground(TreeT& tree,
                         const typename TreeT::ValueType& outsideValue,
                         const typename TreeT::ValueType& insideValue,
                         bool threaded,
                         size_t grainSize)
{
    tree::NodeManager<TreeT> linearTree(tree);
    ChangeLevelSetBackgroundOp<TreeT> op(outsideValue, insideValue);
    linearTree.processTopDown(op, threaded, grainSize);
}

// If the narrow-band is symmetric only one background value is required    
template <typename TreeT>    
inline void
changeLevelSetBackground(TreeT& tree,
                         const typename TreeT::ValueType& background,
                         bool threaded,
                         size_t grainSize)
{
    changeLevelSetBackground(tree, background, math::negative(background), threaded, grainSize);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_CHANGEBACKGROUND_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
