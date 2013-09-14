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
/// @file tree/Util.h

#ifndef OPENVDB_TREE_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/math/Math.h> // for isNegative and negative
#include <openvdb/Types.h> // for Index typedef

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief Helper class for use with Tree::pruneOp() to replace constant branches
/// (to within the provided tolerance) with more memory-efficient tiles
template<typename ValueType, Index TerminationLevel = 0>
struct TolerancePrune
{
    TolerancePrune(const ValueType& tol): tolerance(tol) {}

    template<typename ChildType>
    bool operator()(ChildType& child)
    {
        return (ChildType::LEVEL < TerminationLevel) ? false : this->isConstant(child);
    }

    template<typename ChildType>
    bool isConstant(ChildType& child)
    {
        child.pruneOp(*this);
        return child.isConstant(value, state, tolerance);
    }

    bool            state;
    ValueType       value;
    const ValueType tolerance;
};


/// @brief Helper class for use with Tree::pruneOp() to replace inactive branches
/// with more memory-efficient inactive tiles with the provided value
/// @details This is more specialized but faster than a TolerancePrune.
template<typename ValueType>
struct InactivePrune
{
    InactivePrune(const ValueType& val): value(val) {}

    template <typename ChildType>
    bool operator()(ChildType& child) const
    {
        child.pruneOp(*this);
        return child.isInactive();
    }

    static const bool state = false;
    const ValueType   value;
};


/// @brief Helper class for use with Tree::pruneOp() to prune any branches
/// whose values are all inactive and replace each with an inactive tile
/// whose value is equal in magnitude to the background value and whose sign
/// is equal to that of the first value encountered in the (inactive) child
///
/// @details This operation is faster than a TolerancePrune and useful for
/// narrow-band level set applications where inactive values are limited
/// to either the inside or the outside value.
template<typename ValueType>
struct LevelSetPrune
{
    LevelSetPrune(const ValueType& background): outside(background) {}

    template <typename ChildType>
    bool operator()(ChildType& child)
    {
        child.pruneOp(*this);
        if (!child.isInactive()) return false;
        value = math::isNegative(child.getFirstValue()) ? math::negative(outside) : outside;
        return true;
    }

    static const bool state = false;
    const ValueType   outside;
    ValueType         value;
};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
