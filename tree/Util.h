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

#include <openvdb/math/Math.h>//for zeroVal

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief Helper class for the tree nodes to replace constant tree
/// branches (to within the provided tolerance) with a more memory
/// efficient tile.   
template<typename ValueType>
struct TolerancePrune {
    TolerancePrune(const ValueType &tol) : tolerance(tol) {}
    template <typename ChildType>
    bool operator()(ChildType &child) {
        child.pruneOp(*this);
        return child.isConstant(value, state, tolerance);
    }
    bool      state;
    ValueType value;
    const ValueType tolerance;
};
    
/// @brief Helper class for the tree nodes to replace inactive tree
/// branches with a more memory efficient inactive tiles with the
/// provided value. Specialized but faster then the tolerance based
/// pruning defined above.
template<typename ValueType>
struct InactivePrune {
    InactivePrune(const ValueType &val) : value(val) {}
    template <typename ChildType>
    bool operator()(ChildType &child) const {
        child.pruneOp(*this);
        return child.isInactive();
    }
    static const bool state = false;
    const ValueType   value;
};    
    
/// @brief Prune any descendants whose values are all inactive and replace them
/// with inactive tiles having values with a magnitude equal to the background 
/// value and a sign equal to the first value encountered in the (inactive) child.
///
/// @note This method is faster then tolerance based prune and
/// useful for narrow-band level set applications where inactive
/// values are limited to either an inside or outside value. Also note
/// that this methods
template<typename ValueType>
struct LevelSetPrune {
    LevelSetPrune(const ValueType &background) : outside(background) {}
    template <typename ChildType>
    bool operator()(ChildType &child) {
        child.pruneOp(*this);
        if (!child.isInactive()) return false;
        value = child.getFirstValue() < zeroVal<ValueType>() ? -outside : outside;
        return true;
    }
    static const bool state = false;
    const ValueType   outside;
    ValueType         value;
};
    
} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
