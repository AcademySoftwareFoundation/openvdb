///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
/// @file UT_VDBTools.h
/// @author FX R&D Simulation team
/// @brief Less commonly-used utility classes and functions for OpenVDB plugins

#ifndef OPENVDB_HOUDINI_UT_VDBTOOLS_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_UT_VDBTOOLS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include "Utils.h" // for GridPtr

namespace openvdb_houdini {

/// @brief GridTransformOp is a functor class for use with UTvdbProcessTypedGrid()
/// that samples an input grid into an output grid of the same type through
/// a given affine transform.
/// @details The output grid's transform is unchanged by this operation.
/// @sa GridResampleOp, GridResampleToMatchOp
/// @par Example:
/// @code
/// const Grid& inGrid = ...; // generic reference to a grid of any type
///
/// // Create a new, empty output grid of the same (so far, unknown) type
/// // as the input grid and with the same transform and metadata.
/// GridPtr outGrid = inGrid.copyGridWithNewTree();
///
/// // Initialize a GridTransformer with the parameters of an affine transform.
/// openvdb::tools::GridTransformer xform(pivot, scale, rotate, ...);
///
/// // Resolve the input grid's type and resample it into the output grid,
/// // using a second-order sampling kernel.
/// GridTransformOp<openvdb::tools::QuadraticSampler> op(outGrid, xform);
/// UTvdbProcessTypedGrid(UTvdbGetGridType(inGrid), inGrid, op);
/// @endcode
template<typename Sampler>
class GridTransformOp
{
public:
    /// @param outGrid  a generic pointer to an output grid of the same type
    ///     as the grid to be resampled
    /// @param t        a @c GridTransformer that defines an affine transform
    /// @note GridTransformOp makes an internal copy of the @c GridTransformer
    /// and supplies the copy with a default Interrupter that replaces any
    /// existing interrupter.
    GridTransformOp(GridPtr& outGrid, const openvdb::tools::GridTransformer& t):
        mOutGrid(outGrid), mTransformer(t) {}

    template<typename GridType>
    void operator()(const GridType& inGrid)
    {
        typename GridType::Ptr outGrid = openvdb::gridPtrCast<GridType>(mOutGrid);

        Interrupter interrupter;
        mTransformer.setInterrupter(interrupter);

        mTransformer.transformGrid<Sampler, GridType>(inGrid, *outGrid);
    }

private:
    GridPtr mOutGrid;
    openvdb::tools::GridTransformer mTransformer;
};


////////////////////////////////////////


/// @brief GridResampleOp is a functor class for use with processTypedGrid()
/// that samples an input grid into an output grid of the same type through
/// a given transform.
/// @details The output grid's transform is unchanged by this operation.
/// @sa GridTransformOp, GridResampleToMatchOp
/// @par Example:
/// @code
/// namespace {
///     // Class that implements GridResampler's Transformer interface
///     struct MyXform
///     {
///         bool isAffine() const { ... }
///         openvdb::Vec3d transform(const openvdb::Vec3d&) const { ... }
///         openvdb::Vec3d invTransform(const openvdb::Vec3d&) const { ... }
///     };
/// }
///
/// const Grid& inGrid = ...; // generic reference to a grid of any type
///
/// // Create a new, empty output grid of the same (so far, unknown) type
/// // as the input grid and with the same transform and metadata.
/// GridPtr outGrid = inGrid.copyGridWithNewTree();
///
/// // Resolve the input grid's type and resample it into the output grid,
/// // using a trilinear sampling kernel.
/// GridResampleOp<openvdb::tools::BoxSampler, MyXform> op(outGrid, MyXform());
/// processTypedGrid(inGrid, op);
/// @endcode
template<typename Sampler, typename TransformerType>
class GridResampleOp
{
public:
    /// @param outGrid  a generic pointer to an output grid of the same type
    ///     as the grid to be resampled
    /// @param t        an object that implements <tt>GridResampler</tt>'s
    ///     Transformer interface
    /// @note GridResampleOp makes an internal copy of @a t.
    GridResampleOp(GridPtr& outGrid, const TransformerType& t):
        mOutGrid(outGrid), mTransformer(t) {}

    template<typename GridType>
    void operator()(const GridType& inGrid)
    {
        typename GridType::Ptr outGrid = openvdb::gridPtrCast<GridType>(mOutGrid);

        openvdb::tools::GridResampler resampler;

        Interrupter interrupter;
        resampler.setInterrupter(interrupter);

        resampler.transformGrid<Sampler>(mTransformer, inGrid, *outGrid);
    }

private:
    GridPtr mOutGrid;
    const TransformerType mTransformer;
};


////////////////////////////////////////


/// @brief GridResampleToMatchOp is a functor class for use with
/// UTvdbProcessTypedGrid() that samples an input grid into an output grid
/// of the same type such that, after resampling, the input and output grids
/// coincide, but the output grid's transform is unchanged.
/// @sa GridTransformOp, GridResampleOp
/// @par Example:
/// @code
/// const Grid& inGrid = ...; // generic reference to a grid of any type
///
/// // Create a new, empty output grid of the same (so far, unknown) type as
/// // the input grid and with the same metadata, but with a different transform.
/// GridPtr outGrid = inGrid.copyGridWithNewTree();
/// outGrid->setTransform(myTransform);
///
/// // Resolve the input grid's type and resample it into the output grid,
/// // using a second-order sampling kernel.
/// GridResampleToMatchOp<openvdb::tools::QuadraticSampler> op(outGrid);
/// UTvdbProcessTypedGrid(UTvdbGetGridType(inGrid), inGrid, op);
/// @endcode
template<typename Sampler>
class GridResampleToMatchOp
{
public:
    GridResampleToMatchOp(GridPtr outGrid): mOutGrid(outGrid) {}

    template<typename GridType>
    void operator()(const GridType& inGrid)
    {
        typename GridType::Ptr outGrid = openvdb::gridPtrCast<GridType>(mOutGrid);
        Interrupter interrupter;
        openvdb::tools::resampleToMatch<Sampler>(inGrid, *outGrid, interrupter);
    }

private:
    GridPtr mOutGrid;
};

} // namespace openvdb_houdini

#endif // OPENVDB_HOUDINI_UT_VDBTOOLS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
