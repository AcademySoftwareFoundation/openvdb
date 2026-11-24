// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth, Rohan Sawhney
///
/// @file ShrinkWrap.h
///
/// @brief Generates a LOD family of watertight shrink wrap level set surfaces
///        (or meshes) from a soup of polygons.
///
/// @details Details of this algorithm are given in an upcoming publication.

#ifndef VDB_TOOL_SHRINKWRAP_HAS_BEEN_INCLUDED
#define VDB_TOOL_SHRINKWRAP_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/Assert.h>

#include <openvdb/tools/Composite.h> // for csgUnion
#include <openvdb/tools/GridTransformer.h> // for GridTransformer
#include <openvdb/tools/MeshToVolume.h>// for meshToUnsignedDistanceField
#include <openvdb/tools/VolumeToMesh.h>// for volumeToMesh
#include <openvdb/tools/LevelSetFilter.h>// for Filter
#include <openvdb/tools/LevelSetTracker.h>// for LevelSetTracker
#include <openvdb/tools/LevelSetUtil.h>// for distanceFieldToSDF and active-voxel segmentation

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

/// @brief Simple structure for a polygon soup
/// @details bbox is allowed to be invalid, in which case it will be derived from vtx
struct PolySoup {
    std::vector<Vec3s> vtx;
    std::vector<Vec3I> tri;
    std::vector<Vec4I> quad;
    math::BBox<Vec3f>  bbox;
};

/// @brief Class that implements the actual shrink wrap algorithm
/// @tparam GridType Template parameter of the desired shrink wrap grids
template <typename GridType = FloatGrid>
class ShrinkWrap;

/// @brief Convert a polygon soup to its finest shrink wrapped level set.
///
/// @return A shared pointer to grid of type @c GridType containing a narrow-band level set
///         that shrink wraps the input polygons.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @param poly                  Struct with polygon soup, that will be destroyed (moved)
/// @param minVoxelSize          Finest grid spacing, in world units.
/// @param closingThreshold      Target gap closing size in world units (defaults to zero).
/// @param maxErodeIterations    Maximum erosion steps per voxel size (defaults to 128).
///                              Ignored when useThresholdSeed is true.
/// @param useConservativeOffset Use an offset radius of sqrt(3) voxel widths for
///                              conservative containment; otherwise use one voxel width.
/// @param useThresholdSeed      Use the closing threshold to choose the starting offset,
///                              then perform one erosion-and-union step at each finer voxel size.
/// @param halfWidth             Half the width of the narrow band, in voxel units.
/// @param progress              Optional pointer to progress bar.
template<typename GridType, class ProgressT = void>
typename GridType::Ptr
shrinkWrap(PolySoup &&poly,
           float minVoxelSize,
           float closingThreshold = 0.0f,
           int maxErodeIterations = 128,
           bool useConservativeOffset = false,
           bool useThresholdSeed = true,
           float halfWidth = float(LEVEL_SET_HALF_WIDTH),
           ProgressT *progress = nullptr);

/// @brief Convert a soup of polygons to a LOD sequence of shrink wrapped level set volumes.
///
/// @return a vector of grids of type @c GridType containing a narrow-band level set
///         at various resolution shrink wrapping the input polygon mesh. The first
///         element in this vector has the highest resolution.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @note   This function takes ownership of the contents of vtx, tri, and quad.
///
/// @param vtx                   Vector of world-space vertex positions.
/// @param tri                   Vector of triangle indices.
/// @param quad                  Vector of quad indices.
/// @param bbox                  Input bounds; if invalid, derive them from the vertices.
/// @param minVoxelSize          Finest grid spacing, in world units.
/// @param closingThreshold      Target gap closing size in world units (defaults to zero).
/// @param maxErodeIterations    Maximum erosion steps per voxel size (defaults to 128).
///                              Ignored when useThresholdSeed is true.
/// @param useConservativeOffset Use an offset radius of sqrt(3) voxel widths for
///                              conservative containment; otherwise use one voxel width.
/// @param useThresholdSeed      Use the closing threshold to choose the starting offset,
///                              then perform one erosion-and-union step at each finer voxel size.
/// @param halfWidth             Half the width of the narrow band, in voxel units.
/// @param progress              Optional pointer to a progress bar.
template<typename GridType, class ProgressT = void>
std::vector<typename GridType::Ptr>
shrinkWrap(std::vector<Vec3s>& vtx,
           std::vector<Vec3I>& tri,
           std::vector<Vec4I>& quad,
           const math::BBox<Vec3f> &bbox,
           float minVoxelSize,
           float closingThreshold = 0.0f,
           int maxErodeIterations = 128,
           bool useConservativeOffset = false,
           bool useThresholdSeed = true,
           float halfWidth = float(LEVEL_SET_HALF_WIDTH),
           ProgressT *progress = nullptr);

/////////////////////////////////////////////////////////////////////////////////////

/// @brief This class implements our shrink wrap algorithm. Normally the free-standing
///        function called above should be used instead of this class.
/// @tparam GridType Grid type of the generated level set surfaces (defaults to FloatGrid)
template<typename GridType>
class ShrinkWrap
{
public:
    /// @brief Constructor with a specified voxel size.
    /// @param poly            Polygon soup that will be moved to this instance.
    /// @param minVoxelSize    Finest grid spacing, in world units.
    /// @param width           Half-width of the output narrow-band level set, in voxel units.
    ShrinkWrap(PolySoup &&poly, float minVoxelSize,
               float width = float(LEVEL_SET_HALF_WIDTH));

    /// @brief Start from an offset selected by the closing threshold, then shrink wrap
    ///        to the finest voxel size with one erosion-union step at each finer grid.
    /// @param closingThreshold      Target gap closing size in world units.
    /// @param useConservativeOffset Use radius sqrt(3)*dx instead of dx for each offset.
    /// @param progress              Optional pointer to a progress bar.
    template<class ProgressT = void>
    void process(float closingThreshold = 0.0f,
                 bool useConservativeOffset = false,
                 ProgressT *progress = nullptr);

    /// @brief Start from the coarsest available offset, then shrink wrap to the finest
    ///        voxel size. At each finer grid, repeat erosion-union steps until changes
    ///        become small or that grid's iteration limit is reached.
    /// @tparam ProgressT Template parameter of the optional progress bar.
    /// @param closingThreshold      Target gap closing size in world units.
    /// @param maxErodeIterations    Maximum erosion steps per voxel size.
    /// @param useConservativeOffset Use an offset radius of sqrt(3) voxel widths for
    ///                              conservative containment; otherwise use one voxel width.
    /// @param progress              Optional pointer to a progress bar.
    template<class ProgressT = void>
    void process(float closingThreshold,
                 int maxErodeIterations,
                 bool useConservativeOffset = false,
                 ProgressT *progress = nullptr);

    /// @brief Number of shrink wrap grids generated, i.e. depth of the LOD hierarchy.
    size_t gridCount() const {return mGrids.size();}

    /// @brief   Returns a shared pointer to a particular shrink wrap grid
    /// @param n Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///          n = gridCount-1 is the coarsest voxel sampling.
    typename GridType::Ptr grid(int n = 0) const {return mGrids[n];}

    /// @brief Vector with shared pointers to all the shrink wrap grids
    /// @note  The grids are arranged fine to coarse, grids()[0] is the finest.
    std::vector<typename GridType::Ptr> grids() const {return mGrids;}

    /// @brief Generate an adaptive polygon mesh from a particular shrink wrap SDF
    /// @param n          Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///                   n = gridCount-1 is the coarsest voxel sampling.
    /// @param adaptivity Optional adaptivity parameter used for meshing. A value of zero
    ///                   means adaptivity is disabled, i.e. uniform quads are produced.
    /// @param isoValue   Iso-value used for the mesh generation.
    /// @return Reference to the internal PolySoup populated with the generated mesh.
    const PolySoup& mesh(int n = 0, float adaptivity = 0.005f, float isoValue = 0.0f)
    {
        tools::volumeToMesh(*mGrids[n], mPoly.vtx, mPoly.tri, mPoly.quad, isoValue, adaptivity);
        return mPoly;
    }

    /// @brief Static method that computes the bounding box of a list of vertex coordinates.
    /// @param vtx Vector of vertex coordinates.
    static math::BBox<Vec3f> getBBox(const std::vector<Vec3s> &vtx);

    /// @brief Build an enclosing offset using unsigned distances and exterior flood filling.
    /// @param dx                    Voxel size (world units) of the output level set.
    /// @param useConservativeOffset Use radius sqrt(3)*dx for conservative containment.
    ///                              Otherwise, use radius dx.
    /// @return Shared pointer to the newly created level set grid.
    auto offset(float dx, bool useConservativeOffset = false);

    /// @brief Returns the finest/smallest voxel size, in world units, i.e. the
    ///        voxel size of the final (highest-resolution) level set surface.
    float minVoxelSize() const {return mMinVoxelSize;}

    /// @brief Half the input's longest bounding-box extent, in world units.
    float maxVoxelSize() const {return mMaxVoxelSize;}

    /// @brief Returns the half-width of the output narrow-band level set, in voxel units.
    float halfWidth() const {return mHalfWidth;}

private:
    PolySoup mPoly;
    float mMinVoxelSize, mMaxVoxelSize, mHalfWidth;
    std::vector<typename GridType::Ptr> mGrids;// fine(0) -> coarse grids(size-1)
    bool mIsGridSDF;

    /// @brief Offset radius for the chosen voxel size and containment setting.
    static float offsetRadius(float dx, bool useConservativeOffset) {
        return useConservativeOffset ? float(std::sqrt(3.0))*dx : dx;
    }

    /// @brief Private method that resamples grid onto fineOffset's transform.
    auto upsample(const GridType &grid, const GridType &fineOffset, double upsampleFactor);

    /// @brief Erode by the given world-space distance, then union with the finer offset.
    auto constrainedErode(GridType &grid, const GridType &fineOffset, double erosionDistance);
};// ShrinkWrap<GridType>

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
ShrinkWrap<GridType>::ShrinkWrap(PolySoup &&poly, float minVoxelSize, float width):
    mPoly(std::move(poly)),
    mMinVoxelSize(minVoxelSize),
    mHalfWidth(width)
{
    if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
        OPENVDB_THROW(TypeError, "ShrinkWrap: supported only for scalar floating-point grids");
    }
    if (!math::isFinite(mHalfWidth) || !(mHalfWidth > 0.0f)) {
        OPENVDB_THROW(ValueError, "ShrinkWrap: halfWidth must be finite and positive");
    }
    if (!math::isFinite(mMinVoxelSize) || !(mMinVoxelSize > 0.0f)) {
        OPENVDB_THROW(ValueError, "ShrinkWrap: minVoxelSize must be finite and positive");
    }
    if (!mPoly.bbox) mPoly.bbox = ShrinkWrap::getBBox(mPoly.vtx);
    // Require the input's bounding box to have nonzero size in at least one
    // direction. Empty input and a single point fail this check; flat input passes.
    const float maxLength = mPoly.bbox.extents()[mPoly.bbox.maxExtent()];
    if (!(maxLength > 0.0f)) {
        OPENVDB_THROW(ValueError, "ShrinkWrap: bounding box has non-positive extent (no input geometry?)");
    }
    mMaxVoxelSize = maxLength / 2.0f;
    if (!math::isFinite(mMaxVoxelSize) || !(mMaxVoxelSize > 0.0f)) {
        OPENVDB_THROW(ArithmeticError, "ShrinkWrap: computed voxel size is not finite and positive");
    }
}// vdb_tool::ShrinkWrap::ShrinkWrap()

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
template<class ProgressT>
void ShrinkWrap<GridType>::process(float closingThreshold,
                                   bool useConservativeOffset,
                                   ProgressT *progress)
{
    if (!math::isFinite(closingThreshold) || closingThreshold < 0.0f) {
        OPENVDB_THROW(ValueError, "ShrinkWrap::process: closingThreshold must be finite and nonnegative");
    }

    auto myProgress = [&](const std::string &s){
        if constexpr(!std::is_same<ProgressT,void>::value) {
            if (progress) (*progress)(s);
        }
    };

    // Choose the voxel sizes for the offset hierarchy. Refinement starts
    // from the coarsest offset and works down to mMinVoxelSize. Use half
    // the closing threshold for the starting offset radius, reducing its
    // voxel size when conservative offsets are enabled. Finer offsets use
    // power-of-two multiples of mMinVoxelSize.
    const float scale = useConservativeOffset ? float(std::sqrt(3.0)) : 1.0f;
    const float seedDx = std::max(mMinVoxelSize,
        std::min(0.5f*closingThreshold/scale, mMaxVoxelSize));
    std::vector<float> voxelSizes;
    for (float dx = mMinVoxelSize; dx < seedDx; dx *= 2.0f) {
        voxelSizes.push_back(dx);
    }
    voxelSizes.push_back(seedDx);

    // Build the independent offsets for the selected voxel sizes.
    for (float dx: voxelSizes) {
        myProgress("Offset: dx=" + std::to_string(dx) + ", range: " +
            std::to_string(mMinVoxelSize) + " -> " + std::to_string(seedDx));
        mGrids.push_back(this->offset(dx, useConservativeOffset));
    }

    mIsGridSDF = true;
    if (mGrids.size() == 1) return; // The finest offset is already the requested output.

    // Use the coarsest offset to seed coarse-to-fine shrink wrapping.
    auto grid = mGrids.back();
    mGrids.pop_back();

    // Refine from coarse to fine with one erosion-union step per voxel size.
    // Upsample by the ratio of voxel sizes and erode by the difference in offset radii.
    // Each union preserves the volume enclosed by the finer offset.
    for (auto iter = mGrids.rbegin(); iter != mGrids.rend(); ++iter) {
        const GridType &fineOffset = **iter;
        const double coarseDx = grid->voxelSize()[0];
        const double fineDx = fineOffset.voxelSize()[0];
        const double upsampleFactor = coarseDx/fineDx;
        const double erosionDistance = scale*(coarseDx - fineDx);

        grid = this->upsample(*grid, fineOffset, upsampleFactor);
        myProgress("Shrink wrap iteration=1, voxel size=" + std::to_string(fineDx) +
                   ", erosion distance=" + std::to_string(erosionDistance));
        grid = this->constrainedErode(*grid, fineOffset, erosionDistance);
        *iter = grid;
    }
}// vdb_tool::ShrinkWrap::process()

/////////////////////////////////////////////////////////////////////////////////////

namespace shrinkwrap_detail {

/// @brief Finds the largest value difference while visiting one tree's nodes.
/// @details A VDB tree stores individual voxels in leaf nodes. A block whose
///          voxels all have the same value can be stored as a single tile.
///          Internal nodes group smaller blocks as tiles or child nodes.
///          NodeManager calls this object for each node. Parallel workers keep
///          separate maximum differences, which join() combines.
template<typename TreeT>
struct FieldResidualOp
{
    const TreeT& other;    // Tree to compare against.
    bool skipShared;      // Skip previously compared node pairs on the second pass.
    double maximum = 0.0; // Largest value difference seen by this worker.

    FieldResidualOp(const TreeT& tree, bool skip):
        other(tree), skipShared(skip) {}

    // Each new parallel worker starts with its own maximum of zero.
    FieldResidualOp(const FieldResidualOp& op, tbb::split):
        other(op.other), skipShared(op.skipShared) {}

    // Keep the largest difference reported by any worker.
    void join(const FieldResidualOp& op)
    {
        maximum = std::max(maximum, op.maximum);
    }

    void add(double a, double b)
    {
        maximum = std::max(maximum, std::abs(a - b));
    }

    // The root can store large blocks as tiles. Compare one sample per tile
    // here; if the other tree splits that block into smaller pieces, the pass
    // over that tree checks the values in those pieces.
    void operator()(const typename TreeT::RootNodeType& node)
    {
        for (auto it = node.cbeginValueAll(); it; ++it) {
            add(double(*it), double(other.getValue(it.getCoord())));
        }
    }

    // Compare one leaf or internal node. Smaller blocks are visited separately.
    template<typename NodeT>
    void operator()(const NodeT& node)
    {
        // Find a node covering exactly the same block in the other tree.
        const auto* rhs = other.template probeConstNode<NodeT>(node.origin());

        // On the second pass, matching nodes have already been compared.
        // Returning skips only this node: NodeManager still visits the smaller
        // blocks below it, including those stored only in this tree.
        if (rhs && skipShared) return;

        // No matching node means the other tree uses one constant for this
        // entire block: either a tile value or the default background value.
        const double tile = rhs ? 0.0 : double(other.getValue(node.origin()));

        if constexpr (NodeT::LEVEL == 0) {
            // Leaf: compare each voxel with the other leaf or the constant above.
            // Include inactive voxels, which still store field values.
            const auto* a = node.buffer().data();
            const auto* b = rhs ? rhs->buffer().data() : nullptr;

            for (openvdb::Index i = 0; i < NodeT::SIZE; ++i) {
                add(double(a[i]), b ? double(b[i]) : tile);
            }

        } else {
            // Internal node: entries are either tiles or children containing
            // smaller blocks. This iterator visits only the tiles.
            for (auto it = node.cbeginValueAll(); it; ++it) {
                // If the other entry is a child, let the pass over that tree
                // compare all of its stored values against this tile.
                if (rhs && rhs->isChildMaskOn(it.pos())) continue;

                add(double(*it), rhs ? double(rhs->getValueUnsafe(it.pos())) : tile);
            }
        }
    }
};

} // namespace shrinkwrap_detail

/// @brief Largest absolute difference between corresponding samples, in voxel widths.
/// @details Used to check whether consecutive erosion steps have stopped changing
///          the field. Both active and inactive samples contribute to the result.
/// @pre Both grids have finite scalar values and the same voxel spacing, origin,
///      and orientation. Voxel spacing must be uniform in all directions.
template<typename GridType>
double maxFieldResidual(const GridType& a, const GridType& b)
{
    using TreeT = typename GridType::TreeType;

    const auto scan = [](const TreeT& src, const TreeT& dst, bool skipShared) {
        // Visit nodes in parallel and combine their maximum differences.
        openvdb::tree::NodeManager<const TreeT> nodes(src);
        shrinkwrap_detail::FieldResidualOp<TreeT> op(dst, skipShared);
        nodes.reduceBottomUp(op, /*threaded=*/true);
        return op.maximum;
    };

    // Account for the default value in regions with no stored nodes.
    const double background = std::abs(double(a.background()) - double(b.background()));

    // One tree may store a block as a single value while the other stores
    // individual voxels. Check both trees, skipping repeated node comparisons.
    const double ab = scan(a.tree(), b.tree(), false);
    const double ba = scan(b.tree(), a.tree(), true);

    // Divide by voxel width to express the change in voxel units.
    return std::max({background, ab, ba}) / a.voxelSize()[0];
}

template<typename GridType>
template<class ProgressT>
void ShrinkWrap<GridType>::process(float closingThreshold,
                                   int maxErodeIterations,
                                   bool useConservativeOffset,
                                   ProgressT *progress)
{
    if (!math::isFinite(closingThreshold) || closingThreshold < 0.0f) {
        OPENVDB_THROW(ValueError, "ShrinkWrap::process: closingThreshold must be finite and nonnegative");
    }
    if (maxErodeIterations < 1) {
        OPENVDB_THROW(ValueError, "ShrinkWrap::process: maxErodeIterations must be at least one");
    }

    auto myProgress = [&](const std::string &s){
        if constexpr(!std::is_same<ProgressT,void>::value) {
            if (progress) (*progress)(s);
        }
    };

    // Fine to coarse offset generation
    const float maxDx = std::max(mMinVoxelSize, mMaxVoxelSize);
    for (float dx = mMinVoxelSize; dx <= maxDx; dx *= 2.0f) {
        myProgress("Offset: dx=" + std::to_string(dx) + ", range: " +
            std::to_string(mMinVoxelSize)+" -> " + std::to_string(maxDx));
        mGrids.push_back(this->offset(dx, useConservativeOffset));
    }

    mIsGridSDF = true;
    if (mGrids.size() == 1) return; // The finest offset is already the requested output.

    // Use the coarsest offset to seed coarse-to-fine shrink wrapping.
    auto grid = mGrids.back();
    mGrids.pop_back();

    // Maximum absolute field change, measured in voxel widths.
    constexpr double RESIDUAL_TOLERANCE = 1e-3;
    const float scale = useConservativeOffset ? float(std::sqrt(3.0)) : 1.0f;

    // Allow only one erosion step when the offset radius is at or below
    // one quarter of the closing threshold.
    const auto erosionLimit = [closingThreshold, maxErodeIterations, scale](float dx) {
        return dx > 0.25f*closingThreshold/scale ? maxErodeIterations : 1; // TODO: subject to change
    };

    for (auto iter = mGrids.rbegin(); iter != mGrids.rend(); ++iter) {
        const GridType &fineOffset = **iter;

        // Upsample the grid to the next finer voxel size.
        grid = this->upsample(*grid, fineOffset, 2.0);// grid(dx) -> grid(dx/2)

        // Iteratively shrinkwrap until the erosion budget is exhausted or
        // the maximum field residual falls below the tolerance.
        const float dx = float(grid->voxelSize()[0]);
        const int maxIters = erosionLimit(dx);

        // Each iteration erodes by one voxel width, scaled by sqrt(3) for
        // conservative offsets, then unions the result with the finer offset.
        for (int iteration = 0; iteration < maxIters; ++iteration) {
            myProgress("Shrink wrap iteration=" + std::to_string(iteration + 1) +
                       ", limit=" + std::to_string(maxIters) +
                       ", voxel size=" + std::to_string(dx));

            // Compare consecutive post-union fields. Erosion modifies its input,
            // so preserve a deep copy before the next erosion step. Skip the
            // first step and any step that will exhaust the erosion budget.
            typename GridType::Ptr previousGrid;
            if (iteration > 0 && iteration + 1 < maxIters) previousGrid = grid->deepCopy();

            grid = this->constrainedErode(*grid, fineOffset, scale*dx);

            if (iteration + 1 >= maxIters) break;
            if (previousGrid && maxFieldResidual(*previousGrid, *grid) <= RESIDUAL_TOLERANCE) break;
        }
        *iter = grid;
    }// loop from coarse to fine voxel sizes
}// vdb_tool::ShrinkWrap::process()

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
math::BBox<Vec3f> ShrinkWrap<GridType>::getBBox(const std::vector<Vec3s> &vtx)
{
    using RangeT = tbb::blocked_range<std::vector<Vec3s>::const_iterator>;
    RangeT range(vtx.begin(), vtx.end(), 1024);
    struct BBoxOp {
        math::BBox<Vec3f> bbox;
        BBoxOp() : bbox() {}
        BBoxOp(BBoxOp& s, tbb::split) : bbox(s.bbox) {}
        void operator()(const RangeT& r) {for (auto p=r.begin(); p!=r.end(); ++p) bbox.expand(*p);}
        void join(BBoxOp& rhs) {bbox.expand(rhs.bbox);}
    } tmp;
#if 0
    tmp(range);// serial
#else
    tbb::parallel_reduce(range, tmp);// parallel
#endif
    return tmp.bbox;
}// vdb_tool::ShrinkWrap::getBBox

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto ShrinkWrap<GridType>::offset(float dx, bool useConservativeOffset)
{
    // Build an enclosing offset using distances to the input geometry, without
    // relying on normals or surface orientation. The offset radius equals the
    // voxel size dx, or sqrt(3)*dx with useConservativeOffset to provide the conservative
    // containment margin.
    //
    // Sample distances on the grid and process disconnected groups of samples
    // separately. Voxels within radius of the input form a barrier. Identify
    // exterior space by following connected voxels outside the barrier, then
    // mark the barrier and any regions cut off from the exterior as inside.
    // This fills enclosed spaces, including those sealed by the offset.
    //
    // Remove volumes already enclosed by others and combine the rest. Finally,
    // reconstruct signed distances near the boundary to produce the output level set.

    using ValueT = typename GridType::ValueType;
    using TreeT = typename GridType::TreeType;
    using BoolTreeT = typename TreeT::template ValueConverter<bool>::Type;

    if (!math::isFinite(dx) || !(dx > 0.0f)) {
        OPENVDB_THROW(ValueError, "ShrinkWrap::offset: voxel size must be finite and positive");
    }
    const ValueT radius = ValueT(this->offsetRadius(dx, useConservativeOffset));
    auto xform = math::Transform::createLinearTransform(dx);

    // Filled regions must remain strictly negative, including at the offset boundary.
    const ValueT interiorFillMagnitude = std::max(radius - ValueT(dx), ValueT(1e-3 * dx));

    // Convert the requested output half-band width to world units.
    const ValueT bandWidth = ValueT(mHalfWidth * dx);

    // Create an empty output grid. Its positive background represents exterior
    // space beyond the distance band.
    auto result = GridType::create(bandWidth);
    result->setTransform(xform->copy());
    result->setGridClass(GRID_LEVEL_SET);

    // Sample through the offset radius and leave an exterior collar for flood
    // filling and reconstruction. meshToUnsignedDistanceField takes voxel units.
    const float udfWidth = std::max(mHalfWidth + float(radius) / dx, 3.0f);
    auto udf = tools::meshToUnsignedDistanceField<GridType>(
                *xform, mPoly.vtx, mPoly.tri, mPoly.quad, udfWidth);
    if (udf->empty()) return result;

    // Split the active UDF voxels into components connected through shared
    // faces (six neighbors). Each component retains its original distances.
    std::vector<typename GridType::Ptr> components;
    tools::segmentActiveVoxels(*udf, components);
    udf.reset(); // The components own copies of their samples; release the original grid.

    std::vector<Coord> exteriorSeeds;
    exteriorSeeds.reserve(components.size());
    for (auto& component : components) {
        // Collect voxels on or outside the requested offset: UDF >= radius.
        // Mask membership uses active flags, not the stored Boolean values.
        BoolTreeT nonBarrierMask(false);
        Coord minNonBarrierCoord = Coord::max();
        {
            auto nonBarrierAcc = tree::ValueAccessor<BoolTreeT>(nonBarrierMask);
            for (auto it = component->beginValueOn(); it; ++it) {
                const ValueT value = *it - radius;
                it.setValue(value);
                if (value >= ValueT(0)) {
                    const Coord ijk = it.getCoord();
                    nonBarrierAcc.setValueOn(ijk);
                    if (ijk < minNonBarrierCoord) minNonBarrierCoord = ijk;
                }
            }
        }

        // Retain only the region containing the smallest non-barrier coordinate.
        // This assumes each UDF component has one connected exterior region.
        const auto rejectSegment = [&minNonBarrierCoord](const auto& segment) {
            return !segment.contains(minNonBarrierCoord);
        };
        std::vector<typename BoolTreeT::Ptr> regionMasks;
        tools::extractActiveVoxelSegmentMasks(nonBarrierMask, regionMasks, rejectSegment);
        if (regionMasks.empty()) {
            OPENVDB_THROW(RuntimeError, "ShrinkWrap::offset: missing UDF exterior collar");
        }
        nonBarrierMask.clear();

        // Disjoint segments ensure that only one can contain the selected coordinate.
        auto exteriorMask = regionMasks.front();
        exteriorSeeds.push_back(minNonBarrierCoord);
        regionMasks.clear();

        // Use UDF - radius for exterior and barrier voxels so interpolated
        // zero crossings occur where the sampled UDF reaches the target offset.
        {
            auto exteriorAcc = tree::ValueAccessor<const BoolTreeT>(*exteriorMask);
            for (auto it = component->beginValueOn(); it; ++it) {
                // The stored value already contains UDF - radius.
                const ValueT shifted = *it;
                if (shifted < ValueT(0) || exteriorAcc.isValueOn(it.getCoord())) {
                    // Exterior values are nonnegative; barrier values are negative.
                    it.setValue(shifted);

                } else {
                    // Fill enclosed regions with strictly negative values.
                    it.setValue(-std::max(shifted, interiorFillMagnitude));
                }
            }
        }
        exteriorMask.reset();

        // Set the background magnitude, then propagate the active signs into
        // inactive voxels and tiles: -bandWidth inside, +bandWidth outside.
        // These inactive values are clamped, not actual distances to the surface.
        tools::changeLevelSetBackground(component->tree(), bandWidth);
        tools::signedFloodFill(component->tree());
        component->setGridClass(GRID_LEVEL_SET);
    }

    // Test each component's saved exterior voxel against the other grids.
    // A negative value in grid j is used to classify component i as nested in j.
    // This inference relies on the separated UDF bands, consistent signs,
    // and the assumption that the filled volumes are nested or disjoint.
    const size_t count = components.size();
    std::vector<bool> nested(count, false);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < count; ++j) {
            if (i != j && components[j]->tree().getValue(exteriorSeeds[i]) < ValueT(0)) {
                nested[i] = true;
                break;
            }
        }
    }

    // Merge retained components and release each component after use.
    for (size_t i = 0; i < count; ++i) {
        if (!nested[i]) tools::csgUnion(*result, *components[i]);
        components[i].reset();
    }

    // Expand the active band to provide samples for distance reconstruction.
    tools::dilateActiveValues(result->tree(), math::Ceil(mHalfWidth),
                              tools::NN_FACE, tools::EXPAND_TILES);

    // Reconstruct signed distances from the current zero crossings.
    tools::LevelSetTracker<GridType> tracker(*result);
#if 1//first-order
    tracker.setSpatialScheme(math::FIRST_BIAS);
    tracker.setTemporalScheme(math::TVD_RK1);
#else// higher order
    tracker.setSpatialScheme(math::HJWENO5_BIAS);
    tracker.setTemporalScheme(math::TVD_RK3);
#endif
    tracker.setNormCount(math::Ceil(mHalfWidth));
    tracker.normalize();

    // Prune the grid.
    tracker.prune();

    return result;
}// vdb_tool::ShrinkWrap<GridType>::offset

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto ShrinkWrap<GridType>::upsample(const GridType &grid,
                                    const GridType &fineOffset,
                                    double upsampleFactor)
{
#if 0
    auto outGrid = createLevelSet<GridType>(fineOffset.voxelSize()[0], mHalfWidth);
    outGrid->setTransform(fineOffset.transform().copy());
    tools::resampleToMatch<tools::BoxSampler>(grid, *outGrid);
#else
    using ValueT = typename GridType::ValueType;

    // Match the finer offset's voxel spacing, origin, and orientation.
    auto outTransform = fineOffset.transform().copy();

    // Preserve the input band's width in world units during reconstruction.
    auto outGrid = GridType::create(grid.background());
    outGrid->setTransform(outTransform);
    outGrid->setGridClass(GRID_LEVEL_SET);

    // The offset grids share an origin and axes; only their spacing differs.
    math::Mat4d indexTransform = math::Mat4d::identity();
    indexTransform.preScale(Vec3d(upsampleFactor));

    // Interpolate directly onto the aligned fine grid.
    tools::GridTransformer transformer(indexTransform);
    transformer.transformGrid<tools::BoxSampler>(grid, *outGrid);

    // Correct distance errors. This can slightly move the zero surface.
    tools::LevelSetTracker<GridType> tracker(*outGrid);
#if 1//first-order
    tracker.setSpatialScheme(math::FIRST_BIAS);
    tracker.setTemporalScheme(math::TVD_RK1);
#else// higher order
    tracker.setSpatialScheme(math::HJWENO5_BIAS);
    tracker.setTemporalScheme(math::TVD_RK3);
#endif
    tracker.setNormCount(math::Ceil(mHalfWidth));
    tracker.normalize();

    // Restore the requested half-band width in world units.
    const ValueT bandWidth = ValueT(mHalfWidth * outGrid->voxelSize()[0]);
    tools::changeLevelSetBackground(outGrid->tree(), bandWidth);

    // Clamp and deactivate out-of-band samples, then compact the tree.
    tracker.prune();
#endif
    mIsGridSDF = true;
    return outGrid;
}// vdb_tool::ShrinkWrap<GridType>::upsample

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto ShrinkWrap<GridType>::constrainedErode(GridType &grid,
                                            const GridType &fineOffset,
                                            double erosionDistance)
{
    tools::LevelSetFilter<GridType> filter(grid);
    filter.setNormCount(math::Ceil(mHalfWidth));
#if 1//first-order
    filter.setSpatialScheme(math::FIRST_BIAS);
    filter.setTemporalScheme(math::TVD_RK1);
#else// higher order
    filter.setSpatialScheme(math::HJWENO5_BIAS);
    filter.setTemporalScheme(math::TVD_RK3);
#endif
    if (mIsGridSDF == false) {
        filter.normalize();
        filter.prune();// is this needed?
    }
    filter.offset(static_cast<typename GridType::ValueType>(erosionDistance));
    mIsGridSDF = false;// the CSG operation messed up the SDF
    return tools::csgUnionCopy(grid, fineOffset);
}// vdb_tool::ShrinkWrap<GridType>::constrainedErode

//////////////////////////////////////////////////////////////////////////

template<typename GridType, class ProgressT>
typename GridType::Ptr
shrinkWrap(PolySoup &&poly,
           float minVoxelSize,
           float closingThreshold,
           int maxErodeIterations,
           bool useConservativeOffset,
           bool useThresholdSeed,
           float halfWidth,
           ProgressT *progress)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "shrinkWrap requires an SDF grid with floating-point values");
    ShrinkWrap<GridType> tmp(std::move(poly), minVoxelSize, halfWidth);
    if (useThresholdSeed) {
        tmp.process(closingThreshold, useConservativeOffset, progress);
    } else {
        tmp.process(closingThreshold, maxErodeIterations, useConservativeOffset, progress);
    }
    return tmp.grid();
}

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType, class ProgressT>
std::vector<typename GridType::Ptr>
shrinkWrap(std::vector<Vec3s>& vtx,
           std::vector<Vec3I>& tri,
           std::vector<Vec4I>& quad,
           const math::BBox<Vec3f> &bbox,
           float minVoxelSize,
           float closingThreshold,
           int maxErodeIterations,
           bool useConservativeOffset,
           bool useThresholdSeed,
           float halfWidth,
           ProgressT *progress)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "shrinkWrap requires an SDF grid with floating-point values");
    PolySoup poly{std::move(vtx), std::move(tri), std::move(quad), bbox};
    ShrinkWrap<GridType> tmp(std::move(poly), minVoxelSize, halfWidth);
    if (useThresholdSeed) {
        tmp.process(closingThreshold, useConservativeOffset, progress);
    } else {
        tmp.process(closingThreshold, maxErodeIterations, useConservativeOffset, progress);
    }
    return tmp.grids();
}

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // VDB_TOOL_SHRINKWRAP_HAS_BEEN_INCLUDED
