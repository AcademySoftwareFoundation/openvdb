// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @author Greg Hurst
///
/// @file ConvexVoxelizer.h
///
/// @brief Base class used to generate the narrow-band level set of a convex region.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_CONVEXVOXELIZER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_CONVEXVOXELIZER_HAS_BEEN_INCLUDED

#include <openvdb/math/Math.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/util/NullInterrupter.h>

#include <openvdb/tools/Merge.h>
#include <openvdb/tools/SignedFloodFill.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <type_traits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace lvlset {

/// @brief Internal class used by derived @c ConvexVoxelizer classes that make use of PointPartitioner.
template<typename VectorType>
struct PointArray
{
    using PosType = VectorType;

    PointArray() = default;

    PointArray(const std::vector<VectorType>& vec)
    : mVec(vec)
    {
    }

    inline const VectorType& operator[](const Index& i) { return mVec[i]; }

    inline size_t size() const { return mVec.size(); };

    inline void getPos(size_t n, VectorType& xyz) const { xyz = mVec[n]; }

private:

    const std::vector<VectorType>& mVec;

}; // struct PointArray

} // namespace lvlset

/// @brief Base class used to generate a grid of type @c GridType containing a narrow-band level set
/// representation of a convex region.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c Derived is the derived class that implements the base class (curiously recurring template pattern).
///
/// @par Example of derived level set sphere class
/// @code
/// template <typename GridType>
/// class SphereVoxelizer : public ConvexVoxelizer<GridType, SphereVoxelizer<GridType>>
/// {
///     using GridPtr = typename GridType::Ptr;
///
///     using BaseT = ConvexVoxelizer<GridType, SphereVoxelizer<GridType>>;
///     using BaseT::mXYData;
///     using BaseT::tileCeil;
///
///     using ValueT = typename BaseT::ValueT;
///     using Vec3T  = typename BaseT::Vec3T;
///
/// public:
///
///     friend class ConvexVoxelizer<GridType, SphereVoxelizer<GridType>>;
///
///     SphereVoxelizer(GridPtr& grid, const bool& threaded = true)
///     : BaseT(grid, threaded)
///     {
///     }
///
///     template <typename ScalarT>
///     void
///     operator()(const math::Vec3<ScalarT>& pt, const ScalarT& r)
///     {
///         static_assert(std::is_floating_point<ScalarT>::value);
///
///         if (r <= 0)
///             return;
///
///         initialize<ScalarT>(pt, r);
///
///         BaseT::iterate();
///     }
///
/// private:
///
///     inline ValueT
///     signedDistance(const Vec3T& p) const
///     {
///         return (p - mPt).length() - mRad;
///     }
///
///     inline void
///     setXYRangeData(const Index& step = 1)
///     {
///         mXYData.reset(mX - mORad, mX + mORad, step);
///
///         for (ValueT x = tileCeil(mX - mORad, step); x <= mX + mORad; x += step)
///             mXYData.expandYRange(x, BaseT::circleBottom(mX, mY, mORad, x),
///                 BaseT::circleTop(mX, mY, mORad, x));
///     }
///
///     std::function<bool(ValueT&, ValueT&, const ValueT&, const ValueT&)> sphereBottomTop =
///     [this](ValueT& zb, ValueT& zt, const ValueT& x, const ValueT& y)
///     {
///         zb = BaseT::sphereBottom(mX, mY, mZ, mORad, x, y);
///         zt = BaseT::sphereTop(mX, mY, mZ, mORad, x, y);
///
///         return std::isfinite(zb) && std::isfinite(zt);
///     };
///
///     template <typename ScalarT>
///     inline void
///     initialize(const math::Vec3<ScalarT>& pt, const ScalarT& r)
///     {
///         const ValueT vx = BaseT::voxelSize(),
///                      hw = BaseT::halfWidth();
///
///         // sphere data in index space
///         mPt = Vec3T(pt)/vx;
///         mRad = ValueT(r)/vx;
///
///         mX = mPt.x(); mY = mPt.y(); mZ = mPt.z();
///
///         // padded radius used to populate the outer halfwidth of the sdf
///         mORad  = mRad + hw;
///
///         BaseT::bottomTop = sphereBottomTop;
///     }
///
///     Vec3T mPt;
///     ValueT mRad, mORad, mX, mY, mZ;
/// };
///
/// // usage:
///
/// // initialize level set grid with voxel size 0.1 and half width 3.0
/// FloatGrid::Ptr grid = createLevelSet<GridT>(0.1f, 3.0f);
///
/// // populate grid with a sphere centered at (0, 1, 2) and radius 5
/// SphereVoxelizer<FloatGrid> op(grid);
/// op(Vec3s(0.0f, 1.0f, 2.0f), 5.0f);
///
/// @endcode
template <typename GridType, typename Derived, typename InterruptType = util::NullInterrupter>
class ConvexVoxelizer
{
    using GridPtr = typename GridType::Ptr;

    using TreeT = typename GridType::TreeType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    using NodeChainT = typename RootT::NodeChainType;

    using AccessorT = typename GridType::Accessor;

protected:

    using ValueT = typename GridType::ValueType;
    using Vec3T  = math::Vec3<ValueT>;
    using Vec2T  = math::Vec2<ValueT>;

    static_assert(std::is_floating_point<ValueT>::value);

public:

    /// @brief Constructor
    ///
    /// @param grid scalar grid to populate the level set in
    /// @param threaded center of the sphere in world units
    /// @param interrupter pointer to optional interrupter. Use template
    /// argument util::NullInterrupter if no interruption is desired.
    ///
    /// @note The voxel size and half width are determined from the input grid,
    /// meaning the voxel size and background value need to be set prior to voxelization
    ConvexVoxelizer(GridPtr& grid, const bool& threaded = false, InterruptType* interrupter = nullptr)
    : mGrid(grid)
    , mVox(ValueT((grid->voxelSize())[0]))
    , mHw(ValueT(grid->background()/(grid->voxelSize())[0]))
    , mBg(grid->background())
    , mNegBg(-(grid->background()))
    , mSerial(!threaded)
    , mInterrupter(interrupter)
    {
    }

    virtual ~ConvexVoxelizer() = default;

    /// @brief Return the voxel size of the grid.
    inline ValueT voxelSize() const { return mVox; }

    /// @brief Return the half width of the narrow-band level set.
    inline ValueT halfWidth() const { return mHw; }

private:

    class CacheLastLeafAccessor;

protected:

    // ------------ Main APIs for derived classes ------------

    /// @brief The function the derived class calls to create the level set,
    /// working in index space other than setting signed distance values.
    ///
    /// @note This function handles both parallel and serial iterations. If running in serial mode,
    /// it flood fills the tile topology immediately; otherwise, it avoids duplicating nontrivial
    /// tree topology over multiple threads. This method also checks for background tiles
    /// that are too thin to fit and delegates accordingly.
    inline void
    iterate()
    {
        static const Index LEAFDIM = LeafT::DIM;

        // objects too thin to have negative background tiles
        if (!invokeTileCanFit(LEAFDIM)) {
            thinIterate();
            return;
        }

        // iterate over all non root nodes
        using ChainT = typename NodeChainT::PopBack;

        // if we're working in parallel, we avoid flood filling the tile topology until the end
        // this avoids duplicating nontrivial tree topology over multiple threads
        if (mSerial)
            iterateTile<ChainT>();

        iterateLeaf();

        if (!checkInterrupter())
            return;

        if (!mSerial)
            tools::signedFloodFill(getTree());
    }

    // ------------ derived classes need to implement these ------------

    /// @brief Determines the x bounds in index space of the convex region dilated by the half width.
    /// For each x value in index space, the y range in index space of the dilated region is computed.
    /// This function should store the data in @c mXYData.
    ///
    /// @param step The step size for setting the XY range data, defaults to 1.
    /// @note Function to be implemented by derived classes to set XY range data.
    /// This function is called at most 4 times within @c iterate().
    void setXYRangeData(const Index&) {}

    /// @brief Checks if the tile of a given dimension can possibly fit within the region.
    ///
    /// The derived class does not need to implement it if the default behavior is acceptable,
    /// which assumes a tile can always possibly fit.
    ///
    /// @param dim The dimension of the tile in which to check if the tile fits.
    /// @note This is meant as a short-circuting method: if a tile of a given dimension
    /// can't fit then @c iterate will not try to populate the level set with background
    /// tiles of this dimension.
    /// @return true if the tile can possibly fit; otherwise false.
    inline bool tileCanFit(const Index&) const { return true; }

    // distance in index space
    /// @brief Computes the signed distance from a point to the convex region in index space.
    ///
    /// @param p The point in 3D space for which to compute the signed distance.
    inline ValueT signedDistance(const Vec3T&) const { return ValueT(0); }

    /// @brief Computes the signed distance for tiles in index space,
    /// considering the center of the tile.
    /// This method is optional to override and defaults to @c signedDistance.
    ///
    /// @param p The point at the center of the tile in 3D space.
    /// @note This can be useful for cases that build objects from multiple primitives, e.g.
    /// dilated mesh is built by constructing and unioning _open_ prisms and _open_ tube wedges.
    /// A tile might not fully fit in an open prism but might fit in the union of a prism and wedge,
    /// and so in this case it might make sense to use the sdf for an offset triangle on tiles
    /// during the open prism scan.
    inline ValueT
    tilePointSignedDistance(const Vec3T& p) const
    {
        return static_cast<const Derived*>(this)->signedDistance(p);
    }

    /// @brief Find where a vertical infinite line intersects
    /// a convex region dilated by the half width.
    ///
    /// @param[out] zb Reference to the z ordinate where the bottom intersection occurs.
    /// @param[out] zt Reference to the z ordinate where the top intersection occurs.
    /// @param[in] x The x ordinate of the infinte line.
    /// @param[in] y The y ordinate of the infinte line.
    /// @return true if an intersection occurs; otherwise false.
    /// @note The derived class can override this lambda to implement different behavior for degenerate cases.
    std::function<bool(ValueT&, ValueT&, const ValueT&, const ValueT&)> bottomTop =
        [](ValueT&, ValueT&, const ValueT&, const ValueT&) { return false; };

    // ------------ utilities ------------

    /// @brief Rounds an input scalar up to the nearest valid ordinate of tile of a specified size.
    /// @param x Input value.
    /// @param step Tile step size.
    /// @return The ceiling of the value based on the tile size.
    inline static ValueT
    tileCeil(const ValueT& x, const ValueT& step)
    {
        const ValueT offset = ValueT(0.5) * (step - ValueT(1));

        return step == ValueT(1)
            ? static_cast<ValueT>(math::Ceil(perturbDown(x)))
            : step * static_cast<ValueT>(math::Ceil(perturbDown((x - offset)/step))) + offset;
    }

    /// @brief Rounds an input scalar up to the nearest valid ordinate of tile of a specified size.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param x Input value.
    /// @param step Tile step size.
    /// @return The ceiling of the value based on the tile size.
    template <typename T>
    inline static ValueT
    tileCeil(const ValueT& x, const T& step)
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        const ValueT s = static_cast<ValueT>(step);

        return tileCeil(x, s);
    }

    /// @brief Rounds an input scalar down to the nearest valid ordinate of tile of a specified size.
    /// @param x Input value.
    /// @param step Tile step size.
    /// @return The ceiling of the value based on the tile size.
    inline static ValueT
    tileFloor(const ValueT& x, const ValueT& step)
    {
        const ValueT offset = ValueT(0.5) * (step - ValueT(1));

        return step == ValueT(1)
            ? static_cast<ValueT>(math::Floor(perturbUp(x)))
            : step * static_cast<ValueT>(math::Floor(perturbUp((x - offset)/step))) + offset;
    }

    /// @brief Rounds an input scalar down to the nearest valid ordinate of tile of a specified size.
    /// @tparam T Any integral type (int, unsigned int, size_t, etc.)
    /// @param x Input value.
    /// @param step Tile step size.
    /// @return The ceiling of the value based on the tile size.
    template <typename T>
    inline static ValueT
    tileFloor(const ValueT& x, const T& step)
    {
        static_assert(std::is_integral<T>::value, "Index must be an integral type");

        const ValueT s = static_cast<ValueT>(step);

        return tileFloor(x, s);
    }

    /// @brief Computes the bottom y-coordinate of a circle at a given x position.
    /// @param x0 X-coordinate of the circle's center.
    /// @param y0 Y-coordinate of the circle's center.
    /// @param r Radius of the circle.
    /// @param x X-coordinate for which to compute the bottom y-coordinate.
    /// @return The y-coordinate at the bottom of the circle for the given x position.
    inline static ValueT
    circleBottom(const ValueT& x0, const ValueT& y0,
                 const ValueT& r, const ValueT& x)
    {
        return y0 - math::Sqrt(math::Pow2(r) - math::Pow2(x-x0));
    }

    /// @brief Computes the top y-coordinate of a circle at a given x position.
    /// @param x0 X-coordinate of the circle's center.
    /// @param y0 Y-coordinate of the circle's center.
    /// @param r Radius of the circle.
    /// @param x X-coordinate for which to compute the top y-coordinate.
    /// @return The y-coordinate at the top of the circle for the given x position.
    inline static ValueT
    circleTop(const ValueT& x0, const ValueT& y0,
              const ValueT& r, const ValueT& x)
    {
        return y0 + math::Sqrt(math::Pow2(r) - math::Pow2(x-x0));
    }

    /// @brief Computes the bottom z-coordinate of a sphere at a given (x, y) position.
    /// @param x0 X-coordinate of the sphere's center.
    /// @param y0 Y-coordinate of the sphere's center.
    /// @param z0 Z-coordinate of the sphere's center.
    /// @param r Radius of the sphere.
    /// @param x X-coordinate for which to compute the bottom z-coordinate.
    /// @param y Y-coordinate for which to compute the bottom z-coordinate.
    /// @return The z-coordinate at the bottom of the sphere for the given (x, y) position.
    inline static ValueT
    sphereBottom(const ValueT& x0, const ValueT& y0, const ValueT& z0,
                 const ValueT& r, const ValueT& x, const ValueT& y)
    {
        return z0 - math::Sqrt(math::Pow2(r) - math::Pow2(x-x0) - math::Pow2(y-y0));
    }

    /// @brief Computes the top z-coordinate of a sphere at a given (x, y) position.
    /// @param x0 X-coordinate of the sphere's center.
    /// @param y0 Y-coordinate of the sphere's center.
    /// @param z0 Z-coordinate of the sphere's center.
    /// @param r Radius of the sphere.
    /// @param x X-coordinate for which to compute the top z-coordinate.
    /// @param y Y-coordinate for which to compute the top z-coordinate.
    /// @return The z-coordinate at the top of the sphere for the given (x, y) position.
    inline static ValueT
    sphereTop(const ValueT& x0, const ValueT& y0, const ValueT& z0,
              const ValueT& r, const ValueT& x, const ValueT& y)
    {
        return z0 + math::Sqrt(math::Pow2(r) - math::Pow2(x-x0) - math::Pow2(y-y0));
    }

    // ------------ nested classes ------------

    /// @brief Class that stores endpoints of a y range for each x value within a specified range and step size.
    /// @details This class tracks y ranges (defined by ymin and ymax) for each x value over a defined interval,
    /// using a configurable step size.
    /// It allows updating, expanding, and resetting the y ranges, as well as merging data from other instances
    /// and trimming invalid entries.
    class XYRangeData
    {

    public:

        XYRangeData() = default;

        /// @brief Constructor that sets the x range to span a given interval with a specific step size.
        /// This initializes all y ranges as empty.
        /// @param xmin The lower bound of the x range.
        /// @param xmax The upper bound of the x range.
        /// @param step The step size between x values. Defaults to 1.
        XYRangeData(const ValueT& xmin, const ValueT& xmax, const Index& step = 1)
        {
            reset(xmin, xmax, step);
        }

        /// @brief Expands the y range for a given x value by updating the minimum and maximum
        /// y values if the new ones extend the current range.
        /// @param x The x value.
        /// @param ymin The new minimum y value to compare with and possibly update
        /// the current minimum at x.
        /// @param ymax The new maximum y value to compare with and possibly update
        /// the current maximum at x.
        inline void
        expandYRange(const ValueT& x, const ValueT& ymin, const ValueT& ymax)
        {
            expandYMin(x, ymin);
            expandYMax(x, ymax);
        }

        /// @brief Sets the minimum y value for a given x value, if the provided ymin
        /// is smaller than the current value.
        /// @param x The x value.
        /// @param ymin The minimum y value to possibly be set.
        inline void
        expandYMin(const ValueT& x, const ValueT& ymin)
        {
            const Index i = worldToIndex(x);

            if (std::isfinite(ymin) && ymin < mYMins[i])
                mYMins[i] = ymin;
        }

        /// @brief Sets the maximum y value for a given x value, if the provided ymax
        /// is larger than the current value.
        /// @param x The x value.
        /// @param ymax The maximum y value to possibly be set.
        inline void
        expandYMax(const ValueT& x, const ValueT& ymax)
        {
            const Index i = worldToIndex(x);

            if (std::isfinite(ymax) && ymax > mYMaxs[i])
                mYMaxs[i] = ymax;
        }

        /// @brief Expands the y range for a given x by adjusting ymin or ymax if the
        /// given y is smaller or larger.
        /// @param x The x value.
        /// @param y The y value to use for expanding the range.
        inline void
        expandYRange(const ValueT& x, const ValueT& y)
        {
            if (std::isfinite(y)) {
                const Index i = worldToIndex(x);

                if (y < mYMins[i])
                    mYMins[i] = y;

                if (y > mYMaxs[i])
                    mYMaxs[i] = y;
            }
        }

        /// @brief Set the minimum y value for a given x value,
        /// even if it is larger than the current value.
        /// @param x The x value.
        /// @param ymin The minimum y value to reset.
        inline void
        setYMin(const ValueT& x, const ValueT& ymin)
        {
            const Index i = worldToIndex(x);

            mYMins[i] = ymin;
        }

        /// @brief Set the maximum y value for a given x value,
        /// even if it is smaller than the current value.
        /// @param x The x value.
        /// @param ymax The maximum y value to reset.
        inline void
        setYMax(const ValueT& x, const ValueT& ymax)
        {
            const Index i = worldToIndex(x);

            mYMaxs[i] = ymax;
        }

        /// @brief Clears the y range for a given x value, setting it to an empty interval.
        /// @param x The x value.
        inline void
        clearYRange(const ValueT& x)
        {
            const Index i = worldToIndex(x);

            mYMins[i] = MAXVALUE;
            mYMaxs[i] = MINVALUE;
        }

        /// @brief Clears the data container
        inline void
        clear()
        {
            mStep = 1;
            mStepInv = ValueT(1);

            mXStart = MAXVALUE;
            mXEnd = MINVALUE;

            mSize = 0;

            mYMins.assign(mSize, MAXVALUE);
            mYMaxs.assign(mSize, MINVALUE);
        }

        /// @brief Resets the x range to span a given interval with a specific step size.
        /// This initializes all y ranges as empty.
        /// @param xmin The lower bound of the x range.
        /// @param xmax The upper bound of the x range.
        /// @param step The step size between x values. Defaults to 1.
        inline void
        reset(const ValueT& xmin, const ValueT& xmax, const Index& step = 1)
        {
            assert(step != 0);

            mStep = step;
            mStepInv = ValueT(1)/static_cast<ValueT>(mStep);

            mXStart = tileCeil(xmin, mStep);
            mXEnd = tileFloor(xmax, mStep);

            mSize = 1 + indexDistance(mXEnd, mXStart);

            mYMins.assign(mSize, MAXVALUE);
            mYMaxs.assign(mSize, MINVALUE);
        }

        /// @brief Retrieves the step size used for the x values.
        /// @return The step size.
        inline Index step() const { return mStep; }

        /// @brief Returns the number of x points in the current range.
        /// @return The size of the x range.
        inline Index size() const { return mSize; }

        /// @brief Retrieves the starting x value in the range.
        /// @return The start of the x range.
        inline ValueT start() const { return mXStart; }

        /// @brief Retrieves the ending x value in the range.
        /// @return The end of the x range.
        inline ValueT end() const { return mXEnd; }

        /// @brief Converts an index to its corresponding x value.
        /// @param i The index value.
        /// @return The corresponding x value.
        inline ValueT getX(const Index& i) const { return indexToWorld(i); }

        /// @brief Gets the minimum y value for a given index.
        /// @param i The index value.
        /// @return The minimum y value.
        inline ValueT getYMin(const Index& i) const { assert(i < mSize); return mYMins[i]; }

        /// @brief Gets the maximum y value for a given index.
        /// @param i The index value.
        /// @return The maximum y value.
        inline ValueT getYMax(const Index& i) const { assert(i < mSize); return mYMaxs[i]; }

        /// @brief Gets the minimum y value for a given x value.
        /// @param x The x value.
        /// @return The minimum y value at the given x.
        /// @note @c x is rounded to the nearest value in the x range.
        inline ValueT getYMin(const ValueT& x) const { return mYMins[worldToIndex(x)]; }

        /// @brief Gets the maximum y value for a given x value.
        /// @param x The x value.
        /// @return The maximum y value at the given x.
        /// @note @c x is rounded to the nearest value in the x range.
        inline ValueT getYMax(const ValueT& x) const { return mYMaxs[worldToIndex(x)]; }

        /// @brief Retrieves the x, ymin, and ymax values for a given index.
        /// @param x Output parameter for the x value.
        /// @param ymin Output parameter for the minimum y value.
        /// @param ymax Output parameter for the maximum y value.
        /// @param i The index to query.
        inline void
        XYData(ValueT& x, ValueT& ymin, ValueT& ymax, const Index& i) const
        {
            x = indexToWorld(i);
            ymin = mYMins[i];
            ymax = mYMaxs[i];
        }

        /// @brief Returns @c true if the container has no x values or if all y ranges are empty.
        inline bool isEmpty() const
        {
            if (mSize == 0)
                return true;

            for (Index i = 0; i < mSize; ++i) {
                if (mYMins[i] <= mYMaxs[i])
                    return false;
            }

            return true;
        }

        /// @brief Merges another XYRangeData into the current instance by combining y ranges
        /// over the overlapping x range.
        /// @param xydata The XYRangeData to merge with.
        inline void
        merge(const XYRangeData& xydata)
        {
            assert(mStep == xydata.step());

            const ValueT start = xydata.start(), end = xydata.end();

            const std::vector<ValueT>& ymins = xydata.mYMins;
            const std::vector<ValueT>& ymaxs = xydata.mYMaxs;

            if (start < mXStart) {
                const Index n = indexDistance(mXStart, start);
                mYMins.insert(mYMins.begin(), n, MAXVALUE);
                mYMaxs.insert(mYMaxs.begin(), n, MINVALUE);
                mXStart = start;
            }

            if (mXEnd < end) {
                const Index m = indexDistance(end, mXEnd);
                mYMins.insert(mYMins.end(), m, MAXVALUE);
                mYMaxs.insert(mYMaxs.end(), m, MINVALUE);
                mXEnd = end;
            }

            mSize = 1 + indexDistance(mXEnd, mXStart);

            const Index offset = start < mXStart ? 0 : indexDistance(start, mXStart);
            for (Index i = 0, j = offset; i < ymins.size(); ++i, ++j) {
                if (mYMins[j] > ymins[i]) { mYMins[j] = ymins[i]; }
                if (mYMaxs[j] < ymaxs[i]) { mYMaxs[j] = ymaxs[i]; }
            }
        }

        /// @brief Trims the x range by removing empty or invalid y ranges from the beginning and end.
        /// Truncates the range if all values are invalid.
        inline void
        trim()
        {
            Index i = 0;
            while(i < mSize) {
                if (mYMins[i] > mYMaxs[i]) i++;
                else break;
            }

            if (i == mSize) {
                mSize = 0; mXStart = ValueT(0); mXEnd = ValueT(0);
                mYMins.clear(); mYMaxs.clear();
                return;
            }

            Index j = 0;
            while(j < mSize) {
                const Index k = mSize - (j + 1);
                if (mYMins[k] > mYMaxs[k]) j++;
                else break;
            }

            if (i == 0 && j == 0)
                return;

            mSize -= i + j;
            mXStart += ValueT(i * mStep);
            mXEnd -= ValueT(j * mStep);

            if (i > 0) {
                mYMins.erase(mYMins.begin(), mYMins.begin() + i);
                mYMaxs.erase(mYMaxs.begin(), mYMaxs.begin() + i);
            }

            if (j > 0) {
                mYMins.erase(mYMins.end() - j, mYMins.end());
                mYMaxs.erase(mYMaxs.end() - j, mYMaxs.end());
            }
        }

    private:

        inline static const ValueT
            MINVALUE = std::numeric_limits<ValueT>::lowest(),
            MAXVALUE = std::numeric_limits<ValueT>::max();

        inline Index
        indexDistance(const ValueT& a, const ValueT& b)
        {
            return Index(math::Round(mStepInv*math::Abs(a - b)));
        }

        inline Index
        worldToIndex(const ValueT& x) const
        {
            const Index i = Index(math::Round(mStepInv*(x - mXStart)));
            assert(i < mSize);

            return i;
        }

        inline ValueT
        indexToWorld(const Index i) const
        {
            assert(i < mSize);

            return mXStart + ValueT(i * mStep);
        }

        Index mStep, mSize;

        ValueT mStepInv, mXStart, mXEnd;

        std::vector<ValueT> mYMins, mYMaxs;

    }; // class XYRangeData

    // ------------ protected members ------------

    XYRangeData mXYData;

private:

#define EPS 0.0005f
    inline static ValueT perturbDown(const ValueT& x) { return x - ValueT(EPS); }
    inline static ValueT perturbUp(const ValueT& x) { return x + ValueT(EPS); }
#undef EPS

    inline static ValueT
    voxelCeil(const ValueT& x)
    {
        return static_cast<ValueT>(math::Ceil(perturbDown(x)));
    }

    inline static ValueT
    voxelFloor(const ValueT& x)
    {
        return static_cast<ValueT>(math::Floor(perturbUp(x)));
    }

    // skips the need for negative tile population and internal leap frogging
    inline void thinIterate()
    {
        invokeSetXYRangeData();

        // false means disable internal leap frogging
        iterateXYZ<false>();
    }

    template <typename ChainT>
    inline void iterateTile()
    {
        using NodeT        = typename ChainT::Back;
        using PoppedChainT = typename ChainT::PopBack;

        static const Index DIM = NodeT::DIM;

        // only attempt to add negative background tiles at this level if they can fit
        if (invokeTileCanFit(DIM)) {
            invokeSetXYRangeData(DIM);

            tileIterateXYZ<NodeT>();
        }

        if constexpr (!std::is_same_v<PoppedChainT, openvdb::TypeList<>>) {
            iterateTile<PoppedChainT>();
        }
    }

    inline void
    iterateLeaf()
    {
        invokeSetXYRangeData();

        // true means enable internal leap frogging
        iterateXYZ<true>();
    }

    template <bool LeapFrog = false>
    void
    iterateXYZ()
    {
        if (mXYData.isEmpty())
            return;

        // borrowing parallel logic from tools/LevelSetSphere.h

        const Index n = mXYData.size();

        if (mSerial) {
            CacheLastLeafAccessor acc(getTree());
            for (Index i = 0; i < n; ++i) {
                if (mInterrupter && !(i & ((1 << 7) - 1)) && !checkInterrupter())
                    return;

                iterateYZ<LeapFrog>(i, acc);
            }
        } else {
            tbb::enumerable_thread_specific<TreeT> pool(getTree());

            auto kernel = [&](const tbb::blocked_range<Index>& rng) {
                TreeT &tree = pool.local();
                CacheLastLeafAccessor acc(tree);

                if (!checkInterrupter())
                    return;

                for (Index i = rng.begin(); i != rng.end(); ++i) {
                    if constexpr (LeapFrog)
                        iterateNoTilesYZ(i, acc);
                    else
                        iterateYZ<false>(i, acc);
                }
            };

            tbb::parallel_for(tbb::blocked_range<Index>(Index(0), n, Index(128)), kernel);
            using RangeT = tbb::blocked_range<typename tbb::enumerable_thread_specific<TreeT>::iterator>;
            struct Op {
                const bool mDelete;
                TreeT *mTree;
                Op(TreeT &tree) : mDelete(false), mTree(&tree) {}
                Op(const Op& other, tbb::split) : mDelete(true), mTree(new TreeT(other.mTree->background())) {}
                ~Op() { if (mDelete) delete mTree; }
                void operator()(RangeT &r) { for (auto i=r.begin(); i!=r.end(); ++i) this->merge(*i);}
                void join(Op &other) { this->merge(*(other.mTree)); }
                void merge(TreeT &tree) { mTree->merge(tree, MERGE_ACTIVE_STATES); }
            } op( getTree() );
            tbb::parallel_reduce(RangeT(pool.begin(), pool.end(), 4), op);
        }
    }


    // for a given x value, create a filled slice of the object by populating
    //   active voxels and inactive negative background voxels
    // if we're leap frogging, we may assume the negative background tiles are already populated
    // for each x ordinate and y-scan range
    //   find the z-range for each y and then populate the grid with distance values
    template <bool LeapFrog = false>
    inline void
    iterateYZ(const Index& i, CacheLastLeafAccessor& acc)
    {
        // initialize x value and y-range
        ValueT x, yb, yt;
        mXYData.XYData(x, yb, yt, i);

        if (!std::isfinite(yb) || !std::isfinite(yt))
            return;

        ValueT zb, zt;

        for (ValueT y = voxelCeil(yb); y <= perturbUp(yt); ++y) {
            if (!bottomTop(zb, zt, x, y))
                continue;

            Coord ijk(Int32(x), Int32(y), Int32(0));
            Vec3T p(x, y, ValueT(0));

            ijk[2] = Int32(voxelCeil(zb))-1;
            acc.reset(ijk);

            for (ValueT z = voxelCeil(zb); z <= perturbUp(zt); ++z) {
                ijk[2] = Int32(z);
                const ValueT val = acc.template getValue<1>(ijk);

                if (val == mNegBg) {
                    if constexpr (LeapFrog) acc.template leapUp<false>(ijk, z);
                    continue;
                }

                p[2] = z;
                const ValueT dist = mVox * invokeSignedDistance(p);

                if (dist <= mNegBg) {
                    acc.template setValueOff<1,false>(ijk, mNegBg);
                } else if (dist < val) {
                    acc.template setValueOn<1,false>(ijk, dist);
                } else { // dist >= val
                    acc.template checkReset<1>(ijk);
                }
            }
        }
    }

    // for a given x value, create a hollow slice of the object by only populating active voxels
    // for each x ordinate and y-scan range
    //   find the z-range for each y and then populate the grid with distance values
    inline void
    iterateNoTilesYZ(const Index& i, CacheLastLeafAccessor& acc)
    {
        // initialize x value and y-range
        ValueT x, yb, yt;
        mXYData.XYData(x, yb, yt, i);

        if (!std::isfinite(yb) || !std::isfinite(yt))
            return;

        ValueT zb, zt;

        for (ValueT y = voxelCeil(yb); y <= perturbUp(yt); ++y) {
            if (!bottomTop(zb, zt, x, y))
                continue;

            Coord ijk(Int32(x), Int32(y), Int32(0));
            Vec3T p(x, y, ValueT(0));

            bool early_break = false;
            ValueT z_stop;

            ijk[2] = Int32(voxelCeil(zb))-1;
            acc.reset(ijk);
            for (ValueT z = voxelCeil(zb); z <= perturbUp(zt); ++z) {
                ijk[2] = Int32(z);
                p[2] = z;
                const ValueT dist = mVox * invokeSignedDistance(p);

                if (dist <= mNegBg) {
                    early_break = true;
                    z_stop = z;
                    break;
                } else if (dist < mBg) {
                    acc.template setValueOn<1>(ijk, dist);
                } else { // dist >= mBg
                    acc.template checkReset<1>(ijk);
                }
            }
            if (early_break) {
                ijk[2] = Int32(voxelFloor(zt))+1;
                acc.reset(ijk);
                for (ValueT z = voxelFloor(zt); z > z_stop; --z) {
                    ijk[2] = Int32(z);
                    p[2] = z;
                    const ValueT dist = mVox * invokeSignedDistance(p);

                    if (dist <= mNegBg) {
                        break;
                    } else if (dist < mBg) {
                        acc.template setValueOn<-1>(ijk, dist);
                    } else { // dist >= mBg
                        acc.template checkReset<-1>(ijk);
                    }
                }
            }
        }
    }

    template <typename NodeT>
    void
    tileIterateXYZ()
    {
        if (mXYData.isEmpty())
            return;

        AccessorT acc(getTree());
        for (Index i = 0; i < mXYData.size(); ++i) {
            if (mInterrupter && !(i & ((1 << 7) - 1)) && !checkInterrupter())
                return;

            tileIterateYZ<NodeT>(i, acc);
        }
    }

    template <typename NodeT>
    inline void
    tileIterateYZ(const Index& i, AccessorT& acc)
    {
        // initialize x value and y-range
        ValueT x, yb, yt;
        mXYData.XYData(x, yb, yt, i);

        if (!std::isfinite(yb) || !std::isfinite(yt))
            return;

        static const Index TILESIZE = NodeT::DIM;

        ValueT zb, zt;

        for (ValueT y = tileCeil(yb, TILESIZE); y <= perturbUp(yt); y += TILESIZE) {
            if (!bottomTop(zb, zt, x, y))
                continue;

            Coord ijk(Int32(x), Int32(y), Int32(0));
            Vec3T p(x, y, ValueT(0));

            bool tiles_added = false;
            ValueT z = tileCeil(zb, TILESIZE) - 2*TILESIZE;
            while (z <= tileFloor(zt, TILESIZE) + TILESIZE) {
                ijk[2] = Int32(z);
                p[2] = z;

                if (leapFrogToNextTile<NodeT, 1>(ijk, z, acc))
                    continue;

                if (addTile<NodeT>(p, ijk, acc)) tiles_added = true;
                else if (tiles_added) break;
            }
        }
    }

    template <typename NodeT, int dir>
    inline bool
    leapFrogToNextTile(const Coord& ijk, ValueT& z, AccessorT& acc) const
    {
        static const int offset  = NodeT::DIM;
        static const int nodeDepth = int(TreeT::DEPTH - NodeT::LEVEL - 1);

        // we have not encountered an already populated tile
        if (acc.getValue(ijk) != mNegBg) {
            z += dir*offset;
            return false;
        }

        const int depth = acc.getValueDepth(ijk);

        // tile is not larger than current node
        if (depth >= nodeDepth) {
            z += dir*offset;
            return false;
        }

        const ValueT sz = ValueT(mTileSizes[depth]);

        z = dir > 0
            ? sz * ValueT(math::Ceil(z/sz)) + ValueT(0.5)*(offset-1)
            : sz * ValueT(math::Floor(z/sz)) - ValueT(0.5)*(offset+1);

        return true;
    }

    // add negative background tile inside the object if it fits and return true iff it was added
    template<typename NodeT>
    inline bool
    addTile(const Vec3T& p, const Coord& ijk, AccessorT& acc)
    {
        static const Index LEVEL = NodeT::LEVEL + 1;

        if (tileFits<NodeT>(p)) {
            acc.addTile(LEVEL, ijk, mNegBg, false);
            return true;
        } else {
            return false;
        }
    }

    template <typename NodeT>
    inline bool
    tileFits(const Vec3T& p) const
    {
        static const Index TILESIZE = NodeT::DIM;

        static const ValueT R1 = ValueT(0.500)*(TILESIZE-1),
                            R2 = ValueT(0.866)*(TILESIZE-1);

        const ValueT dist = invokeTilePointSignedDistance(p);

        // fast positive criterion: circumsribed ball is in the object
        if (dist <= -R2-mHw)
            return true;

        // fast negative criterion: inscribed ball is not in the object
        else if (dist >= -R1-mHw)
            return false;

        // convexity: the tile is in the object iff all corners are in the object
        return invokeTilePointSignedDistance(p + Vec3T(-R1, -R1, -R1)) < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(-R1, -R1, R1))  < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(-R1, R1, -R1))  < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(-R1, R1, R1))   < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(R1, -R1, -R1))  < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(R1, -R1, R1))   < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(R1, R1, -R1))   < -mHw
            && invokeTilePointSignedDistance(p + Vec3T(R1, R1, R1))    < -mHw;
    }

    inline void
    invokeSetXYRangeData(const Index& step = 1)
    {
        mXYData.clear();
        static_cast<Derived*>(this)->setXYRangeData(step);
    }

    inline bool
    invokeTileCanFit(const Index& dim) const
    {
        return static_cast<const Derived*>(this)->tileCanFit(dim);
    }

    inline ValueT
    invokeSignedDistance(const Vec3T& p) const
    {
        return static_cast<const Derived*>(this)->signedDistance(p);
    }

    inline ValueT
    invokeTilePointSignedDistance(const Vec3T& p) const
    {
        return static_cast<const Derived*>(this)->tilePointSignedDistance(p);
    }

    // misc

    static inline std::vector<int>
    treeTileSizes()
    {
        // iterate over all non root nodes
        using ChainT = typename NodeChainT::PopBack;

        std::vector<int> sizes;
        doTreeTileSizes<ChainT>(sizes);

        return sizes;
    }

    template <typename ChainT>
    static inline void
    doTreeTileSizes(std::vector<int>& sizes)
    {
        using NodeT        = typename ChainT::Back;
        using PoppedChainT = typename ChainT::PopBack;

        sizes.push_back(NodeT::DIM);

        if constexpr (!std::is_same_v<PoppedChainT, openvdb::TypeList<>>) {
            doTreeTileSizes<PoppedChainT>(sizes);
        }
    }

    inline bool
    checkInterrupter()
    {
        if (util::wasInterrupted(mInterrupter)) {
            openvdb::thread::cancelGroupExecution();
            return false;
        }
        return true;
    }

    inline TreeT& getTree() { return mGrid->tree(); }

    // ------------ private nested classes ------------

    /// @brief A class that caches access to the last leaf node.
    /// @tparam TreeT The type of the tree being accessed.
    /// @note This class optimizes repeated accesses to the same
    /// leaf node by caching the last accessed leaf.
    class CacheLastLeafAccessor
    {
        using NodeMaskType = util::NodeMask<LeafT::LOG2DIM>;

    public:

        /// @brief Constructs the CacheLastLeafAccessor and initializes
        /// the internal accessor with the provided tree.
        /// @param tree Reference to the tree being accessed.
        CacheLastLeafAccessor(TreeT& tree)
        : mAcc(tree), mTileSizes(treeTileSizes())
        {
        }

        /// @brief Resets the cache by caching new leaf data for the given coordinates.
        /// @param ijk The coordinate for which to cache the new leaf data.
        inline void reset(const Coord& ijk) { cacheNewLeafData(ijk); }

        /// @brief Checks if the given coordinates are at a new leaf position,
        /// and resets the cache if needed.
        /// @tparam ZDir The direction in the Z-axis (default: 0, other choices are -1 and 1).
        /// @param ijk The coordinate to check and potentially reset.
        template<int ZDir = 0>
        inline void checkReset(const Coord& ijk)
        {
            if (atNewLeafPos<ZDir>(ijk))
                cacheNewLeafData(ijk);
        }

        /// @brief Retrieves the value at the given coordinate,
        /// checking and resetting the cache if needed.
        /// @tparam ZDir The direction in the Z-axis (default: 0, other choices are -1 and 1).
        /// @tparam Check If true, checks if the coordinate is at a new leaf position (default: true).
        /// @param ijk The coordinate for which to retrieve the value.
        /// @return The value at the specified coordinate.
        template<int ZDir = 0, bool Check = true>
        inline ValueT getValue(const Coord& ijk)
        {
            if (Check && atNewLeafPos<ZDir>(ijk))
                cacheNewLeafData(ijk);

            return mLeaf
                ? mLeafData[coordToOffset(ijk)]
                : mAcc.getValue(ijk);
        }

        /// @brief Sets the value at the given coordinates and marks the voxel as active,
        /// checking the cache if needed.
        /// @tparam ZDir The direction in the Z-axis (default: 0, other choices are -1 and 1).
        /// @tparam Check If true, checks if the coordinate is at a new leaf position (default: true).
        /// @param ijk The coordinate where the value is to be set.
        /// @param val The value to set at the specified coordinate.
        template<int ZDir = 0, bool Check = true>
        inline void setValueOn(const Coord& ijk, const ValueT& val)
        {
            if (Check && atNewLeafPos<ZDir>(ijk))
                cacheNewLeafData(ijk);

            if (mLeaf) {
                const Index n = coordToOffset(ijk);
                mLeafData[n] = val;
                mValueMask->setOn(n);
            } else {
                mAcc.setValueOn(ijk, val);
                cacheNewLeafData(ijk);
            }
        }

        /// @brief Sets the value at the given coordinate and marks the voxel as inactive,
        /// checking the cache if needed.
        /// @tparam ZDir The direction in the Z-axis (default: 0, other choices are -1 and 1).
        /// @tparam Check If true, checks if the coordinate is at a new leaf position (default: true).
        /// @param ijk The coordinates where the value is to be set.
        /// @param val The value to set at the specified coordinate.
        template<int ZDir = 0, bool Check = true>
        inline void setValueOff(const Coord& ijk, const ValueT& val)
        {
            if (Check && atNewLeafPos<ZDir>(ijk))
                cacheNewLeafData(ijk);

            if (mLeaf) {
                const Index n = coordToOffset(ijk);
                mLeafData[n] = val;
                mValueMask->setOff(n);
            } else {
                mAcc.setValueOff(ijk, val);
                cacheNewLeafData(ijk);
            }
        }

        /// @brief Return @c true if the value of a voxel resides at the (possibly cached)
        /// leaf level of the tree, i.e., if it is not a tile value.
        /// @tparam ZDir The direction in the Z-axis (default: 0, other choices are -1 and 1).
        /// @tparam Check If true, checks if the coordinate is at a new leaf position (default: true).
        /// @param ijk The coordinate of the voxel to check.
        /// @return True if the voxel exists, otherwise false.
        template<int ZDir = 0, bool Check = true>
        inline bool isVoxel(const Coord& ijk)
        {
            return Check && atNewLeafPos<ZDir>(ijk) ? mLeaf != nullptr : mAcc.isVoxel(ijk);
        }

        /// @brief If the input coordinate lies within a tile,
        /// the input z value is set to the top of the tile bounding box, in index space.
        /// @tparam Check If true, checks if the voxel exists before leaping (default: true).
        /// @param ijk The coordinate to be examined.
        /// @param z The Z-coordinate to be adjusted.
        template<bool Check = true>
        inline void leapUp(const Coord& ijk, ValueT& z)
        {
            if (isVoxel<1,Check>(ijk))
                return;

            const int depth = mAcc.getValueDepth(ijk);
            const ValueT sz = ValueT(mTileSizes[depth]);

            z = sz * ValueT(math::Ceil((z+ValueT(1))/sz)) - ValueT(1);
        }

    private:

        static const Index
            DIM = LeafT::DIM,
            LOG2DIM = LeafT::LOG2DIM;

        template<int ZDir>
        bool atNewLeafPos(const Coord& ijk) const
        {
            if constexpr (ZDir == -1) {
                // assumes value just above has been cached already!
                return (ijk[2] & (DIM-1u)) == DIM-1u;
            } else if constexpr (ZDir == 1) {
                // assumes value just below has been cached already!
                return (ijk[2] & (DIM-1u)) == 0;
            } else {
                return Coord::lessThan(ijk, mOrigin)
                    || Coord::lessThan(mOrigin.offsetBy(DIM-1u), ijk);
            }
        }

        inline void cacheNewLeafData(const Coord& ijk)
        {
            mLeaf = mAcc.probeLeaf(ijk);
            mOrigin = ijk & (~(DIM - 1));

            if (mLeaf) {
                mLeafData = mLeaf->buffer().data();
                mValueMask = &(mLeaf->getValueMask());
            } else {
                mLeafData = nullptr;
            }
        }

        inline static Index coordToOffset(const Coord& ijk)
        {
            return ((ijk[0] & (DIM-1u)) << 2*LOG2DIM)
                +  ((ijk[1] & (DIM-1u)) <<  LOG2DIM)
                +   (ijk[2] & (DIM-1u));
        }

        AccessorT mAcc;
        LeafT*    mLeaf;

        ValueT*       mLeafData;
        NodeMaskType* mValueMask;
        Coord         mOrigin;

        const std::vector<int> mTileSizes;

    }; // class CacheLastLeafAccessor

    // ------------ private members ------------

    GridPtr mGrid;

    const std::vector<int> mTileSizes = treeTileSizes();

    const ValueT mVox, mHw, mBg, mNegBg;

    // misc

    const bool mSerial;

    InterruptType* mInterrupter;

}; // class ConvexVoxelizer


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_CONVEXVOXELIZER_HAS_BEEN_INCLUDED
