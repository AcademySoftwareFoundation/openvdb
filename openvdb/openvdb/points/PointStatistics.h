// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointStatistics.h
///
/// @brief Functions to perform multi threaded reductions and analysis of
///   arbitrary point attribute types. Each function imposes various
///   requirements on the point ValueType (such as expected operators) and
///   supports arbitrary point filters.
///

#ifndef OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/tree/LeafManager.h>

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Evaluates the minimum and maximum values of a point attribute.
/// @details Performs parallel reduction by comparing values using their less
///   than and greater than operators. If the PointDataGrid is empty or the
///   filter evalutes to empty, zeroVal<ValueT>() is returned for both values.
/// @note The ValueT of the attribute must be copy constructible. This method
///   will throw if the templated ValueT does not match the given attribute.
///   For vectors and matrices, this results in per component comparisons.
///   See evalExtents for magnitudes or more custom control.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return min,max value pair
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
std::pair<ValueT, ValueT>
evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the average value of a point attribute.
/// @details Performs parallel reduction by cumulative moving average. The
///   reduction arithmetic and return value precision evaluates to:
///      ConvertElementType<ValueT, double>::Type
///   which, for POD and VDB math types, is ValueT at double precision. If the
///   PointDataGrid is empty or the filter evalutes to empty, zeroVal<ValueT>()
///   is returned.
/// @note The ConvertElementType of the attribute must be copy constructible,
///   support the same type + - * operators and * / operators from a double.
///   This method will throw if ValueT does not match the given attribute. The
///   function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return the average value
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
typename ConvertElementType<ValueT, double>::Type
evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the total value of a point attribute.
/// @details Performs parallel reduction by summing all values. The reduction
///   arithmetic and return value precision evaluates to:
///      PromoteType<ValueT>::Highest
///   which, for POD and VDB math types, is ValueT at its highest bit precision.
///   If the PointDataGrid is empty or the filter evalutes to empty,
///   zeroVal<ValueT>() is returned.
/// @note The PromoteType of the attribute must be copy constructible, support
///   the same type + operator. This method will throw if ValueT does not match
///   the given attribute. The function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param filter     a filter to apply to points
/// @return the total value
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
typename PromoteType<ValueT>::Highest
accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter = NullFilter());

/// @brief Evaluates the minimum and maximum values of a point attribute and
///   returns whether the values are valid. Optionally constructs localised
///   min and max value trees.
/// @details Performs parallel reduction by comparing values using their less
///   than and greater than operators. This method will return true if min and
///   max have been set, false otherwise (when no points existed or a filter
///   evaluated to empty).
/// @note The ValueT of the attribute must also be copy constructible. This
///   method will throw if the templated ValueT does not match the given
///   attribute. For vectors and matrices, this results in per component
///   comparisons. See evalExtents for magnitudes or more custom control.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param min        the computed min value
/// @param max        the computed max value
/// @param filter     a filter to apply to points
/// @param minTree    if provided, builds a tiled tree of localised min results
/// @param maxTree    if provided, builds a tiled tree of localised max results
/// @return true if min and max have been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT>
bool evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    ValueT& min,
    ValueT& max,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* minTree = nullptr,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* maxTree = nullptr);

/// @brief Evaluates the average value of a point attribute and returns whether
///   the value is valid. Optionally constructs localised average value trees.
/// @details Performs parallel reduction by cumulative moving average. The
///   reduction arithmetic and return value precision evaluates to:
///      ConvertElementType<ValueT, double>::Type
///   which, for POD and VDB math types, is ValueT at double precision. This
///   method will return true average has been set, false otherwise (when no
///   points existed or a filter evaluated to empty).
/// @note The ConvertElementType of the attribute must be copy constructible,
///   support the same type + - * operators and * / operators from a double.
///   This method will throw if ValueT does not match the given attribute. The
///   function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param average    the computed averaged value at double precision
/// @param filter     a filter to apply to points
/// @param averageTree  if provided, builds a tiled tree of localised avg results.
/// @return true if average has been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
/// @par Example:
/// @code
///    using namespace openvdb;
///    using namespace openvdb::points
///
///    // average and store per leaf values in a new tree
///    ConvertElementType<uint8_t, double>::Type avg;  // evaluates to double
///    PointDataTree::ValueConverter<decltype(avg)>::Type avgTree; // double tree of averages
///    bool success = evalAverage<uint8_t>(tree, "attrib", avg, NullFilter(), &avgTree);
/// @endcode
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT,
    typename ResultTreeT = typename ConvertElementType<ValueT, double>::Type>
bool evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    typename ConvertElementType<ValueT, double>::Type& average,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* averageTree = nullptr);

/// @brief Evaluates the total value of a point attribute and returns whether
///   the value is valid. Optionally constructs localised total value trees.
/// @details Performs parallel reduction by summing all values. The reduction
///   arithmetic and return value precision evaluates to:
///      PromoteType<ValueT>::Highest
///   which, for POD and VDB math types, is ValueT at its highest bit precision.
///   This method will return true total has been set, false otherwise (when no
///   points existed or a filter evaluated to empty).
/// @note The PromoteType of the attribute must be copy constructible, support
///   the same type + operator. This method will throw if ValueT does not match
///   the given attribute. The function is deterministic.
/// @warning if "P" is provided, the result is undefined.
/// @param points     the point tree
/// @param attribute  the attribute to reduce
/// @param total      the computed total value
/// @param filter     a filter to apply to points
/// @param totalTree  if provided, builds a tiled tree of localised total results.
/// @return true if total has been set, false otherwise. Can be false if
///   no points were processed or if the tree was empty.
/// @par Example:
/// @code
///    using namespace openvdb;
///    using namespace openvdb::points;
///
///    // accumulate and store per leaf values in a new tree
///    PromoteType<uint8_t>::Highest total;  // evaluates to uint64_t
///    PointDataTree::ValueConverter<decltype(total)>::Type totalTree; // uint64_t tree of totals
///    bool success = accumulate<uint8_t>(tree, "attrib", total, NullFilter(), &totalTree);
/// @endcode
template <typename ValueT,
    typename CodecT = UnknownCodec,
    typename FilterT = NullFilter,
    typename PointDataTreeT,
    typename ResultTreeT = typename PromoteType<ValueT>::Highest>
bool accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    typename PromoteType<ValueT>::Highest& total,
    const FilterT& filter = NullFilter(),
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* totalTree = nullptr);

///////////////////////////////////////////////////
///////////////////////////////////////////////////

namespace statistics_internal
{

/// @brief  Scalar extent op to evaluate the min/max values of a
///    single integral or floating point attribute type
template <typename ValueT>
struct ScalarMinMax
{
    using ExtentT = std::pair<ValueT, ValueT>;
    ScalarMinMax(const ValueT& init) : mMinMax(init, init) {}
    ScalarMinMax(const ExtentT& init) : mMinMax(init) {}
    inline void operator()(const ValueT& b)
    {
        mMinMax.first = std::min(mMinMax.first, b);
        mMinMax.second = std::max(mMinMax.second, b);
    }
    inline void operator()(const ExtentT& b)
    {
        mMinMax.first = std::min(mMinMax.first, b.first);
        mMinMax.second = std::max(mMinMax.second, b.second);
    }
    inline const ExtentT& get() const { return mMinMax; }
    ExtentT mMinMax;
};

/// @brief  Vector squared magnitude op to evaluate the min/max of a
///   vector attribute and return the result as a scalar of the
///   appropriate precision
template <typename VecT, bool MagResult = true>
struct MagnitudeExtent
    : public ScalarMinMax<typename ValueTraits<VecT>::ElementType>
{
    using ElementT = typename ValueTraits<VecT>::ElementType;
    using ExtentT = typename ScalarMinMax<ElementT>::ExtentT;
    using BaseT = ScalarMinMax<ElementT>;
    MagnitudeExtent(const VecT& init) : BaseT(init.lengthSqr()) {}
    MagnitudeExtent(const ExtentT& init) : BaseT(init) {}
    inline void operator()(const VecT& b) { this->BaseT::operator()(b.lengthSqr()); }
    inline void operator()(const ExtentT& b) { this->BaseT::operator()(b); }
};

/// @brief  Vector squared magnitude op to evaluate the min/max of a
///   vector attribute and return the result as the original vector
template <typename VecT>
struct MagnitudeExtent<VecT, false>
{
    using ElementT = typename ValueTraits<VecT>::ElementType;
    using ExtentT = std::pair<VecT, VecT>;
    MagnitudeExtent(const VecT& init)
        : mLengths(), mMinMax(init, init) {
        mLengths.first = init.lengthSqr();
        mLengths.second = mLengths.first;
    }
    MagnitudeExtent(const ExtentT& init)
        : mLengths(), mMinMax(init) {
        mLengths.first = init.first.lengthSqr();
        mLengths.second = init.second.lengthSqr();
    }
    inline const ExtentT& get() const { return mMinMax; }
    inline void operator()(const VecT& b)
    {
        const ElementT l = b.lengthSqr();
        if (l < mLengths.first) {
            mLengths.first = l;
            mMinMax.first = b;
        }
        else if (l > mLengths.second) {
            mLengths.second = l;
            mMinMax.second = b;
        }
    }
    inline void operator()(const ExtentT& b)
    {
        ElementT l = b.first.lengthSqr();
        if (l < mLengths.first) {
            mLengths.first = l;
            mMinMax.first = b.first;
        }
        l = b.second.lengthSqr();
        if (l > mLengths.second) {
            mLengths.second = l;
            mMinMax.second = b.second;
        }
    }

    std::pair<ElementT, ElementT> mLengths;
    ExtentT mMinMax;
};

/// @brief  Vector component-wise op to evaluate the min/max of
///   vector components and return the result as a vector of
///   equal size and precision
template <typename VecT>
struct ComponentExtent
{
    using ExtentT = std::pair<VecT, VecT>;
    ComponentExtent(const VecT& init) : mMinMax(init, init) {}
    ComponentExtent(const ExtentT& init) : mMinMax(init) {}
    inline const ExtentT& get() const { return mMinMax; }
    inline void operator()(const VecT& b)
    {
        mMinMax.first = math::minComponent(mMinMax.first, b);
        mMinMax.second =  math::maxComponent(mMinMax.second, b);
    }
    inline void operator()(const ExtentT& b)
    {
        mMinMax.first = math::minComponent(mMinMax.first, b.first);
        mMinMax.second = math::maxComponent(mMinMax.second, b.second);
    }

    ExtentT mMinMax;
};

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename ExtentOp,
    typename PointDataTreeT>
bool evalExtents(const PointDataTreeT& points,
        const std::string& attribute,
        typename ExtentOp::ExtentT& ext,
        const FilterT& filter,
        typename PointDataTreeT::template ValueConverter
            <typename ExtentOp::ExtentT::first_type>::Type* const minTree = nullptr,
        typename PointDataTreeT::template ValueConverter
            <typename ExtentOp::ExtentT::second_type>::Type* const maxTree = nullptr)
{
    static_assert(std::is_base_of<TreeBase, PointDataTreeT>::value,
        "PointDataTreeT in instantiation of evalExtents is not an openvdb Tree type");

    struct ResultType {
        typename ExtentOp::ExtentT ext;
        bool data = false;
    };

    tree::LeafManager<const PointDataTreeT> manager(points);
    if (manager.leafCount() == 0) return false;
    const size_t idx = manager.leaf(0).attributeSet().find(attribute);
    if (idx == AttributeSet::INVALID_POS) return false;

    // track results per leaf for min/max trees
    std::vector<std::unique_ptr<typename ExtentOp::ExtentT>> values;
    if (minTree || maxTree) values.resize(manager.leafCount());

    const ResultType result = tbb::parallel_reduce(manager.leafRange(),
        ResultType(),
        [idx, &filter, &values]
            (const auto& range, ResultType in) -> ResultType
        {
            for (auto leaf = range.begin(); leaf; ++leaf) {
                AttributeHandle<ValueT, CodecT> handle(leaf->constAttributeArray(idx));
                if (handle.size() == 0) continue;
                if (filter.state() == index::ALL) {
                    const size_t size = handle.isUniform() ? 1 : handle.size();
                    ExtentOp op(handle.get(0));
                    for (size_t i = 1; i < size; ++i) {
                        assert(i < size_t(std::numeric_limits<Index>::max()));
                        op(handle.get(Index(i)));
                    }
                    if (!values.empty()) {
                        values[leaf.pos()].reset(new typename ExtentOp::ExtentT(op.get()));
                    }
                    if (in.data) op(in.ext);
                    in.data = true;
                    in.ext = op.get();
                }
                else {
                    auto iter = leaf->beginIndexOn(filter);
                    if (!iter) continue;
                    ExtentOp op(handle.get(*iter));
                    ++iter;
                    for (; iter; ++iter) op(handle.get(*iter));
                    if (!values.empty()) {
                        values[leaf.pos()].reset(new typename ExtentOp::ExtentT(op.get()));
                    }
                    if (in.data) op(in.ext);
                    in.data = true;
                    in.ext = op.get();
                }
            }

            return in;
        },
        [](const ResultType& a, const ResultType& b) -> ResultType {
            if (!b.data) return a;
            if (!a.data) return b;
            ExtentOp op(a.ext); op(b.ext);
            ResultType t;
            t.ext = op.get();
            t.data = true;
            return t;
        });

    // set minmax trees only if a new value was set - if the value
    // hasn't changed, leave it as inactive background (this is
    // only possible if a point leaf exists with no points or if a
    // filter is provided but is not hit for a given leaf)
    if (minTree || maxTree) {
        manager.foreach([minTree, maxTree, &values]
            (const auto& leaf, const size_t idx) {
                const auto& v = values[idx];
                if (v == nullptr) return;
                const Coord& origin = leaf.origin();
                if (minTree) minTree->addTile(1, origin, v->first, true);
                if (maxTree) maxTree->addTile(1, origin, v->second, true);
            }, false);
    }

    if (result.data) ext = result.ext;
    return result.data;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT,
    typename std::enable_if<ValueTraits<ValueT>::IsVec, int>::type = 0>
bool evalExtents(const PointDataTreeT& points,
    const std::string& attribute,
    ValueT& min,
    ValueT& max,
    const FilterT& filter,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* minTree,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* maxTree)
{
    typename ComponentExtent<ValueT>::ExtentT ext;
    const bool s = evalExtents<ValueT, CodecT, FilterT,
        ComponentExtent<ValueT>, PointDataTreeT>
            (points, attribute, ext, filter, minTree, maxTree);
    if (s) min = ext.first, max = ext.second;
    return s;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT,
    typename std::enable_if<!ValueTraits<ValueT>::IsVec, int>::type = 0>
bool evalExtents(const PointDataTreeT& points,
    const std::string& attribute,
    ValueT& min,
    ValueT& max,
    const FilterT& filter,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* minTree,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* maxTree)
{
    typename ScalarMinMax<ValueT>::ExtentT ext;
    const bool s = evalExtents<ValueT, CodecT, FilterT,
        ScalarMinMax<ValueT>, PointDataTreeT>
            (points, attribute, ext, filter, minTree, maxTree);
    if (s) min = ext.first, max = ext.second;
    return s;
}

} // namespace statistics_internal

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT>
bool evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    ValueT& min,
    ValueT& max,
    const FilterT& filter,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* minTree,
    typename PointDataTreeT::template ValueConverter<ValueT>::Type* maxTree)
{
    return statistics_internal::evalExtents<ValueT, CodecT, FilterT, PointDataTreeT>
            (points, attribute, min, max, filter, minTree, maxTree);
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT,
    typename ResultTreeT>
bool evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    typename ConvertElementType<ValueT, double>::Type& average,
    const FilterT& filter,
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* averageTree)
{
    using ResultT = typename ConvertElementType<ValueT, double>::Type;

    struct Sample
    {
        Sample(const ResultT& _avg, size_t _size) : avg(_avg), size(_size) {}

        void add(const ResultT& val)
        {
            ++size;
            const ResultT delta = val - avg;
            avg = avg + (delta / static_cast<double>(size));
        }

        void add(const Sample& other)
        {
            assert(other.size > 0);
            const double denom = 1.0 / static_cast<double>(size + other.size);
            const ResultT delta = other.avg - avg;
            avg = avg + (denom * delta * static_cast<double>(other.size));
            size += other.size;
        }

        ResultT avg; size_t size;
    };

    static_assert(std::is_base_of<TreeBase, PointDataTreeT>::value,
        "PointDataTreeT in instantiation of evalAverage is not an openvdb Tree type");
    static_assert(std::is_constructible<ResultT, ValueT>::value,
        "Target value in points::evalAverage is not constructible from the source value type.");

    tree::LeafManager<const PointDataTreeT> manager(points);
    if (manager.leafCount() == 0) return false;
    const size_t idx = manager.leaf(0).attributeSet().find(attribute);
    if (idx == AttributeSet::INVALID_POS) return false;

    std::vector<std::unique_ptr<Sample>> values;
    values.resize(manager.leafCount());
    tbb::parallel_for(manager.leafRange(),
        [idx, &filter, &values] (const auto& range) {
            for (auto leaf = range.begin(); leaf; ++leaf) {
                AttributeHandle<ValueT, CodecT> handle(leaf->constAttributeArray(idx));
                size_t size = handle.size();
                if (size == 0) continue;
                if (filter.state() == index::ALL) {
                    std::unique_ptr<Sample> S(new Sample(ResultT(handle.get(0)), 1));
                    if (handle.isUniform()) {
                        S->avg = S->avg / static_cast<double>(size);
                        S->size = size;
                    }
                    else {
                        for (size_t i = 1; i < size; ++i) {
                            assert(i < size_t(std::numeric_limits<Index>::max()));
                            S->add(ResultT(handle.get(Index(i))));
                        }
                    }
                    values[leaf.pos()] = std::move(S);
                }
                else {
                    auto iter = leaf->beginIndexOn(filter);
                    if (!iter) continue;
                    std::unique_ptr<Sample> S(new Sample(ResultT(handle.get(*iter)), 1));
                    ++iter;
                    for (; iter; ++iter, ++size) {
                        S->add(ResultT(handle.get(*iter)));
                    }
                    values[leaf.pos()] = std::move(S);
                }
            }
        });

    auto iter = values.cbegin();
    while (iter != values.cend() && !(*iter)) ++iter;
    if (iter == values.cend()) return false;
    assert(*iter);

    // serial deterministic reduction of floating point samples
    Sample S = **iter;
    ++iter;
    for (; iter != values.cend(); ++iter) {
        if (*iter) S.add(**iter);
    }
    average = S.avg;

    // set average tree only if a new value was set - if the value
    // hasn't changed, leave it as inactive background (this is
    // only possible if a point leaf exists with no points or if a
    // filter is provided but is not hit for a given leaf)
    if (averageTree) {
        manager.foreach([averageTree, &values]
            (const auto& leaf, const size_t idx) {
                const auto& S = values[idx];
                if (S == nullptr) return;
                const Coord& origin = leaf.origin();
                averageTree->addTile(1, origin, S->avg, true);
            }, false);
    }

    return true;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT,
    typename ResultTreeT>
bool accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    typename PromoteType<ValueT>::Highest& total,
    const FilterT& filter,
    typename PointDataTreeT::template ValueConverter<ResultTreeT>::Type* totalTree)
{
    using ResultT = typename PromoteType<ValueT>::Highest;
    using ElementT = typename ValueTraits<ResultT>::ElementType;

    static_assert(std::is_base_of<TreeBase, PointDataTreeT>::value,
        "PointDataTreeT in instantiation of accumulate is not an openvdb Tree type");
    static_assert(std::is_constructible<ResultT, ValueT>::value,
        "Target value in points::accumulate is not constructible from the source value type.");

    tree::LeafManager<const PointDataTreeT> manager(points);
    if (manager.leafCount() == 0) return false;
    const size_t idx = manager.leaf(0).attributeSet().find(attribute);
    if (idx == AttributeSet::INVALID_POS) return false;

    std::vector<std::unique_ptr<ResultT>> values;
    values.resize(manager.leafCount());
    tbb::parallel_for(manager.leafRange(),
        [idx, &filter, &values](const auto& range) {
            for (auto leaf = range.begin(); leaf; ++leaf) {
                AttributeHandle<ValueT, CodecT> handle(leaf->constAttributeArray(idx));
                if (handle.size() == 0) continue;
                if (filter.state() == index::ALL) {
                    const size_t size = handle.isUniform() ? 1 : handle.size();
                    auto total = ResultT(handle.get(0));
                    for (size_t i = 1; i < size; ++i) {
                        assert(i < size_t(std::numeric_limits<Index>::max()));
                        total += ResultT(handle.get(Index(i)));
                    }
                    values[leaf.pos()].reset(new ResultT(total));
                }
                else {
                    auto iter = leaf->beginIndexOn(filter);
                    if (!iter) continue;
                    auto total = ResultT(handle.get(*iter));
                    ++iter;
                    for (; iter; ++iter) total += ResultT(handle.get(*iter));
                    values[leaf.pos()].reset(new ResultT(total));
                }
            }
        });

    auto iter = values.cbegin();
    while (iter != values.cend() && !(*iter)) ++iter;
    if (iter == values.cend()) return false;
    assert(*iter);
    total = **iter; ++iter;

    if (std::is_integral<ElementT>::value) {
        using RangeT = tbb::blocked_range<const std::unique_ptr<ResultT>*>;
        // reasonable grain size for accumulation of single to matrix types
        total = tbb::parallel_reduce(RangeT(&(*iter), (&values.back())+1, 32), total,
           [](const RangeT& range, ResultT p) -> ResultT {
                for (const auto& r : range) if (r) p += *r;
                return p;
           }, std::plus<ResultT>());
    }
    else {
        for (; iter != values.cend(); ++iter) {
            if (*iter) total += (**iter);
        }
    }

    // set total tree only if a new value was set - if the value
    // hasn't changed, leave it as inactive background (this is
    // only possible if a point leaf exists with no points or if a
    // filter is provided but is not hit for a given leaf)
    if (totalTree) {
        manager.foreach([totalTree, &values]
            (const auto& leaf, const size_t idx) {
                const auto& v = values[idx];
                if (v == nullptr) return;
                const Coord& origin = leaf.origin();
                totalTree->addTile(1, origin, *v, true);
            }, false);
    }

    return true;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT>
std::pair<ValueT, ValueT>
evalMinMax(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter)
{
    std::pair<ValueT, ValueT> results {
        zeroVal<ValueT>(), zeroVal<ValueT>()
    };
    evalMinMax<ValueT, CodecT, FilterT, PointDataTreeT>
        (points, attribute, results.first, results.second, filter);
    return results;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT>
typename ConvertElementType<ValueT, double>::Type
evalAverage(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter)
{
    using ConvertedT = typename ConvertElementType<ValueT, double>::Type;
    ConvertedT result = zeroVal<ConvertedT>();
    evalAverage<ValueT, CodecT, FilterT, PointDataTreeT>(points, attribute, result, filter);
    return result;
}

template <typename ValueT,
    typename CodecT,
    typename FilterT,
    typename PointDataTreeT>
typename PromoteType<ValueT>::Highest
accumulate(const PointDataTreeT& points,
    const std::string& attribute,
    const FilterT& filter)
{
    using PromotedT = typename PromoteType<ValueT>::Highest;
    PromotedT result = zeroVal<PromotedT>();
    accumulate<ValueT, CodecT, FilterT, PointDataTreeT>(points, attribute, result, filter);
    return result;
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_STATISTICS_HAS_BEEN_INCLUDED
