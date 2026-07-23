// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pySampler.cc
/// @brief nanobind wrappers for sampling grids

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <memory>
#include <vector>
#include "pyTypeCasters.h"

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace {

class LevelSetSampler
{
public:
    explicit LevelSetSampler(FloatGrid::ConstPtr grid)
        : mImpl(new Impl<FloatGrid>(grid))
    {
    }

#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    explicit LevelSetSampler(DoubleGrid::ConstPtr grid)
        : mImpl(new Impl<DoubleGrid>(grid))
    {
    }
#endif

    double sampleWS(const Vec3d& point) const
    {
        return mImpl->sampleWS(point);
    }

    nb::object sampleBatchWS(nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> points) const
    {
        return mImpl->sampleBatchWS(points);
    }

    nb::object sampleBatchWS(nb::ndarray<double, nb::shape<-1, 3>, nb::device::cpu> points) const
    {
        return mImpl->sampleBatchWS(points);
    }

private:
    struct BaseImpl
    {
        virtual ~BaseImpl() = default;
        virtual double sampleWS(const Vec3d& point) const = 0;
        virtual nb::object sampleBatchWS(
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> points) const = 0;
        virtual nb::object sampleBatchWS(
            nb::ndarray<double, nb::shape<-1, 3>, nb::device::cpu> points) const = 0;
    };

    template<typename GridType>
    struct Impl: public BaseImpl
    {
        using ValueT = typename GridType::ValueType;
        using SamplerT = tools::GridSampler<GridType, tools::BoxSampler>;

        explicit Impl(typename GridType::ConstPtr grid)
            : mGrid(grid)
        {
            if (!mGrid) {
                OPENVDB_THROW(ValueError, "LevelSetSampler requires a non-null grid");
            }
            if (mGrid->getGridClass() != GRID_LEVEL_SET) {
                OPENVDB_THROW(TypeError, "LevelSetSampler requires a level set grid");
            }
        }

        double sampleWS(const Vec3d& point) const override
        {
            SamplerT sampler(*mGrid);
            return static_cast<double>(sampler.wsSample(point));
        }

        nb::object sampleBatchWS(
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> points) const override
        {
            return this->sampleBatch(points);
        }

        nb::object sampleBatchWS(
            nb::ndarray<double, nb::shape<-1, 3>, nb::device::cpu> points) const override
        {
            return this->sampleBatch(points);
        }

        template<typename CoordT>
        nb::object sampleBatch(nb::ndarray<CoordT, nb::shape<-1, 3>, nb::device::cpu> points) const
        {
            SamplerT sampler(*mGrid);
            auto values = new std::vector<ValueT>(points.shape(0));
            for (size_t n = 0, size = points.shape(0); n < size; ++n) {
                (*values)[n] = sampler.wsSample(Vec3d(points(n, 0), points(n, 1), points(n, 2)));
            }

            nb::capsule valuesDeleter(values, [](void* p) noexcept {
                delete static_cast<std::vector<ValueT>*>(p);
            });
            return nb::cast(nb::ndarray<nb::numpy, ValueT>(
                values->data(), {values->size()}, valuesDeleter));
        }

        typename GridType::ConstPtr mGrid;
    };

    std::unique_ptr<BaseImpl> mImpl;
};

} // namespace

void
exportLevelSetSampler(nb::module_ m)
{
    auto cls = nb::class_<LevelSetSampler>(m, "LevelSetSampler",
        "Reusable world-space sampler for level set grids.");

    cls.def(nb::init<FloatGrid::ConstPtr>(), nb::arg("grid"),
            "Construct a sampler for a FloatGrid level set.");

#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    cls.def(nb::init<DoubleGrid::ConstPtr>(), nb::arg("grid"),
            "Construct a sampler for a DoubleGrid level set.");
#endif

    cls.def("sampleWS", &LevelSetSampler::sampleWS, nb::arg("point"),
            "Sample the signed distance value at a world-space point.");

    cls.def("sampleBatchWS",
            [](const LevelSetSampler& sampler,
               nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu> points) {
                return sampler.sampleBatchWS(points);
            },
            nb::arg("points"),
            "Sample signed distance values at an Nx3 array of world-space points.");

    cls.def("sampleBatchWS",
            [](const LevelSetSampler& sampler,
               nb::ndarray<double, nb::shape<-1, 3>, nb::device::cpu> points) {
                return sampler.sampleBatchWS(points);
            },
            nb::arg("points"),
            "Sample signed distance values at an Nx3 array of world-space points.");
}
