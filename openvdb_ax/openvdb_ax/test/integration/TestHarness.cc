// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"
#include "util.h"

#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>

#include <openvdb/points/PointConversion.h>

namespace unittest_util
{

std::string loadText(const std::string& codeFileName)
{
    std::ostringstream sstream;
    std::ifstream fs(codeFileName);

    if (fs.fail()) {
        throw std::runtime_error(std::string("Failed to open ") + std::string(codeFileName));
    }

    sstream << fs.rdbuf();
    return sstream.str();
}

bool wrapExecution(openvdb::points::PointDataGrid& grid,
                   const std::string& codeFileName,
                   const std::string * const group,
                   openvdb::ax::Logger& logger,
                   const openvdb::ax::CustomData::Ptr& data,
                   const openvdb::ax::CompilerOptions& opts,
                   const bool createMissing)
{
    using namespace openvdb::ax;

    Compiler compiler(opts);
    const std::string code = loadText(codeFileName);
    ast::Tree::ConstPtr syntaxTree = ast::parse(code.c_str(), logger);
    PointExecutable::Ptr executable = compiler.compile<PointExecutable>(*syntaxTree, logger, data);
    if (!executable) return false;
    executable->setCreateMissing(createMissing);
    if (group) executable->setGroupExecution(*group);
    executable->execute(grid);
    return true;
}

bool wrapExecution(openvdb::GridPtrVec& grids,
                   const std::string& codeFileName,
                   openvdb::ax::Logger& logger,
                   const openvdb::ax::CustomData::Ptr& data,
                   const openvdb::ax::CompilerOptions& opts,
                   const bool createMissing)
{
    using namespace openvdb::ax;

    Compiler compiler(opts);
    const std::string code = loadText(codeFileName);

    ast::Tree::ConstPtr syntaxTree = ast::parse(code.c_str(), logger);
    VolumeExecutable::Ptr executable = compiler.compile<VolumeExecutable>(*syntaxTree, logger, data);
    if (!executable) return false;
    executable->setCreateMissing(createMissing);
    executable->setValueIterator(VolumeExecutable::IterType::ON);
    executable->execute(grids);
    return true;
}

void AXTestHarness::addInputGroups(const std::vector<std::string> &names,
                                   const std::vector<bool> &defaults)
{
    for (size_t i = 0; i < names.size(); i++) {
        for (auto& grid : mInputPointGrids) {
            openvdb::points::appendGroup(grid->tree(), names[i]);
            openvdb::points::setGroup(grid->tree(), names[i], defaults[i]);
        }
    }
}

void AXTestHarness::addExpectedGroups(const std::vector<std::string> &names,
                                      const std::vector<bool> &defaults)
{
    for (size_t i = 0; i < names.size(); i++) {
        for (auto& grid : mOutputPointGrids) {
            openvdb::points::appendGroup(grid->tree(), names[i]);
            openvdb::points::setGroup(grid->tree(), names[i], defaults[i]);
        }
    }
}

bool AXTestHarness::executeCode(const std::string& codeFile,
                                const std::string* const group,
                                const bool createMissing)
{
    bool success = false;
    if (mUsePoints) {
        for (auto& grid : mInputPointGrids) {
            mLogger.clear();
            success = wrapExecution(*grid, codeFile, group, mLogger, mCustomData, mOpts, createMissing);
            if (!success) break;
        }
    }

    if (mUseVolumes) {
        mLogger.clear();
        success = wrapExecution(mInputVolumeGrids, codeFile, mLogger, mCustomData, mOpts, createMissing);
    }
    return success;
}

template <typename T>
void AXTestHarness::addInputPtAttributes(const std::vector<std::string>& names,
                          const std::vector<T>& values)
{
    for (size_t i = 0; i < names.size(); i++) {
        for (auto& grid : mInputPointGrids) {
            openvdb::points::appendAttribute<T>(grid->tree(), names[i], values[i]);
       }
    }
}

template <typename T>
void AXTestHarness::addInputVolumes(const std::vector<std::string>& names,
                     const std::vector<T>& values)
{
    using GridType = typename openvdb::BoolGrid::ValueConverter<T>::Type;

    for (size_t i = 0; i < names.size(); i++) {
        typename GridType::Ptr grid = GridType::create();
        grid->denseFill(mVolumeBounds, values[i], true/*active*/);
        grid->setName(names[i]);
        mInputVolumeGrids.emplace_back(grid);
    }
}

template <typename T>
void AXTestHarness::addExpectedPtAttributes(const std::vector<std::string>& names,
                             const std::vector<T>& values)
{
    for (size_t i = 0; i < names.size(); i++) {
        for (auto& grid : mOutputPointGrids) {
            openvdb::points::appendAttribute<T>(grid->tree(), names[i], values[i]);
       }
    }
}

template <typename T>
void AXTestHarness::addExpectedVolumes(const std::vector<std::string>& names,
                        const std::vector<T>& values)
{
    using GridType = typename openvdb::BoolGrid::ValueConverter<T>::Type;

    for (size_t i = 0; i < names.size(); i++) {
        typename GridType::Ptr grid = GridType::create();
        grid->denseFill(mVolumeBounds, values[i], true/*active*/);
        grid->setName(names[i] + "_expected");
        mOutputVolumeGrids.emplace_back(grid);
    }
}

bool AXTestHarness::checkAgainstExpected(std::ostream& sstream)
{
    unittest_util::ComparisonSettings settings;
    bool success = true;

    if (mUsePoints) {
        std::stringstream resultStream;
        unittest_util::ComparisonResult result(resultStream);

        const size_t count = mInputPointGrids.size();
        for (size_t i = 0; i < count; ++i) {
            const auto& input = mInputPointGrids[i];
            const auto& expected = mOutputPointGrids[i];
            const bool pass =
                unittest_util::compareGrids(result, *expected, *input, settings, nullptr);
            if (!pass) sstream << resultStream.str() << std::endl;
            success &= pass;
        }
    }

    if (mUseVolumes) {
        for (size_t i = 0; i < mInputVolumeGrids.size(); i++) {
            std::stringstream resultStream;
            unittest_util::ComparisonResult result(resultStream);
            const bool volumeSuccess =
                unittest_util::compareUntypedGrids(result, *mOutputVolumeGrids[i],
                    *mInputVolumeGrids[i], settings, nullptr);
            success &= volumeSuccess;
            if (!volumeSuccess)  sstream << resultStream.str() << std::endl;
        }
    }

    return success;
}

void AXTestHarness::testVolumes(const bool enable)
{
    mUseVolumes = enable;
}

void AXTestHarness::testPoints(const bool enable)
{
    mUsePoints = enable;
}

void AXTestHarness::reset(const openvdb::Index64 ppv, const openvdb::CoordBBox& bounds)
{
    using openvdb::points::PointDataGrid;
    using openvdb::points::NullCodec;

    mInputPointGrids.clear();
    mOutputPointGrids.clear();
    mInputVolumeGrids.clear();
    mOutputVolumeGrids.clear();

    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(1.0);
    openvdb::MaskGrid::Ptr mask = openvdb::MaskGrid::create();
    mask->setTransform(transform);
    mask->sparseFill(bounds, true, true);
    openvdb::points::PointDataGrid::Ptr points =
        openvdb::points::denseUniformPointScatter(*mask, static_cast<float>(ppv));
    mask.reset();

    mInputPointGrids.emplace_back(points);
    mOutputPointGrids.emplace_back(points->deepCopy());
    mOutputPointGrids.back()->setName("custom_expected");

    mVolumeBounds = bounds;

    mLogger.clear();
}

void AXTestHarness::reset()
{
    using openvdb::points::PointDataGrid;
    using openvdb::points::NullCodec;

    mInputPointGrids.clear();
    mOutputPointGrids.clear();
    mInputVolumeGrids.clear();
    mOutputVolumeGrids.clear();

    std::vector<openvdb::Vec3d> coordinates =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    openvdb::math::Transform::Ptr transform1 =
        openvdb::math::Transform::createLinearTransform(1.0);

    openvdb::points::PointDataGrid::Ptr onePointGrid =
        openvdb::points::createPointDataGrid<NullCodec, PointDataGrid>
            (std::vector<openvdb::Vec3d>{coordinates[0]}, *transform1);

    onePointGrid->setName("1_point");
    mInputPointGrids.emplace_back(onePointGrid);
    mOutputPointGrids.emplace_back(onePointGrid->deepCopy());
    mOutputPointGrids.back()->setName("1_point_expected");

    openvdb::math::Transform::Ptr transform2 =
        openvdb::math::Transform::createLinearTransform(0.1);

    openvdb::points::PointDataGrid::Ptr fourPointGrid =
        openvdb::points::createPointDataGrid<NullCodec, PointDataGrid>
            (coordinates, *transform2);

    fourPointGrid->setName("4_points");
    mInputPointGrids.emplace_back(fourPointGrid);
    mOutputPointGrids.emplace_back(fourPointGrid->deepCopy());
    mOutputPointGrids.back()->setName("4_points_expected");

    mVolumeBounds = openvdb::CoordBBox({0,0,0}, {0,0,0});

    mLogger.clear();
}

template <typename ValueT>
using ConverterT = typename openvdb::BoolGrid::ValueConverter<ValueT>::Type;

void AXTestHarness::resetInputsToZero()
{
    for (auto& grid : mInputPointGrids) {
        openvdb::tree::LeafManager<openvdb::points::PointDataTree> manager(grid->tree());
        manager.foreach([](openvdb::points::PointDataTree::LeafNodeType& leaf, size_t) {
            const size_t attrs = leaf.attributeSet().size();
            const size_t pidx = leaf.attributeSet().descriptor().find("P");
            for (size_t idx = 0; idx < attrs; ++idx) {
                if (idx == pidx) continue;
                leaf.attributeArray(idx).collapse();
            }
        });
    }

    /// @todo: share with volume executable when the move to header files is made
    ///        for customization of grid types.
    using SupportedTypeList = openvdb::TypeList<
        ConverterT<double>,
        ConverterT<float>,
        ConverterT<int64_t>,
        ConverterT<int32_t>,
        ConverterT<int16_t>,
        ConverterT<bool>,
        ConverterT<openvdb::math::Vec2<double>>,
        ConverterT<openvdb::math::Vec2<float>>,
        ConverterT<openvdb::math::Vec2<int32_t>>,
        ConverterT<openvdb::math::Vec3<double>>,
        ConverterT<openvdb::math::Vec3<float>>,
        ConverterT<openvdb::math::Vec3<int32_t>>,
        ConverterT<openvdb::math::Vec4<double>>,
        ConverterT<openvdb::math::Vec4<float>>,
        ConverterT<openvdb::math::Vec4<int32_t>>,
        ConverterT<openvdb::math::Mat3<double>>,
        ConverterT<openvdb::math::Mat3<float>>,
        ConverterT<openvdb::math::Mat4<double>>,
        ConverterT<openvdb::math::Mat4<float>>,
        ConverterT<std::string>>;

    for (auto& grid : mInputVolumeGrids) {
        const bool success = grid->apply<SupportedTypeList>([](auto& typed) {
            using GridType = typename std::decay<decltype(typed)>::type;
            openvdb::tree::LeafManager<typename GridType::TreeType> manager(typed.tree());
            manager.foreach([](typename GridType::TreeType::LeafNodeType& leaf, size_t) {
                leaf.fill(openvdb::zeroVal<typename GridType::ValueType>());
            });
        });
        if (!success) {
            throw std::runtime_error("Unable to reset input grid of an unsupported type");
        }
    }
}


#define REGISTER_HARNESS_METHODS(T) \
template void AXTestHarness::addInputPtAttributes<T>(const std::vector<std::string>&, const std::vector<T>&); \
template void AXTestHarness::addInputVolumes<T>(const std::vector<std::string>&, const std::vector<T>&); \
template void AXTestHarness::addExpectedPtAttributes<T>(const std::vector<std::string>&, const std::vector<T>&); \
template void AXTestHarness::addExpectedVolumes<T>(const std::vector<std::string>&, const std::vector<T>&);

REGISTER_HARNESS_METHODS(double)
REGISTER_HARNESS_METHODS(float)
REGISTER_HARNESS_METHODS(int64_t)
REGISTER_HARNESS_METHODS(int32_t)
REGISTER_HARNESS_METHODS(int16_t)
REGISTER_HARNESS_METHODS(bool)
REGISTER_HARNESS_METHODS(openvdb::math::Vec2<double>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec2<float>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec2<int32_t>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec3<double>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec3<float>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec3<int32_t>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec4<double>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec4<float>)
REGISTER_HARNESS_METHODS(openvdb::math::Vec4<int32_t>)
REGISTER_HARNESS_METHODS(openvdb::math::Mat3<double>)
REGISTER_HARNESS_METHODS(openvdb::math::Mat3<float>)
REGISTER_HARNESS_METHODS(openvdb::math::Mat4<double>)
REGISTER_HARNESS_METHODS(openvdb::math::Mat4<float>)
REGISTER_HARNESS_METHODS(std::string)

}


