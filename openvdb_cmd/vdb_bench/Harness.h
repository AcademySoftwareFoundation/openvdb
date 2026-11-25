// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Harness.h
///
/// @brief Timing, checksums and test grids shared by the vdb_bench groups.

#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VelocityFields.h>

#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace bench {

using namespace openvdb;

////////////////////////////////////////
// Options and timing

struct Options
{
    std::string group;
    int threads = 8;
    int repeat = 3;
    /// How many times every group runs, zero for 1 or, for the group all, 2
    int passes = 0;
    float voxelSize = 1.0f / 1024.0f;
    bool csv = false;
    /// Write every row to this CSV file at the end
    std::string savePath;
    /// Compare with the rows of this CSV file, written by savePath or csv
    std::string baselinePath;
    /// Write the summary to this file, empty for standard output
    std::string summaryPath;
    /// Only run rows whose "tool/name" contains this text
    std::string only;
    /// The dragon.vdb file, empty to skip the rows that need it
    std::string dragonPath;
};

struct Timing
{
    double minimum = 0.0; // milliseconds
    double median = 0.0;
};

template<typename PrepareFunction, typename RunFunction>
Timing
measure(int repeat, PrepareFunction&& prepare, RunFunction&& run)
{
    using Clock = std::chrono::steady_clock;
    {
        auto state = prepare();
        run(state); // warm-up
    }
    std::vector<double> milliseconds;
    for (int pass = 0; pass < repeat; ++pass) {
        auto state = prepare();
        const Clock::time_point start = Clock::now();
        run(state);
        const Clock::time_point end = Clock::now();
        milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    Timing timing;
    timing.minimum = milliseconds.front();
    timing.median = milliseconds[milliseconds.size() / 2];
    return timing;
}

////////////////////////////////////////
// Checksums

/// 64-bit FNV-1a hash of what a tool produced, plus a count and a sum so that
/// a mismatch shows how large it is.
class Digest
{
public:
    template<typename T>
    void add(T value)
    {
        static_assert(std::is_trivially_copyable_v<T>, "digest of a plain value");
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        for (unsigned char byte : bytes) {
            mHash ^= byte;
            mHash *= 1099511628211ull;
        }
    }

    /// For results of parallel reductions, whose last bits depend on the order of the sum
    void addApproximate(double value, int digits = 5)
    {
        char text[32];
        std::snprintf(text, sizeof(text), "%.*e", digits, value);
        for (const char* c = text; *c != '\0'; ++c) this->add(*c);
        mSum += value;
        ++mCount;
    }

    template<typename ValueT>
    void addValue(const ValueT& value)
    {
        if constexpr (std::is_arithmetic_v<ValueT>) {
            this->add(value);
            mSum += double(value);
        } else {
            for (int i = 0; i < int(ValueT::size); ++i) {
                this->add(value[i]);
                mSum += double(value[i]);
            }
        }
    }

    /// Active coordinates and the bits of their values, in iterator order
    template<typename GridT>
    void addGrid(const GridT& grid)
    {
        for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
            const Coord xyz = iter.getCoord();
            this->add(xyz[0]);
            this->add(xyz[1]);
            this->add(xyz[2]);
            this->addValue(*iter);
            ++mCount;
        }
    }

    /// For results that depend on which thread wrote last, where only the size and the
    /// first digits of the sum are stable
    template<typename GridT>
    void addGridCoarse(const GridT& grid)
    {
        double sum = 0.0;
        for (auto iter = grid.cbeginValueOn(); iter; ++iter) sum += double(*iter);
        this->add(grid.activeVoxelCount());
        this->addApproximate(sum, 1);
    }

    template<typename ArrayT>
    void addArray(const ArrayT& array)
    {
        for (const auto& element : array) {
            this->addValue(element);
        }
        mCount += uint64_t(array.size());
    }

    uint64_t hash() const { return mHash; }
    uint64_t count() const { return mCount; }
    double sum() const { return mSum; }

private:
    uint64_t mHash = 14695981039346656037ull;
    uint64_t mCount = 0;
    double mSum = 0.0;
};

struct Mesh
{
    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    std::vector<Vec4I> quads;

    void addTo(Digest& digest) const
    {
        digest.addArray(points);
        digest.addArray(triangles);
        digest.addArray(quads);
    }
};

////////////////////////////////////////
// Rows and reporting

struct Row
{
    std::string group;
    std::string tool;
    std::string name;
    int threads = 0;
    Timing timing;
    Digest digest;

    std::string hashText() const
    {
        char text[32];
        std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(digest.hash()));
        return text;
    }

    /// One line of the CSV file, see csvHeader()
    std::string csvLine() const
    {
        std::ostringstream line;
        line << group << ',' << tool << ',' << name << ',' << threads << ','
            << std::setprecision(8) << timing.minimum << ',' << timing.median << ','
            << hashText() << ',' << digest.count() << ',' << std::setprecision(12) << digest.sum();
        return line.str();
    }

    static const char* csvHeader() { return "group,tool,name,threads,min_ms,median_ms,hash,count,sum"; }
};

class Reporter
{
public:
    explicit Reporter(bool csv): mCsv(csv)
    {
        if (mCsv) std::cout << Row::csvHeader() << std::endl;
    }

    /// Rows are written as they finish, so a run that dies keeps what it had
    void add(const Row& row)
    {
        if (mCsv) {
            std::cout << row.csvLine() << std::endl;
        } else {
            std::ostringstream line;
            line << std::left << std::setw(36) << row.tool << std::setw(44) << row.name << std::right
                << std::fixed << std::setprecision(2) << std::setw(12) << row.timing.minimum << " ms  "
                << row.hashText() << "  " << row.digest.count();
            std::cout << line.str() << std::endl;
        }
    }

private:
    bool mCsv;
};

/// Everything a group needs to time a tool and report it
class Context
{
public:
    /// Every recorded row is also appended to @a rows, for the summary
    Context(const Options& options, std::vector<Row>& rows)
        : mOptions(options), mRows(rows), mReporter(options.csv), mArena(options.threads) {}

    const Options& options() const { return mOptions; }

    bool wanted(const std::string& tool, const std::string& name) const
    {
        return mOptions.only.empty() || (tool + "/" + name).find(mOptions.only) != std::string::npos;
    }

    /// Time run(state) inside the thread arena. prepare() builds a fresh state outside it.
    /// @param repeat  zero means the repeat count of the options
    template<typename PrepareFunction, typename RunFunction>
    Timing time(PrepareFunction&& prepare, RunFunction&& run, int repeat = 0)
    {
        return measure(repeat > 0 ? repeat : mOptions.repeat, prepare,
            [&](auto& state) { mArena.execute([&]() { run(state); }); });
    }

    void record(const std::string& tool, const std::string& name, const Timing& timing,
        const Digest& digest)
    {
        Row row;
        row.group = mOptions.group;
        row.tool = tool;
        std::string clean = name;
        std::replace(clean.begin(), clean.end(), ',', ' ');
        row.name = clean;
        row.threads = mOptions.threads;
        row.timing = timing;
        row.digest = digest;
        if (mProgressShown) {
            std::cerr << "\r\033[K" << std::flush;
            mProgressShown = false;
        }
        mReporter.add(row);
        mRows.push_back(row);
    }

    /// @brief Show which row is running, on a line that record() erases. Nothing is shown when
    /// standard error is not a terminal, so a log has one line per row.
    void progress(const std::string& tool, const std::string& name)
    {
        if (!isTerminal()) return;
        std::cerr << "\r\033[K" << mOptions.group << ": " << tool << " / " << name << std::flush;
        mProgressShown = true;
    }

private:
    static bool isTerminal()
    {
#ifdef _WIN32
        return _isatty(_fileno(stderr)) != 0;
#else
        return isatty(fileno(stderr)) != 0;
#endif
    }

    Options mOptions;
    bool mProgressShown = false;
    std::vector<Row>& mRows;
    Reporter mReporter;
    tbb::task_arena mArena;
};

/// The voxel size as it appears in a row name, for example 1/512
inline std::string
voxelLabel(float voxelSize)
{
    return "1/" + std::to_string(int(std::lround(1.0f / voxelSize)));
}

/// A grid and the text that names it in a row
struct NamedGrid
{
    std::string name;
    FloatGrid::Ptr grid;
};

////////////////////////////////////////
// Grids. All of them are built outside the timed region.

template<typename GridT>
typename GridT::Ptr
makeSphere(float voxelSize, float halfWidth = 3.0f)
{
    return tools::createLevelSetSphere<GridT>(0.15f, Vec3f(0.35f), voxelSize, halfWidth);
}

inline FloatGrid::Ptr
makeIcosahedron(float voxelSize, float halfWidth = 3.0f)
{
    return tools::createLevelSetIcosahedron<FloatGrid>(0.15f, Vec3f(0.35f), voxelSize, halfWidth);
}

inline FloatGrid::Ptr
makeCube(float voxelSize, float halfWidth = 3.0f)
{
    return tools::createLevelSetCube<FloatGrid>(0.25f, Vec3f(0.35f), voxelSize, halfWidth);
}

/// Perturb the distance values so that normalize has work to do
inline FloatGrid::Ptr
distort(FloatGrid::Ptr grid, float voxelSize)
{
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        const float noise = float(((xyz[0] * 7 + xyz[1] * 13 + xyz[2] * 29) % 11) - 5) / 5.0f;
        iter.setValue(*iter + 0.4f * voxelSize * noise);
    }
    return grid;
}

inline FloatGrid::Ptr
makeFogSphere(float voxelSize)
{
    FloatGrid::Ptr grid = makeSphere<FloatGrid>(voxelSize);
    tools::sdfToFogVolume(*grid);
    return grid;
}

/// The unit direction away from the center of the sphere, on the voxels of the sphere.
/// Built from the formula, so it does not depend on the code being measured.
inline Vec3SGrid::Ptr
makeRadialField(const FloatGrid& sphere)
{
    Vec3SGrid::Ptr field = Vec3SGrid::create(Vec3f(0.0f));
    field->setTransform(sphere.transform().copy());
    Vec3SGrid::Accessor accessor = field->getAccessor();
    for (auto iter = sphere.cbeginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        Vec3f direction = Vec3f(sphere.transform().indexToWorld(xyz)) - Vec3f(0.35f);
        const float length = direction.length();
        if (length > 0.0f) direction /= length;
        accessor.setValue(xyz, direction);
    }
    return field;
}

/// The Enright velocity sampled on a dense box that covers the unit cube. The voxel
/// size is fixed so that the cost does not depend on the voxel size of the level set.
inline Vec3SGrid::Ptr
makeVelocityGrid(float voxelSize = 1.0f / 128.0f)
{
    Vec3SGrid::Ptr grid = Vec3SGrid::create(Vec3f(0.0f));
    grid->setTransform(math::Transform::createLinearTransform(voxelSize));
    const int dimension = int(std::lround(1.0f / voxelSize));
    const CoordBBox box(Coord(-4), Coord(dimension + 4));
    grid->tree().fill(box, Vec3f(0.0f), true);
    grid->tree().voxelizeActiveTiles();
    const tools::EnrightField<float> field;
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        iter.setValue(field(grid->transform().indexToWorld(iter.getCoord()), 0.0f));
    }
    return grid;
}

/// A dense box of the given size, active, with smooth varying values
inline FloatGrid::Ptr
makeDenseBox(int dimension, float voxelSize)
{
    FloatGrid::Ptr grid = FloatGrid::create(10.0f);
    grid->setTransform(math::Transform::createLinearTransform(voxelSize));
    grid->setGridClass(GRID_LEVEL_SET);
    grid->tree().fill(CoordBBox(Coord(0), Coord(dimension - 1)), 0.0f, true);
    grid->tree().voxelizeActiveTiles();
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        const Coord xyz = iter.getCoord();
        iter.setValue(std::sin(0.11f * float(xyz[0])) + std::cos(0.07f * float(xyz[1])) +
            0.05f * float(xyz[2]) - 3.0f);
    }
    return grid;
}

/// Read the first float grid of a file.
/// Returns null, with a note on stderr, when there is no path or the file is missing.
inline FloatGrid::Ptr
loadFloatGrid(const std::string& path)
{
    if (path.empty()) return nullptr;
    if (!std::ifstream(path).good()) {
        std::cerr << "skipping " << path << " (not found)" << std::endl;
        return nullptr;
    }
    io::File file(path);
    file.open();
    GridPtrVecPtr all = file.getGrids();
    file.close();
    for (GridBase::Ptr base : *all) {
        if (FloatGrid::Ptr grid = gridPtrCast<FloatGrid>(base)) return grid;
    }
    std::cerr << "skipping " << path << " (no float grid)" << std::endl;
    return nullptr;
}

////////////////////////////////////////
// Groups

void runLevelSet(Context& context);
void runAnalysis(Context& context);
void runOperators(Context& context);
void runFilter(Context& context);
void runPoisson(Context& context);
void runSampling(Context& context);
void runMeshing(Context& context);
void runAccessor(Context& context);

const char* const kGroupNames[] = {
    "levelset", "analysis", "operators", "filter", "poisson", "sampling", "meshing", "accessor"};

} // namespace bench
