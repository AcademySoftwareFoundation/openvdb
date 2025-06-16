// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/fileformat/ReaderE57.cpp

    \author Petra Hapalova
    \brief
*/

#include "ReaderE57.h"

#include <E57SimpleData.h>

#include <functional>
#include <memory>
#include <string>
#include <cassert>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <mutex>

namespace pnanovdb_fileformat
{
static constexpr size_t BUFFER_SIZE = 1024;

using Uint8_getter = std::function<uint8_t(int)>;
using Float_getter = std::function<float(int)>;
using Double_getter = std::function<double(int)>;

template <class T, class Compare>
constexpr const T& my_clamp(const T& v, const T& lo, const T& hi, Compare comp)
{
    assert(!comp(hi, lo));
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template <class T>
constexpr const T& my_clamp(const T& v, const T& lo, const T& hi)
{
    return my_clamp(v, lo, hi, std::less<T>{});
}

//-----------------------------------------------------------------------------
template <typename T>
struct Getter;

//-----------------------------------------------------------------------------
template <>
struct Getter<uint8_t>
{
    static Uint8_getter create(uint8_t const val)
    {
        return [val](int i) -> uint8_t { return val; };
    }

    static Uint8_getter create(std::shared_ptr<std::vector<double> const> doubles, double const offset, double const range)
    {
        return [doubles, offset, range](int i) -> uint8_t
        {
            auto const d = doubles->at(i);
            return static_cast<uint8_t>(my_clamp((d - offset) / range * 255.0, 0.0, 255.0));
        };
    }

    static Uint8_getter create(std::shared_ptr<std::vector<int64_t> const> i64s, int64_t const, int64_t const)
    {
        return [i64s](int i) -> uint8_t
        {
            int64_t const i64 = i64s->at(i);
            constexpr int64_t imin = 0;
            constexpr int64_t imax = 255;
            return static_cast<uint8_t>(my_clamp(i64, imin, imax));
        };
    }
};

//-----------------------------------------------------------------------------
template <>
struct Getter<float>
{
    static Float_getter create(float const val)
    {
        return [val](int i) -> float { return val; };
    }

    static Float_getter create(std::shared_ptr<std::vector<double> const> doubles, double const offset, double const range)
    {
        return [doubles, offset, range](int i) -> float
        {
            auto const d = doubles->at(i);
            return static_cast<float>((d - offset) / range);
        };
    }

    static Float_getter create(std::shared_ptr<std::vector<int64_t> const> i64s,
                               int64_t const min_val,
                               int64_t const max_val)
    {
        return [i64s, min_val, max_val](int i) -> float
        {
            auto const i64 = i64s->at(i);
            if (i64 <= min_val)
            {
                return 0.0;
            }
            else if (i64 >= max_val)
            {
                return 1.0;
            }
            else
            {
                return static_cast<float>(i64 - min_val) / static_cast<float>(max_val - min_val);
            }
        };
    }
};
//-----------------------------------------------------------------------------
template <>
struct Getter<double>
{
    static Double_getter create(double const val)
    {
        return [val](int i) -> double { return val; };
    }

    static Double_getter create(std::shared_ptr<std::vector<double> const> doubles, double const offset, double const range)
    {
        return [doubles, offset, range](int i) -> double { return (doubles->at(i) - offset) / range; };
    }

    static Double_getter create(std::shared_ptr<std::vector<int64_t> const> i64s,
                                int64_t const min_val,
                                int64_t const max_val)
    {
        return [i64s, min_val, max_val](int i) -> double
        {
            auto const i64 = i64s->at(i);
            if (i64 <= min_val)
            {
                return 0.0;
            }
            else if (i64 >= max_val)
            {
                return 1.0;
            }
            else
            {
                return static_cast<double>(i64 - min_val) / static_cast<double>(max_val - min_val);
            }
        };
    }

    static Double_getter create_cartesian_x(std::function<double(int)> r,
                                            std::function<double(int)> theta,
                                            std::function<double(int)> phi)
    {
        return [r, theta, phi](int i) -> double { return r(i) * cos(phi(i)) * cos(theta(i)); };
    }

    static Double_getter create_cartesian_y(std::function<double(int)> r,
                                            std::function<double(int)> theta,
                                            std::function<double(int)> phi)
    {
        return [r, theta, phi](int i) -> double { return r(i) * cos(phi(i)) * sin(theta(i)); };
    }

    static Double_getter create_cartesian_z(std::function<double(int)> r, std::function<double(int)> phi)
    {
        return [r, phi](int i) -> double { return r(i) * sin(phi(i)); };
    }
};

//-----------------------------------------------------------------------------
using Point_transformer = std::function<std::array<float, 3>(double, double, double)>;

std::array<float, 3> identity_transformer(double x, double y, double z)
{
    return std::array<float, 3>{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };
}

using Affine_transformation = double[3][4];
std::array<float, 3> affine_transformer(Affine_transformation const& mat, double const x, double const y, double const z)
{
    return std::array<float, 3> {
        static_cast<float>(mat[0][0] * x + mat[0][1] * y +
                           mat[0][2] * z + mat[0][3]),
        static_cast<float>(mat[1][0] * x + mat[1][1] * y +
                           mat[1][2] * z + mat[1][3]),
        static_cast<float>(mat[2][0] * x + mat[2][1] * y +
                           mat[2][2] * z + mat[2][3]) };
}

//-----------------------------------------------------------------------------

class Scan_context
{
private:
    const e57::ImageFile& image_file;
    e57::StructureNode scan;
    e57::CompressedVectorNode points;
    e57::StructureNode prototype;
    e57::IntensityLimits intensityLimits;
    e57::ColorLimits colorLimits;
    e57::CartesianBounds cartesianBounds;
    bool has_bounds = false;

    std::vector<std::shared_ptr<std::vector<double>>> double_buffers;
    std::vector<std::shared_ptr<std::vector<int64_t>>> i64_buffers;
    std::vector<e57::SourceDestBuffer> source_dest_buffers;

    void read_intensity_limits()
    {
        if (scan.isDefined("intensityLimits"))
        {
            e57::StructureNode intbox(scan.get("intensityLimits"));

            switch (intbox.get("intensityMaximum").type())
            {
            case e57::TypeScaledInteger:
            {
                intensityLimits.intensityMaximum = e57::ScaledIntegerNode(intbox.get("intensityMaximum")).scaledValue();
                intensityLimits.intensityMinimum = e57::ScaledIntegerNode(intbox.get("intensityMinimum")).scaledValue();
                break;
            }
            case e57::TypeFloat:
            {
                intensityLimits.intensityMaximum = e57::FloatNode(intbox.get("intensityMaximum")).value();
                intensityLimits.intensityMinimum = e57::FloatNode(intbox.get("intensityMinimum")).value();
                break;
            }
            case e57::TypeInteger:
            {
                intensityLimits.intensityMaximum = (double)e57::IntegerNode(intbox.get("intensityMaximum")).value();
                intensityLimits.intensityMinimum = (double)e57::IntegerNode(intbox.get("intensityMinimum")).value();
                break;
            }
            default:
                throw std::runtime_error{"intensityMaximum has unknown type"};
            }
        }
    }

    void read_color_limits()
    {
        if (scan.isDefined("colorLimits"))
        {
            e57::StructureNode colorbox(scan.get("colorLimits"));

            switch (colorbox.get("colorRedMaximum").type())
            {
            case e57::TypeScaledInteger:
            {
                colorLimits.colorRedMaximum = e57::ScaledIntegerNode(colorbox.get("colorRedMaximum")).scaledValue();
                colorLimits.colorRedMinimum = e57::ScaledIntegerNode(colorbox.get("colorRedMinimum")).scaledValue();
                colorLimits.colorGreenMaximum = e57::ScaledIntegerNode(colorbox.get("colorGreenMaximum")).scaledValue();
                colorLimits.colorGreenMinimum = e57::ScaledIntegerNode(colorbox.get("colorGreenMinimum")).scaledValue();
                colorLimits.colorBlueMaximum = e57::ScaledIntegerNode(colorbox.get("colorBlueMaximum")).scaledValue();
                colorLimits.colorBlueMinimum = e57::ScaledIntegerNode(colorbox.get("colorBlueMinimum")).scaledValue();
                break;
            }
            case e57::TypeFloat:
            {
                colorLimits.colorRedMaximum = e57::FloatNode(colorbox.get("colorRedMaximum")).value();
                colorLimits.colorRedMinimum = e57::FloatNode(colorbox.get("colorRedMinimum")).value();
                colorLimits.colorGreenMaximum = e57::FloatNode(colorbox.get("colorGreenMaximum")).value();
                colorLimits.colorGreenMinimum = e57::FloatNode(colorbox.get("colorGreenMinimum")).value();
                colorLimits.colorBlueMaximum = e57::FloatNode(colorbox.get("colorBlueMaximum")).value();
                colorLimits.colorBlueMinimum = e57::FloatNode(colorbox.get("colorBlueMinimum")).value();
                break;
            }
            case e57::TypeInteger:
            {
                colorLimits.colorRedMaximum = (double)e57::IntegerNode(colorbox.get("colorRedMaximum")).value();
                colorLimits.colorRedMinimum = (double)e57::IntegerNode(colorbox.get("colorRedMinimum")).value();
                colorLimits.colorGreenMaximum = (double)e57::IntegerNode(colorbox.get("colorGreenMaximum")).value();
                colorLimits.colorGreenMinimum = (double)e57::IntegerNode(colorbox.get("colorGreenMinimum")).value();
                colorLimits.colorBlueMaximum = (double)e57::IntegerNode(colorbox.get("colorBlueMaximum")).value();
                colorLimits.colorBlueMinimum = (double)e57::IntegerNode(colorbox.get("colorBlueMinimum")).value();
                break;
            }
            default:
                throw std::runtime_error("colorRedMaximum has unknown type");
            }
        }
    }

    bool read_cartesian_bounds()
    {
        if (scan.isDefined("cartesianBounds"))
        {
            e57::StructureNode bbox(scan.get("cartesianBounds"));

            switch (bbox.get("xMinimum").type())
            {
            case e57::TypeScaledInteger:
            {
                cartesianBounds.xMinimum = e57::ScaledIntegerNode(bbox.get("xMinimum")).scaledValue();
                cartesianBounds.xMaximum = e57::ScaledIntegerNode(bbox.get("xMaximum")).scaledValue();
                cartesianBounds.yMinimum = e57::ScaledIntegerNode(bbox.get("yMinimum")).scaledValue();
                cartesianBounds.yMaximum = e57::ScaledIntegerNode(bbox.get("yMaximum")).scaledValue();
                cartesianBounds.zMinimum = e57::ScaledIntegerNode(bbox.get("zMinimum")).scaledValue();
                cartesianBounds.zMaximum = e57::ScaledIntegerNode(bbox.get("zMaximum")).scaledValue();
                break;
            }
            case e57::TypeFloat:
            {
                cartesianBounds.xMinimum = e57::FloatNode(bbox.get("xMinimum")).value();
                cartesianBounds.xMaximum = e57::FloatNode(bbox.get("xMaximum")).value();
                cartesianBounds.yMinimum = e57::FloatNode(bbox.get("yMinimum")).value();
                cartesianBounds.yMaximum = e57::FloatNode(bbox.get("yMaximum")).value();
                cartesianBounds.zMinimum = e57::FloatNode(bbox.get("zMinimum")).value();
                cartesianBounds.zMaximum = e57::FloatNode(bbox.get("zMaximum")).value();
                break;
            }
            case e57::TypeInteger:
            {
                cartesianBounds.xMinimum = (double)e57::IntegerNode(bbox.get("xMinimum")).value();
                cartesianBounds.xMaximum = (double)e57::IntegerNode(bbox.get("xMaximum")).value();
                cartesianBounds.yMinimum = (double)e57::IntegerNode(bbox.get("yMinimum")).value();
                cartesianBounds.yMaximum = (double)e57::IntegerNode(bbox.get("yMaximum")).value();
                cartesianBounds.zMinimum = (double)e57::IntegerNode(bbox.get("zMinimum")).value();
                cartesianBounds.zMaximum = (double)e57::IntegerNode(bbox.get("zMaximum")).value();
                break;
            }
            default:
                throw std::runtime_error("xMinimum has unknown type");
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    Scan_context() = delete;
    Scan_context(const e57::ImageFile& _image_file,
                 e57::StructureNode _scan,
                 e57::CompressedVectorNode _points,
                 e57::StructureNode _prototype)
        : image_file(_image_file), scan(_scan), points(_points), prototype(_prototype)
    {
        intensityLimits.intensityMinimum = 0.;
        intensityLimits.intensityMaximum = 255.;
        colorLimits.colorRedMinimum = 0.;
        colorLimits.colorRedMaximum = 255.;
        colorLimits.colorGreenMinimum = 0.;
        colorLimits.colorGreenMaximum = 255.;
        colorLimits.colorBlueMinimum = 0.;
        colorLimits.colorBlueMaximum = 255.;

        read_intensity_limits();
        read_color_limits();
        has_bounds = read_cartesian_bounds();
    }

    double bounded_range(double range) const
    {
        if (range <= 0.)
        {
            return 1.;
        }
        return range;
    }

public:
    static Scan_context create(const e57::ImageFile& image_file, e57::VectorNode data_3d, int64_t const scan_i)
    {
        e57::StructureNode scan{ data_3d.get(scan_i) };
        if (!scan.isDefined("points"))
        {
            throw std::runtime_error("Scan is missing points node");
        }
        e57::CompressedVectorNode points{ scan.get("points") };
        e57::StructureNode prototype{ points.prototype() };
        return Scan_context{ image_file, scan, points, prototype };
    }

    template <typename T>
    std::function<T(int)> create_required_getter(std::string const& name, double const offset = 0., double const range = 1.)
    {
        if (!prototype.isDefined(name))
        {
            throw std::runtime_error("Missing necessary attribute\n");
        }
        e57::Node node{ prototype.get(name) };
        switch (node.type())
        {
        case e57::TypeScaledInteger:
        case e57::TypeFloat:
        {
            auto double_buffer = std::make_shared<std::vector<double>>();
            double_buffer->resize(BUFFER_SIZE);
            double_buffers.push_back(double_buffer);
            source_dest_buffers.emplace_back(image_file, name, double_buffer->data(), BUFFER_SIZE, true, true);
            return Getter<T>::create(double_buffer, offset, range);
        }
        case e57::TypeInteger:
        {
            auto i64_buffer = std::make_shared<std::vector<int64_t>>();
            i64_buffer->resize(BUFFER_SIZE);
            i64_buffers.push_back(i64_buffer);
            source_dest_buffers.emplace_back(image_file, name, i64_buffer->data(), BUFFER_SIZE, true, true);
            e57::IntegerNode const integer_node{ node };

            return Getter<T>::create(i64_buffer, integer_node.minimum(), integer_node.maximum());
        }

        default:
            throw std::runtime_error("Attribute is not a numeric type");
        }
    }

    template <typename T>
    std::function<T(int)> create_getter(std::string const& name,
                                        T const default_value,
                                        double const offset = 0.,
                                        double const range = 1.)
    {
        if (!prototype.isDefined(name))
        {
            /*
            DEBUG_LOG(
                "Warning: attribute {} is missing, will use default value: "
                "{}\n",
                name, default_value);
            */
            return Getter<T>::create(default_value);
        }
        else
        {
            return create_required_getter<T>(name, offset, range);
        }
    }

    Point_transformer create_point_transformer()
    {
        if (scan.isDefined("pose")) {
            Affine_transformation mat = {
              1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0
            };

            //printf("Pose found in scan\n");
            e57::StructureNode pose{ scan.get("pose") };
            if (pose.isDefined("rotation")) {
                e57::StructureNode rotation{ pose.get("rotation") };
                auto const w = e57::FloatNode(rotation.get("w")).value();
                auto const x = e57::FloatNode(rotation.get("x")).value();
                auto const y = e57::FloatNode(rotation.get("y")).value();
                auto const z = e57::FloatNode(rotation.get("z")).value();
                //printf("Rotation (%.2f, %.2f, %.2f, %.2f)\n", w, x, y, z);

                mat[0][0] = 1.0 - 2.0 * y * y - 2.0 * z * z;
                mat[0][1] = 2.0 * x * y - 2.0 * z * w;
                mat[0][2] = 2.0 * x * z + 2.0 * y * w;

                mat[1][0] = 2.0 * x * y + 2.0 * z * w;
                mat[1][1] = 1.0 - 2.0 * x * x - 2.0 * z * z;
                mat[1][2] = 2.0 * y * z + 2.0 * x * w;

                mat[2][0] = 2.0 * x * z - 2.0 * y * w;
                mat[2][1] = 2.0 * y * z - 2.0 * x * w;
                mat[2][2] = 1.0 - 2.0 * x * x - 2.0 * y * y;
            }
            if (pose.isDefined("translation")) {
                e57::StructureNode translation{ pose.get("translation") };
                auto const x = e57::FloatNode(translation.get("x")).value();
                auto const y = e57::FloatNode(translation.get("y")).value();
                auto const z = e57::FloatNode(translation.get("z")).value();
                //printf("Translation (%.2f, %.2f, %.2f)\n", x, y, z);

                mat[0][3] = x;
                mat[1][3] = y;
                mat[2][3] = z;
            }

            return [mat](double x, double y, double z) -> std::array<float, 3> {
                return affine_transformer(mat, x, y, z);
                };
        }
        else {
            return identity_transformer;
        }
    }

    e57::CompressedVectorReader create_reader()
    {
        assert(image_file.isOpen());

        return points.reader(source_dest_buffers);
    }

    double intensity_offset() const
    {
        return intensityLimits.intensityMinimum;
    }

    double intensity_range() const
    {
        return bounded_range(intensityLimits.intensityMaximum - intensityLimits.intensityMinimum);
    }

    double color_red_offset() const
    {
        return colorLimits.colorRedMinimum;
    }

    double color_red_range() const
    {
        return bounded_range(colorLimits.colorRedMaximum - colorLimits.colorRedMinimum);
    }

    double color_green_offset() const
    {
        return colorLimits.colorGreenMinimum;
    }

    double color_green_range() const
    {
        return bounded_range(colorLimits.colorGreenMaximum - colorLimits.colorGreenMinimum);
    }

    double color_blue_offset() const
    {
        return colorLimits.colorBlueMinimum;
    }

    double color_blue_range() const
    {
        return bounded_range(colorLimits.colorBlueMaximum - colorLimits.colorBlueMinimum);
    }

    bool is_cartesian_valid() const
    {
        return prototype.isDefined("cartesianX") && prototype.isDefined("cartesianY") &&
               prototype.isDefined("cartesianZ");
    }

    bool are_normals_valid() const
    {
        return prototype.isDefined("nor:normalX") && prototype.isDefined("nor:normalY") &&
               prototype.isDefined("nor:normalZ");
    }

    bool is_color_valid() const
    {
        return prototype.isDefined("colorRed") && prototype.isDefined("colorGreen") && prototype.isDefined("colorBlue");
    }

    static std::string get_scan_name(e57::StructureNode& scan)
    {
        if (scan.isDefined("name"))
        {
            return e57::StringNode(scan.get("name")).value();
        }
        return std::string();
    }

    int64_t get_record_count() const
    {
        return points.childCount();
    }
};

typedef std::shared_ptr<e57::ImageFile> ImageFilePtr;
std::mutex xerces_mutex;

bool read_e57(const char* filename, size_t* array_size, float** positions_array, float** colors_array, float** normals_array)
{
    if (positions_array == nullptr || colors_array == nullptr)
    {
        return false;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        printf("Error: Could not open file: %s\n", filename);
        return false;
    }
    file.close();

    ImageFilePtr imfPtr = nullptr;
    {
        // image file can be created one at time, there is a mutex in the xerces XML reader used by libE57
        std::lock_guard<std::mutex> lock(xerces_mutex);
        imfPtr = std::make_shared<e57::ImageFile>(filename, "r");
    }
    if (!imfPtr)
    {
        return false;
    }

    e57::StructureNode root = imfPtr->root();
    if (!root.isDefined("data3D"))
    {
        return false;
    }

    std::vector<Scan_context> scan_contexts;
    e57::VectorNode const data_3d{ root.get("data3D") };
    size_t arraySize = 0u;
    for (int64_t scan_i = 0; scan_i < data_3d.childCount(); ++scan_i)
    {
        auto context = Scan_context::create(*imfPtr, data_3d, scan_i);
        scan_contexts.push_back(context);
        arraySize += 3 * size_t(context.get_record_count());
    }

    *positions_array = new float[arraySize];
    *colors_array = new float[arraySize];
    if (normals_array)
    {
        *normals_array = new float[arraySize];
    }

    for (auto& context : scan_contexts)
    {
        int64_t scan_point_count = context.get_record_count();

        // create position getter functions
        std::array<std::function<double(int)>, 3> const cartesian
        {
            context.create_required_getter<double>("cartesianX"),
            context.create_required_getter<double>("cartesianY"),
            context.create_required_getter<double>("cartesianZ")
        };

        // create color getter functions or use intensity values if there are no colors
        std::array<std::function<float(int)>, 3> const color([&]()
            {
                std::array<std::function<float(int)>, 3> result{ nullptr, nullptr, nullptr };
                if (context.is_color_valid())
                {
                    result[0] = context.create_getter<float>("colorRed", 0.f, context.color_red_offset(),
                        context.color_red_range());
                    result[1] = context.create_getter<float>("colorGreen", 0.f, context.color_green_offset(),
                        context.color_green_range());
                    result[2] = context.create_getter<float>("colorBlue", 0.f, context.color_blue_offset(),
                        context.color_blue_range());
                }
                else
                {
                    result[0] = context.create_getter<float>("intensity", 0.f, context.intensity_offset(),
                        context.intensity_range());
                    result[1] = result[0];
                    result[2] = result[0];
                }
                return result;
            }());

        // create normal getter functions if normals are present
        std::array<std::function<float(int)>, 3> const normal([&]()
            {
                std::array<std::function<float(int)>, 3> result{ nullptr, nullptr, nullptr };
                for (auto i = 0; i < imfPtr->extensionsCount(); ++i) {
                    if (imfPtr->extensionsPrefix(i) == "nor") {
                        if (context.are_normals_valid()) {
                            result[0] = context.create_getter<float>("nor:normalX", 0.f, 0.f, 1.f);
                            result[1] = context.create_getter<float>("nor:normalY", 0.f, 0.f, 1.f);
                            result[2] = context.create_getter<float>("nor:normalZ", 0.f, 0.f, 1.f);
                        }
                        break;
                    }
                }
                return result;
            }());

        bool const valid_normals = normal[0] != nullptr;
        if (valid_normals)
        {
            printf("Normals found in scan\n");
        }

        Point_transformer const point_transformer = context.create_point_transformer();
        auto reader = context.create_reader();

        size_t total_count = 0;
        while (auto count = reader.read())
        {
            for (uint64_t point_i = 0; point_i < count; ++point_i)
            {
                uint64_t total_i = 3 * (total_count + point_i);
                if (total_i + 2 > 3 * scan_point_count)
                {
                    break;
                }

                std::array<float, 3> position = point_transformer(
                    cartesian[0](point_i),
                    cartesian[1](point_i),
                    cartesian[2](point_i)
                );

                for (int i = 0; i < 3; ++i)
                {
                    (*positions_array)[total_i + i] = position[i];
                    (*colors_array)[total_i + i] = color[i](point_i);
                    if (normals_array)
                    {
                        (*normals_array)[total_i + i] = valid_normals ? normal[i](point_i) : 0.f;
                    }
                }
            }
            total_count += count;
            if (array_size)
            {
                *array_size = 3 * total_count;
            }
        }
        if (total_count != scan_point_count)
        {
            printf("Warning: points read in scan does not correspond to the point count: %zu != %zu\n", total_count, scan_point_count);
        }

        //printf("points read in scan: %zu\n", total_count);
    }
    return true;
}

void e57_to_float(const char* filename, size_t* array_size, float** positions_array, float** colors_array, float** normals_array)
{
    //printf("Loading '%s'... \n", filename);
    try
    {
        read_e57(filename, array_size, positions_array, colors_array, normals_array);
    }
    catch (e57::E57Exception& exc)
    {
        printf("Failed to read e57 file \"%s\": %s", filename, exc.what());

        std::stringstream stream;
        exc.report(nullptr, 0, nullptr, stream);
        printf("%s\n", stream.str().c_str());
    }
}
}
