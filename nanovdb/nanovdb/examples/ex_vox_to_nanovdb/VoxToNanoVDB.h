// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"
#if defined(_MSC_VER)
#include <io.h>
#endif

namespace detail {

inline const ogt_vox_scene* load_vox_scene(const char* filename, uint32_t scene_read_flags = 0)
{
#if defined(_MSC_VER) && _MSC_VER >= 1400
    FILE* fp;
    if (0 != fopen_s(&fp, filename, "rb"))
        fp = 0;
#else
    FILE* fp = fopen(filename, "rb");
#endif
    if (!fp)
        return NULL;
    fseek(fp, 0, SEEK_END);
    uint32_t buffer_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* buffer = new uint8_t[buffer_size];
    size_t bytes = fread(buffer, buffer_size, 1, fp);
    fclose(fp);
    const ogt_vox_scene* scene = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags);
    delete[] buffer; // the buffer can be safely deleted once the scene is instantiated.
    return scene;
}

inline nanovdb::Vec4f matMult4x4(const float* mat, const nanovdb::Vec4f& rhs)
{
#define _mat(m, r, c) m[c * 4 + r]

    return nanovdb::Vec4f(_mat(mat, 0, 0) * rhs[0] + _mat(mat, 0, 1) * rhs[1] + _mat(mat, 0, 2) * rhs[2] + _mat(mat, 0, 3) * rhs[3],
                          _mat(mat, 1, 0) * rhs[0] + _mat(mat, 1, 1) * rhs[1] + _mat(mat, 1, 2) * rhs[2] + _mat(mat, 1, 3) * rhs[3],
                          _mat(mat, 2, 0) * rhs[0] + _mat(mat, 2, 1) * rhs[1] + _mat(mat, 2, 2) * rhs[2] + _mat(mat, 2, 3) * rhs[3],
                          _mat(mat, 3, 0) * rhs[0] + _mat(mat, 3, 1) * rhs[1] + _mat(mat, 3, 2) * rhs[2] + _mat(mat, 3, 3) * rhs[3]);
#undef _mat
}

inline ogt_vox_transform matMult4x4(const float* m, const float* n)
{
#define _mat(m, c, r) m[c * 4 + r]

    return ogt_vox_transform{
        _mat(m, 0, 0) * _mat(n, 0, 0) + _mat(m, 0, 1) * _mat(n, 1, 0) + _mat(m, 0, 2) * _mat(n, 2, 0) + _mat(m, 0, 3) * _mat(n, 3, 0),
        _mat(m, 0, 0) * _mat(n, 0, 1) + _mat(m, 0, 1) * _mat(n, 1, 1) + _mat(m, 0, 2) * _mat(n, 2, 1) + _mat(m, 0, 3) * _mat(n, 3, 1),
        _mat(m, 0, 0) * _mat(n, 0, 2) + _mat(m, 0, 1) * _mat(n, 1, 2) + _mat(m, 0, 2) * _mat(n, 2, 2) + _mat(m, 0, 3) * _mat(n, 3, 2),
        _mat(m, 0, 0) * _mat(n, 0, 3) + _mat(m, 0, 1) * _mat(n, 1, 3) + _mat(m, 0, 2) * _mat(n, 2, 3) + _mat(m, 0, 3) * _mat(n, 3, 3),

        _mat(m, 1, 0) * _mat(n, 0, 0) + _mat(m, 1, 1) * _mat(n, 1, 0) + _mat(m, 1, 2) * _mat(n, 2, 0) + _mat(m, 1, 3) * _mat(n, 3, 0),
        _mat(m, 1, 0) * _mat(n, 0, 1) + _mat(m, 1, 1) * _mat(n, 1, 1) + _mat(m, 1, 2) * _mat(n, 2, 1) + _mat(m, 1, 3) * _mat(n, 3, 1),
        _mat(m, 1, 0) * _mat(n, 0, 2) + _mat(m, 1, 1) * _mat(n, 1, 2) + _mat(m, 1, 2) * _mat(n, 2, 2) + _mat(m, 1, 3) * _mat(n, 3, 2),
        _mat(m, 1, 0) * _mat(n, 0, 3) + _mat(m, 1, 1) * _mat(n, 1, 3) + _mat(m, 1, 2) * _mat(n, 2, 3) + _mat(m, 1, 3) * _mat(n, 3, 3),

        _mat(m, 2, 0) * _mat(n, 0, 0) + _mat(m, 2, 1) * _mat(n, 1, 0) + _mat(m, 2, 2) * _mat(n, 2, 0) + _mat(m, 2, 3) * _mat(n, 3, 0),
        _mat(m, 2, 0) * _mat(n, 0, 1) + _mat(m, 2, 1) * _mat(n, 1, 1) + _mat(m, 2, 2) * _mat(n, 2, 1) + _mat(m, 2, 3) * _mat(n, 3, 1),
        _mat(m, 2, 0) * _mat(n, 0, 2) + _mat(m, 2, 1) * _mat(n, 1, 2) + _mat(m, 2, 2) * _mat(n, 2, 2) + _mat(m, 2, 3) * _mat(n, 3, 2),
        _mat(m, 2, 0) * _mat(n, 0, 3) + _mat(m, 2, 1) * _mat(n, 1, 3) + _mat(m, 2, 2) * _mat(n, 2, 3) + _mat(m, 2, 3) * _mat(n, 3, 3),

        _mat(m, 3, 0) * _mat(n, 0, 0) + _mat(m, 3, 1) * _mat(n, 1, 0) + _mat(m, 3, 2) * _mat(n, 2, 0) + _mat(m, 3, 3) * _mat(n, 3, 0),
        _mat(m, 3, 0) * _mat(n, 0, 1) + _mat(m, 3, 1) * _mat(n, 1, 1) + _mat(m, 3, 2) * _mat(n, 2, 1) + _mat(m, 3, 3) * _mat(n, 3, 1),
        _mat(m, 3, 0) * _mat(n, 0, 2) + _mat(m, 3, 1) * _mat(n, 1, 2) + _mat(m, 3, 2) * _mat(n, 2, 2) + _mat(m, 3, 3) * _mat(n, 3, 2),
        _mat(m, 3, 0) * _mat(n, 0, 3) + _mat(m, 3, 1) * _mat(n, 1, 3) + _mat(m, 3, 2) * _mat(n, 2, 3) + _mat(m, 3, 3) * _mat(n, 3, 3),
    };
#undef _mat
}

ogt_vox_transform getXform(const ogt_vox_scene& scene, const ogt_vox_instance& instance)
{
    ogt_vox_transform transform = instance.transform; //{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    auto groupIndex = instance.group_index;
    while (groupIndex != 0 && groupIndex != k_invalid_group_index) {
        const auto& group = scene.groups[groupIndex];
        transform = matMult4x4((const float*)&transform, (const float*)&group.transform);
        groupIndex = group.parent_group_index;
    }

    return transform;
}

bool isVisible(const ogt_vox_scene& scene, const ogt_vox_instance& instance)
{
    if (instance.hidden)
        return false;

    if (scene.layers[instance.layer_index].hidden)
        return false;

    auto groupIndex = instance.group_index;
    while (groupIndex != 0 && groupIndex != k_invalid_group_index) {
        const auto& group = scene.groups[groupIndex];
        if (group.hidden)
            return false;
        if (scene.layers[group.layer_index].hidden)
            return false;
        groupIndex = group.parent_group_index;
        printf("group.parent_group_index = %d\n", groupIndex);
    }
    return true;
}

} // namespace detail

/// @brief load a .vox file.
template<typename BufferT = nanovdb::HostBuffer>
nanovdb::GridHandle<BufferT> convertVoxToNanoVDB(const std::string& inFilename, const std::string& modelName)
{
#if 0
    // just debugging the xforms!
    {
        ogt_vox_transform translate{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1};
        ogt_vox_transform scale{10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 1};
        ogt_vox_transform translate2{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1};
        ogt_vox_transform xform = detail::matMult4x4((float*)&scale, (float*)&translate);
        xform = detail::matMult4x4((float*)&translate2, (float*)&xform);
        auto              v = detail::matMult4x4((float*)&xform, nanovdb::Vec4f(0, 1, 0, 1));
        std::cout << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }
#endif

    try {
        if (const auto* scene = detail::load_vox_scene(inFilename.c_str())) {
            // we just merge into one grid...
            nanovdb::tools::build::Grid<nanovdb::math::Rgba8> grid(nanovdb::math::Rgba8(),modelName,nanovdb::GridClass::VoxelVolume);
            auto acc = grid.getAccessor();

            auto processModelFn = [&](int modelIndex, const ogt_vox_transform& xform) {
                const auto* model = scene->models[modelIndex];

                uint32_t voxel_index = 0;
                for (uint32_t z = 0; z < model->size_z; ++z) {
                    for (uint32_t y = 0; y < model->size_y; ++y) {
                        for (uint32_t x = 0; x < model->size_x; ++x, ++voxel_index) {
                            if (uint8_t color_index = model->voxel_data[voxel_index]) {
                                ogt_vox_rgba rgba = scene->palette.color[color_index];
                                auto ijk = nanovdb::Coord::Floor(detail::matMult4x4((float*)&xform, nanovdb::Vec4f(x, y, z, 1)));
                                acc.setValue(nanovdb::Coord(ijk[0], ijk[2], -ijk[1]), *reinterpret_cast<nanovdb::math::Rgba8*>(&rgba));
                            }
                        }
                    }
                }
            };

            if (scene->num_instances > 0) {
                printf("scene processing begin... %d instances\n", scene->num_instances);

                for (uint32_t instanceIndex = 0; instanceIndex < scene->num_instances; instanceIndex++) {
                    const auto& instance = scene->instances[instanceIndex];
                    uint32_t    modelIndex = instance.model_index;

                    //printf("instance[%d].model_index = %d\n", instanceIndex, instance.model_index);
                    //printf("instance[%d].layer_index = %d\n", instanceIndex, instance.layer_index);
                    //printf("instance[%d].group_index = %d\n", instanceIndex, instance.group_index);
#if 1
                    if (!detail::isVisible(*scene, instance))
                        continue;

                    auto xform = detail::getXform(*scene, instance);
#else
                    auto xform = instance.transform;
#endif
                    processModelFn(modelIndex, xform);
                }
            } else {
                printf("scene processing begin... %d models\n", scene->num_models);

                ogt_vox_transform xform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

                for (uint32_t modelIndex = 0; modelIndex < scene->num_models; modelIndex++) {
                    processModelFn(modelIndex, xform);
                    xform.m30 += 30;
                }
            }

            printf("scene processing end.\n");
            ogt_vox_destroy_scene(scene);
            return nanovdb::tools::createNanoGrid(grid);
        } else {
            std::ostringstream ss;
            ss << "Invalid file \"" << inFilename << "\"";
            throw std::runtime_error(ss.str());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return nanovdb::GridHandle<BufferT>();
}
