// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_RENDER_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_RENDER_HAS_BEEN_INCLUDED

#include "../Utilities/Render.h"


/* OpenVDBGrid public member function list

mma::ImageRef<mma::im_byte_t> renderGrid(
    double isovalue, mma::RGBRef color, mma::RGBRef color2, mma::RGBRef color3, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
    mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed
)

mma::ImageRef<mma::im_byte_t> renderGridPBR(
    double isovalue, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, bool is_closed,
    mma::RGBRef baseColorFront, mma::RGBRef baseColorBack, mma::RGBRef baseColorClosed,
    double metallic, double rough, double ani, double ref,
    mma::RGBRef coatColor, double coatRough, double coatAni, double coatRef,
    double fac_spec, double fac_diff, double fac_coat
)

mma::ImageRef<mma::im_byte_t> renderGridVectorColor(
    double isovalue, OpenVDBGrid<Vec3s> cGrid,
    OpenVDBGrid<Vec3s> cGrid2, OpenVDBGrid<Vec3s> cGrid3, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
    mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed
)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::ImageRef<mma::im_byte_t>
openvdbmma::OpenVDBGrid<V>::renderGrid(
    double isovalue, mma::RGBRef color, mma::RGBRef color2, mma::RGBRef color3, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
    mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed) const
{
    scalar_type_assert<V>();

    if (resolution.size() != 2)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    const int w = resolution[0];
    const int h = resolution[1];

    if (w <= 0 || h <= 0)
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    const std::shared_ptr<tools::Film::RGBA> colorRGBA =
        std::make_shared<tools::Film::RGBA>(openvdbmma::render::mmaRGBToColor(color));
    const std::shared_ptr<tools::Film::RGBA> color2RGBA =
        std::make_shared<tools::Film::RGBA>(openvdbmma::render::mmaRGBToColor(color2));
    const std::shared_ptr<tools::Film::RGBA> color3RGBA =
        std::make_shared<tools::Film::RGBA>(openvdbmma::render::mmaRGBToColor(color3));
    const tools::Film::RGBA backgroundRGBA = openvdbmma::render::mmaRGBToColor(background);

    openvdbmma::render::RenderGridMma<wlGridType> renderer(grid(), w, h);

    renderer.setIsoValue(isovalue);
    renderer.setFrame(frame);

    renderer.setColor(colorRGBA);
    renderer.setColor2(color2RGBA);
    renderer.setColor3(color3RGBA);
    renderer.setBackground(backgroundRGBA);

    renderer.setTranslate(translate);
    renderer.setLookAt(lookat);
    renderer.setUp(up);
    renderer.setRange(range);
    renderer.setFOV(fov);
    renderer.setIsClosed(is_closed);

    renderer.setShader(shader);
    renderer.setCamera(camera);

    renderer.setSamples(samples);
    renderer.setStep(step);
    renderer.setLightDir(lightdir);

    renderer.setDepthParameters(depthParams);

    return renderer.renderImage();
}

template<typename V>
mma::ImageRef<mma::im_byte_t>
openvdbmma::OpenVDBGrid<V>::renderGridPBR(
    double isovalue, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, bool is_closed,
    mma::RGBRef baseColorFront, mma::RGBRef baseColorBack, mma::RGBRef baseColorClosed,
    double metallic, double rough, double ani, double ref,
    mma::RGBRef coatColor, double coatRough, double coatAni, double coatRef,
    double fac_spec, double fac_diff, double fac_coat) const
{
    scalar_type_assert<V>();

    if (resolution.size() != 2)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    const int w = resolution[0];
    const int h = resolution[1];

    if (w <= 0 || h <= 0)
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    const std::shared_ptr<tools::Film::RGBA> colorRGBA =
        std::make_shared<tools::Film::RGBA>(tools::Film::RGBA(1.0, 1.0, 1.0, 1.0));
    const std::shared_ptr<tools::Film::RGBA> color2RGBA =
        std::make_shared<tools::Film::RGBA>(tools::Film::RGBA(1.0, 1.0, 1.0, 1.0));
    const tools::Film::RGBA backgroundRGBA = openvdbmma::render::mmaRGBToColor(background);

    openvdbmma::render::RenderGridMma<wlGridType> renderer(grid(), w, h);

    renderer.setIsoValue(isovalue);
    renderer.setFrame(frame);

    renderer.setColor(colorRGBA);
    renderer.setColor2(color2RGBA);
    renderer.setBackground(backgroundRGBA);

    renderer.setTranslate(translate);
    renderer.setLookAt(lookat);
    renderer.setUp(up);
    renderer.setRange(range);
    renderer.setFOV(fov);

    renderer.setCamera(camera);
    renderer.setSamples(samples);

    const Vec3R baseColorFrontVec(baseColorFront[0], baseColorFront[1], baseColorFront[2]);
    const Vec3R baseColorBackVec(baseColorBack[0], baseColorBack[1], baseColorBack[2]);
    const Vec3R baseColorClosedVec(baseColorClosed[0], baseColorClosed[1], baseColorClosed[2]);
    const Vec3R coatColorVec(coatColor[0], coatColor[1], coatColor[2]);

    renderer.setPBRShader(
        baseColorFrontVec, baseColorBackVec, baseColorClosedVec,
        metallic, rough, ani, ref,
        coatColorVec, coatRough, coatAni, coatRef,
        fac_spec, fac_diff, fac_coat, is_closed
    );

    return renderer.renderImage();
}

template<typename V>
mma::ImageRef<mma::im_byte_t>
openvdbmma::OpenVDBGrid<V>::renderGridVectorColor(
    double isovalue, OpenVDBGrid<Vec3s> cGrid,
    OpenVDBGrid<Vec3s> cGrid2, OpenVDBGrid<Vec3s> cGrid3, mma::RGBRef background,
    mma::RealVectorRef translate, mma::RealVectorRef lookat, mma::RealVectorRef up,
    mma::RealVectorRef range, mma::RealVectorRef fov, mint shader, mint camera, mint samples,
    mma::IntVectorRef resolution, double frame, mma::RealVectorRef depthParams,
    mma::RealVectorRef lightdir, mma::RealVectorRef step, bool is_closed) const
{
    scalar_type_assert<V>();

    using wlVectorType    = Vec3s;
    using wlVectorTree    = typename openvdb::tree::Tree4<wlVectorType, 5, 4, 3>::Type;
    using wlVectorGrid    = openvdb::Grid<wlVectorTree>;
    using wlVectorGridPtr = typename wlVectorGrid::Ptr;

    if (resolution.size() != 2)
        throw mma::LibraryError(LIBRARY_DIMENSION_ERROR);

    const int w = resolution[0];
    const int h = resolution[1];

    if (w <= 0 || h <= 0)
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    const tools::Film::RGBA backgroundRGBA = openvdbmma::render::mmaRGBToColor(background);

    openvdbmma::render::RenderGridMma<wlGridType, wlVectorGrid, wlVectorGridPtr> renderer(grid(), w, h);

    renderer.setIsoValue(isovalue);
    renderer.setFrame(frame);

    renderer.setColor(cGrid.grid());
    renderer.setColor2(cGrid2.grid());
    renderer.setColor3(cGrid3.grid());
    renderer.setBackground(backgroundRGBA);

    renderer.setTranslate(translate);
    renderer.setLookAt(lookat);
    renderer.setUp(up);
    renderer.setRange(range);
    renderer.setFOV(fov);
    renderer.setIsClosed(is_closed);

    renderer.setShader(shader);
    renderer.setCamera(camera);

    renderer.setSamples(samples);
    renderer.setStep(step);
    renderer.setLightDir(lightdir);

    renderer.setDepthParameters(depthParams);

    return renderer.renderImage();
}

#endif // OPENVDBLINK_OPENVDBGRID_RENDER_HAS_BEEN_INCLUDED
