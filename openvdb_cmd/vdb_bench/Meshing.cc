// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Meshing.cc
///
/// @brief Group meshing: volume to mesh, mesh to volume and the level set utilities.

#include "Harness.h"

#include <openvdb/tools/Composite.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <memory>

namespace bench {

namespace {

// Meshing a fine grid is slow, so these rows use a coarser voxel size than the rest
const float kMeshVoxelSize = 1.0f / 512.0f;

void
addMesher(Digest& digest, const tools::VolumeToMesh& mesher)
{
    for (size_t i = 0; i < mesher.pointListSize(); ++i) {
        digest.addValue(mesher.pointList()[i]);
    }
    digest.add(uint64_t(mesher.pointListSize()));
    for (size_t pool = 0; pool < mesher.polygonPoolListSize(); ++pool) {
        const tools::PolygonPool& polygons = mesher.polygonPoolList()[pool];
        for (size_t n = 0; n < polygons.numQuads(); ++n) {
            digest.addValue(polygons.quad(n));
            digest.add(polygons.quadFlags(n));
        }
        for (size_t n = 0; n < polygons.numTriangles(); ++n) {
            digest.addValue(polygons.triangle(n));
            digest.add(polygons.triangleFlags(n));
        }
        digest.add(uint64_t(polygons.numQuads()));
        digest.add(uint64_t(polygons.numTriangles()));
    }
}

void
meshRow(Context& context, const std::string& name, const FloatGrid& grid, double adaptivity,
    FloatGrid::ConstPtr reference = nullptr)
{
    const std::string tool = "volumeToMesh";
    if (!context.wanted(tool, name)) return;
    context.progress(tool, name);
    std::unique_ptr<tools::VolumeToMesh> last;
    const Timing timing = context.time(
        [&]() {
            std::unique_ptr<tools::VolumeToMesh> mesher(new tools::VolumeToMesh(0.0, adaptivity));
            if (reference) mesher->setRefGrid(reference, adaptivity);
            return mesher;
        },
        [&](std::unique_ptr<tools::VolumeToMesh>& mesher) {
            (*mesher)(grid);
            last = std::move(mesher);
        });
    Digest digest;
    addMesher(digest, *last);
    context.record(tool, name, timing, digest);
}

void
runVolumeToMesh(Context& context)
{
    const float voxelSize = kMeshVoxelSize;
    const std::string label = voxelLabel(voxelSize);
    FloatGrid::Ptr sphere = makeSphere<FloatGrid>(voxelSize);
    FloatGrid::Ptr icosahedron = makeIcosahedron(voxelSize);
    FloatGrid::Ptr dragon = loadFloatGrid(context.options().dragonPath);

    for (double adaptivity : {0.0, 0.5}) {
        const std::string suffix = std::string(" adaptivity ") + (adaptivity == 0.0 ? "0" : "0.5");
        meshRow(context, "sphere " + label + suffix, *sphere, adaptivity);
        meshRow(context, "icosahedron " + label + suffix, *icosahedron, adaptivity);
        if (dragon) meshRow(context, "dragon" + suffix, *dragon, adaptivity);
    }

    // The sphere with a corner cut off, meshed against the whole sphere. This is the
    // reference based scheme that handles the seams of fractured pieces.
    FloatGrid::Ptr cube = tools::createLevelSetCube<FloatGrid>(0.12f, Vec3f(0.5f, 0.35f, 0.35f), voxelSize, 3.0f);
    FloatGrid::Ptr fragment = tools::csgDifferenceCopy(*sphere, *cube);
    meshRow(context, "sphere minus cube reference " + label + " adaptivity 0.5", *fragment, 0.5, sphere);
}

/// Quads of a mesh in the index space of a transform
struct IndexMesh
{
    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;
};

void
append(IndexMesh& mesh, const FloatGrid& grid, const math::Transform& target, const Vec3d& shift)
{
    std::vector<Vec3s> worldPoints;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(grid, worldPoints, quads, 0.0);
    const Index offset = Index(mesh.points.size());
    for (const Vec3s& point : worldPoints) {
        mesh.points.push_back(Vec3s(target.worldToIndex(Vec3d(point) + shift)));
    }
    for (const Vec4I& quad : quads) {
        mesh.quads.push_back(Vec4I(quad[0] + offset, quad[1] + offset, quad[2] + offset, quad[3] + offset));
    }
}

void
runMeshToVolume(Context& context)
{
    const float voxelSize = kMeshVoxelSize;
    const std::string label = voxelLabel(voxelSize);
    const math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

    // The source meshes come from a coarser sphere, and are not part of the timing
    FloatGrid::Ptr source = makeSphere<FloatGrid>(1.0f / 256.0f);
    IndexMesh single;
    append(single, *source, *transform, Vec3d(0.0));
    IndexMesh overlapping;
    append(overlapping, *source, *transform, Vec3d(0.0));
    append(overlapping, *source, *transform, Vec3d(0.1, 0.0, 0.0));

    struct Case { std::string name; const IndexMesh* mesh; float width; };
    const Case cases[] = {
        {"sphere width 3", &single, 3.0f},
        {"sphere width 10", &single, 10.0f},
        {"two overlapping spheres width 3", &overlapping, 3.0f}};

    const std::string tool = "meshToVolume";
    for (const Case& test : cases) {
        const std::string name = test.name + " " + label;
        if (!context.wanted(tool, name)) continue;
        context.progress(tool, name);
        const tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> adapter(test.mesh->points, test.mesh->quads);
        FloatGrid::Ptr last;
        const Timing timing = context.time(
            []() { return 0; },
            [&](int&) { last = tools::meshToVolume<FloatGrid>(adapter, *transform, test.width, test.width); });
        Digest digest;
        digest.addGrid(*last);
        context.record(tool, name, timing, digest);
    }
}

void
runLevelSetUtil(Context& context)
{
    const float voxelSize = kMeshVoxelSize;
    const std::string label = voxelLabel(voxelSize);
    std::vector<NamedGrid> grids;
    grids.push_back({"sphere " + label, makeSphere<FloatGrid>(voxelSize)});
    // Two surfaces that do not touch, for the segmentation
    {
        FloatGrid::Ptr second = tools::createLevelSetSphere<FloatGrid>(0.1f, Vec3f(0.7f, 0.35f, 0.35f), voxelSize, 3.0f);
        grids.push_back({"two spheres " + label, tools::csgUnionCopy(*grids.front().grid, *second)});
    }
    if (FloatGrid::Ptr dragon = loadFloatGrid(context.options().dragonPath)) grids.push_back({"dragon", dragon});

    for (const NamedGrid& named : grids) {
        {
            const std::string tool = "extractIsosurfaceMask";
            if (context.wanted(tool, named.name)) {
                context.progress(tool, named.name);
                BoolGrid::Ptr result;
                const Timing timing = context.time(
                    []() { return 0; },
                    [&](int&) { result = tools::extractIsosurfaceMask(*named.grid, 0.0f); });
                Digest digest;
                digest.addGrid(*result);
                context.record(tool, named.name, timing, digest);
            }
        }
        {
            const std::string tool = "extractEnclosedRegion";
            if (context.wanted(tool, named.name)) {
                context.progress(tool, named.name);
                BoolGrid::Ptr result;
                const Timing timing = context.time(
                    []() { return 0; },
                    [&](int&) { result = tools::extractEnclosedRegion(*named.grid, 0.0f); });
                Digest digest;
                digest.addGrid(*result);
                context.record(tool, named.name, timing, digest);
            }
        }
        {
            const std::string tool = "segmentSDF";
            if (context.wanted(tool, named.name)) {
                context.progress(tool, named.name);
                std::vector<FloatGrid::Ptr> segments;
                const Timing timing = context.time(
                    []() { return 0; },
                    [&](int&) { segments.clear(); tools::segmentSDF(*named.grid, segments); });
                Digest digest;
                for (const FloatGrid::Ptr& segment : segments) digest.addGrid(*segment);
                digest.add(uint64_t(segments.size()));
                context.record(tool, named.name, timing, digest);
            }
        }
    }
}

} // anonymous namespace

void
runMeshing(Context& context)
{
    runVolumeToMesh(context);
    runMeshToVolume(context);
    runLevelSetUtil(context);
}

} // namespace bench
