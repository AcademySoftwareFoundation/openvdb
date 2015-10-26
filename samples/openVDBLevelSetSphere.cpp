#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
int main()
{
    openvdb::initialize();
    // Create a FloatGrid and populate it with a narrow-band
    // signed distance field of a sphere.
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
            /*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3),
            /*voxel size=*/0.5, /*width=*/4.0);
    // Associate some metadata with the grid.
    grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    // Name the grid "LevelSetSphere".
    grid->setName("LevelSetSphere");
    // Create a VDB file object.
    openvdb::io::File file("mygrids.vdb");
    // Add the grid pointer to a container.
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    // Write out the contents of the container.
    file.write(grids);
    file.close();
}