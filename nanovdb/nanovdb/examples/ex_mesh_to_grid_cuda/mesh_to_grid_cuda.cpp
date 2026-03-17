// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/util/CpuTimer.h>
#include <openvdb/tools/MeshToVolume.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>

#include <thrust/universal_vector.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template<typename BuildT>
void mainMeshToGrid(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid);

void readOBJ(const std::string& filename,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec3I>& triangles,
    std::vector<openvdb::Vec4I>& quads)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        OPENVDB_THROW(openvdb::IoError, "Failed to open OBJ file: " + filename);
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            points.push_back(openvdb::Vec3s(x, y, z));
        } else if (type == "f") {
            std::vector<int> faceIndices;
            std::string vertexData;

            while (iss >> vertexData) {
                // Isolate the vertex index (everything before the first slash)
                size_t slashPos = vertexData.find('/');
                std::string indexStr = vertexData.substr(0, slashPos);

                if (indexStr.empty()) continue;

                int raw_idx = std::stoi(indexStr);
                int actual_idx = 0;

                // Handle negative indices: relative to the number of points parsed so far
                if (raw_idx < 0) {
                    actual_idx = points.size() + raw_idx;
                } else {
                    // Standard positive indices: OBJ is 1-based, convert to 0-based for C++
                    actual_idx = raw_idx - 1;
                }

                // Strict bounds checking to prevent segfaults
                if (actual_idx < 0 || actual_idx >= points.size()) {
                    OPENVDB_THROW(openvdb::ValueError,
                        "OBJ parse error on line " + std::to_string(lineNumber) +
                        ": Face index out of bounds (Raw: " + std::to_string(raw_idx) +
                        ", Computed: " + std::to_string(actual_idx) + ", Total Points: " +
                        std::to_string(points.size()) + ")");
                }

                faceIndices.push_back(actual_idx);
            }

            // Add to the appropriate OpenVDB list
            if (faceIndices.size() == 3) {
                triangles.push_back(openvdb::Vec3I(faceIndices[0], faceIndices[1], faceIndices[2]));
            } else if (faceIndices.size() == 4) {
                quads.push_back(openvdb::Vec4I(faceIndices[0], faceIndices[1], faceIndices[2], faceIndices[3]));
            } else if (faceIndices.size() > 4) {
                std::cerr << "Warning on line " << lineNumber << ": Skipping face with "
                          << faceIndices.size() << " vertices. Triangulate your mesh!" << std::endl;
            }
        }
    }
}

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using BuildT = nanovdb::ValueOnIndex;

    openvdb::util::CpuTimer cpuTimer;

    try {

        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input.obj [output.vdb]\n");
        std::string inputFile = argv[1];
        std::string outputFile = "output.vdb";
        if (argc > 2)
            outputFile = argv[2];
        float voxelSize = 0.001f;
        if (argc > 3)
            voxelSize = std::stof(argv[3]);

        std::vector<openvdb::Vec3s> openvdb_points;
        std::vector<openvdb::Vec3I> openvdb_triangles;
        std::vector<openvdb::Vec4I> quads;

        // Read the OBJ file
        std::cout << "Reading " << inputFile << "..." << std::endl;
        readOBJ(inputFile, openvdb_points, openvdb_triangles, quads);
        std::cout << "Loaded " << openvdb_points.size() << " vertices, "
                  << openvdb_triangles.size() << " triangles, and "
                  << quads.size() << " quads." << std::endl;

        // Initialize OpenVDB
        openvdb::initialize();

        // Setup Grid Transform (Voxel Size)
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        // Convert Mesh to Unsigned Distance Field (UDF)
        // halfband specifies the half-width of the narrow band in voxel units
        float halfband = 3.0f;
        cpuTimer.start("Converting mesh to OpenVDB unsigned distance field");
        openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid>(
            *transform, openvdb_points, openvdb_triangles, quads, halfband);
        cpuTimer.stop();

        // Write the Grid to a VDB File
        grid->setName("UnsignedDistanceField");
        grid->print(std::cout, 2);
        std::cout << "Writing to " << outputFile << "..." << std::endl;
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        openvdb::io::File file(outputFile);
        file.write(grids);
        file.close();

        // Cast the raw pointers from the std::vector data
        const auto* nano_pts_data = reinterpret_cast<const nanovdb::Vec3f*>(openvdb_points.data());
        const auto* nano_tris_data = reinterpret_cast<const nanovdb::Vec3i*>(openvdb_triangles.data());

        // Initialize the thrust vectors using the casted pointer ranges
        thrust::universal_vector<nanovdb::Vec3f> nanovdb_points(nano_pts_data, nano_pts_data + openvdb_points.size());
        thrust::universal_vector<nanovdb::Vec3i> nanovdb_triangles(nano_tris_data, nano_tris_data + openvdb_triangles.size());

        // Convert OpenVDB transform to nanovdb::Map
        const auto openvdb_mat4 = transform->baseMap()->getAffineMap()->getMat4();
        nanovdb::Map map;
        map.set(openvdb_mat4, openvdb_mat4.inverse());

        mainMeshToGrid<BuildT>(
            nanovdb_points.data().get(),
            nanovdb_points.size(),
            nanovdb_triangles.data().get(),
            nanovdb_triangles.size(),
            map,
            grid);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
