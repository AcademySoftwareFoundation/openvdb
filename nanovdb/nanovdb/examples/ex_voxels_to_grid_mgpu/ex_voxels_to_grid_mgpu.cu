// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>

namespace
{

template<typename BufferT = nanovdb::HostBuffer>
std::vector<nanovdb::Coord>
initializeCoordListFromCSV(const std::string& filename)
{
    std::ifstream file;
    file.open(filename);
    std::string line;
    std::vector<nanovdb::Coord> coords;
    while (getline(file, line)) {
        std::istringstream stream(line);
        nanovdb::Coord coord;
        stream >> coord.x(); stream.ignore(1, ',' );
        stream >> coord.y(); stream.ignore(1, ',' );
        stream >> coord.z();
        coords.emplace_back(coord);
    }
    file.close();
    return coords;
}

}

/// @brief Demonstrates how to create a NanoVDB grid from voxel coordinates on the GPU
void testVoxelsToGrid(const std::string& filename)
{
    nanovdb::cuda::DeviceMesh deviceMesh;

    std::cout << "CSV index source : " << filename << std::endl;
    std::cout << "==================================================" << std::endl;
    std::vector<nanovdb::Coord> hostCoords = initializeCoordListFromCSV(filename);
    std::cout << "Number of coords : " << hostCoords.size() << std::endl;

    size_t coordCount = hostCoords.size();
    nanovdb::Coord* coords = nullptr;
    cudaMallocManaged(&coords, coordCount * sizeof(nanovdb::Coord));
    cudaMemcpy(coords, hostCoords.data(), coordCount * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);

    nanovdb::tools::cuda::DistributedPointsToGrid<nanovdb::ValueOnIndex> distributedConverter(deviceMesh);
    auto distributedHandle = distributedConverter.getHandle(coords, coordCount);

    cudaFree(coords);
}

int main(int argc, char** argv)
{
    if (argc != 2)
        throw std::runtime_error("Expected CSV filename as command-line argument");
    std::string filename(argv[1]);
    testVoxelsToGrid(filename);

    return 0;
}
