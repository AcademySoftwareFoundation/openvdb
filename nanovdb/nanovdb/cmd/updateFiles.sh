#!/bin/bash
#Usage process all files in this directory or optionally specify a target directory

# Define directory in which to find files
dir="."
if [ "$1" ]; then
    dir="$1"
fi

# Check if dir is not a directory
if [ ! -d "$dir" ]; then
  echo -e "\nUsage: '$0 <directory>'\n"
  exit 1
fi

# E.g.: func1 $file "math" "Coord" "Vec3" "Vec4"
func1 () {
    for ((i=3; i<=$#; i++)); do
        arg="s/nanovdb::${!i}/nanovdb::$2::${!i}/g"
        #echo "sed -i $arg $1"
        sed -i $arg $1
    done
}

# E.G.: func2 file namespace old new : nanovdb::old -> nanovdb::namespace::new in file
func2 () {
    arg="s/nanovdb::$3/nanovdb::$2::$4/g"
    #echo "sed -i $arg $1"
    sed -i $arg $1
}

# E.G.: func3 file path1/old.h path2/new.h  <nanovdb/path1/old.h> -> <nanovdb/path2/new.h> in file
func3 () {
    arg="s;<nanovdb/$2>;<nanovdb/$3>;g"
    #echo "sed -i $arg $1"
    sed -i $arg $1
}

# E.g.: func4 file old new   new -> old
func4 () {
    arg="s;$2;$3;g"
    #echo "sed -i $arg $1"
    sed -i $arg $1
}

# Loop through files in the target directory
for file in $(find "$dir" -name '*.h' -or -name '*.cuh' -or -name '*.cc' -or -name '*.cu' -or -name '*.cpp'); do
  if [ -f "$file" ]; then
    echo "Processing file: $file"
    func1 $file math Ray "DDA<" HDDA "Vec3<" "Vec4<" "BBox<" ZeroCrossing TreeMarcher PointTreeMarcher\
                "BoxStencil<" "CurvatureStencil<" "GradStencil<" "WenoStencil<" AlignUp Min Max Abs Clamp\
                Sqrt Sign "Maximum<" "Delta<" "RoundDown<" "pi<" "isApproxZero<" "Round<" createSampler "SampleFromVoxels<"
    func1 $file tools createNanoGrid StatsMode createLevelSetSphere\
                createFogVolumeSphere createFogVolumeSphere createFogVolumeSphere\
                createFogVolumeTorus createLevelSetBox CreateNanoGrid updateGridStats\
                evalChecksum validateChecksum checkGrid Extrema
    func1 $file util is_floating_point findLowestOn findHighestOn Range streq strcpy strcat "empty("\
                Split invoke forEach reduce prefixSum is_same is_specialization PtrAdd PtrDiff
    func4 $file "nanovdb::build::" "nanovdb::tools::build::"
    func4 $file "nanovdb::BBoxR" "nanovdb::Vec3dBBox"
    func4 $file "nanovdb::BBox<nanovdb::Vec3d>" "nanovdb::Vec3dBbox"
    func2 $file cuda cudaCreateNodeManager createNodeManager
    func2 $file cuda cudaVoxelsToGrid voxelsToGrid
    func2 $file cuda cudaPointsToGrid pointsToGrid
    func2 $file math DitherLUT DitherLUT
    func2 $file math PackedRGBA8 Rgba8
    func2 $file math Rgba8 Rgba8
    func2 $file util CpuTimer Timer
    func2 $file util GpuTimer "cuda::Timer"
    func2 $file util CountOn countOn
    func3 $file "util/GridHandle.h" "GridHandle.h"
    func3 $file "util/GridHandle.h" "HostBuffer.h"
    func3 $file "util/BuildGrid.h"   "tools/GridBuilder.h"
    func3 $file "util/GridBuilder.h" "tools/GridBuilder.h"
    func3 $file "util/IO.h" "io/IO.h"
    func3 $file "util/CSampleFromVoxels.h" "math/CSampleFromVoxels.h"
    func3 $file "util/DitherLUT.h" "math/DitherLUT.h"
    func3 $file "util/HDDA.h" "math/HDDA.h"
    func3 $file "util/Ray.h" "math/Ray.h"
    func3 $file "util/SampleFromVoxels.h" "math/SampleFromVoxels.h"
    func3 $file "util/Stencils.h" "nanovdb/math/Stencils.h"
    func3 $file "util/CreateNanoGrid.h" "tools/CreateNanoGrid.h"
    func3 $file "util/Primitives.h" "tools/CreatePrimitives.h"
    func3 $file "util/GridChecksum.h" "tools/GridChecksum.h"
    func3 $file "util/GridStats.h" "tools/GridStats.h"
    func3 $file "util/GridChecksum.h" "tools/GridChecksum.h"
    func3 $file "util/GridValidator.h" "tools/GridValidator.h"
    func3 $file "util/NanoToOpenVDB.h" "tools/NanoToOpenVDB.h"
    func3 $file "util/cuda/CudaGridChecksum.cuh" "tools/cuda/CudaGridChecksum.cuh"
    func3 $file "util/cuda/CudaGridStats.cuh" "tools/cuda/CudaGridStats.cuh"
    func3 $file "util/cuda/CudaGridValidator.cuh" "tools/cuda/CudaGridValidator.cuh"
    func3 $file "util/cuda/CudaIndexToGrid.cuh" "tools/cuda/CudaIndexToGrid.cuh"
    func3 $file "util/cuda/CudaPointsToGrid.cuh" "tools/GridChecksum.cuh"
    func3 $file "util/cuda/CudaSignedFloodFill.cuh" "tools/cuda/CudaSignedFloodFill.cuh"
    func3 $file "util/cuda/CudaDeviceBuffer.h" "cuda/DeviceBuffer.h"
    func3 $file "util/cuda/CudaGridHandle.cuh" "cuda/GridHandle.cuh"
    func3 $file "util/cuda/CudaUtils.h" "util/cuda/Util.h"
    func3 $file "util/cuda/GpuTimer.h" "util/cuda/Timer.h"
  fi
done