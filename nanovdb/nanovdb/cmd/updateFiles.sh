#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
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

# E.g.: func5 file regex new   ERE regex -> new in file
func5 () {
    arg="s/$2/$3/g"
    #echo "sed -E -i $arg $1"
    sed -E -i "$arg" $1
}

# Loop through files in the target directory
for file in $(find "$dir" -name '*.h' -or -name '*.cuh' -or -name '*.cc' -or -name '*.cu' -or -name '*.cpp'); do
  if [ -f "$file" ]; then
    echo "Processing file: $file"
    func1 $file math Ray "DDA<" HDDA "Vec3<" "Vec4<" "BBox<" TreeMarcher PointTreeMarcher\
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
    func4 $file "nanovdb::BBox<nanovdb::Vec3d>" "nanovdb::Vec3dBBox"
    func2 $file cuda cudaCreateNodeManager createNodeManager
    func2 $file math DitherLUT DitherLUT
    func2 $file math PackedRGBA8 Rgba8
    func2 $file math Rgba8 Rgba8
    func2 $file util CpuTimer Timer
    func2 $file util GpuTimer "cuda::Timer"
    func2 $file util CountOn countOn
    func2 $file tools nanoToOpenVDB nanoToOpenVDB
    func2 $file tools openToNanoVDB openToNanoVDB
    # Renamed functions, whether or not the old spelling is namespace-qualified
    func5 $file "(nanovdb::)?(math::)?\bZeroCrossingNode\b" "nanovdb::math::zeroCrossingNode"
    func5 $file "(nanovdb::)?(math::)?\bZeroCrossing\b" "nanovdb::math::zeroCrossing"
    func5 $file "(nanovdb::)?\bFindLowestOn\b" "nanovdb::util::findLowestOn"
    func5 $file "(nanovdb::)?\bFindHighestOn\b" "nanovdb::util::findHighestOn"
    func5 $file "(nanovdb::)?\bmapToGridType\b" "nanovdb::toGridType"
    func5 $file "(nanovdb::)?\bmapToGridClass\b" "nanovdb::toGridClass"
    # pointsToGrid has no default voxel size, so a call that relied on it needs an explicit 1.0
    func5 $file "(nanovdb::)?\bcudaPointsToGrid\b" "nanovdb::tools::cuda::pointsToGrid"
    func5 $file "(nanovdb::)?\bcudaVoxelsToGrid\b" "nanovdb::tools::cuda::voxelsToGrid"
    func5 $file "(nanovdb::)?\bcudaIndexToGrid\b" "nanovdb::tools::cuda::indexToGrid"
    func5 $file "(nanovdb::)?\bcudaCreateNanoGrid\b" "nanovdb::tools::cuda::indexToGrid"
    func5 $file "(nanovdb::)?\bcudaSignedFloodFill\b" "nanovdb::tools::cuda::signedFloodFill"
    func5 $file "(nanovdb::)?\bcudaAddBlindData\b" "nanovdb::tools::cuda::addBlindData"
    func5 $file "(nanovdb::)?\bcudaGridStats\b" "nanovdb::tools::cuda::updateGridStats"
    func5 $file "(nanovdb::)?\bcudaIsValid\b" "nanovdb::tools::cuda::isValid"
    func5 $file "\bcudaLambdaKernel\b" "nanovdb::util::cuda::lambdaKernel"
    # These now take a grid pointer instead of a reference, so the argument may also need an &
    func5 $file "\b(nanovdb::(tools::)?|tools::)gridStats\b" "nanovdb::tools::updateGridStats"
    func5 $file "\b(nanovdb::(tools::)?|tools::)checksum\b" "nanovdb::tools::evalChecksum"
    func5 $file "\bisPartial\(\)" "isHalf()"
    func3 $file "util/GridHandle.h" "GridHandle.h"
    func3 $file "util/BuildGrid.h"   "tools/GridBuilder.h"
    func3 $file "util/GridBuilder.h" "tools/GridBuilder.h"
    func3 $file "util/IO.h" "io/IO.h"
    func3 $file "util/CSampleFromVoxels.h" "math/CSampleFromVoxels.h"
    func3 $file "util/DitherLUT.h" "math/DitherLUT.h"
    func3 $file "util/HDDA.h" "math/HDDA.h"
    func3 $file "util/Ray.h" "math/Ray.h"
    func3 $file "util/SampleFromVoxels.h" "math/SampleFromVoxels.h"
    func3 $file "util/Stencils.h" "math/Stencils.h"
    func3 $file "util/CreateNanoGrid.h" "tools/CreateNanoGrid.h"
    func3 $file "util/Primitives.h" "tools/CreatePrimitives.h"
    func3 $file "util/GridChecksum.h" "tools/GridChecksum.h"
    func3 $file "util/GridStats.h" "tools/GridStats.h"
    func3 $file "util/GridChecksum.h" "tools/GridChecksum.h"
    func3 $file "util/GridValidator.h" "tools/GridValidator.h"
    func3 $file "util/NanoToOpenVDB.h" "tools/NanoToOpenVDB.h"
    func3 $file "util/OpenToNanoVDB.h" "tools/CreateNanoGrid.h"
    func3 $file "util/CpuTimer.h" "util/Timer.h"
    func3 $file "util/HostBuffer.h" "HostBuffer.h"
    func3 $file "util/NodeManager.h" "NodeManager.h"
    func3 $file "util/cuda/CudaGridChecksum.cuh" "tools/cuda/GridChecksum.cuh"
    func3 $file "util/cuda/CudaGridStats.cuh" "tools/cuda/GridStats.cuh"
    func3 $file "util/cuda/CudaGridValidator.cuh" "tools/cuda/GridValidator.cuh"
    func3 $file "util/cuda/CudaIndexToGrid.cuh" "tools/cuda/IndexToGrid.cuh"
    func3 $file "util/cuda/CudaPointsToGrid.cuh" "tools/cuda/PointsToGrid.cuh"
    func3 $file "util/cuda/CudaSignedFloodFill.cuh" "tools/cuda/SignedFloodFill.cuh"
    func3 $file "util/cuda/CudaAddBlindData.cuh" "tools/cuda/AddBlindData.cuh"
    func3 $file "util/cuda/CudaNodeManager.cuh" "cuda/NodeManager.cuh"
    func3 $file "util/cuda/CudaDeviceBuffer.h" "cuda/DeviceBuffer.h"
    func3 $file "util/cuda/CudaGridHandle.cuh" "cuda/GridHandle.cuh"
    func3 $file "util/cuda/CudaUtils.h" "util/cuda/Util.h"
    func3 $file "util/cuda/GpuTimer.h" "util/cuda/Timer.h"
  fi
done
