// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file main.cpp
///
/// @brief Level set benchmark test
///
////////////////////////////////////////////////////////////////////////////////

// the following files are from OpenVDB
#include <openvdb/util/CpuTimer.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/FastSweeping.h>

// note the file is not part of OpenVDB. It was developed specifically for this test
#define OPENVDB_INSTANTIATE_LEVELSETTRACKERNEW
#include <LevelSetPropagate.h>
#include <VelocityExtension.h>

// the following files are from NanoVDB
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/NanoToOpenVDB.h>

#define USE_NEW_SPEED

// Define the scalar speed function used by the benchmark
template <typename GridT>
class SpeedFunction {
  static const float sAlpha, sBeta, sLateralRatio;
  mutable openvdb::math::BoxStencil<GridT> mStencil;// mutable since operator() must be const
public:
  SpeedFunction(const GridT &levelSet) : mStencil(levelSet) {}
  SpeedFunction(const SpeedFunction &other) : mStencil(other.mStencil.grid()) {}
  float operator()(const openvdb::Vec3R &xyz) const {
    mStencil.moveTo(xyz);
    auto grad = mStencil.gradient(xyz);
    grad.normalize();
    auto speed = grad[2] < sAlpha ? 0.0f : (grad[2] - sAlpha)*sBeta;
    return (1.0f - sLateralRatio) * speed - sLateralRatio;
  }
  float fromGradient(const openvdb::Vec3R &gradient) const {
    auto grad = gradient;
    grad.normalize();
    auto speed = grad[2] < sAlpha ? 0.0f : (grad[2] - sAlpha)*sBeta;
    return (1.0f - sLateralRatio) * speed - sLateralRatio;
  }
}; // SpeedFunction class
// Initiate static member data of the speed function
template <typename GridT>
const float SpeedFunction<GridT>::sAlpha = std::cos(openvdb::math::pi<float>()*(0.5f - 30.0f/180.f));
template <typename GridT>
const float SpeedFunction<GridT>::sBeta = 1.0f/(sAlpha - 1.0f);
template <typename GridT>
const float SpeedFunction<GridT>::sLateralRatio = 0.2f;

int main(int argc, char *argv[])
{
  using GridT = openvdb::FloatGrid;
  openvdb::initialize();
  openvdb::util::CpuTimer timer;
  int exitStatus = EXIT_SUCCESS;
  std::vector<std::string> names;// executable, input.vdb, output.vdb

  try {
    float dt = 0.25f, tMin = 0.0f, tMax = 100.0f, backgroundSpeed = 0.0f;
    int frameCounter = 0, frameWrite = 0;
    for (int i=0; i<argc; ++i) {
      std::string str(argv[i]), cpy(str);
      std::transform(cpy.begin(), cpy.end(), cpy.begin(),[](unsigned char c){ return std::tolower(c); });
      if (cpy == "-dt") {
        if (++i == argc) OPENVDB_THROW(openvdb::ValueError, "error: missing argument to option \""+str+"\"\n");
        dt = std::stof(argv[i]);
      } else if (cpy == "-tmin") {
         if (++i == argc) OPENVDB_THROW(openvdb::ValueError, "error: missing argument to option \""+str+"\"\n");
        tMin = std::stof(argv[i]);
      } else if (cpy == "-tmax") {
         if (++i == argc) OPENVDB_THROW(openvdb::ValueError, "error: missing argument to option \""+str+"\"\n");
        tMax = std::stof(argv[i]);
      } else if (cpy == "-write") {
         if (++i == argc) OPENVDB_THROW(openvdb::ValueError, "error: missing argument to option \""+str+"\"\n");
        frameWrite = std::stoi(argv[i]);
      } else if (cpy == "-cpu") {
        Benchmark::getInstance().mPlatform = ExecutionPolicy::CPU;
      } else if (str[0]=='-') {
        OPENVDB_THROW(openvdb::ValueError, "error: unknown option \""+str+"\"\n");
      } else {// must be executable, input or output names
        names.push_back(str);
      }
    }
    if (names.size()!=3) OPENVDB_THROW(openvdb::ValueError, "usage: "+names[0]+" [-dx 0.25 -tmin 0 -tmax 100 -write 1] input.vdb output.vdb\n");
    for (int i=1; i<3; ++i) {
      const std::string str(names[i]);
      if (str.substr(str.length()-4)!=".vdb") OPENVDB_THROW(openvdb::ValueError, "expected extension \".vdb\" in \""+str+"\"\n");
    }
    if (frameWrite == 0) frameWrite = int((tMax - tMin)/dt);
    std::cout << "Time step: " << dt << ", min time: " << tMin << ", max time: " << tMax << ", writing every " << frameWrite << " frame\n";

    // Read the initial level set from file
    openvdb::util::CpuTimer timer("Read input VDB file \""+names[1]+"\"");
    openvdb::io::File inFile(names[1]);
    inFile.open(false);// disable delayed loading
    auto baseGrids = inFile.getGrids();
    inFile.close();
    auto levelSet = openvdb::gridPtrCast<GridT>(baseGrids->at(0));
    if (!levelSet) OPENVDB_THROW(openvdb::ValueError, "First grid is not a FloatGrid\n");
    timer.stop();

    // instantiate the LevelSetPropagation class
    openvdb::tools::LevelSetPropagate<GridT> prop(*levelSet);
    prop.setSpatialScheme(openvdb::math::BiasedGradientScheme::HJWENO5_BIAS);
    prop.setTemporalScheme(openvdb::math::TVD_RK2);
    prop.setTrackerSpatialScheme(openvdb::math::BiasedGradientScheme::HJWENO5_BIAS);
    prop.setNormCount(3);
    prop.setTrackerTemporalScheme(openvdb::math::TVD_RK2);
    
    // Initialize NanoVDB
    Benchmark::copyHostOpenVDBToDeviceNanoVDB(*levelSet, Benchmark::getInstance().mHandle);
    if (!Benchmark::compareHostOpenVDBToDeviceNanoVDB(*levelSet, Benchmark::getInstance().mHandle))
        throw std::runtime_error("Inconsistent grids between OpenVDB (host) and NanoVDB (GPU)");
    Benchmark::printGridDiagnostics(Benchmark::getInstance().mHandle);
    Benchmark::getInstance().mBackground = levelSet->background();
    Benchmark::getInstance().mDx = static_cast<GridT::ValueType>(levelSet->voxelSize()[0]);
    if (Benchmark::getInstance().onCPU())
        Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mPhi, Benchmark::getInstance().mBackground);
    else
        Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mPhi, Benchmark::getInstance().mBackground);
    Benchmark::copyOpenVDBDataToNanoVDBSidecar(*levelSet, Benchmark::getInstance().mHandle, Benchmark::getInstance().mPhi);    
    if (!Benchmark::compareOpenVDBDataToNanoVDBSidecar(*levelSet, Benchmark::getInstance().mHandle, Benchmark::getInstance().mPhi))
        throw std::runtime_error("Inconsistent levelset sidecars between OpenVDB (host) and NanoVDB (GPU)");
    if (Benchmark::getInstance().onCPU())
        Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle);
    else
        Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle);

    const std::string base = names[2].substr(0,names[2].length()-4) + "_";
    for (float time = tMin + dt; time <= tMax; time += dt) {

      std::cout << "\n" << std::setfill('=') << std::setw(30) << "> Frame: " << ++frameCounter << ", Time: " 
                << time  << " <" << std::setfill('=') << std::setw(25) << "\n" << std::endl;
      
      // Compute speed grid via NanoVDB
      timer.start("Updating speed [NanoVDB]");
      if (Benchmark::getInstance().onCPU())
          Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mSpeed, 0);
      else
          Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mSpeed, 0);
      if (Benchmark::getInstance().onCPU())
          Benchmark::updateSpeedGrid<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi,
              Benchmark::getInstance().mSpeed, Benchmark::getInstance().mDx, true);
      else
          Benchmark::updateSpeedGrid<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi,
              Benchmark::getInstance().mSpeed, Benchmark::getInstance().mDx, true);
      timer.stop();
      
      const size_t cflSteps = prop.propagate(time - dt, time);
      if (cflSteps!=1) std::cerr << "Performed " << cflSteps << " CFL time step(s)" << std::endl;

      if ((frameCounter % frameWrite) == 0) {
        std::stringstream ss;
        ss << base << std::setfill('0') << std::setw(4) << frameCounter << ".vdb";
        const std::string name = ss.str();        
        timer.start("Write output to \""+name+"\"");
        auto hostGridBuffer = nanovdb::HostBuffer::create(Benchmark::getInstance().mHandle.size());
        if (hostGridBuffer.isEmpty()) throw std::runtime_error("Could not allocate host copy of indexgrid for output");
        cudaCheck(cudaMemcpy(hostGridBuffer.data(), Benchmark::getInstance().mHandle.deviceData(), Benchmark::getInstance().mHandle.size(), cudaMemcpyDeviceToHost));
        auto hostIndexGrid = static_cast<nanovdb::NanoGrid<nanovdb::ValueOnIndex>*>(hostGridBuffer.data());
        auto hostSidecarBuffer = nanovdb::HostBuffer::create(Benchmark::getInstance().mPhi.size());
        if (hostSidecarBuffer.isEmpty()) throw std::runtime_error("Could not allocate host copy of sidecar for output");
        cudaCheck(cudaMemcpy(hostSidecarBuffer.data(), Benchmark::getInstance().mPhi.deviceData(), Benchmark::getInstance().mPhi.size(), cudaMemcpyDeviceToHost));
        auto hostSidecar = static_cast<float*>(hostSidecarBuffer.data());
        std::string gridName = "Level_Set_"+name.substr(names[2].find_last_of("/\\") + 1);
        auto openGrid = nanovdb::tools::nanoToOpenVDB(*hostIndexGrid, hostSidecar, nanovdb::GridClass::LevelSet, gridName.c_str());
        openvdb::io::File outFile(name);
        outFile.write({openGrid});
        outFile.close();
        timer.stop();
      }

    }// loop over time

  } catch (const std::exception& e) {

    OPENVDB_LOG_FATAL(names[0] + ": " + e.what());
    exitStatus = EXIT_FAILURE;

  } catch (...) {

    OPENVDB_LOG_FATAL(names[0] + ": exception of unknown type caught");
    exitStatus = EXIT_FAILURE;
  }

  return exitStatus;

} //end of main
