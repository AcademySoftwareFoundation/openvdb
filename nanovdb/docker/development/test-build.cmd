echo off

set COMPILER_NAME=%1
IF NOT "%COMPILER_NAME%"=="" GOTO FoundCompiler
GOTO NoCompiler
:FoundCompiler
set CUDA_VER=%2
IF NOT "%CUDA_VER%"=="" GOTO FoundCudaVer
GOTO NoCompiler
:FoundCudaVer
set GIT_BRANCH=%3
:NoCompiler

IF "%COMPILER_NAME%"=="" set COMPILER_NAME="clang++"
IF "%CUDA_VER%"=="" set CUDA_VER=10.2

echo Building with compiler: %COMPILER_NAME%
echo CUDA version: %CUDA_VER%

set DOCKER_DIR=%~dp0
set DIST_DIR=%DOCKER_DIR%\__dist
set REPO_DIR=%DOCKER_DIR%\..\..
echo DOCKER_DIR: %DOCKER_DIR%

set IMAGE="oddsocks/nanovdb"

rem -- copy local files to release directory...
mkdir %DIST_DIR%

del %DIST_DIR%\repo.tar /q
IF "%GIT_BRANCH%"=="" GOTO NoRepo
pushd %REPO_DIR%
git archive %GIT_BRANCH% -o %DIST_DIR%\repo.tar
popd
GOTO BuildImage
:NoRepo
pushd %REPO_DIR%
tar --exclude="data" --exclude="out" --exclude=".git" --exclude="__*" -cvf %DIST_DIR%\repo.tar .
popd
GOTO BuildImage

:BuildImage
rem IF EXIST %DIST_DIR%\repo.tar (
    docker build -t %IMAGE%:dev-test-build -f %DOCKER_DIR%\Dockerfile.test-build --build-arg COMPILER=%COMPILER_NAME% --build-arg CUDA_VER=%CUDA_VER% %DOCKER_DIR% 
rem )

GOTO Exit

:Usage
echo "Usage: test-build.cmd <compiler e.g. g++,g++-8,clang++,etc.> <cuda_version=9.2,10.2> <branch|commit>"
GOTO Exit

:Exit
