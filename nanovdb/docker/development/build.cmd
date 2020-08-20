echo off

set DOCKER_DIR=%~dp0
set DIST_DIR=%DOCKER_DIR%\__dist
set REPO_DIR=%DOCKER_DIR%\..\..
echo "DOCKER_DIR: %DOCKER_DIR%"

set IMAGE="oddsocks/nanovdb"

rem -- copy local files to release directory...
mkdir %DIST_DIR%\scripts
pushd %DOCKER_DIR%\..

rem del %DIST_DIR%\scripts.tar /q
rem tar -cvf %DIST_DIR%\scripts.tar scripts
del /s /q %DIST_DIR%\scripts\*
xcopy /S /E /Y scripts\* %DIST_DIR%\scripts\

popd

rem -- build the image.
docker build -t %IMAGE%:dev-base -f %DOCKER_DIR%\Dockerfile.base %DOCKER_DIR%

:Exit
