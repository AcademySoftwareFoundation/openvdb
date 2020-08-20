echo off

rem IF "%1"=="" GOTO Usage

set GIT_BRANCH=%1
set DOCKER_DIR=%~dp0
set DIST_DIR=%DOCKER_DIR%\__dist
set REPO_DIR=%DOCKER_DIR%\..\..
echo "DOCKER_DIR: %DOCKER_DIR%"

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
    docker build -t %IMAGE%:dev-test -f %DOCKER_DIR%\Dockerfile.test %DOCKER_DIR%
rem )

GOTO Exit

:Usage
echo "Usage: build.cmd <branch|commit>"
GOTO Exit

:Exit
