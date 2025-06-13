@echo off

:: global env vars for cmake
set VCPKG_TARGET_TRIPLET="x64-windows"

setlocal

set PROJECT_NAME=nanovdb_editor
set PROJECT_DIR=%~dp0
set BUILD_DIR=%PROJECT_DIR%build
set PYTHON_BUILD_DIR=%PROJECT_DIR%pymodule\nanovdb_editor\lib
set CONFIG_FILE=config.ini
set SLANG_DEBUG_OUTPUT=OFF
set CLEAN_SHADERS=OFF
set DEBUG_PYTHON=OFF

set clean_build=0
set release=0
set debug=0
set verbose=0
set python=0

:parse_args
set args=%1
if "%args%"=="" goto end_parse_args
if "%args:~0,1%"=="-" (
    set "arg=%args:~1%"
    shift
    goto parse_arg
)

echo Invalid option: %args%
goto Usage

:parse_arg
if "%arg%"=="" goto parse_args
if "%arg:~0,1%"=="x" (
    set clean_build=1
    goto check_next_char
)
if "%arg:~0,1%"=="r" (
    set release=1
    goto check_next_char
)
if "%arg:~0,1%"=="d" (
    set debug=1
    goto check_next_char
)
if "%arg:~0,1%"=="v" (
    set verbose=1
    goto check_next_char
)
if "%arg:~0,1%"=="s" (
    set SLANG_DEBUG_OUTPUT=ON
    set CLEAN_SHADERS=ON
    goto check_next_char
)
if "%arg:~0,1%"=="p" (
    set python=1
    goto check_next_char
)
if "%arg:~0,1%"=="h" (
    goto Usage
)

echo Invalid option: %arg%
goto Usage

:check_next_char
set "arg=%arg:~1%"
goto parse_arg

:end_parse_args

set only_python=0

:: set defaults
if %release%==0 (
    if %debug%==0 (
        if %python%==1 (
            set only_python=1
        ) else (
            set release=1
        )
    )
)

:: set env vars from a config file
for /f "tokens=1,2 delims==" %%i in (%PROJECT_DIR%%CONFIG_FILE%) do (
  set %%i=%%j
)

:: check the required variables has been set
if not defined MSVS_VERSION (
    echo MSVS_VERSION not set, please set Visual Studio CMake generator name
    goto Error
)
if not defined VCPKG_ROOT (
    echo VCPKG_ROOT not set, please set path to vcpkg
    goto Error
)

goto Build

:Success
echo -- Build of %PROJECT_NAME% completed
exit /b 0

:Error
echo Failure while building %PROJECT_NAME%
exit /b %errorlevel%

:Usage
echo Usage: build [-x] [-r] [-d] [-v] [-s] [-p]
echo        -x  Perform a clean build
echo        -r  Build in release (default)
echo        -d  Build in debug
echo        -v  Enable CMake verbose output
echo        -s  Compile slang into ASM
echo        -p  Build and install python module, set also -d to build in debug
exit /b 1

:Build

if %clean_build%==1 (
    echo -- Performing a clean build...
    if exist %BUILD_DIR% (
        rmdir /s /q %BUILD_DIR%
        echo -- Deleted %BUILD_DIR%
    )
    set CLEAN_SHADERS=ON
)

if %verbose%==1 (
    set CMAKE_VERBOSE=--verbose
 ) else (
    set CMAKE_VERBOSE=
 )

:: need to create config directories in advance
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

if %release%==1 (
    call :CreateConfigDir Release
)
if %debug%==1 (
    call :CreateConfigDir Debug
    if exist %PYTHON_BUILD_DIR% (
        rmdir /s /q %PYTHON_BUILD_DIR%
    )
    if %release%==0 (
        set DEBUG_PYTHON=ON
    )
)

set VCPKG_CMAKE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

if %only_python%==0 (
    echo -- Building %PROJECT_NAME%...

    set SLANG_PROFILE_ARG=
    if defined SLANG_PROFILE (
        set SLANG_PROFILE_ARG=-DNANOVDB_EDITOR_SLANG_PROFILE=%SLANG_PROFILE%
    )

    cmake -G %MSVS_VERSION% %PROJECT_DIR% -B %BUILD_DIR% ^
        -DCMAKE_TOOLCHAIN_FILE=%VCPKG_CMAKE% ^
        -DNANOVDB_EDITOR_USE_VCPKG=ON ^
        -DNANOVDB_EDITOR_CLEAN_SHADERS=%CLEAN_SHADERS% ^
        -DNANOVDB_EDITOR_SLANG_DEBUG_OUTPUT=%SLANG_DEBUG_OUTPUT% ^
        -DNANOVDB_EDITOR_DEBUG_PYTHON=%DEBUG_PYTHON% ^
        %SLANG_PROFILE_ARG%
)

if %release%==1 (
    call :BuildConfig Release
)
if %debug%==1 (
    call :BuildConfig Debug
)
if %python%==1 (
    echo -- Installing python module...
    cd pymodule
    python setup.py bdist_wheel
    pip install .
    cd ..
)

goto Success

:CreateConfigDir
set BUILD_DIR_CONFIG=%BUILD_DIR%\%1
if not exist %BUILD_DIR_CONFIG% mkdir %BUILD_DIR_CONFIG%
exit /b 1

:BuildConfig
set CONFIG=%1
echo -- Building config %CONFIG%...
cmake --build %BUILD_DIR% --config %CONFIG% %CMAKE_VERBOSE%

if %errorlevel% neq 0 (
    echo Failure while building %CONFIG%
    goto Error
) else (
    echo Built config %CONFIG%
)
exit /b 1
