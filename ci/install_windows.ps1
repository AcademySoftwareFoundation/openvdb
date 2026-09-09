# Enable verbose and stop on error
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# Required dependencies
$vcpkgPackages = @(
    "zlib",
    "libpng",
    "openexr",
    "tbb",
    "gtest",
    "cppunit",
    "blosc",
    "glfw3",
    "glew",
    "python3",
    "jemalloc",
    "boost-iostreams",
    "boost-interprocess",
    "boost-algorithm"
)

# Update vcpkg
vcpkg update

# Allow the vcpkg command to fail once so we can retry with the latest
try {
    vcpkg install $vcpkgPackages
} catch {
    Write-Host "vcpkg install failed, retrying with latest ports..."
    # Retry the installation with updated ports
    Push-Location $env:VCPKG_INSTALLATION_ROOT
    git pull
    Pop-Location
    vcpkg update
    vcpkg install $vcpkgPackages
}

Write-Host "vcpkg install completed successfully"

# nanobind comes from source rather than vcpkg: the vcpkg port floats with the
# runner's baseline (nanobind 3.0 broke the CUDA python bindings under
# --Werror=all-warnings), while every other CI platform pins the version
# through ci/install_nanobind.sh. Pin the same version here, installed into
# the vcpkg tree so the toolchain finds it exactly as it found the port.
$nanobindVersion = "2.5.0"
git clone --recurse-submodules --depth 1 --branch "v$nanobindVersion" https://github.com/wjakob/nanobind.git
cmake -S nanobind -B nanobind\build -DNB_TEST=OFF "-DCMAKE_INSTALL_PREFIX=$env:VCPKG_INSTALLATION_ROOT\installed\$env:VCPKG_DEFAULT_TRIPLET"
if ($LASTEXITCODE -ne 0) { throw "nanobind configure failed" }
cmake --install nanobind\build
if ($LASTEXITCODE -ne 0) { throw "nanobind install failed" }

Write-Host "nanobind $nanobindVersion install completed successfully"
