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
    "boost-algorithm",
    "nanobind"
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
