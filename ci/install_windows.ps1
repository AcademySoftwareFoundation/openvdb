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
    "jemalloc"
)

$maxAttempts = 3

# curl's schannel backend reports an unreachable CRL/OCSP responder as a
# certificate verification failure (error 60), which vcpkg then treats as
# permanent. Downgrade a missing revocation answer to a warning; the rest of
# certificate validation still applies.
$env:VCPKG_SSL_REVOKE_BEST_EFFORT = "1"

# Update vcpkg
vcpkg update

$installed = $false

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    vcpkg install $vcpkgPackages

    # A failing native command does not raise a terminating error, so the exit
    # code has to be inspected explicitly rather than relying on try/catch.
    if ($LASTEXITCODE -eq 0) {
        $installed = $true
        break
    }

    if ($attempt -eq $maxAttempts) {
        break
    }

    # vcpkg fetches port sources directly from upstream hosts and won't retry
    # downloads it classifies as permanent failures, so a single flaky TLS
    # handshake aborts the whole install.
    Write-Host "vcpkg install failed (attempt $attempt of $maxAttempts), retrying..."
    Start-Sleep -Seconds 15

    # Refresh the ports before the last attempt in case the failure is caused
    # by a stale port rather than the network.
    if ($attempt -eq ($maxAttempts - 1)) {
        Write-Host "Retrying with latest ports..."
        Push-Location $env:VCPKG_INSTALLATION_ROOT
        git pull
        Pop-Location
        vcpkg update
    }
}

if (-not $installed) {
    throw "vcpkg install failed after $maxAttempts attempts"
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
