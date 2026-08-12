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
