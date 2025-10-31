# Enable verbose and stop on error
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# Install numpy for tests
& "$env:VCPKG_INSTALLATION_ROOT\installed\$env:VCPKG_DEFAULT_TRIPLET\tools\python3\python.exe" -m ensurepip --upgrade
& "$env:VCPKG_INSTALLATION_ROOT\installed\$env:VCPKG_DEFAULT_TRIPLET\tools\python3\python.exe" -m pip install numpy

Write-Host "pip numpy install completed successfully"
