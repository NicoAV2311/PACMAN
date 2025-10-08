# Build script for PacPy on Windows using PyInstaller
# Usage: run this from the project folder in PowerShell:
#   .\build.ps1
# Requirements: Python (same version you used to develop), pip install pyinstaller

$exeName = "PACMAN"
$src = "PACMAN.py"
$dataDir = "pacman_data"

# Clean previous builds
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }

# Build one-folder by default; uncomment --onefile for single EXE (larger, needs more testing)
# Note: Use the proper -add-data separator on Windows: source;target
$addData = "$dataDir;$dataDir"

Write-Host "Building $exeName with PyInstaller..."
pyinstaller --noconfirm --clean `
    --name $exeName `
    --add-data $addData `
    --windowed `
    $src

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed. See output above."; exit $LASTEXITCODE
}

# Optionally create a zip of the dist folder for submission
$zipName = "$exeName`_dist.zip"
if (Test-Path .\dist\$exeName) {
    if (Test-Path $zipName) { Remove-Item $zipName -Force }
    Compress-Archive -Path .\dist\$exeName\* -DestinationPath $zipName -Force
    Write-Host "Created $zipName"
}

Write-Host "Build finished. Dist folder: .\dist\$exeName"