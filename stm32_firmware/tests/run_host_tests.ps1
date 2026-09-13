param(
    [string]$BuildDir = (Join-Path $PSScriptRoot 'build-host-msvc'),
    [string]$Generator = 'Visual Studio 16 2019'
)

$ErrorActionPreference = 'Stop'
$SourceDir = $PSScriptRoot

cmake -S $SourceDir -B $BuildDir -G $Generator -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

ctest --test-dir $BuildDir -C Debug --output-on-failure
exit $LASTEXITCODE
