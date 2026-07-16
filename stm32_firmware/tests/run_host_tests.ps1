param()

$ErrorActionPreference = 'Stop'
$SourceDir = $PSScriptRoot
$BuildDir = Join-Path $SourceDir 'build'

cmake -S $SourceDir -B $BuildDir -G 'Visual Studio 16 2019' -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

ctest --test-dir $BuildDir -C Debug --output-on-failure
exit $LASTEXITCODE
