param(
    [string]$Project = (Join-Path $PSScriptRoot "..\MDK-ARM\stm32_firmware.uvprojx"),
    [string]$Target = "stm32_firmware"
)

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path -LiteralPath $Project).Path
$projectDir = Split-Path -Parent $projectPath
$logPath = Join-Path $projectDir "build_keil.log"

$candidates = @()
if ($env:KEIL_UV4) { $candidates += $env:KEIL_UV4 }
$candidates += @(
    "C:\Keil_v5\UV4\UV4.exe",
    "C:\Keil\UV4\UV4.exe",
    "C:\Program Files\Keil_v5\UV4\UV4.exe"
)
$uv4 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $uv4) { throw "UV4.exe not found; set KEIL_UV4 to its full path" }

if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }
$arguments = @("-b", $projectPath, "-t", $Target, "-j0", "-o", $logPath)
$process = Start-Process -FilePath $uv4 -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
if ($process.ExitCode -ne 0) { throw "Keil exited with code $($process.ExitCode)" }
if (-not (Test-Path -LiteralPath $logPath)) { throw "Keil did not create $logPath" }

$log = Get-Content -LiteralPath $logPath -Raw
$summary = [regex]::Match($log, '(\d+) Error\(s\), (\d+) Warning\(s\)')
if (-not $summary.Success) { throw "Cannot find Keil error/warning summary in $logPath" }
$errors = [int]$summary.Groups[1].Value
$warnings = [int]$summary.Groups[2].Value
if ($errors -ne 0 -or $warnings -ne 0) {
    Get-Content -LiteralPath $logPath
    throw "Keil build failed quality gate: $errors error(s), $warnings warning(s)"
}

$outDir = Join-Path $projectDir "stm32_firmware"
$artifacts = @("stm32_firmware.hex", "stm32_firmware.axf", "stm32_firmware.map")
Write-Host "Keil build passed: 0 errors, 0 warnings"
foreach ($name in $artifacts) {
    $path = Join-Path $outDir $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing build artifact: $path" }
    $item = Get-Item -LiteralPath $path
    Write-Host ("{0}  {1} bytes" -f $item.FullName, $item.Length)
}

$mapPath = Join-Path $outDir "stm32_firmware.map"
$map = Get-Content -LiteralPath $mapPath
foreach ($symbol in @("g_system_ticks", "g_imu", "g_mahony", "g_servos", "g_safety_state")) {
    $line = $map | Select-String -Pattern ("^\s+" + [regex]::Escape($symbol) + "\s+(0x[0-9a-fA-F]+)\s+Data\b") | Select-Object -First 1
    if ($line) { Write-Host ("{0}={1}" -f $symbol, $line.Matches[0].Groups[1].Value) }
}

Write-Host "build_log=$logPath"
