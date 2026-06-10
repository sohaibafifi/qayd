# Register this qayd MiniZinc bundle with the local MiniZinc installation.
#
# Standalone: works from wherever the bundle was unpacked, no toolchain
# needed. Writes %USERPROFILE%\.minizinc\solvers\qayd.msc with absolute paths
# into this directory, so the bundle must stay where it is after installing
# (re-run this script if you move it).
#
# Run with:  powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$SolversDir = Join-Path $env:USERPROFILE ".minizinc\solvers"

if (-not (Test-Path (Join-Path $Here "qayd-fzn.exe"))) {
    Write-Error "qayd-fzn.exe not found next to install.ps1"
}
if (-not (Test-Path (Join-Path $Here "mznlib"))) {
    Write-Error "mznlib\ not found next to install.ps1"
}

New-Item -ItemType Directory -Force -Path $SolversDir | Out-Null

# The bundled qayd.msc uses relative paths; rewrite them as absolute so the
# installed copy works from any working directory. JSON wants forward
# slashes, which MiniZinc accepts on Windows.
$HereFs = $Here -replace "\\", "/"
$Msc = Get-Content (Join-Path $Here "qayd.msc") -Raw
$Msc = $Msc -replace '"\./qayd-fzn"', ('"' + $HereFs + '/qayd-fzn.exe"')
$Msc = $Msc -replace '"\./mznlib"', ('"' + $HereFs + '/mznlib"')
Set-Content -Path (Join-Path $SolversDir "qayd.msc") -Value $Msc

Write-Host "installed: $SolversDir\qayd.msc"
Write-Host "try: minizinc --solver qayd <model.mzn>"
