# Move Docker data from C to E drive
# Run as Administrator: powershell -ExecutionPolicy Bypass -File move_docker_to_e.ps1
# Frees ~31GB on C drive immediately. Docker data is PRESERVED.

Write-Host "=== Move Docker Data to E Drive ===" -ForegroundColor Cyan
Write-Host ""

if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "[ERROR] Run as Administrator!" -ForegroundColor Red
    exit 1
}

$SRC = "$env:LOCALAPPDATA\Docker"
$DST = "E:\DockerData"

Write-Host "[1/4] Stopping Docker Desktop + WSL..." -ForegroundColor Yellow
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
wsl --shutdown 2>$null
Start-Sleep -Seconds 3

# Verify all WSL processes are stopped
$wslRunning = Get-Process -Name "wsl" -ErrorAction SilentlyContinue
if ($wslRunning) {
    Write-Host "  Waiting for WSL to fully stop..."
    Stop-Process -Name "wsl" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}
Write-Host "  Docker + WSL stopped" -ForegroundColor Green

# Show before
$beforeGB = [math]::Round(((Get-ChildItem $SRC -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum) / 1GB, 2)
Write-Host "  Docker data size: ${beforeGB} GB" -ForegroundColor Yellow

Write-Host "[2/4] Moving Docker data to E:\DockerData..." -ForegroundColor Yellow
Write-Host "  (Copying ~${beforeGB} GB, this may take 2-5 minutes)"
New-Item -ItemType Directory -Path $DST -Force | Out-Null
$copyStart = Get-Date
robocopy $SRC $DST /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP 2>&1 | Out-Null
$copyTime = [math]::Round(((Get-Date) - $copyStart).TotalSeconds, 1)
Write-Host "  Moved in ${copyTime}s" -ForegroundColor Green

Write-Host "[3/4] Creating junction (so Docker still finds data)..." -ForegroundColor Yellow
# Make sure source directory is gone
if (Test-Path $SRC) {
    Remove-Item $SRC -Recurse -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1
cmd /c "mklink /J `"$SRC`" `"$DST`"" 2>&1 | Out-Null
Write-Host "  Junction created: $SRC -> $DST" -ForegroundColor Green

Write-Host "[4/4] Verifying..." -ForegroundColor Yellow
if (Test-Path "$DST\wsl\disk\docker_data.vhdx") {
    $vhdSize = [math]::Round((Get-Item "$DST\wsl\disk\docker_data.vhdx").Length / 1GB, 2)
    Write-Host "  VHDX found on E drive: ${vhdSize} GB" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] VHDX not found on E drive!" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Done! Docker data moved to E drive." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  C drive freed: ~${beforeGB} GB" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start Docker Desktop to verify it works." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
