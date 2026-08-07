# Docker VHDX compact script - fstrim + proper diskpart approach
# Run as Administrator: powershell -ExecutionPolicy Bypass -File compact_docker.ps1

Write-Host "=== Docker VHDX Compactor ===" -ForegroundColor Cyan
Write-Host ""

if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "[ERROR] Run as Administrator!" -ForegroundColor Red
    exit 1
}

$VHDX = "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx"

Write-Host "[1/5] Stopping Docker + WSL..." -ForegroundColor Yellow
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
wsl --shutdown 2>$null
Start-Sleep -Seconds 3
Write-Host "  Done" -ForegroundColor Green

$old = [math]::Round((Get-Item $VHDX).Length / 1GB, 2)
Write-Host "  Current: ${old} GB" -ForegroundColor Yellow

Write-Host "[2/5] Attaching VHDX to Ubuntu..." -ForegroundColor Yellow
wsl --mount $VHDX --vhd --bare 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Failed to attach VHDX" -ForegroundColor Red
    exit 1
}
Start-Sleep -Seconds 2
Write-Host "  Attached" -ForegroundColor Green

Write-Host "[3/5] Mounting + fstrim..." -ForegroundColor Yellow
# Find the new disk (last disk device without mountpoint)
$result = wsl -d Ubuntu -u root sh -c "
DEVS=\$(lsblk -o NAME,TYPE,MOUNTPOINT -n | grep 'disk' | grep -v '/' | tail -1 | cut -d' ' -f1)
echo \"DISK=/\$DEVS\"
mkdir -p /mnt/dd
mount -o discard /dev/\$DEVS /mnt/dd 2>/dev/null || mount -o discard /dev/\${DEVS}1 /mnt/dd 2>/dev/null
if [ \$? -ne 0 ]; then
    echo 'MOUNT_FAILED'
    exit 1
fi
df -h /mnt/dd | tail -1
fstrim -v /mnt/dd
umount /mnt/dd
echo 'OK'
" 2>&1
Write-Host $result
Write-Host "  fstrim done" -ForegroundColor Green

Write-Host "[4/5] Detaching from WSL..." -ForegroundColor Yellow
wsl --unmount 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Host "  Detached" -ForegroundColor Green

Write-Host "[5/5] diskpart compact (no attach, no readonly)..." -ForegroundColor Yellow
@"
select vdisk file="$VHDX"
compact vdisk
exit
"@ | diskpart 2>&1 | Out-Null
Write-Host "  Compact done" -ForegroundColor Green

Write-Host ""
$new = [math]::Round((Get-Item $VHDX).Length / 1GB, 2)
$saved = $old - $new
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Before: ${old} GB" -ForegroundColor Yellow
Write-Host "  After:  ${new} GB" -ForegroundColor Green
Write-Host "  Saved:  ${saved} GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($saved -le 1) {
    Write-Host "Compaction saved little space. The VHDX is internally" -ForegroundColor Red
    Write-Host "fragmented. Recommend the nuclear option:" -ForegroundColor Red
    Write-Host "  1. Export postgres data (pg_dump)" -ForegroundColor White
    Write-Host "  2. Delete docker_data.vhdx" -ForegroundColor White
    Write-Host "  3. Restart Docker Desktop (creates clean VHDX)" -ForegroundColor White
    Write-Host "  4. Rebuild images and restore DB" -ForegroundColor White
} else {
    Write-Host "Start Docker Desktop to resume work." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
