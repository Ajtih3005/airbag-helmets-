param (
    [string]$Target = "raspberrypi.local",
    [string]$User = "pi"
)

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  SMART AIRBAG HELMET -- PUSH PACKAGE TO PI" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Bundling latest code into layer_pi/..." -ForegroundColor Yellow
python layer_pi/bundle_for_pi.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Bundling failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/4] Cleaning old files on Pi (${User}@${Target}:~/smart_helmet/layer_pi)..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no "${User}@${Target}" "mkdir -p ~/smart_helmet && rm -rf ~/smart_helmet/layer_pi"

Write-Host "`n[3/4] SCP transfer of fresh layer_pi/ to ${User}@${Target}:~/smart_helmet/..." -ForegroundColor Yellow
scp -o StrictHostKeyChecking=no -r layer_pi "${User}@${Target}:~/smart_helmet/"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] SCP transfer failed." -ForegroundColor Red
    Write-Host "Check that the Pi is reachable, SSH is enabled, and Wi-Fi is connected." -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] Clean deployment complete!" -ForegroundColor Green
Write-Host "`nTo run on Raspberry Pi:" -ForegroundColor White
Write-Host "  ssh ${User}@${Target}" -ForegroundColor Gray
Write-Host "  cd ~/smart_helmet/layer_pi" -ForegroundColor Gray
Write-Host "  python pi_ws_client.py --server ws://<LAPTOP_IP>:5500/ws --hardware --arm" -ForegroundColor Gray
