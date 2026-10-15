@echo off
setlocal

echo ===================================================
echo   SMART AIRBAG HELMET -- PUSH PACKAGE TO RASPBERRY PI
echo ===================================================

set PI_TARGET=%1
if "%PI_TARGET%"=="" (
    set /p PI_TARGET="Enter Raspberry Pi hostname or IP [default: raspberrypi.local]: "
)
if "%PI_TARGET%"=="" (
    set PI_TARGET=raspberrypi.local
)

set PI_USER=%2
if "%PI_USER%"=="" (
    set PI_USER=pi
)

echo.
echo [1/4] Bundling latest code into layer_pi/...
python layer_pi/bundle_for_pi.py
if errorlevel 1 (
    echo [ERROR] Bundling failed. Check Python environment.
    exit /b 1
)

echo.
echo [2/4] Cleaning old files on Pi (%PI_USER%@%PI_TARGET%:~/smart_helmet/layer_pi)...
ssh -o StrictHostKeyChecking=no %PI_USER%@%PI_TARGET% "mkdir -p ~/smart_helmet && rm -rf ~/smart_helmet/layer_pi"
if errorlevel 1 (
    echo [WARN] Could not clean remote directory via SSH (proceeding to copy)...
)

echo.
echo [3/4] Transferring fresh layer_pi/ to %PI_USER%@%PI_TARGET%:~/smart_helmet/...
scp -o StrictHostKeyChecking=no -r layer_pi %PI_USER%@%PI_TARGET%:~/smart_helmet/
if errorlevel 1 (
    echo.
    echo [ERROR] SCP transfer failed.
    echo Make sure:
    echo   1. The Pi is powered on and connected to the same Wi-Fi/network.
    echo   2. SSH is enabled on the Pi (sudo raspi-config -> Interfaces -> SSH).
    echo   3. The IP or hostname '%PI_TARGET%' is reachable (try: ping %PI_TARGET%).
    exit /b 1
)

echo.
echo [4/4] Clean deployment complete!
echo.
echo To run on the Raspberry Pi:
echo   ssh %PI_USER%@%PI_TARGET%
echo   cd ~/smart_helmet/layer_pi
echo   pip install -r requirements.txt
echo   # For hardware testing with real relay on Pin 17 (auto-closes in 10s):
echo   python pi_ws_client.py --server ws://<YOUR_LAPTOP_IP>:5500/ws --hardware --arm
echo ===================================================
endlocal
