@echo off
echo ============================================
echo   Smart Helmet - Push to Raspberry Pi
echo ============================================
echo.

set PI=ajith@sthavishtah
set SRC=C:\Users\ajith\OneDrive\Desktop\a123\layer_pi

echo [1/3] Deleting old files on Pi...
ssh %PI% "rm -rf ~/smart_helmet/layer_pi && mkdir -p ~/smart_helmet"
if %errorlevel% neq 0 (
    echo ERROR: Could not connect to Pi.
    pause
    exit /b 1
)
echo      Done!
echo.

echo [2/3] Pushing layer_pi to Pi...
scp -r "%SRC%" %PI%:~/smart_helmet/layer_pi
if %errorlevel% neq 0 (
    echo ERROR: File transfer failed.
    pause
    exit /b 1
)
echo      Done!
echo.

echo [3/3] Installing dependencies on Pi...
ssh %PI% "cd ~/smart_helmet/layer_pi && pip install -r requirements.txt 2>/dev/null; echo 'Dependencies done'"
echo      Done!
echo.

echo ============================================
echo   ALL DONE! layer_pi is on the Pi.
echo ============================================
pause
