@echo off
cd /d "%~dp0"

if not exist "..\testcompyuv" mkdir "..\testcompyuv"

x265 --input-res 480x320 --fps 30 --input "..\testrawyuv\BSD500val.yuv" -o val.hevc --crf 35 --keyint 1 --preset medium
if errorlevel 1 goto error

TAppDecoder.exe -b val.hevc -o "..\testcompyuv\BSD500valc.yuv"
if errorlevel 1 goto error

echo.
echo Done: yuvcompression\val.hevc -^> testcompyuv\BSD500valc.yuv
pause
exit /b 0

:error
echo.
echo Error: check x265.exe / TAppDecoder.exe and testrawyuv\BSD500val.yuv
pause
exit /b 1
