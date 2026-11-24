@echo off
cd /d "%~dp0"

if not exist "..\traincompyuv" mkdir "..\traincompyuv"

x265 --input-res 480x320 --fps 30 --input "..\trainrawyuv\BSD500train.yuv" -o train.hevc --crf 35 --keyint 1 --preset medium
if errorlevel 1 goto error

TAppDecoder.exe -b train.hevc -o "..\traincompyuv\BSD500trainc.yuv"
if errorlevel 1 goto error

echo.
echo Done: yuvcompression\train.hevc -^> traincompyuv\BSD500trainc.yuv
pause
exit /b 0

:error
echo.
echo Error: check x265.exe / TAppDecoder.exe and trainrawyuv\BSD500train.yuv
pause
exit /b 1
