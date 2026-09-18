@echo off
setlocal
cd /d "%~dp0"
title FVG Predictive Strength - Research Lab

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 bootstrap.py
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python bootstrap.py
    goto :end
)

echo.
echo Python 3.10 or newer is required.
echo Install Python from https://www.python.org/downloads/
echo During installation, enable "Add Python to PATH".
echo Then double-click START_HERE.bat again.
echo.
pause

:end
if errorlevel 1 (
  echo.
  echo The launcher stopped because of an error. Read the message above.
  pause
)
endlocal
