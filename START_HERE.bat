@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title FVG Predictive Strength - Research Lab

set "FVG_PYTHON="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>nul && set "FVG_PYTHON=py -3.13"
  if not defined FVG_PYTHON py -3.12 -c "import sys" >nul 2>nul && set "FVG_PYTHON=py -3.12"
  if not defined FVG_PYTHON py -3.11 -c "import sys" >nul 2>nul && set "FVG_PYTHON=py -3.11"
)

if not defined FVG_PYTHON (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
    if not errorlevel 1 set "FVG_PYTHON=python"
  )
)

if not defined FVG_PYTHON (
  echo.
  echo Python 3.11, 3.12, or 3.13 ^(64-bit^) is required.
  echo Install it from https://www.python.org/downloads/
  echo Enable "Add Python to PATH" during installation, then try again.
  echo.
  pause
  exit /b 1
)

call %FVG_PYTHON% "%~dp0bootstrap.py"
set "FVG_EXIT=%errorlevel%"
if not "%FVG_EXIT%"=="0" (
  echo.
  echo The launcher stopped with error code %FVG_EXIT%.
  echo Read the message above; this window will remain open.
  pause
)
endlocal & exit /b %FVG_EXIT%
