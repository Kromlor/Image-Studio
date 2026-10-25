@echo off
setlocal EnableExtensions
title Image Studio - One-time install
cd /d "%~dp0"

rem Keep every download and temp file on this drive, next to the app.
if not exist "%~dp0pip-cache" mkdir "%~dp0pip-cache"
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "PIP_CACHE_DIR=%~dp0pip-cache"
set "TMP=%~dp0tmp"
set "TEMP=%~dp0tmp"

echo ==========================================================
echo    Image Studio v1.1 - one-time install
echo ==========================================================
echo  This downloads about 35 GB and can take 20 to 60 minutes.
echo  Leave this window open until it says INSTALL COMPLETE.
echo  If it gets interrupted, just run it again - it resumes.
echo.

rem ---- the app's own Python already set up? skip ahead ----
if exist "venv\Scripts\python.exe" goto :haveenv

call :findpy
if defined PY goto :makeenv

rem ---- no usable Python 3.11-3.13: install 3.12 ----
where py >nul 2>&1
if errorlevel 1 goto :wingetpy
echo  Installing Python 3.12 with the Python install manager...
py install 3.12
call :findpy
if defined PY goto :makeenv

:wingetpy
echo  Installing Python 3.12 with winget...
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
call :findpy
if defined PY goto :makeenv
echo.
echo  Python 3.12 is installed, but this window can't see it yet.
echo  Close this window and double-click "Install Image Studio" again.
echo.
pause
exit /b 0

:makeenv
echo  Using Python: %PY%
echo.
echo  [1/4] Setting up a private Python space for the app...
%PY% -m venv venv
if not exist "venv\Scripts\python.exe" goto :fail

:haveenv
set "VPY=%~dp0venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet

echo.
echo  [2/4] Installing the graphics-card engine - about 3 GB...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fail

echo.
echo  [3/4] Installing the app parts...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo  [4/4] Downloading the FLUX Schnell picture engine - about 33 GB...
"%VPY%" setup_model.py
if errorlevel 1 goto :fail

echo.
echo ==========================================================
echo    INSTALL COMPLETE
echo ==========================================================
echo  Double-click "Start Image Studio" to begin.
echo  The very first start takes a few extra minutes while the
echo  engine is packed to fit your graphics card. After that it
echo  starts much faster.
echo.
pause
exit /b 0

rem ---- find a Python that really runs: 3.12, then 3.11, then 3.13 ----
:findpy
set "PY="
for %%V in (3.12 3.11 3.13) do call :trypy %%V
exit /b 0

:trypy
if defined PY exit /b 0
set "OUT="
for /f "delims=" %%A in ('py -%1 -c "print(4242)" 2^>nul') do set "OUT=%%A"
if "%OUT%"=="4242" set "PY=py -%1"
exit /b 0

:fail
echo.
echo  ----------------------------------------------------------
echo   The install stopped before finishing.
echo   Run it again - it picks up where it left off.
echo   If it stops at the same spot twice, take a screenshot of
echo   this window and send it to Claude.
echo  ----------------------------------------------------------
echo.
pause
exit /b 1
