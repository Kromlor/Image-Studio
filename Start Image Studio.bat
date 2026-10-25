@echo off
title Image Studio
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" goto :run
echo.
echo  Image Studio isn't installed yet.
echo  Double-click "Install Image Studio" first.
echo.
pause
exit /b 1

:run
set "PIP_CACHE_DIR=%~dp0pip-cache"
rem ---- after an update, install any new app parts (only when requirements.txt changed) ----
fc /b requirements.txt "venv\installed-requirements.txt" >nul 2>&1
if not errorlevel 1 goto :start
echo.
echo  Updating Image Studio's parts - this happens once after an update...
"venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 goto :updatefail
copy /y requirements.txt "venv\installed-requirements.txt" >nul

:start
echo.
echo  Image Studio is running. Your browser will open in a moment.
echo  Keep this window open while you use it.
echo  Close this window to shut Image Studio down.
echo.
"venv\Scripts\python.exe" server.py
if errorlevel 1 pause
exit /b 0

:updatefail
echo.
echo  The update couldn't finish - check your internet connection.
echo  Starting with what's already installed...
goto :start
