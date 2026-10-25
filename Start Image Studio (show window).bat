@echo off
title Image Studio
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo  Image Studio isn't installed yet. Double-click "Install Image Studio" first.
  pause
  exit /b 1
)
echo.
echo  Image Studio is running with its window showing (for troubleshooting).
echo  Keep this window open while you use it. Close it to shut Image Studio down.
echo.
"venv\Scripts\python.exe" server.py
if errorlevel 1 pause
