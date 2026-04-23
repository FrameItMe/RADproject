@echo off
setlocal enabledelayedexpansion

cd /d %~dp0

where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found. Install Python 3 first.
    pause
    exit /b 1
  )
  set PYTHON=python
) else (
  set PYTHON=py -3
)

if not exist .venv (
  %PYTHON% -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist logs mkdir logs
start "Yanoey Flask Server" /b cmd /c "python app.py > logs\\server.log 2>&1"

timeout /t 3 /nobreak >nul
start http://127.0.0.1:5050

echo Server started. Close this window to stop the app.
pause >nul