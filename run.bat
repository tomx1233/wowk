@echo off
REM Timbreforge launcher (Windows).
REM Creates .venv on first run, installs requirements, then starts the app.
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    py -3 -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    echo Installing dependencies ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

".venv\Scripts\python.exe" -m timbreforge %*
goto :eof

:error
echo.
echo Setup failed. Check that Python 3.10+ is installed and that you have
echo network access for the first dependency install.
pause
exit /b 1
